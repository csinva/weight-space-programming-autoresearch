"""
Interpretable transformer for decimal-to-binary-8bit.

Task: prompt "068=" -> answer "01000100" (8-bit binary of the 3-digit decimal, MSB first).
Inputs cover 0..255. Layout: positions 0,1,2 = hundreds, tens, units digits; 3 = '='.
Answer bit k is predicted at position 3+k.

Rules of the game:
  * No training: all weights are written in closed form in write_weights().
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


# ---------------------------------------------------------------------------
# Agent's knobs: residual layout and sweep/ablation switches (defaults = full circuit)
# ---------------------------------------------------------------------------

TOK = 0                 # 0..10  token one-hot (digits 0-9, '=' = 10)
POS = 11                # 11..26 position one-hot
N_POS = 16              # number of position channels
SC = POS + N_POS        # 27: scalar N = 100h + 10t + u, gathered by attention
GB = SC + 1             # 28..35: answer bit k of N (k=0 is MSB)
O1B = GB + 8            # 36..43: gated indicator "bit k is 1" at answer slot k
D_MODEL = 45            # residual width (>= O1B + 8 = 44, divisible by 3 heads)
Q_SCALE = 8.0           # attention sharpness (score = Q_SCALE^2 / sqrt(d_head))
PLACE = (100, 10, 1)    # place value of the hundreds, tens, units digit
KINKS = tuple(range(255))  # kink thresholds n in ReLU(N - n); unit count = len(KINKS)
KINK_SHIFT = 0.0        # added to every kink bias (0 = exact)
LOGIT_SCALE = 10.0      # head gain (logit margin)
GATE_BY_SLOT = False    # gate bit indicators by answer-slot one-hot (False = ablation)
HEADS_ON = (True, True, True)  # which gather heads write their digit (False = ablation)
POS_EMB_ON = True       # position embeddings on (False = ablation)
LSB_FROM_ATTN = False   # if True, bit 7 (units parity) comes from attention instead of kinks
NOISE_SIGMA = 0.0       # multiplicative weight noise (robustness test)


# ---------------------------------------------------------------------------
# Architecture (no LayerNorm: every block is a plain residual stream)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = D_MODEL,
        n_heads: int = 3,
        d_ff: tuple = (len(KINKS), 8),
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = len(d_ff)
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, f) for f in d_ff])
        self.final_ln = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training."""
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        dh = model.d_model // model.n_heads

        # Embeddings: one-hot token and position in fixed channels.
        for d in range(model.vocab_size):
            model.token_emb.weight[d, TOK + d] = 1.0
        if POS_EMB_ON:
            for p in range(min(N_POS, model.max_seq_len)):
                model.pos_emb.weight[p, POS + p] = 1.0

        # Block 0 attention: head j reads digit j (position j) and writes place_value * digit into SC.
        # Summing the three heads gives N = 100h + 10t + u in one scalar channel.
        attn = model.blocks[0].attn
        for j in range(3):
            base = j * dh
            for i in range(11):                      # query: constant (token one-hot sums to 1)
                attn.W_q.weight[base, TOK + i] = Q_SCALE
            attn.W_k.weight[base, POS + j] = Q_SCALE  # key: only position j is attended
            if HEADS_ON[j]:
                for d in range(10):
                    attn.W_v.weight[base + 1, TOK + d] = float(PLACE[j] * d)
                attn.W_o.weight[SC, base + 1] = 1.0
        if LSB_FROM_ATTN:
            base = 2 * dh                            # head 2 also carries units parity into bit 7
            for d in range(10):
                attn.W_v.weight[base + 2, TOK + d] = float(d % 2)
            attn.W_o.weight[GB + 7, base + 2] = 1.0

        # Block 0 MLP: one shared layer of ReLU kinks in N, then every bit is a linear read-out.
        # For integer N in [0,255] any f(N) = f(0) + sum_n D_n * ReLU(N - n), with
        # D_n = second difference of f at n. Unit i = ReLU(N - KINKS[i]).
        mlp0 = model.blocks[0].mlp
        for i, n in enumerate(KINKS):
            mlp0.fc1.weight[i, SC] = 1.0
            mlp0.fc1.bias[i] = KINK_SHIFT - float(n)
        for k in range(8):
            if LSB_FROM_ATTN and k == 7:
                continue
            fk = [float((S >> (7 - k)) & 1) for S in range(256)]
            for i, n in enumerate(KINKS):
                first = fk[n + 1] - fk[n]
                prev = (fk[n] - fk[n - 1]) if n >= 1 else 0.0
                mlp0.fc2.weight[GB + k, i] = first - prev
            mlp0.fc2.bias[GB + k] = fk[0]

        # Block 1 MLP: gate by answer slot. Slot k lives at position 3+k.
        mlp1 = model.blocks[1].mlp
        for k in range(8):
            mlp1.fc1.weight[k, GB + k] = 1.0          # unit k: bit k == 1 (and slot k if gated)
            if GATE_BY_SLOT:
                mlp1.fc1.weight[k, POS + 3 + k] = 1.0
            mlp1.fc1.bias[k] = -1.0 if GATE_BY_SLOT else 0.0
            mlp1.fc2.weight[O1B + k, k] = 1.0

        # Head: '1' logit = gated indicator. '0' logit = gain at an answer slot, minus gain if the indicator fired.
        for k in range(8):
            model.head.weight[0, POS + 3 + k] = LOGIT_SCALE
            model.head.weight[0, O1B + k] = -LOGIT_SCALE
            model.head.weight[1, O1B + k] = LOGIT_SCALE

        if NOISE_SIGMA > 0:
            gen = torch.Generator().manual_seed(0)
            for p in model.parameters():
                p.mul_(1.0 + NOISE_SIGMA * torch.randn(p.shape, generator=gen))


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "Abl_NoSlotGate"
model_description = "Ablation of LeanPWL: remove answer-slot gating (block-1 units fire on the bit alone). Shows the position one-hot is essential."


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def upsert_overall_results(rows: list[dict], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["task"]) for r in rows}
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_shorthand_name"), row.get("task")) not in new_keys:
                    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows])
    print(f"Overall results saved → {path}")



def build_model(task) -> SimpleTransformer:
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task name (see src/task.py)", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task":        args.task,
        "accuracy":    f"{accuracy:.4f}",
        "status":      "",
        "model_shorthand_name":  model_shorthand_name,
        "n_params":    f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
