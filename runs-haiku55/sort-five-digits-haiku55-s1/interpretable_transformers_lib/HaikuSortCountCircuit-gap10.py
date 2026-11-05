"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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

# Residual-stream channel layout used by write_weights (d_model = 64)
D = 64
TOK = 0       # 0..10  token one-hot (digits 0-9, '=' is 10)
POS = 11      # 11..26 position one-hot (16 positions)
CNT = 27      # 27..35 C_m = #input digits <= m, m = 0..8 (written by attention)
ZCH = 36      # companion channel = -sum_m C_m (keeps the residual sum constant for LayerNorm)
OUT = 37      # 37..46 one-hot of the sorted digit at the current output step
PAD = 47      # 47..63 constant pad that pins the LayerNorm scale
PAD_VAL = 100.0       # magnitude of the zero-mean constant pad (sets LayerNorm scale)
GATE_BIG = 30.0       # strength of the position gate on MLP units
ATTN_GAP = 10.0       # score gap between input-position keys and all other keys
Z_COMPANION = True    # write -sum_m C_m into channel ZCH (keeps residual sum constant)
N_PAD_CH = 8          # number of pad channels (half +PAD_VAL, half -PAD_VAL)
EQ_SUPPRESS = False   # head row for '=' is -pad (else logit 0)
CANCEL_OFFSET = True  # cancel the LayerNorm mean offset inside attention values
DROP_NINE = False     # omit the constant unit that handles digit 9
N_UNITS = 95  # 9 thresholds x 5 steps x 2 ReLUs + 5 gate units

# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "HaikuSortCountCircuit-gap10"
model_description = "Base (K=100, 8 pads, no '=' suppression) with attention score gap 10 (was 40). Tests how sharp the input-only attention must be between 0 and 40."


def _uidx(m: int, j: int, k: int) -> int:
    return (m * 5 + j) * 2 + k


def _cidx(j: int) -> int:
    return 90 + j


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
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
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 95,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
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
    """Populate `model`'s parameters in-place. No training allowed.

    Default implementation: leave the random initialization untouched.
    The agent should replace this with hand-designed weights that implement
    the task (e.g. 5-digit addition).
    """
    # Keep deterministic so reruns of the same code produce the same model.
    torch.manual_seed(0)
    assert model.n_layers == 1 and model.n_heads == 1 and model.d_model == D and model.d_ff == N_UNITS
    assert task.stoi["="] == 10 and all(task.stoi[str(d)] == d for d in range(10))
    K = PAD_VAL
    mu0 = 2.0 / D                                  # zero-mean pad => mean of residual at input is 2/D
    sigma0 = math.sqrt((2 + N_PAD_CH * K * K) / D - mu0 * mu0)
    B = 8.0 * ATTN_GAP / (K - mu0)                 # key gain: input-position keys get +ATTN_GAP score
    BIG = GATE_BIG
    cancel = 1.0 if CANCEL_OFFSET else 0.0

    with torch.no_grad():
        # ---- embeddings ----
        E = torch.zeros_like(model.token_emb.weight)
        for t in range(model.vocab_size):
            E[t, t] = 1.0                          # token one-hot
        h_pad = N_PAD_CH // 2
        E[:, PAD:PAD + h_pad] = K                  # constant zero-mean pad (+K half, -K half)
        E[:, PAD + h_pad:PAD + N_PAD_CH] = -K
        model.token_emb.weight.copy_(E)
        Pe = torch.zeros_like(model.pos_emb.weight)
        for p in range(Pe.shape[0]):
            Pe[p, POS + p] = 1.0                   # position one-hot
        model.pos_emb.weight.copy_(Pe)

        # ---- LayerNorms: gamma = sigma0 so LN(x) = x - mu0 (exact, since norms are constant) ----
        blk = model.blocks[0]
        for ln in (blk.ln1, blk.ln2, model.final_ln):
            ln.weight.fill_(sigma0)
            ln.bias.zero_()

        # ---- attention: uniform over the 5 input positions, value = thermometer of digit ----
        Wq = torch.zeros(D, D); Wk = torch.zeros(D, D); Wv = torch.zeros(D, D); Wo = torch.zeros(D, D)
        Wq[0, PAD] = 1.0                           # constant query
        for p in range(5):
            Wk[0, POS + p] = B                     # keys at input positions get score gap 40
        for m in range(9):
            r = CNT + m
            Wv[r, : m + 1] = 1.0                   # [digit <= m]
            Wv[r, PAD] = cancel * mu0 * (m + 1) / (K - mu0)  # cancels constant offset from LN
        if Z_COMPANION:
            Wv[ZCH, :10] = torch.arange(10, dtype=torch.float32) - 9.0   # -(sum_m C_m) companion
            Wv[ZCH, PAD] = cancel * mu0 * (-45.0) / (K - mu0)
        for r in range(CNT, ZCH + 1):
            Wo[r, r] = 5.0                         # undo the 1/5 averaging
        if not Z_COMPANION:
            Wo[ZCH, ZCH] = 0.0
        blk.attn.W_q.weight.copy_(Wq)
        blk.attn.W_k.weight.copy_(Wk)
        blk.attn.W_v.weight.copy_(Wv)
        blk.attn.W_o.weight.copy_(Wo)

        # ---- MLP: threshold units gated by output position ----
        W1 = torch.zeros(N_UNITS, D); b1 = torch.zeros(N_UNITS)
        W2 = torch.zeros(D, N_UNITS)
        for m in range(9):
            for j in range(5):
                for k in range(2):
                    i = _uidx(m, j, k)
                    W1[i, CNT + m] = 1.0
                    W1[i, POS + 5 + j] = BIG
                    b1[i] = (-j - k - BIG) + mu0 * (1 + BIG)     # relu(C_m - j - k + BIG*(pos-1))
        for j in range(5):
            i = _cidx(j)
            W1[i, POS + 5 + j] = 1.0
            b1[i] = mu0                                          # relu(pos_{5+j})
        for j in range(5):
            for d in range(10):
                o = OUT + d
                if d <= 8:
                    W2[o, _uidx(d, j, 0)] += 1.0
                    W2[o, _uidx(d, j, 1)] -= 1.0
                if d == 9 and not DROP_NINE:
                    W2[o, _cidx(j)] += 1.0
                if d >= 1:
                    W2[o, _uidx(d - 1, j, 0)] -= 1.0
                    W2[o, _uidx(d - 1, j, 1)] += 1.0
        blk.mlp.fc1.weight.copy_(W1); blk.mlp.fc1.bias.copy_(b1)
        blk.mlp.fc2.weight.copy_(W2); blk.mlp.fc2.bias.zero_()

        # ---- readout: logit_d = output channel d; '=' suppressed ----
        Wh = torch.zeros_like(model.head.weight)
        for d in range(10):
            Wh[d, OUT + d] = 1.0
        Wh[10, PAD] = -1.0 if EQ_SUPPRESS else 0.0
        model.head.weight.copy_(Wh)



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
