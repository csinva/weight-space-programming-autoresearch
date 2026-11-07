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


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

N_CAND = 150  # candidate divisors d = 1..N_CAND (one head-slot / MLP group per d)


# exact number of tent neurons: 2 numbers x (#targets k in {0,d,2d} <= 3(d-1)) x 3 relus per tent
N_TENTS = sum(2 * (1 + 3 * (len(range(0, 3 * (d - 1) + 1, d)) - 1)) for d in range(1, N_CAND + 1))
JOINT = True   # one tent bank on V = S_a + 3d*S_b tests (d|a and d|b) at once -> no separate AND layer
N_TENTS_J = sum(1 + 3 * (len([(i, j) for i in range(3) for j in range(3) if i * d <= 3 * (d - 1) and j * d <= 3 * (d - 1)]) - 1)
                for d in range(1, N_CAND + 1))
ATT_SHARP = 40.0  # attention logit gap between the source position and all others
USE_NULL_FIX = True # ablation switch: LayerNorm mean correction through the always-zero NULL dim
LN_PIN = 2.0e6  # magnitude C of the +/-C dims that pin the LayerNorm scale


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)

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
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int, has_attn: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model) if has_attn else None
        self.attn = CausalSelfAttention(d_model, n_heads, d_head) if has_attn else None
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = (3 if JOINT else 4) * N_CAND + 42,
        n_heads: int = 6,
        d_head: int = N_CAND,
        d_ffs: tuple = (N_TENTS_J, 3 * N_CAND) if JOINT else (N_TENTS, N_CAND, 3 * N_CAND),
        attn_layers: tuple = (True, False) if JOINT else (True, False, False),
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_layers = len(d_ffs)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_head, f, a) for f, a in zip(d_ffs, attn_layers)]
        )
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
    N = N_CAND                  # candidate divisors d = 1..N
    C = LN_PIN                  # large +/-C constant dims pin the LayerNorm scale
    D = model.d_model
    dh = model.d_head
    assert dh == N and model.n_heads == 6
    # ---- residual stream layout --------------------------------------------
    TOK, POS = 0, 12            # one-hot token (12), one-hot position (16)
    CP, CM, NULL = 28, 29, 30   # +C, -C, and an always-zero dim
    SA, SB = 31, 31 + N         # SA[d]: (a mod d)-ish sum of digit residues; SB likewise for b
    G = 31 + 2 * N              # G[d] = [d|a] + [d|b]  in {0,1,2}
    GG = 31 + (2 if JOINT else 3) * N   # GG[d] = [d|a and d|b]
    OUT = GG + N                # 10 dims: output digit evidence
    assert OUT + 10 <= D
    gamma = math.sqrt(2 * C * C / D)   # makes LayerNorm ~ identity (minus mean)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for ln in [model.final_ln] + [m for b in model.blocks for m in (b.ln1, b.ln2) if m is not None]:
            ln.weight.fill_(gamma)

        for t in range(12):
            model.token_emb.weight[t, TOK + t] = 1.0
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, POS + p] = 1.0
            model.pos_emb.weight[p, CP] = C
            model.pos_emb.weight[p, CM] = -C

        # ---- Layer 0 attention: head i copies the digit at a fixed position ----
        # and writes, for every d, (digit * place) mod d  into SA[d] / SB[d].
        src = [0, 1, 2, 4, 5, 6]
        place = [100, 10, 1, 100, 10, 1]
        att = model.blocks[0].attn
        for i in range(6):
            att.W_q.weight[i * dh, CP] = 1.0 / C
            att.W_k.weight[i * dh, POS + src[i]] = ATT_SHARP * math.sqrt(dh)
            for d in range(1, N + 1):
                for t in range(10):
                    att.W_v.weight[i * dh + d - 1, TOK + t] = (t * place[i]) % d
                base = SA if i < 3 else SB
                att.W_o.weight[base + d - 1, i * dh + d - 1] = 1.0

        # ---- Layer 0 MLP: tent functions detect S in {0, d, 2d} -> divisible ----
        mlp = model.blocks[0].mlp
        u = 0
        if JOINT:
            for d in range(1, N + 1):
                tg = [(i * d, j * d) for i in range(3) for j in range(3)
                      if i * d <= 3 * (d - 1) and j * d <= 3 * (d - 1)]
                for (ka, kb) in tg:
                    kv = ka + 3 * d * kb       # V = S_a + 3d*S_b encodes the pair uniquely
                    if kv == 0:
                        mlp.fc1.weight[u, SA + d - 1] = -1.0
                        mlp.fc1.weight[u, SB + d - 1] = -3.0 * d
                        mlp.fc1.bias[u] = 1.0
                        mlp.fc2.weight[GG + d - 1, u] = 1.0
                        u += 1
                        continue
                    for j, coef in ((1, 1.0), (0, -2.0), (-1, 1.0)):
                        mlp.fc1.weight[u, SA + d - 1] = 1.0
                        mlp.fc1.weight[u, SB + d - 1] = 3.0 * d
                        mlp.fc1.bias[u] = j - kv
                        mlp.fc2.weight[GG + d - 1, u] = coef
                        u += 1
        for d in range(1, N + 1):
            if JOINT:
                break
            for base in (SA, SB):
                for k in range(0, 3 * (d - 1) + 1, d):
                    if k == 0:
                        # S >= 0 is an integer, so [S == 0] = relu(1 - S): one neuron
                        mlp.fc1.weight[u, base + d - 1] = -1.0
                        mlp.fc1.bias[u] = 1.0
                        mlp.fc2.weight[G + d - 1, u] = 1.0
                        u += 1
                        continue
                    for j, coef in ((1, 1.0), (0, -2.0), (-1, 1.0)):
                        mlp.fc1.weight[u, base + d - 1] = 1.0
                        mlp.fc1.bias[u] = j - k
                        mlp.fc2.weight[G + d - 1, u] = coef
                        u += 1
        assert u <= mlp.fc1.out_features, u

        # ---- Layer 1 MLP: AND of the two divisibilities ----
        if not JOINT:
            mlp = model.blocks[1].mlp
            for d in range(1, N + 1):
                mlp.fc1.weight[d - 1, G + d - 1] = 1.0
                mlp.fc1.bias[d - 1] = -1.0
                mlp.fc2.weight[GG + d - 1, d - 1] = 1.0

        # ---- last MLP: pick the largest common divisor; write its digit by position ----
        mlp = model.blocks[-1].mlp
        u = 0
        for d in range(1, N + 1):
            for p, digit in ((7, d // 100), (8, (d // 10) % 10), (9, d % 10)):
                mlp.fc1.weight[u, GG + d - 1] = 1.0
                for d2 in range(d + 1, N + 1):
                    mlp.fc1.weight[u, GG + d2 - 1] = -1.0
                mlp.fc1.weight[u, POS + p] = 1.0
                mlp.fc1.bias[u] = -1.0
                mlp.fc2.weight[OUT + digit, u] = 1.0
                u += 1

        # ---- head ----
        for dg in range(10):
            model.head.weight[dg, OUT + dg] = 1.0

        # ---- LayerNorm mean correction: make every reading row sum to zero via the NULL dim ----
        readers = []
        for b in model.blocks:
            if b.attn is not None:
                readers += [b.attn.W_q.weight, b.attn.W_k.weight, b.attn.W_v.weight]
            readers.append(b.mlp.fc1.weight)
        readers.append(model.head.weight)
        if USE_NULL_FIX:
            for W in readers:
                W[:, NULL] = -W.sum(dim=1)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "JointTent2LayerTight_N150"
model_description = (f"N={N_CAND}, null_fix={USE_NULL_FIX}, C={LN_PIN:g}. Attention (L0 only) gathers digits and writes (digit*place mod d) "
                     "per candidate d; MLP tents test d|a, d|b; AND layer; last MLP selects largest common d and emits its digit by position.")


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
