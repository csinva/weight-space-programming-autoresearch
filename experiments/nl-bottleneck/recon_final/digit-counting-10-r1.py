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

class CausalSelfAttention(nn.Module):
    """Single head, d_head = d_model. Score = x_q^T W_qk x_k; output = attn @ (x W_ov^T)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_qk = nn.Parameter(torch.zeros(d_model, d_model))
        self.W_ov = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        scores = x @ self.W_qk @ x.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        return attn @ (x @ self.W_ov.T)


class Block(nn.Module):
    """Attention-only block: no LayerNorm, no MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class SimpleTransformer(nn.Module):
    """2-layer attention-only transformer, width 3 (cos, sin, flag)."""

    def __init__(self, vocab_size: int, max_seq_len: int = 16, d_model: int = 3, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

C, S, F_ = 0, 1, 2  # channels: cos, sin, flag


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for prm in model.parameters():
        prm.zero_()
    stoi = task.stoi
    eq = stoi["="]
    V = model.vocab_size

    # --- embeddings ---------------------------------------------------------
    tok = model.token_emb.weight
    for d in range(10):
        th = 2 * math.pi * d / 10
        tok[stoi[str(d)], C] = math.cos(th)
        tok[stoi[str(d)], S] = math.sin(th)
    E = (0.0, 2.0)                      # '=' sits off the unit circle (radius 2)
    tok[eq, C], tok[eq, S] = E
    tok[eq, F_] = -1.0                  # flag -1 at '='
    pos = model.pos_emb.weight
    pos[10, F_] = 1.0                   # query digit: flag +1
    P = (2.0, 0.0)                      # tens-digit position pushed off the circle
    pos[12, C], pos[12, S] = P
    pos[12, F_] = -1.0                  # generated tens digit: flag -1

    # --- layer 1: same-digit match, value = flag -> flag --------------------
    # match beats nearest non-match by s*(1 - cos 36deg) = 40 logits
    s1 = 40.0 / (1 - math.cos(2 * math.pi / 10))
    L1 = model.blocks[0].attn
    L1.W_qk[C, C] = s1
    L1.W_qk[S, S] = s1
    L1.W_ov[F_, F_] = 1.0

    # --- layer 2: score -60 f_q f_k, value = flag -> flag -------------------
    L2 = model.blocks[1].attn
    L2.W_qk[F_, F_] = -60.0
    L2.W_ov[F_, F_] = 1.0

    # --- readout: nearest prototype on rho = f + 1 --------------------------
    # logit_v = Cf_v * f + A_v * cos + B_v * sin ; circle channels give constants
    p = [1.0 / (u + 1) for u in range(11)]
    M = 20.0
    Kt0 = torch.full((V,), -M)          # constants at units step (after tens '0')
    Keq = torch.full((V,), -M)          # constants at tens step ('=')
    Cf = torch.zeros(V)
    for u in range(10):
        Cf[stoi[str(u)]] = 2 * p[u]
        Kt0[stoi[str(u)]] = 2 * p[u] - p[u] ** 2   # -> 2 p_u rho - p_u^2
    # tens step: '0' vs '1' threshold midway between prototypes of k=9 and k=10
    thr = 0.5 * (p[9] + p[10])
    Keq[stoi["0"]] = 0.0
    Keq[stoi["1"]] = -(1.0 - thr)       # logit_0 - logit_1 = rho - thr
    t0 = torch.tensor([1.0 + P[0], 0.0 + P[1]])
    e = torch.tensor(E)
    AB = torch.linalg.solve(torch.stack([t0, e]), torch.stack([Kt0, Keq]))  # exact 2x2 inverse
    W = model.head.weight
    W[:, C] = AB[0]
    W[:, S] = AB[1]
    W[:, F_] = Cf
    # After tens '1' (context t1 = 0.936 t0 + 0.294 e) the -M constants on
    # digits >= 2 and the '0'-vs-'1' gap at '=' force units '0'.


model_shorthand_name = "CircleFlagCount_v1"
model_description = ("2-layer attn-only width-3: circle digit matching counts via softmax into flag (1+1/(k+1)); "
                     "layer2 copies query flag to '='/tens (rho-1); nearest-prototype linear readout, circle channels give constants.")


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
