"""
Interpretable transformer for character-level sequence tasks.

Task here: multiplication-five-digits  ("AAAAA*BBBBB=" -> "PPPPPPPPPP").

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.
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
# Attention-only, float64, 7-channel residual stream, no LayerNorm, no MLP.
# Residual channels:
S_, R1, R2, LOGD, DP, BA, X = range(7)
#  S_   : positional log-place score (log 10^k), plus sink offset at '*'
#  R1,R2: 2-D region code  A=(1,0)  B=(0,1)  '*'=(0,0)  '='/answer=(-1,-1)
#  LOGD : log(digit) of the token (digit 0 and '=' -> -L)
#  DP   : eps*digit (token)  +  place 10^(20-t) at answer positions (pos)
#  BA   : layer-1 output  b*A  at B positions
#  X    : remainder register  P - V + 0.5
DT = torch.float64


class Head(nn.Module):
    """Single attention head with explicit q/k dim, scalar value, no 1/sqrt(d) scaling."""

    def __init__(self, d_model: int, d_qk: int):
        super().__init__()
        self.W_q = nn.Parameter(torch.zeros(d_qk, d_model, dtype=DT))
        self.W_k = nn.Parameter(torch.zeros(d_qk, d_model, dtype=DT))
        self.W_v = nn.Parameter(torch.zeros(1, d_model, dtype=DT))
        self.W_o = nn.Parameter(torch.zeros(d_model, 1, dtype=DT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        q = x @ self.W_q.T
        k = x @ self.W_k.T
        v = x @ self.W_v.T
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        return (attn @ v) @ self.W_o.T


class AttnLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_qk: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_model, d_qk) for _ in range(n_heads)])

    def forward(self, x):
        return x + sum(h(x) for h in self.heads)


class SimpleTransformer(nn.Module):
    """Two attention-only layers (1 head, then 2 heads), float64, no LN/MLP."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 7, d_qk: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Parameter(torch.zeros(vocab_size, d_model, dtype=DT))
        self.pos_emb = nn.Parameter(torch.zeros(max_seq_len, d_model, dtype=DT))
        self.layers = nn.ModuleList([AttnLayer(d_model, 1, d_qk), AttnLayer(d_model, 2, d_qk)])
        self.head = nn.Parameter(torch.zeros(vocab_size, d_model, dtype=DT))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        h = self.token_emb[ids] + self.pos_emb[:T][None]
        for layer in self.layers:
            h = layer(h)
        return h @ self.head.T


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    ln10 = math.log(10.0)
    M = 1000.0             # region-separation margin in score space
    L = 100.0              # "log 0" surrogate
    EPS = 1e-6             # token digit scale in DP channel
    S1 = 1e17              # layer-1 sink weight
    S2 = 1e22              # layer-2 head-2 sink weight
    logS1, logS2 = math.log(S1), math.log(S2)
    SIG = 2 * M + logS2    # S_ value at the sink position '*'
    st = task.stoi

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        te, pe = model.token_emb, model.pos_emb
        # ---- token embedding
        for d in range(10):
            te[st[str(d)], LOGD] = math.log(d) if d > 0 else -L
            te[st[str(d)], DP] = EPS * d
        te[st["*"], LOGD] = 0.0
        te[st["="], LOGD] = -L
        # ---- positional embedding
        for i in range(5):                      # A digits, positions 0..4
            pe[i, S_] = (4 - i) * ln10
            pe[i, R1] = 1.0
        pe[5, S_] = SIG                         # sink '*'
        for t in range(6, 11):                  # B digits
            pe[t, S_] = (10 - t) * ln10
            pe[t, R2] = 1.0
        for p in range(11, task.seq_len):       # '=' and answer region
            pe[p, R1] = -1.0
            pe[p, R2] = -1.0
            pe[p, X] = 0.5
            if p >= 12:
                pe[p, S_] = (21 - p) * ln10     # place value of the digit emitted at p
            if p <= 20:
                pe[p, DP] += 10.0 ** (20 - p)   # place of the digit predicted at p

        # ---- Layer 1: B digit b reads b*A via exponential sink trick
        h = model.layers[0].heads[0]
        # k = [s, r1, r2];  q = [r2, logd + (SIG-logS1)*r2, -M*r2]
        h.W_k[0, S_] = 1.0; h.W_k[1, R1] = 1.0; h.W_k[2, R2] = 1.0
        h.W_q[0, R2] = 1.0
        h.W_q[1, LOGD] = 1.0; h.W_q[1, R2] = SIG - logS1
        h.W_q[2, R2] = -M
        h.W_v[0, DP] = 1.0                       # eps * a_i
        h.W_o[BA, 0] = S1 / EPS                  # rescale to b*A

        # ---- Layer 2 head 1: P = 11111 * weighted mean of b_t*A, weights ∝ 10^(10-t)
        h = model.layers[1].heads[0]
        h.W_k[0, S_] = 1.0; h.W_k[1, R1] = 1.0; h.W_k[2, R2] = 1.0
        # c = -r1 = 1 at answer queries; q = c*[1, -2M, 4M]
        h.W_q[0, R1] = -1.0; h.W_q[1, R1] = 2 * M; h.W_q[2, R1] = -4 * M
        h.W_v[0, BA] = 1.0
        h.W_o[X, 0] = 11111.0

        # ---- Layer 2 head 2: V = sum emitted digits * place via sink trick, subtract
        h = model.layers[1].heads[1]
        h.W_k[0, S_] = 1.0; h.W_k[0, LOGD] = 1.0
        h.W_k[1, R1] = 1.0; h.W_k[1, R2] = 1.0
        h.W_q[0, R1] = -1.0; h.W_q[1, R1] = M    # q = c*[1, -M]
        h.W_v[0, R1] = -1.0                      # 1 on answer positions, 0 at sink
        h.W_o[X, 0] = -S2

        # ---- Tropical readout: logit_k = k*X - place*k(k+1)/2
        for k in range(10):
            model.head[st[str(k)], X] = float(k)
            model.head[st[str(k)], DP] = -k * (k + 1) / 2.0
        model.head[st["*"], R1] = 1000.0          # -1000 in answer region
        model.head[st["="], R1] = 1000.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "attn2_sink_tropical_v1"
model_description = ("Two attention-only float64 layers on 7 channels: L1 sink-trick head computes b*A at B digits; "
                     "L2 head1 sums 10^(10-t)-weighted b*A (x11111) to P, head2 sink-trick reads emitted value V; "
                     "tropical readout argmax floor((P-V+0.5)/place).")


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
