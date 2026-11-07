"""
Interpretable transformer for character-level sequence tasks.

Task: 5-digit multiplication, prompt "50494*99346=" -> answer "5016376924"
(10 digits, most significant first). Weights are written by hand (no training).

Circuit (float64, no LayerNorm):
  block 0  attention : 10 heads, each reads one fixed prompt position (a_i / b_j digit value)
           MLP       : exact digit products a_i*b_j via squares ( ab = (s^2 - d^2)/4 with
                       s=a+b, d=a-b, squares as piecewise-linear ReLU sums ), summed into
                       column sums c_q = sum_{i+j=q} a_i b_j
  block 1  MLP       : position-gated sum u_p = sum_{q<=p} c_q 10^(q-p) for the output digit p
  block 2  MLP       : floor(u_p) mod 10 through steep ramp thresholds -> one-hot digit logits
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


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_qk: int = 1, d_v: int = 1):
        super().__init__()
        self.n_heads, self.d_qk, self.d_v = n_heads, d_qk, d_v
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, self.d_qk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.d_qk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.d_v).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_qk)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * self.d_v)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int = 0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads) if n_heads > 0 else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, d_ffs, n_heads: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, f, n_heads if i == 0 else 0) for i, f in enumerate(d_ffs)]
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# residual-stream channel layout
CONST, VAL = 0, 1            # constant 1 ; digit value of the current token
POS = 2                      # 22 one-hot position channels
A0 = POS + 22                # a_0..a_4  (digits of first factor, a_0 = units)
B0 = A0 + 5                  # b_0..b_4
C0 = B0 + 5                  # c_0..c_8  column sums
U = C0 + 9                   # u_p
LOG0 = U + 1                 # 10 digit logits
D_MODEL = LOG0 + 10

K = 1e10                     # ramp steepness for exact floor
EPS = 1e-10
V_MAX = 451


def write_weights(model: SimpleTransformer, task) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        te, pe = model.token_emb.weight, model.pos_emb.weight
        for d in range(10):
            te[task.stoi[str(d)], VAL] = d
        for t in range(pe.shape[0]):
            pe[t, POS + t] = 1.0
        pe[:, CONST] = 1.0

        # ---- block 0 attention: head h reads one fixed prompt position
        att = model.blocks[0].attn
        S = 60.0
        for h in range(10):
            if h < 5:
                i, src_pos, ch = h, 4 - h, A0 + h
            else:
                j = h - 5
                src_pos, ch = 10 - j, B0 + j
            att.W_q.weight[h, CONST] = S
            att.W_k.weight[h, POS + src_pos] = 1.0
            att.W_v.weight[h, VAL] = 1.0
            att.W_o.weight[ch, h] = 1.0

        # ---- block 0 MLP: a_i*b_j into column c_{i+j}
        m = model.blocks[0].mlp
        u = 0
        for i in range(5):
            for j in range(5):
                q = C0 + i + j
                # s^2 = relu(s) + 2 sum_{k=1..17} relu(s-k)
                for k in range(0, 18):
                    m.fc1.weight[u, A0 + i] = 1.0
                    m.fc1.weight[u, B0 + j] = 1.0
                    m.fc1.bias[u] = -k
                    m.fc2.weight[q, u] = (1.0 if k == 0 else 2.0) / 4
                    u += 1
                # d^2 = |d|^2, both signs
                for sgn in (1.0, -1.0):
                    for k in range(0, 9):
                        m.fc1.weight[u, A0 + i] = sgn
                        m.fc1.weight[u, B0 + j] = -sgn
                        m.fc1.bias[u] = -k
                        m.fc2.weight[q, u] = -(1.0 if k == 0 else 2.0) / 4
                        u += 1
        assert u == model.blocks[0].mlp.fc1.weight.shape[0]

        # ---- block 1 MLP: u_p = sum_{q<=p} c_q 10^(q-p), gated by output position
        m = model.blocks[1].mlp
        u = 0
        for p in range(10):
            pos = 20 - p            # position 11+k predicts digit k, p = 9-k
            for q in range(min(p, 8) + 1):
                m.fc1.weight[u, C0 + q] = 1.0
                m.fc1.weight[u, POS + pos] = 1000.0
                m.fc1.bias[u] = -1000.0
                m.fc2.weight[U, u] = 10.0 ** (q - p)
                u += 1
        m.fc2.bias[U] = EPS

        # ---- block 2 MLP: digit = floor(u) mod 10 via thresholds
        m = model.blocks[2].mlp
        u = 0
        for v in range(V_MAX + 1):
            for off in (0.0, 1.0):
                m.fc1.weight[u, U] = K
                m.fc1.bias[u] = -K * v - off
                sgn = 1.0 if off == 0.0 else -1.0     # step = relu(x) - relu(x-1)
                m.fc2.weight[LOG0 + v % 10, u] = sgn
                if v >= 1:
                    m.fc2.weight[LOG0 + (v - 1) % 10, u] = -sgn
                u += 1

        for d in range(10):
            model.head.weight[task.stoi[str(d)], LOG0 + d] = 1.0


model_shorthand_name = "SquaresColumnSumThreshold"
model_description = (
    "float64 3-block net: 10 fixed-position heads gather digits; ReLU squares give exact a_i*b_j "
    "column sums; gated 10^(q-p) sum + steep ramp thresholds give floor(u) mod 10 per output digit."
)


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
    d_ffs = [25 * (18 + 18), sum(min(p, 8) + 1 for p in range(10)), 2 * (V_MAX + 1)]
    model = SimpleTransformer(task.vocab_size, max_seq_len, D_MODEL, d_ffs, n_heads=10).double()
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
