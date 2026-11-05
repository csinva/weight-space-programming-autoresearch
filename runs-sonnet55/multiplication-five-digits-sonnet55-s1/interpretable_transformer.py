"""
Interpretable transformer for 5-digit x 5-digit multiplication.
Prompt "48622*30927=" -> 10-digit zero-padded product, most significant digit first.

No training: every weight is written in closed form in write_weights().

Circuit (schoolbook multiplication, carries counted with ReLU steps):
  L0 attn : 10 heads, each reading one fixed position (a_0..a_4 at 0..4, b_0..b_4 at 6..10).
            a_i arrives as 4 bits, b_j as a scalar.
  L0 MLP  : 100 gated units  2^t * b_j * bit_t(a_i)  (exact digit products) summed into the 9 column
            sums c_k = sum_{i+j=k} a_i b_j ; scratch slots are erased and reused.
  carry   : for each column k (LSB first), t = c_k + cy_k and cy_{k+1} = floor(t/10), computed in two MLP
            layers: (1) 5 * #{n: t >= 50n}, (2) #{n: t - 50h >= 10n}; each step [x>=T] = relu(x-T+1)-relu(x-T).
  readout : digit m = c_m + cy_m - 10 cy_{m+1}; a position gate (2-hot row/col match) selects the digit that
            belongs to the current output position; shared steps [s>=v] turn it into logits (STEP_v - STEP_{v+1}).
LayerNorm is removed so the arithmetic stays exact; attention exists only in block 0.
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
# Architecture: no LayerNorm; blocks may omit attention; per-block d_ff.
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

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        out = (scores.softmax(dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, use_attn):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads) if use_attn else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_heads, d_ffs, attn_layers):
        super().__init__()
        self.vocab_size, self.max_seq_len, self.d_model = vocab_size, max_seq_len, d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, f, i in attn_layers) for i, f in enumerate(d_ffs)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Residual stream layout (D=40, 10 heads x 4 dims)
#   TOK  0-3   : 4-bit code of the digit token (erased by the product layer, then reused for carries)
#   POSR 4-8, POSC 9-13 : position p as 2-hot (row p//5, col p%5)
#   A    14-33 : bits of a_0..a_4 (4 each);  B 34-38 : scalar b_0..b_4  (both erased by the product layer)
#   C    14-22 : column sums (written into the erased A dims);  IND 23-31 : step values;  SEL 32 : chosen digit
#   CY   : carry into column k (k=1..9) lives in CYS[k-1] = B dims 34-38 then TOK dims 0-3
# ---------------------------------------------------------------------------

D, NH = 40, 10
DH = D // NH
NMAX = [8, 17, 26, 35, 44, 36, 27, 18, 9]  # largest possible carry out of each column
TOK, POSR, POSC, A0, B0 = 0, 4, 9, 14, 34
C0, IND0, SEL = 14, 23, 32
CYS = [34, 35, 36, 37, 38, 0, 1, 2, 3]
S = 60.0      # attention logit scale (full-match minus half-match gap ~ S/sqrt(DH))
SPLIT = 50    # carry: first count multiples of SPLIT in t, then multiples of 10 in the remainder


def cy(k):
    return CYS[k - 1]


def build_model(task) -> SimpleTransformer:
    L = max(task.seq_len, 16)
    assert L <= 25
    carry_ffs = [f for k in range(9)
                 for f in (2 * (NMAX[k] // (SPLIT // 10)), 2 * min(SPLIT // 10 - 1, NMAX[k])) if f > 0]
    ffs = [100 + 29] + carry_ffs + [10, 18]
    model = SimpleTransformer(task.vocab_size, L, D, NH, ffs, attn_layers={0})
    write_weights(model, task)
    model.eval()
    return model


def write_weights(model, task):
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        L = model.max_seq_len
        for v in range(10):  # digit tokens carry their 4-bit code; '*' and '=' are zero vectors
            for t in range(4):
                model.token_emb.weight[task.stoi[str(v)], TOK + t] = (v >> t) & 1
        for p_ in range(L):
            model.pos_emb.weight[p_, POSR + p_ // 5] = 1
            model.pos_emb.weight[p_, POSC + p_ % 5] = 1

        # --- L0 attention: head h reads one fixed position (a_i at i, b_j at 6+j)
        at = model.blocks[0].attn
        for h in range(10):
            target = h if h < 5 else 6 + (h - 5)
            r = DH * h
            for rr in range(5):  # the 5 row dims always sum to 1, so this is a constant query
                at.W_q.weight[r, POSR + rr] = S
            at.W_k.weight[r, POSR + target // 5] = 1
            at.W_k.weight[r, POSC + target % 5] = 1
            if h < 5:  # copy the 4 bits of a_i
                for t in range(4):
                    at.W_v.weight[r + t, TOK + t] = 1
                    at.W_o.weight[A0 + 4 * h + t, r + t] = 1
            else:  # b_j as a scalar = sum_t 2^t bit_t
                for t in range(4):
                    at.W_v.weight[r, TOK + t] = 2 ** t
                at.W_o.weight[B0 + (h - 5), r] = 1

        # --- L0 MLP: unit (i,j,t) = relu(2^t b_j + 9*2^t bit_t(a_i) - 9*2^t) = 2^t b_j if bit_t(a_i) else 0
        m = model.blocks[0].mlp
        u = 0
        for i in range(5):
            for j in range(5):
                for t in range(4):
                    x = 2 ** t
                    m.fc1.weight[u, B0 + j] = x
                    m.fc1.weight[u, A0 + 4 * i + t] = 9 * x
                    m.fc1.bias[u] = -9 * x
                    m.fc2.weight[C0 + 8 - i - j, u] = 1  # operands are written MSB first
                    u += 1
        for s in list(range(A0, A0 + 20)) + list(range(B0, B0 + 5)) + list(range(TOK, TOK + 4)):
            m.fc1.weight[u, s] = 1  # erase scratch (values >= 0 so relu is the identity)
            m.fc2.weight[s, u] = -1
            u += 1

        # --- carry layers: two per column (a column with no possible multiple of SPLIT skips the first)
        bi = 1
        for k in range(9):
            for nn_, scale, kind in ((NMAX[k] // (SPLIT // 10), SPLIT, "A"), (min(SPLIT // 10 - 1, NMAX[k]), 10, "B")):
                if nn_ == 0:
                    continue
                m = model.blocks[bi].mlp
                bi += 1
                for n in range(1, nn_ + 1):
                    for q, (uu, bias) in enumerate(((2 * (n - 1), 1 - n * scale), (2 * (n - 1) + 1, -n * scale))):
                        m.fc1.weight[uu, C0 + k] = 1
                        if k > 0:
                            m.fc1.weight[uu, cy(k)] = 1
                        if kind == "B":
                            m.fc1.weight[uu, cy(k + 1)] = -10  # remove the multiples of SPLIT already counted
                        m.fc1.bias[uu] = bias
                        m.fc2.weight[cy(k + 1), uu] = (SPLIT // 10 if kind == "A" else 1) * (1 - 2 * q)

        # --- readout 1: digit dm (LSB index) is emitted at sequence position 20-dm; gate on row AND column
        m = model.blocks[-2].mlp
        for dm in range(10):
            if dm < 9:
                m.fc1.weight[dm, C0 + dm] = 1
                if dm > 0:
                    m.fc1.weight[dm, cy(dm)] = 1
                m.fc1.weight[dm, cy(dm + 1)] = -10
            else:
                m.fc1.weight[dm, cy(9)] = 1
            m.fc1.weight[dm, POSR + (20 - dm) // 5] = 10
            m.fc1.weight[dm, POSC + (20 - dm) % 5] = 10
            m.fc1.bias[dm] = -20
            m.fc2.weight[SEL, dm] = 1

        # --- readout 2: STEP_v = [s >= v] (v=1..9) at IND0+v-1; logit_v = STEP_v - STEP_{v+1}  (STEP_0=1, STEP_10=0)
        m = model.blocks[-1].mlp
        for v in range(1, 10):
            for k, off in enumerate((1, 0)):
                m.fc1.weight[2 * (v - 1) + k, SEL] = 1
                m.fc1.bias[2 * (v - 1) + k] = -v + off
                m.fc2.weight[IND0 + v - 1, 2 * (v - 1) + k] = 1 - 2 * k
        for v in range(10):
            row = model.head.weight[task.stoi[str(v)]]
            if v >= 1:
                row[IND0 + v - 1] = 1
            else:
                row[POSR:POSR + 5] = 1  # constant 1 (row dims sum to 1)
            if v <= 8:
                row[IND0 + v] = -1


model_shorthand_name = "final_clean_d40"
model_description = "Final: 4-bit token code, binary-gated digit products, D=40 with reused scratch dims, two-level (50/10) step carries, gated step readout; 10 heads, attention only in block 0, no LayerNorm."


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
