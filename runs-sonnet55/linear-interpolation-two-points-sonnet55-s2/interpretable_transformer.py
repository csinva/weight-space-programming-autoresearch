"""
Interpretable transformer for character-level sequence tasks.

Task: linear-interpolation-two-points.  Prompt "x1,yy;x2,yy;q=" -> 2-digit answer
y1 + s*(q-x1) where the integer slope s = (y2-y1)/(x2-x1) is in 1..5.

The model is never trained; weights are written in closed form in write_weights().
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
# Architecture
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=True)   # bias = constant query (target position)
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
    """Residual block without LayerNorm (residual dims are exact, readable quantities).
    Attention is only present when has_attn."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, has_attn: bool):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads) if has_attn else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=23, n_heads=7, n_layers=4, d_ff=(10, 10, 39, 20)):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff[i], i == 0) for i in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Weights.  Positions 11 ('=') and 12 (tens digit) emit the two answer digits.
# Residual layout:
#   TOK 0 digit value | P 1 position p | P2 2 p^2 | dx 3, dy 4, q-x1 5 | A 6 (answer scalar)
#   SD 7..11 slope mismatch |dy - s*dx| | TENS 12 (holds ones digit A mod 10) | OUT 13..22 digit logits
# ---------------------------------------------------------------------------

TOK, P, P2, SLOT, SD, A, TENS, OUT = 0, 1, 2, 3, 7, 6, 12, 13
SRC_POS = [0, 2, 3, 5, 7, 8, 10]   # x1, y1 tens, y1 ones, x2, y2 tens, y2 ones, q
X1, Y1T, Y1O, X2, Y2T, Y2O, Q = range(7)
DXD, DYD, TQD = SLOT, SLOT + 1, SLOT + 2   # derived scalars: x2-x1, y2-y1, q-x1
CONTRIB = {   # head -> {residual dim: coefficient}; also initialises A = y1
    X1: {DXD: -1.0, TQD: -1.0}, Y1T: {DYD: -10.0, A: 10.0}, Y1O: {DYD: -1.0, A: 1.0},
    X2: {DXD: 1.0}, Y2T: {DYD: 10.0}, Y2O: {DYD: 1.0}, Q: {TQD: 1.0},
}


def write_weights(model: SimpleTransformer, task) -> None:
    D, H = model.d_model, model.n_heads
    dh = model.blocks[0].attn.d_head
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        dig = [task.stoi[str(j)] for j in range(10)]
        for p in range(model.max_seq_len):   # position embedding = (p, p^2, 1)
            model.pos_emb.weight[p, P] = float(p)
            model.pos_emb.weight[p, P2] = float(p * p)
        for j in range(10):          # token embedding = the digit's numeric value (1 dim)
            model.token_emb.weight[dig[j], TOK] = float(j)

        # ---- Block 0 attention: head h copies the digit at SRC_POS[h] into SLOT_h
        at = model.blocks[0].attn
        for h, sp in enumerate(SRC_POS):
            r = h * dh
            # score(p) = c*(2*sp*p - p^2) = const - c*(p - sp)^2 : sharp peak at the source position
            c = 20.0 * math.sqrt(dh)
            at.W_q.bias[r] = 2.0 * sp * c
            at.W_q.bias[r + 1] = -c
            at.W_k.weight[r, P] = 1.0
            at.W_k.weight[r + 1, P2] = 1.0
            at.W_v.weight[r, TOK] = 1.0   # value = numeric digit value
            # each head writes its digit value, with a coefficient, into the derived scalars
            for dst, coef in CONTRIB[h].items():
                at.W_o.weight[dst, r] = coef

        def unitvec(i):
            v = torch.zeros(D); v[i] = 1.0
            return v

        DX, DY, TQ = unitvec(DXD), unitvec(DYD), unitvec(TQD)

        def unit(mlp, i, w, b, outs):
            mlp.fc1.weight[i] = w
            mlp.fc1.bias[i] = b
            for d, c in outs:
                mlp.fc2.weight[d, i] = c

        # ---- Block 0 MLP: slope mismatch SD[s-1] = |dy - s*dx| (0 only for the true slope)
        m = model.blocks[0].mlp
        i = 0
        for s in range(1, 6):
            L = DY - s * DX
            for sgn in (1.0, -1.0):   # SD[s-1] = |dy - s*dx| : mismatch of slope hypothesis s
                unit(m, i, sgn * L, 0.0, [(SD + s - 1, 1.0)]); i += 1

        # ---- Block 1 MLP: A += sum_s [mismatch_s == 0] * s*(q-x1)   (gated products; A starts at y1)
        m = model.blocks[1].mlp
        M = 46.0
        i = 0
        for s in range(1, 6):
            for sign in (1.0, -1.0):
                w = sign * s * TQ
                w[SD + s - 1] = -M     # any mismatch >= 1 kills the unit (M > max |s*t| = 45)
                unit(m, i, w, 0.0, [(A, sign)]); i += 1

        # ---- Block 2 MLP: ones digit = A - 10*sum_{k=1..9} [A >= 10k]  (step = 2 ReLUs)
        m = model.blocks[2].mlp
        i = 0
        w = torch.zeros(D); w[A] = 1.0
        unit(m, i, w, 0.0, [(TENS, 1.0)]); i += 1      # + A   (A >= 0)
        for k in range(1, 10):                          # - 10 * [A >= 10k]
            unit(m, i, w, float(1 - 10 * k), [(TENS, -10.0)]); i += 1
            unit(m, i, w, float(-10 * k), [(TENS, 10.0)]); i += 1

        # Tens readout (same MLP, reads A directly): logit_k = -dist(A, [10k, 10k+9]),
        # gated to position 11 by G*(11-p) (0 at position 11, -G at position 12).
        G = 100.0   # gate must exceed the largest un-gated pre-activation (A <= 99)
        for k in range(10):
            for sgn, lo in ((1.0, 10 * k + 9), (-1.0, 10 * k)):
                w = torch.zeros(D); w[A] = sgn; w[P] = -G
                unit(m, i, w, -sgn * lo + 11 * G, [(OUT + k, -1.0)]); i += 1

        # ---- Block 3 MLP: ones readout, logit_k = -|ones - k|, gated to position 12
        m = model.blocks[3].mlp
        O = torch.zeros(D); O[TENS] = 1.0   # TENS dim holds the ones digit A mod 10
        i = 0
        for k in range(10):
            # ones: gate G*(p-12): 0 at position 12, -G at position 11
            w = O.clone(); w[P] = G
            unit(m, i, w, -k - 12 * G, [(OUT + k, -1.0)]); i += 1
            w = -O; w[P] = G
            unit(m, i, w, k - 12 * G, [(OUT + k, -1.0)]); i += 1

        # Margin: digit logits are -dist + 0.5, so the right digit (0.5) strictly beats every
        # non-digit token (logit 0).  Without it the winner ties/loses to ',' at ~1e-5 noise.
        for k in range(10):
            model.blocks[3].mlp.fc2.bias[OUT + k] = 0.5
            model.head.weight[dig[k], OUT + k] = 1.0


model_shorthand_name = "final_margin_sharp20"
model_description = "Final: scalar token/quadratic position embeddings, 7 d_head-2 heads write dx,dy,q-x1,A; slope-mismatch gating (M=46); ones scalar; -dist readouts with +0.5 margin; sharpness 20; d_model 23, 4 blocks."


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
