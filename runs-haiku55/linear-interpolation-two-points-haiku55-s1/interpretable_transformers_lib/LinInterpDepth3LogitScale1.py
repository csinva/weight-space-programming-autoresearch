"""
Interpretable transformer for character-level sequence tasks.

Task `linear-interpolation-two-points`:
  prompt "x1,y1;x2,y2;q=" (12 chars) -> answer "yq" (2 digits), where the two
  points lie on a line y = m*x + b with integer m in 1..5, b in 0..9.
  Answer = y1 + m*(q - x1).

Hand-written circuit (no training, no LayerNorm so the residual stream is linear).
Depth-3 variant: block 2 computes the answer one-hot directly from the slope
indicators, gated by I_m with big-M = GATE_M.
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
    # LayerNorm is replaced by identity so that every residual register holds an exact
    # linear quantity (digit values, indicators) that later layers can read off.
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
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 112,
        n_heads: int = 7,
        n_layers: int = 3,
        d_ff: int = 825,
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

# Residual register layout
POS0 = 0          # 0..15: one-hot position
ONE = 16          # constant 1
VAL = 17          # digit value of the token
FLAG = 19         # 1 at position 12, else 0
X1, X2, XQ, Y2 = 20, 21, 22, 24
Y1_OR_A = 40      # Y1 gathered (block 1); not modified later


def I_REG(m: int) -> int:      # slope indicator, m = 1..5
    return 30 + (m - 1)


def OH_REG(v: int) -> int:     # one-hot of answer value, v = 0..54
    return 41 + v


def TG_REG(t: int) -> int:     # tens digit one-hot, t = 0..5
    return 96 + t


def UG_REG(r: int) -> int:     # units digit one-hot, r = 0..9
    return 102 + r


HEAD_DIM = 16                  # d_head = d_model / n_heads = 112 / 7
BETA = 40.0                    # attention sharpness: score = BETA * [match] / sqrt(16)
GATE_M = 100.0                 # big-M for slope gating (must exceed |A_m - v| + 1, here <= 101)


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. Closed-form, no training."""
    assert model.d_model == 112 and model.n_heads == 7 and model.n_layers == 3
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # --- embeddings -----------------------------------------------------
        for t in range(model.vocab_size):
            model.token_emb.weight[t, ONE] = 1.0
        for d in range(10):                       # digit tokens '0'..'9' have ids 0..9
            model.token_emb.weight[d, VAL] = float(d)
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, POS0 + p] = 1.0
        model.pos_emb.weight[12, FLAG] = 1.0

        # --- layer 1 attention: gather prompt digits to every query position --
        gathers = [
            (0, X1, 1.0),
            (5, X2, 1.0),
            (10, XQ, 1.0),
            (7, Y2, 10.0),
            (8, Y2, 1.0),
            (2, Y1_OR_A, 10.0),
            (3, Y1_OR_A, 1.0),
        ]
        attn = model.blocks[0].attn
        for h, (src, dest, coef) in enumerate(gathers):
            base = h * HEAD_DIM
            attn.W_q.weight[base + src, ONE] = BETA
            for i in range(HEAD_DIM):
                attn.W_k.weight[base + i, POS0 + i] = 1.0
            attn.W_v.weight[base + 0, VAL] = 1.0
            attn.W_o.weight[dest, base + 0] = coef

        # --- layer 1 MLP: slope indicators I_m = [Y2 - Y1 == m*(X2 - X1)] ---------
        mlp = model.blocks[0].mlp
        for m in range(1, 6):
            for k, (shift, c) in enumerate([(1.0, 1.0), (0.0, -2.0), (-1.0, 1.0)]):
                j = (m - 1) * 3 + k
                mlp.fc1.weight[j, Y2] = 1.0
                mlp.fc1.weight[j, Y1_OR_A] = -1.0
                mlp.fc1.weight[j, X2] = -float(m)
                mlp.fc1.weight[j, X1] = float(m)
                mlp.fc1.bias[j] = shift
                mlp.fc2.weight[I_REG(m), j] = c

        # --- layer 2 MLP: one-hot of A_m = Y1 + m*(XQ - X1), gated by I_m -------
        # tri(A_m - v) = [A_m == v] for the true slope m (I_m = 1); for I_m = 0 the
        # -GATE_M term keeps every ReLU at zero, so only the true slope contributes.
        mlp = model.blocks[1].mlp
        j = 0
        for v in range(55):
            for m in range(1, 6):
                for shift, c in [(1.0, 1.0), (0.0, -2.0), (-1.0, 1.0)]:
                    mlp.fc1.weight[j, Y1_OR_A] = 1.0
                    mlp.fc1.weight[j, XQ] = float(m)
                    mlp.fc1.weight[j, X1] = -float(m)
                    mlp.fc1.weight[j, I_REG(m)] = GATE_M
                    mlp.fc1.bias[j] = -float(v) + shift - GATE_M
                    mlp.fc2.weight[OH_REG(v), j] = c
                    j += 1

        # --- layer 3 MLP: split one-hot into tens (pos 11) / units (pos 12) ----
        mlp = model.blocks[2].mlp
        j = 0
        for t in range(6):                      # TG_t = ReLU(sum_{v in tens t} OH_v - FLAG)
            for v in range(10 * t, min(10 * t + 10, 55)):
                mlp.fc1.weight[j, OH_REG(v)] = 1.0
            mlp.fc1.weight[j, FLAG] = -1.0
            mlp.fc2.weight[TG_REG(t), j] = 1.0
            j += 1
        for r in range(10):                     # UG_r = ReLU(sum_{v = r mod 10} OH_v + FLAG - 1)
            for v in range(r, 55, 10):
                mlp.fc1.weight[j, OH_REG(v)] = 1.0
            mlp.fc1.weight[j, FLAG] = 1.0
            mlp.fc1.bias[j] = -1.0
            mlp.fc2.weight[UG_REG(r), j] = 1.0
            j += 1

        # --- output head: digit logits -----------------------------------------
        for t in range(10):
            if t <= 5:
                model.head.weight[t, TG_REG(t)] = 1.0
            model.head.weight[t, UG_REG(t)] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "LinInterpDepth3LogitScale1"
model_description = (
    "Depth-3 variant (BETA=40, gate 100) with head logit scale 1 instead of 10: output margin "
    "check in the 3-block model."
)


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
