"""
Interpretable transformer for character-level sequence tasks.

Task: linear-interpolation-two-points.  "x1,y1;x2,y2;x3=" -> two-digit y3, where
all points lie on y = m*x + b with integer m in 1..5, b in 0..9, x in 0..9.

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py --task linear-interpolation-two-points
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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, has_attn: bool = True):
        super().__init__()
        self.ln1 = nn.Identity()  # LayerNorm removed: channels hold exact numbers
        self.attn = CausalSelfAttention(d_model, n_heads) if has_attn else None
        self.ln2 = nn.Identity()
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
        d_model: int = 19,
        n_heads: int = 19,
        n_layers: int = 2,
        d_ff: int = 15,
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
        # Block 0 reads the prompt (attention + slope test); block 1 is a pure MLP selecting the line.
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, True), Block(d_model, n_heads, 5, False)])
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

# Named channels of the d_model-dimensional state.
BIAS, DIGIT, SCALE = 0, 1, 2           # constant 1 | token digit value | place value of the digit to print
KEY = dict(x1=3, y1=4, x2=5, y2=6, x3=7)   # positional attention keys, one per number
REG = dict(x1=8, y1=9, x2=10, y2=11, x3=12)  # registers filled by attention
GATE = {k: 12 + k for k in range(1, 6)}    # GATE[k] = 1 iff the slope is k
Y = 18                                 # the answer y3
SOURCES = dict(x1={0: 1.0}, y1={2: 10.0, 3: 1.0}, x2={5: 1.0}, y2={7: 10.0, 8: 1.0}, x3={10: 1.0})
FORBID = 1e3    # attention score of positions a head must ignore
M = 200.0       # big-M used to switch off a ReLU unit


def write_weights(model: SimpleTransformer, task) -> None:
    """Read numbers (attention), test slopes (MLP1), select the line (MLP2), floor-divide into digits (head)."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: digits carry their value; positions carry a bias, flag and attention keys.
        for d in range(10):
            model.token_emb.weight[task.stoi[str(d)], DIGIT] = float(d)
        model.pos_emb.weight[:, BIAS] = 1.0
        model.pos_emb.weight[11, SCALE], model.pos_emb.weight[12, SCALE] = 10.0, 1.0
        for name, src_pos in SOURCES.items():
            model.pos_emb.weight[:, KEY[name]] = -FORBID
            for p, w in src_pos.items():
                model.pos_emb.weight[p, KEY[name]] = math.log(w)

        # Block 0 attention: head `h` attends to its number's digits; 10:1 weights decode "tens,units".
        at = model.blocks[0].attn
        for name, src_pos in SOURCES.items():
            h = REG[name]
            at.W_q.weight[h, BIAS] = 1.0
            at.W_k.weight[h, KEY[name]] = 1.0
            at.W_v.weight[h, DIGIT] = 1.0
            at.W_o.weight[h, h] = sum(src_pos.values())

        # Block 0 MLP: triangle bump GATE[k] = 1 iff z_k = (y2-y1) - k(x2-x1) is 0.
        fc1, fc2 = model.blocks[0].mlp.fc1, model.blocks[0].mlp.fc2
        for k in range(1, 6):
            for j, (shift, coef) in enumerate(((1.0, 1.0), (0.0, -2.0), (-1.0, 1.0))):
                u = 3 * (k - 1) + j
                fc1.weight[u, REG["y2"]], fc1.weight[u, REG["y1"]] = 1.0, -1.0
                fc1.weight[u, REG["x2"]], fc1.weight[u, REG["x1"]] = -float(k), float(k)
                fc1.bias[u] = shift
                fc2.weight[GATE[k], u] = coef

        # Block 1 MLP: Y = sum_k relu(y1 + k(x3-x1) - M(1-GATE[k])); only the true line survives.
        fc1, fc2 = model.blocks[1].mlp.fc1, model.blocks[1].mlp.fc2
        for k in range(1, 6):
            u = k - 1
            fc1.weight[u, REG["y1"]] = 1.0
            fc1.weight[u, REG["x3"]], fc1.weight[u, REG["x1"]] = float(k), -float(k)
            fc1.weight[u, GATE[k]] = M
            fc1.bias[u] = -M
            fc2.weight[Y, u] = 1.0

        # Head: floor division. With V = Y - 10*printed_tens + 0.5, the logit
        # d*V - SCALE*d(d+1)/2 rises while V >= SCALE*d, so the argmax is floor(V/SCALE).
        for d in range(10):
            row = model.head.weight[task.stoi[str(d)]]
            row[Y], row[DIGIT], row[BIAS] = float(d), -10.0 * d, 0.5 * d
            row[SCALE] = -d * (d + 1) / 2.0
        for c in ",;=":
            model.head.weight[task.stoi[c], BIAS] = -FORBID


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "AttnOnlyBlock0"
model_description = "Attention kept only in block 0 and block 1 MLP narrowed to 5 units; reading needs one attention layer, so the rest is removed."


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
