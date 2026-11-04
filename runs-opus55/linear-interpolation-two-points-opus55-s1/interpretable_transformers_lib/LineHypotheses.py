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
        # One scalar head per number: q, k, v live in n_heads dims, not d_model dims.
        self.n_heads = n_heads
        self.query = nn.Parameter(torch.zeros(n_heads))  # constant query: routing ignores the state
        self.W_k = nn.Linear(d_model, n_heads, bias=False)
        self.W_v = nn.Linear(d_model, n_heads, bias=False)
        self.W_o = nn.Linear(n_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        q = self.query.view(1, H, 1, 1).expand(B, H, T, 1)
        k = self.W_k(x).view(B, T, H, 1).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, 1).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H)
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
        d_model: int = 13,
        n_heads: int = 5,
        n_layers: int = 2,
        d_ff: int = 10,
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
SCALE, DIGIT = 0, 1                    # place value to print (10 at tens step, 1 elsewhere) | token digit value
REG = dict(x1=2, y1=3, x2=4, y2=5, x3=6)   # number registers; before attention they hold positional keys
PEN = {k: 6 + k for k in range(1, 6)}  # PEN[k] = |z_k|, which is 0 iff the slope is k
Y = 12                                 # the answer y3
SOURCES = dict(x1={0: 1.0}, y1={2: 10.0, 3: 1.0}, x2={5: 1.0}, y2={7: 10.0, 8: 1.0}, x3={10: 1.0})
FORBID = 30.0   # source attention score of each number
M = 100.0       # big-M: a wrong line has |z_k| >= 1 and y1 + k*t <= 99, so M >= 99 switches it off
MARGIN = 0.05   # tie-break offset in the head: exact multiples of SCALE must round up


@torch.no_grad()
def exhaustive_check(model: SimpleTransformer, task) -> None:
    """Diagnostic only: generate greedily for every possible line and query, then print exact accuracy."""
    prompts, answers = [], []
    for m in range(1, 6):
        for b in range(10):
            for x1 in range(10):
                for x2 in range(10):
                    if x1 == x2:
                        continue
                    for x3 in range(10):
                        prompts.append(f"{x1},{m*x1+b:02d};{x2},{m*x2+b:02d};{x3}=")
                        answers.append(f"{m*x3+b:02d}")
    ids = torch.tensor([task.encode(p) for p in prompts])
    for _ in range(task.answer_len):
        ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
    target = torch.tensor([task.encode(a) for a in answers])
    wrong = (ids[:, -task.answer_len:] != target).any(1)
    print(f"EXHAUSTIVE: {len(prompts) - int(wrong.sum())}/{len(prompts)} correct")
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    print(f"NONZERO WEIGHTS: {nonzero}")
    for i in wrong.nonzero().flatten()[:5].tolist():
        print("  MISS:", prompts[i], "target", answers[i], "got", task.decode(ids[i, -task.answer_len:].tolist()))


def write_weights(model: SimpleTransformer, task) -> None:
    """Read numbers (attention), test slopes (MLP1), select the line (MLP2), floor-divide into digits (head)."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: digits carry their value; positions carry a bias, flag and attention keys.
        for d in range(10):
            model.token_emb.weight[task.stoi[str(d)], DIGIT] = float(d)
        model.pos_emb.weight[:, SCALE] = 1.0
        model.pos_emb.weight[11, SCALE] = 10.0
        for name, src_pos in SOURCES.items():
            for p, w in src_pos.items():  # keys are 0 at the output steps, so registers start empty there
                model.pos_emb.weight[p, REG[name]] = FORBID + math.log(w)

        # Block 0 attention: head `h` attends to its number's digits; 10:1 weights decode "tens,units".
        at = model.blocks[0].attn
        for h, (name, src_pos) in enumerate(SOURCES.items()):
            at.query[h] = 1.0
            at.W_k.weight[h, REG[name]] = 1.0
            at.W_v.weight[h, DIGIT] = 1.0
            at.W_o.weight[REG[name], h] = sum(src_pos.values())

        # Block 0 MLP: PEN[k] = relu(z_k) + relu(-z_k) = |z_k|, for z_k = (y2-y1) - k(x2-x1).
        fc1, fc2 = model.blocks[0].mlp.fc1, model.blocks[0].mlp.fc2
        for k in range(1, 6):
            for j, sign in enumerate((1.0, -1.0)):
                u = 2 * (k - 1) + j
                fc1.weight[u, REG["y2"]], fc1.weight[u, REG["y1"]] = sign, -sign
                fc1.weight[u, REG["x2"]], fc1.weight[u, REG["x1"]] = -sign * k, sign * k
                fc2.weight[PEN[k], u] = 1.0

        # Block 1 MLP: Y = sum_k relu(y1 + k(x3-x1) - M*PEN[k]); only the true line survives.
        fc1, fc2 = model.blocks[1].mlp.fc1, model.blocks[1].mlp.fc2
        for k in range(1, 6):
            u = k - 1
            fc1.weight[u, REG["y1"]] = 1.0
            fc1.weight[u, REG["x3"]], fc1.weight[u, REG["x1"]] = float(k), -float(k)
            fc1.weight[u, PEN[k]] = -M
            fc2.weight[Y, u] = 1.0

        # Head: V = Y - 10*printed_tens. Floor division: the logit
        # d*V - SCALE*(d(d+1)/2 - MARGIN*d) rises while V >= SCALE*(d - MARGIN),
        # so the argmax is floor(V/SCALE).
        for d in range(10):
            row = model.head.weight[task.stoi[str(d)]]
            row[Y], row[DIGIT] = float(d), -10.0 * d
            row[SCALE] = -d * (d + 1) / 2.0 + MARGIN * d
        for c in ",;=":
            model.head.weight[task.stoi[c], SCALE] = -FORBID
    exhaustive_check(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "LineHypotheses"
model_description = "One candidate line per slope: Y = sum_k relu(y1 + k(x3-x1) - M|z_k|). 15 ReLUs and a head reading only Y, but 13 channels; compared with scalar slope."


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
