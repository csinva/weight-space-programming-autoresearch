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
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
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

# Named channels of the d_model-dimensional state.
SCALE, DIGIT = 0, 1                    # place value to print (10 at tens step, 1 elsewhere) | token digit value
REG = dict(x1=2, y1=3, x2=4, y2=5, x3=6)   # number registers; before attention they hold positional keys
S, Y = 7, 8                            # the slope m as a literal number | (m-1)*t', the gated part of y3
BIAS, MIRROR, A_POS, A_NEG = 9, 10, 11, 12  # constant 1 for W_q | LayerNorm tamers, see below
SOURCES = dict(x1={0: 1.0}, y1={2: 10.0, 3: 1.0}, x2={5: 1.0}, y2={7: 10.0, 8: 1.0}, x3={10: 1.0})
READ_PLAN = (("x1", "y1", "x2", "y2"), ("x3",))  # 4 heads per block, so x3 is read in block 1
FORBID = 30.0   # attention score of positions a head must ignore
M = 40.0        # big-M: needs M/2 >= max t' = 18 to switch off an adder
A = 1e4         # size of the constant pair; far above every signal, so std is essentially constant


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
        model.pos_emb.weight[11:13, REG["x3"]] = 9.0  # the x3 register holds x3 + 9 at output steps
        for name, src_pos in SOURCES.items():
            for p, w in src_pos.items():  # keys are 0 at the output steps, so registers start empty there
                model.pos_emb.weight[p, REG[name]] = FORBID + math.log(w)

        # Attention: head `h` uses only the first of its 16 dims and attends to one number's
        # digits; 10:1 weights decode "tens,units". W_q cancels the sqrt(16) score divisor.
        model.pos_emb.weight[:, BIAS] = 1.0
        for block, names in zip(model.blocks, READ_PLAN):
            at = block.attn
            for h, name in enumerate(names):
                dim = h * at.d_head
                at.W_q.weight[dim, BIAS] = math.sqrt(at.d_head)
                at.W_k.weight[dim, REG[name]] = 1.0
                at.W_v.weight[dim, DIGIT] = 1.0
                at.W_o.weight[REG[name], dim] = sum(SOURCES[name].values())

        # Block 0 MLP: S = 1 + sum_{k=2..5} (k-1)*bump(z_k), with z_k = (y2-y1) - k(x2-x1).
        fc1, fc2 = model.blocks[0].mlp.fc1, model.blocks[0].mlp.fc2
        for k in range(2, 6):
            for j, (shift, coef) in enumerate(((1.0, 1.0), (0.0, -2.0), (-1.0, 1.0))):
                u = 3 * (k - 2) + j
                fc1.weight[u, REG["y2"]], fc1.weight[u, REG["y1"]] = 1.0, -1.0
                fc1.weight[u, REG["x2"]], fc1.weight[u, REG["x1"]] = -float(k), float(k)
                fc1.bias[u] = shift
                fc2.weight[S, u] = coef * (k - 1)
        fc2.bias[S] = 1.0

        # Block 1 MLP: add t' = x3 + 9 - x1 >= 0 once for each k in 2..m. Each adder is
        # relu(M(S-k+.5) + t') - relu(M(S-k+.5)), which is t' if S >= k and 0 otherwise;
        # the 0.5 keeps both ReLUs far from their kink, so float noise in S cannot leak in.
        fc1, fc2 = model.blocks[1].mlp.fc1, model.blocks[1].mlp.fc2
        for k in range(2, 6):
            a, b = 2 * k - 4, 2 * k - 3
            fc1.weight[a, S], fc1.weight[a, REG["x3"]], fc1.weight[a, REG["x1"]] = M, 1.0, -1.0
            fc1.bias[a] = -M * (k - 0.5)
            fc1.weight[b, S], fc1.bias[b] = M, -M * (k - 0.5)
            fc2.weight[Y, a], fc2.weight[Y, b] = 1.0, -1.0

        # Head: V = y3 - 10*printed_tens, read as the linear formula
        # y3 = y1 + m*t = y1 + t' - 9m + Y, where Y already holds (m-1)t'.
        # Floor division: the logit d*V - SCALE*(d(d+1)/2 - 0.05d) rises while
        # V >= SCALE*(d - 0.05), so the argmax is floor(V/SCALE).
        for d in range(10):
            row = model.head.weight[task.stoi[str(d)]]
            row[Y], row[REG["y1"]], row[REG["x3"]], row[REG["x1"]] = float(d), float(d), float(d), -float(d)
            row[S], row[DIGIT] = -9.0 * d, -10.0 * d
            row[SCALE] = -d * (d + 1) / 2.0 + 0.05 * d
        for c in ",;=":
            model.head.weight[task.stoi[c], SCALE] = -FORBID

        # LayerNorm becomes the identity. Every write to the state also writes its negative
        # sum to MIRROR, so channels always sum to 0 and the mean vanishes; the +A/-A pair
        # fixes std = A*sqrt(2/D), which each LayerNorm's gain multiplies back out.
        model.pos_emb.weight[:, A_POS], model.pos_emb.weight[:, A_NEG] = A, -A
        model.token_emb.weight[:, MIRROR] = -model.token_emb.weight.sum(dim=1)
        model.pos_emb.weight[:, MIRROR] = -model.pos_emb.weight.sum(dim=1)
        for block in model.blocks:
            block.attn.W_o.weight[MIRROR] = -block.attn.W_o.weight.sum(dim=0)
            block.mlp.fc2.weight[MIRROR] = -block.mlp.fc2.weight.sum(dim=0)
            block.mlp.fc2.bias[MIRROR] = -block.mlp.fc2.bias.sum()
        gain = A * math.sqrt(2.0 / model.d_model)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(gain)
    exhaustive_check(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "StockArchitecture"
model_description = "The whole circuit embedded in the unmodified default transformer (d_model 64, 4 heads, 3 layers, d_ff 256, real LayerNorm); block 2 sits idle."


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
