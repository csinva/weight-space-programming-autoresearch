"""
Interpretable transformer for decimal-to-binary-8bit: "197=" -> "11000101".

The model is never trained. write_weights() sets every parameter in closed form.
This variant is feedback-free: attention computes n once, then a stack of 8 MLP
stages runs restoring subtraction, and each output position reads its own stage.
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
    """Only the first block has attention; every block has one subtractor MLP."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_attn: bool):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads) if use_attn else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 13,
                 n_heads: int = 1, n_layers: int = 8, d_ff: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(min(max_seq_len, 11), d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, i == 0) for i in range(n_layers)])
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

@torch.no_grad()
def check_all_inputs(model, task) -> None:
    """Greedy-decode all 256 inputs; a test only, the model never sees the targets."""
    good, margin = 0, float("inf")
    for n in range(256):
        ids = task.encode(f"{n:03d}=")
        for _ in range(task.answer_len):
            logits = model(torch.tensor([ids]))[0, -1]
            top2 = logits.topk(2).values
            margin = min(margin, float(top2[0] - top2[1]))
            ids = ids + [int(logits.argmax())]
        good += task.decode(ids[task.prompt_len:]) == f"{n:08b}"
    print(f"exhaustive: {good}/256, min logit margin {margin:.4f}")


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    rt = math.sqrt(model.d_model)
    VAL, MODE, LOG, R, OUT, GATE = 0, 1, 2, 3, 4, 5   # GATE..GATE+7 = output-position one-hot
    attn, te, pe = model.blocks[0].attn, model.token_emb.weight, model.pos_emb.weight
    for d in range(10):
        te[task.stoi[str(d)], VAL] = d
    pe[0:3, MODE] = -1.0
    pe[3:, MODE] = 1.0
    pe[0:3, LOG] = torch.tensor([math.log(100), math.log(10), 0.0])
    for p in range(3, 11):
        pe[p, GATE + p - 3] = 1.0
    # attention: R = n, digits weighted 100:10:1 and values scaled by their total
    attn.W_q.weight[0, MODE] = rt
    attn.W_k.weight[0, LOG] = 1.0
    attn.W_k.weight[0, MODE] = -M
    attn.W_v.weight[0, VAL] = 111.0
    attn.W_o.weight[R, 0] = 1.0
    # stage j tests bit k = 7 - j. On integer R, relu(R - t + 1) - relu(R - t) is the step R >= t.
    for j, block in enumerate(model.blocks):
        t = 2 ** (7 - j)
        fc1, fc2 = block.mlp.fc1, block.mlp.fc2
        fc1.weight[[0, 1, 2, 3], R] = 1.0
        fc1.bias[0], fc1.bias[1] = 1.0 - t, -t
        # neurons 2, 3 repeat the step but are held off unless this is stage j's own position
        fc1.weight[[2, 3], GATE + j] = BIG
        fc1.bias[2], fc1.bias[3] = 1.0 - t - BIG, -t - BIG
        fc2.weight[R, 0], fc2.weight[R, 1] = -t, t       # restoring subtraction: R -= t * step
        fc2.weight[OUT, 2], fc2.weight[OUT, 3] = 2.0, -2.0  # OUT = 2 * step at own position
    # logit('1') = OUT - 1/2, MODE supplies the constant, logit('0') is its negative
    model.head.weight[task.stoi["1"], OUT] = 1.0
    model.head.weight[task.stoi["1"], MODE] = -0.5
    model.head.weight[task.stoi["0"]] = -model.head.weight[task.stoi["1"]]
    check_all_inputs(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
M = 20.0                     # how strongly attention prefers digits over everything else
BIG = 1000.0                 # holds a gated neuron off at the wrong position
model_shorthand_name = "e38_mlp_ripple_no_feedback"
model_description = "Feedback-free: attention computes n once, then 8 stacked MLP stages run restoring subtraction and each position reads its own stage. Ignores emitted bits."


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
