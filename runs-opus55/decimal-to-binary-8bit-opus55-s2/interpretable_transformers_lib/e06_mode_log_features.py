"""
Interpretable transformer for decimal-to-binary-8bit: "197=" -> "11000101".

The model is never trained. write_weights() sets every parameter in closed form.
Algorithm: greedy binary expansion, MSB first. At output position p (bit k=10-p)
    r = n - sum(emitted bits * 2^j),   emit 1 iff r >= 2^k.
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


class Block(nn.Module):
    """Attention only: no LayerNorm and no MLP, the task is linear in attended tokens."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 6,
                 n_heads: int = 2, n_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(min(max_seq_len, 11), d_model)  # 4 prompt + 7 fed-back bits
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
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
    good = 0
    for n in range(256):
        ids = task.encode(f"{n:03d}=")
        for _ in range(task.answer_len):
            logits = model(torch.tensor([ids]))
            ids = ids + [int(logits[0, -1].argmax())]
        good += task.decode(ids[task.prompt_len:]) == f"{n:08b}"
    print(f"exhaustive: {good}/256")


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    dh = model.d_model // 2
    rt = math.sqrt(dh)
    M = 20.0                     # how strongly a head prefers its own region
    VAL, EQ, MODE, LOG, OUT = 0, 1, 2, 3, 4   # feature dims
    attn, te, pe = model.blocks[0].attn, model.token_emb.weight, model.pos_emb.weight
    # token embedding: numeric value, and an "is '='" flag
    for d in range(10):
        te[task.stoi[str(d)], VAL] = d
    te[task.stoi["="], VAL] = 255 / 512          # padding 2^(k+1) * this = just under 2^k
    te[task.stoi["="], EQ] = 1.0
    # position embedding: MODE -1 on digits, +1 from '=' on; LOG = log(place value)
    pe[0:3, MODE] = -1.0
    pe[3:, MODE] = 1.0
    pe[0:3, LOG] = torch.tensor([math.log(100), math.log(10), 0.0])
    pe[3:, LOG] = torch.tensor([(11 - p) * math.log(2) for p in range(3, 11)])
    # head 0: digits weighted 100:10:1, a = n / 111
    attn.W_q.weight[0, MODE] = rt
    attn.W_k.weight[0, LOG] = 1.0
    attn.W_k.weight[0, MODE] = -M
    attn.W_v.weight[0, VAL] = 1.0
    # head 1: bits weighted by value, '=' takes the unused 2^(k+1) of 256,
    # b = (S + 2^k * 255/256) / 256
    attn.W_q.weight[dh, MODE] = rt
    attn.W_q.weight[dh + 1, LOG] = rt
    attn.W_q.weight[dh + 1, MODE] = -rt * 8 * math.log(2)
    attn.W_k.weight[dh, LOG] = 1.0
    attn.W_k.weight[dh, MODE] = M
    attn.W_k.weight[dh + 1, EQ] = 1.0
    attn.W_v.weight[dh, VAL] = 1.0
    # y = 111 a - 256 b = n - S - 2^k * 255/256, positive iff remainder >= 2^k
    attn.W_o.weight[OUT, 0] = 111.0
    attn.W_o.weight[OUT, dh] = -256.0
    model.head.weight[task.stoi["1"], OUT] = 1.0
    model.head.weight[task.stoi["0"], OUT] = -1.0
    check_all_inputs(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "e06_mode_log_features"
model_description = "d=6: position is just (MODE, log place value), token is (value, is-equals). Both heads key on LOG -/+ M*MODE; zero-sum head logit1=+y, logit0=-y."


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
