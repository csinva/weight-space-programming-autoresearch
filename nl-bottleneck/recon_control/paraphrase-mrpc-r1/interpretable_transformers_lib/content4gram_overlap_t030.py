"""
Snapshot: content4gram_overlap_t030  (paraphrase-mrpc)

Identical to char4gram_overlap_t035 except:
  * the averaging head (layer 3) counts only s2 positions whose 4-gram is
    made of 4 letters (key = beta3*(is_s2 + #letter slots - 5)),
  * THETA = 0.30.
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


class Head(nn.Module):
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        self.W_o = nn.Linear(d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        return self.W_o(scores.softmax(dim=-1) @ v)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int | None):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_model, d_head) for _ in range(n_heads)])
        self.mlp = MLP(d_model, d_ff) if d_ff else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + sum(h(x) for h in self.heads)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


V = 33
NPOS = 128
TOK = 0
POS = TOK + V
P1 = POS + NPOS
P2 = P1 + V
P3 = P2 + V
G0 = P3 + V
NG = 4 * V
FOUND = G0 + NG
RATIO = FOUND + 1
CONST = RATIO + 1
D_MODEL = CONST + 1


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 128, d_model: int = D_MODEL):
        super().__init__()
        assert vocab_size == V and max_seq_len <= NPOS
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads=3, d_head=NPOS, d_ff=NG),
            Block(d_model, n_heads=1, d_head=NG + 1, d_ff=None),
            Block(d_model, n_heads=1, d_head=2, d_ff=None),
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


THETA = 0.30
NGRAM = 4


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    L = task.SENT_LEN
    SEP = L
    S1_LO, S1_HI = NGRAM - 1, L - 1
    S2_LO, S2_HI = L + 1 + NGRAM - 1, 2 * L
    stoi = task.stoi
    letters = [stoi[c] for c in "abcdefghijklmnopqrstuvwxyz"]

    for t in range(V):
        model.token_emb.weight[t, TOK + t] = 1.0
        model.token_emb.weight[t, CONST] = 1.0
    for i in range(model.max_seq_len):
        model.pos_emb.weight[i, POS + i] = 1.0

    def is_s1(i): return S1_LO <= i <= S1_HI
    def is_s2(i): return S2_LO <= i <= S2_HI

    beta = 60.0
    b1 = model.blocks[0]
    for k, slot in zip((1, 2, 3), (P1, P2, P3)):
        h = b1.heads[k - 1]
        for i in range(k, model.max_seq_len):
            h.W_q.weight[i - k, POS + i] = beta
        for i in range(model.max_seq_len):
            h.W_k.weight[i, POS + i] = 1.0
        for t in range(V):
            h.W_v.weight[t, TOK + t] = 1.0
            h.W_o.weight[slot + t, t] = 1.0
    mlp = b1.mlp
    for s, slot in enumerate((TOK, P1, P2, P3)):
        for c in range(V):
            u = s * V + c
            mlp.fc1.weight[u, slot + c] = 1.0
            for i in range(model.max_seq_len):
                if is_s1(i):
                    mlp.fc1.weight[u, POS + i] = 1.0
            mlp.fc1.bias[u] = -1.0
            mlp.fc2.weight[G0 + u, u] = 1.0

    beta2 = 20.0
    h = model.blocks[1].heads[0]
    for s, slot in enumerate((TOK, P1, P2, P3)):
        for c in range(V):
            u = s * V + c
            h.W_q.weight[u, slot + c] = beta2
            h.W_k.weight[u, G0 + u] = 1.0
    h.W_q.weight[NG, CONST] = beta2
    for i in range(model.max_seq_len):
        if is_s1(i):
            h.W_k.weight[NG, POS + i] = -(NGRAM - 0.5)
            h.W_v.weight[0, POS + i] = 1.0
        elif i == SEP:
            h.W_k.weight[NG, POS + i] = 0.0
        else:
            h.W_k.weight[NG, POS + i] = -100.0
    h.W_o.weight[FOUND, 0] = 1.0

    beta3 = 30.0
    h = model.blocks[2].heads[0]
    h.W_q.weight[0, CONST] = beta3
    for i in range(model.max_seq_len):
        h.W_k.weight[0, POS + i] = 1.0 if is_s2(i) else -1.0
    for slot in (TOK, P1, P2, P3):
        for c in letters:
            h.W_k.weight[0, slot + c] += 1.0
    h.W_k.weight[0, CONST] = -5.0
    h.W_v.weight[0, FOUND] = 1.0
    h.W_o.weight[RATIO, 0] = 1.0

    alpha = 100.0
    model.head.weight[:, CONST] = -50.0
    model.head.weight[stoi['0'], CONST] = 0.0
    model.head.weight[stoi['1'], RATIO] = alpha
    model.head.weight[stoi['1'], CONST] = -alpha * THETA


model_shorthand_name = "content4gram_overlap_t030"
model_description = ("Same circuit, but averaging head only counts s2 4-grams made of 4 letters (content-word "
                     "substrings, no spaces); output 1 if fraction found in s1 > 0.30.")


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
