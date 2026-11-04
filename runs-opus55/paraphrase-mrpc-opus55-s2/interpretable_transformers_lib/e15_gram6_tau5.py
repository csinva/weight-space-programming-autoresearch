"""Hand-wired shared-phrase paraphrase detector. No training anywhere.

Prompt: sent1 (60 chars) | sent2 (60 chars) =   -> answer '1' / '0'

  Block 0  window: head k copies the char k steps back, so each position holds
                   its last N chars as N one-hot slots.
  Block 1  match:  each position queries earlier windows. An identical window
                   in sent1 scores N; a sentinel at '|' scores N-0.5, so it
                   catches every query that has no exact match. M = hit sent1.
  Block 2  count:  zero q/k gives uniform attention over the 122 positions, so
                   122*mean(M) - 60 = H, the number of sent2 windows found in
                   sent1 (all 60 sent1 positions trivially set M).
  Head:            answer '1' iff H > TAU.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]

# Hand-chosen constants
N = 6            # window length in chars
TAU = 4.5        # minimum number of shared windows to call it a paraphrase
GAIN = 50.0      # sharpness of the final decision
BETA = 30.0      # softmax sharpness: attention acts as a hard lookup

V, P = 33, 123
POS, CTX = V, V + P              # token one-hot, position one-hot, N char slots
MEAN_M = CTX + N * V + 1         # M is stored just before it
D = MEAN_M + 1
PIPE, EQ, PROMPT_LEN = 60, 121, 122


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o((attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh))


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 123):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, D)
        self.pos_emb = nn.Embedding(max_seq_len, D)
        self.blocks = nn.ModuleList([
            CausalSelfAttention(D, N, P),          # window
            CausalSelfAttention(D, 1, N * V + 2),  # match
            CausalSelfAttention(D, 1, 1),          # count
        ])
        self.head = nn.Linear(D, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(ids) + self.pos_emb(torch.arange(ids.shape[1], device=ids.device))[None]
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


def write_weights(model: SimpleTransformer, task) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.token_emb.weight[:, :V] = torch.eye(V)
        model.pos_emb.weight[:, POS:POS + P] = torch.eye(P)
        bias = list(range(POS, POS + P))                 # sum of pos one-hot == 1

        # Window: head k looks up position max(t-k, 0); '|' only sees itself.
        a = model.blocks[0]
        for k in range(N):
            for t in range(P):
                a.W_q.weight[k * P + (t if t == PIPE else max(t - k, 0)), POS + t] = BETA
                a.W_k.weight[k * P + t, POS + t] = 1.0
            for c in range(V):
                a.W_v.weight[k * P + c, c] = 1.0
                a.W_o.weight[CTX + k * V + c, k * P + c] = 1.0

        # Match: slot dims (padding never counts), then [sentinel, block sent2 keys].
        a = model.blocks[1]
        SENT, BLOCK = N * V, N * V + 1
        for i in range(N * V):
            if i % V != task.stoi["_"]:
                a.W_q.weight[i, CTX + i] = BETA
                a.W_k.weight[i, CTX + i] = 1.0
        a.W_q.weight[SENT, bias] = BETA * (N - 0.5)
        a.W_k.weight[SENT, POS + PIPE] = 1.0
        a.W_q.weight[BLOCK, bias] = -BETA * 1000
        a.W_k.weight[BLOCK, POS + PIPE + 1:POS + P] = 1.0
        a.W_v.weight[0, POS:POS + PIPE] = 1.0            # v: the hit was in sent1
        a.W_o.weight[MEAN_M - 1, 0] = 1.0

        # Count: q = k = 0, so attention is uniform and v averages M.
        a = model.blocks[2]
        a.W_v.weight[0, MEAN_M - 1] = 1.0
        a.W_o.weight[MEAN_M, 0] = 1.0

        # Head: logit('1') - logit('0') = GAIN * (H - TAU). 1000 lifts 0/1 over other chars.
        one, zero = task.stoi["1"], task.stoi["0"]
        model.head.weight[one, MEAN_M] = GAIN * PROMPT_LEN
        model.head.weight[one, bias] = 1000.0 - GAIN * (PIPE + TAU)
        model.head.weight[zero, bias] = 1000.0
        dev_report(model, task)


def dev_report(model, task) -> None:
    """Accuracy on pool examples with the seed=0 eval prompts removed."""
    held_out = {e.prompt for e in task.generate_examples(200, seed=0)}
    ex = {e.prompt: e for e in task.generate_examples(8000, seed=999) if e.prompt not in held_out}
    ex = list(ex.values())
    ids = torch.tensor([task.encode(e.prompt) for e in ex])
    pred = model(ids)[:, EQ].argmax(-1).tolist()
    y = [task.stoi[e.answer] for e in ex]
    one = task.stoi["1"]
    print(f"DEV n={len(y)} acc={sum(p == t for p, t in zip(pred, y)) / len(y):.4f} "
          f"always1={sum(t == one for t in y) / len(y):.4f} pred1={sum(p == one for p in pred) / len(y):.3f}")


model_shorthand_name = "e15_gram6_tau5"
model_description = "Attention-only circuit: answer 1 iff at least 5 6-char windows of sent2 appear verbatim in sent1."


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
