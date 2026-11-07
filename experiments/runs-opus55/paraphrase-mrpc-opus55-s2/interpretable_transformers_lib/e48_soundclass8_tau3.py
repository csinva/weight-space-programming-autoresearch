"""Compact hand-wired shared-phrase paraphrase detector. No training anywhere.

Prompt: sent1 (60 chars) | sent2 (60 chars) =   -> answer '1' / '0'
Rule:   answer '1' iff some N-char window in the first WIN chars of sent2 is
        also in sent1 (more generally: at least TAU such windows).

Chars are CHAR_BITS-bit +-1 codes ('_', '|', '=', '0', '1' are all-zero, so they
can never take part in a match). Positions are POS_BITS-bit +-1 codes. Dot products of
codes equal the bit count only for identical codes, so attention can look up
exactly. The position table stores, for each t, the code of t and of t-k.

  Block 0  window: head k reads the code of t-k and fetches that char.
  Block 1  match:  an identical window in sent1 scores N*CHAR_BITS; a sentinel
                   at '|' scores one less, catching everything inexact.
                   M = the lookup landed in sent1.
  Block 2  count:  '=' attends uniformly to the first WIN positions of sent2,
                   so WIN*mean(M) = H.
  Head:            logit('1') - logit('0') = GAIN * (H - TAU).
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
OFFSETS = list(range(8)) # which chars back from t the window samples
N = len(OFFSETS)
WIN = 40       # only the first WIN positions of sent2 are counted
S1_LIM = 60    # only the first S1_LIM positions of sent1 can be matched
TAU = 2.5        # answer '1' iff more than TAU shared windows
GAIN = 50.0      # sharpness of the final decision
BETA = 30.0      # softmax sharpness: attention acts as a hard lookup
CHAR_BITS, POS_BITS = 3, 7
# Eight sound classes: letters that sound alike share a code (after Soundex).
SOUND_CLASSES = ["aeiouy", "bfpv", "cgjkqsxz", "dt", "lr", "mn", "hw'", " "]
PIPE, EQ = 60, 121

# State layout: name -> dims, in order.
_dims = [("TOK", CHAR_BITS), ("KEY", POS_BITS), ("QRY", N * POS_BITS), ("BIAS", 1), ("IS_PIPE", 1),
         ("IN_S1", 1), ("IN_S2", 1), ("IN_WIN", 1), ("CTX", N * CHAR_BITS), ("M", 1), ("COUNT", 1)]
_offset = 0
SLOT = {}
for _name, _size in _dims:
    SLOT[_name] = _offset
    _offset += _size
D = _offset


def bits(x: int, n: int) -> torch.Tensor:
    return torch.tensor([1.0 if (x >> i) & 1 else -1.0 for i in range(n)])


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
            CausalSelfAttention(D, N, POS_BITS),            # window
            CausalSelfAttention(D, 1, N * CHAR_BITS + 2),   # match
            CausalSelfAttention(D, 1, 1),                   # count
        ])
        self.head = nn.Linear(D, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(ids) + self.pos_emb(torch.arange(ids.shape[1], device=ids.device))[None]
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


def write_weights(model: SimpleTransformer, task) -> None:
    S = SLOT
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        for cls, members in enumerate(SOUND_CLASSES):
            for ch in members:
                model.token_emb.weight[task.stoi[ch], S["TOK"]:S["TOK"] + CHAR_BITS] = bits(cls, CHAR_BITS)

        for t in range(model.pos_emb.num_embeddings):
            e = model.pos_emb.weight[t]
            e[S["KEY"]:S["KEY"] + POS_BITS] = bits(t, POS_BITS)
            for k in range(N):
                src = t if t == PIPE else max(t - OFFSETS[k], 0)     # '|' sees only itself
                e[S["QRY"] + k * POS_BITS:S["QRY"] + (k + 1) * POS_BITS] = bits(src, POS_BITS)
            e[S["BIAS"]] = 1.0
            e[S["IS_PIPE"]] = float(t == PIPE)
            e[S["IN_S1"]] = float(t < PIPE)
            e[S["IN_S2"]] = float(t > PIPE or S1_LIM <= t < PIPE)   # keys that may not match
            e[S["IN_WIN"]] = float(PIPE < t <= PIPE + WIN)

        # Window.
        a = model.blocks[0]
        for k in range(N):
            for b in range(POS_BITS):
                a.W_q.weight[k * POS_BITS + b, S["QRY"] + k * POS_BITS + b] = BETA
                a.W_k.weight[k * POS_BITS + b, S["KEY"] + b] = 1.0
            for b in range(CHAR_BITS):
                a.W_v.weight[k * POS_BITS + b, S["TOK"] + b] = 1.0
                a.W_o.weight[S["CTX"] + k * CHAR_BITS + b, k * POS_BITS + b] = 1.0

        # Match: slot bits, then [sentinel, block keys in sent2].
        a = model.blocks[1]
        W = N * CHAR_BITS
        for i in range(W):
            a.W_q.weight[i, S["CTX"] + i] = BETA
            a.W_k.weight[i, S["CTX"] + i] = 1.0
        a.W_q.weight[W, S["BIAS"]] = BETA * (W - 1)
        a.W_k.weight[W, S["IS_PIPE"]] = 1.0
        a.W_q.weight[W + 1, S["BIAS"]] = -BETA * 1000
        a.W_k.weight[W + 1, S["IN_S2"]] = 1.0
        a.W_v.weight[0, S["IN_S1"]] = 1.0
        a.W_o.weight[S["M"], 0] = 1.0

        # Count.
        a = model.blocks[2]
        a.W_q.weight[0, S["BIAS"]] = BETA
        a.W_k.weight[0, S["IN_WIN"]] = 1.0
        a.W_v.weight[0, S["M"]] = 1.0
        a.W_o.weight[S["COUNT"], 0] = 1.0

        # Head: only '0' and '1' are reachable (1000 lifts them over other chars).
        one, zero = task.stoi["1"], task.stoi["0"]
        model.head.weight[one, S["COUNT"]] = GAIN * WIN
        model.head.weight[one, S["BIAS"]] = 1000.0 - GAIN * TAU
        model.head.weight[zero, S["BIAS"]] = 1000.0
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


model_shorthand_name = "e48_soundclass8_tau3"
model_description = "8 Soundex-style sound classes; answer 1 iff at least 3 8-char sound windows in the first 40 chars of sent2 are in sent1."


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
