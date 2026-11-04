"""Hand-wired word-stem paraphrase detector. No training anywhere.

Prompt: sent1 (60 chars) | sent2 (60 chars) =   -> answer '1' / '0'

Circuit:
  Block 0  window:    head k copies the char k steps back; a 1-neuron MLP ANDs
                      "L letters preceded by a word boundary" into STEM.
  Block 1  match:     each stem queries earlier positions. A sent2 stem found in
                      sent1 -> M=1; otherwise a sentinel at '|' wins -> U=1.
                      Non-stem sent2 positions fall to a null key (M=U=0).
  Block 2  count:     '=' attends uniformly to all 122 positions, so mean(flag)
                      * 122 is a count. This yields, causally:
                      H (shared stems), Miss2 (sent2 stems absent from sent1),
                      Miss1 (sent1 stems with no partner, N1 - H).
  Head:               logit('1') - logit('0') ∝ THETA + W_H*H - W_M2*Miss2 - W_M1*Miss1
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

# Hand-chosen constants
L = 4            # stem length in letters
THETA, W_H, W_M2, W_M1 = 6.5, 0.0, 1.0, 1.0   # decision rule on word counts
GAIN = 50.0      # sharpness of the final decision
BETA = 30.0      # softmax sharpness: attention acts as a hard lookup
DEBUG = True


class Layout:
    V, P = 33, 123
    TOK = 0
    POS = V
    CTX = V + P                     # (L+1) one-hot slots: char k steps back
    STEM = CTX + (L + 1) * V
    M, U = STEM + 1, STEM + 2
    MEAN_M, MEAN_U, MEAN_S = STEM + 3, STEM + 4, STEM + 5
    D = STEM + 6
    PIPE, EQ, NULL, N = 60, 121, 61, 122    # '|', '=', null key, prompt length


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


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 123):
        super().__init__()
        D, V, P = Layout.D, Layout.V, Layout.P
        self.token_emb = nn.Embedding(vocab_size, D)
        self.pos_emb = nn.Embedding(max_seq_len, D)
        self.blocks = nn.ModuleList([
            Block(D, L + 1, P, 1),            # window + stem AND
            Block(D, 1, L * V + 4, 1),        # match against sent1
            Block(D, 1, 3, 1),                # count at '='
        ])
        self.head = nn.Linear(D, vocab_size, bias=False)

    def states(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(ids) + self.pos_emb(torch.arange(ids.shape[1], device=ids.device))[None]
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.states(ids))


def write_weights(model: SimpleTransformer, task) -> None:
    A = Layout
    V, P = A.V, A.P
    LETTERS = [task.stoi[c] for c in "abcdefghijklmnopqrstuvwxyz'"]
    BOUNDS = [task.stoi[" "], task.stoi["|"]]
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        ctx = lambda k, c=0: A.CTX + k * V + c            # slot for char c, k steps back
        model.token_emb.weight[:, A.TOK:A.TOK + V] = torch.eye(V)
        model.pos_emb.weight[:, A.POS:A.POS + P] = torch.eye(P)
        bias = list(range(A.POS, A.POS + P))              # sum of pos one-hot == 1

        # Block 0: head k looks up position max(t-k, 0) and copies its token.
        b = model.blocks[0]
        for k in range(L + 1):
            for t in range(P):
                src = t if t == A.PIPE else max(t - k, 0)  # sentinel's window is pure '|'
                b.attn.W_q.weight[k * P + src, A.POS + t] = BETA
                b.attn.W_k.weight[k * P + t, A.POS + t] = 1.0
            for c in range(V):
                b.attn.W_v.weight[k * P + c, A.TOK + c] = 1.0
                b.attn.W_o.weight[ctx(k, c), k * P + c] = 1.0
        # STEM = AND(L letters, boundary before them). pos L-1 counts as boundary.
        for k in range(L):
            b.mlp.fc1.weight[0, [ctx(k, c) for c in LETTERS]] = 1.0
        b.mlp.fc1.weight[0, [ctx(L, c) for c in BOUNDS]] = 1.0
        b.mlp.fc1.weight[0, A.POS + L - 1] = 1.0
        b.mlp.fc1.bias[0] = -float(L)
        b.mlp.fc2.weight[A.STEM, 0] = 1.0

        # Block 1: key dims = L slots, then [stem, sentinel, null, block].
        a = model.blocks[1].attn
        STEM_D, SENT_D, NULL_D, BLOCK_D = L * V, L * V + 1, L * V + 2, L * V + 3
        for k in range(L):
            for c in range(V):
                a.W_q.weight[k * V + c, ctx(k, c)] = BETA
                a.W_k.weight[k * V + c, ctx(k, c)] = 1.0
        a.W_q.weight[STEM_D, bias] = BETA                 # +1 for keys that are stems
        a.W_k.weight[STEM_D, A.STEM] = 1.0
        a.W_q.weight[SENT_D, bias] = BETA * (L + 0.5)     # sentinel beats partial matches
        a.W_k.weight[SENT_D, A.POS + A.PIPE] = 1.0
        a.W_q.weight[NULL_D, bias] = BETA * (2 * L + 4)   # non-stem queries go to null...
        a.W_q.weight[NULL_D, A.STEM] = -BETA * (2 * L + 4)  # ...stem queries do not
        a.W_k.weight[NULL_D, A.POS + A.NULL] = 1.0
        a.W_q.weight[BLOCK_D, bias] = -BETA * 1000        # sent2 positions never act as keys
        a.W_k.weight[BLOCK_D, A.POS + A.NULL + 1:A.POS + P] = 1.0
        for s in range(A.PIPE):
            a.W_v.weight[0, A.POS + s] = 1.0              # v[0]: hit a sent1 position
        a.W_v.weight[1, A.POS + A.PIPE] = 1.0             # v[1]: hit the sentinel
        a.W_o.weight[A.M, 0] = 1.0
        a.W_o.weight[A.U, 1] = 1.0

        # Block 2: zero q/k -> uniform attention -> v is a mean over the prompt.
        a = model.blocks[2].attn
        for i, (src, dst) in enumerate([(A.M, A.MEAN_M), (A.U, A.MEAN_U), (A.STEM, A.MEAN_S)]):
            a.W_v.weight[i, src] = 1.0
            a.W_o.weight[dst, i] = 1.0

        # Counts from means at '=': every sent1 position sets M, and '|' sets U.
        #   H = N*mM - 60;  Miss2 = N*mU - 1;  Miss1 = N1 - H = N*mS - 2H - Miss2
        # Head: S = THETA + W_H*H - W_M2*Miss2 - W_M1*Miss1, expanded over the means.
        one, zero = task.stoi["1"], task.stoi["0"]
        c_M = A.N * (W_H + 2 * W_M1)
        c_U = A.N * (W_M1 - W_M2)
        c_S = -A.N * W_M1
        const = THETA - A.PIPE * (W_H + 2 * W_M1) - (W_M1 - W_M2)
        model.head.weight[one, A.MEAN_M] = GAIN * c_M
        model.head.weight[one, A.MEAN_U] = GAIN * c_U
        model.head.weight[one, A.MEAN_S] = GAIN * c_S
        model.head.weight[one, bias] = 1000.0 + GAIN * const   # lifts 0/1 above other chars
        model.head.weight[zero, bias] = 1000.0

        if DEBUG:
            check_against_reference(model, task)


def dev_examples(task) -> list:
    """Pool examples with the seed=0 eval prompts removed (tuning data disjoint from eval)."""
    held_out = {e.prompt for e in task.generate_examples(200, seed=0)}
    seen, dev = set(), []
    for e in task.generate_examples(8000, seed=999):
        if e.prompt not in held_out and e.prompt not in seen:
            seen.add(e.prompt)
            dev.append(e)
    return dev


def check_against_reference(model, task) -> None:
    """Compare the circuit's counts to plain Python, then report dev accuracy."""
    A = Layout
    ex = dev_examples(task)
    ids = torch.tensor([task.encode(e.prompt) for e in ex])
    h = model.states(ids)[:, A.EQ]
    Hc = A.N * h[:, A.MEAN_M] - A.PIPE
    Uc = A.N * h[:, A.MEAN_U] - 1
    bad = 0
    for e, H, U in zip(ex, Hc.tolist(), Uc.tolist()):
        a, b = e.prompt[:-1].split("|")
        S1 = {w[:L] for w in a.rstrip("_").split() if len(w) >= L}
        S2 = [w[:L] for w in b.rstrip("_").split() if len(w) >= L]
        hits = sum(w in S1 for w in S2)
        bad += abs(hits - H) > 0.01 or abs(len(S2) - hits - U) > 0.01
    pred = model(ids)[:, A.EQ].argmax(-1).tolist()
    y = [task.stoi[e.answer] for e in ex]
    print(f"count check: mismatches={bad}/{len(ex)}")
    print(f"DEV n={len(y)} acc={sum(p == t for p, t in zip(pred, y)) / len(y):.4f} "
          f"always1={sum(t == task.stoi['1'] for t in y) / len(y):.4f} "
          f"pred1={sum(p == task.stoi['1'] for p in pred) / len(y):.3f}")


model_shorthand_name = "e11_counts_L4_miss12"
model_description = "Causal stem-4 circuit with uniform-attention word counts; answer 1 iff Miss1+Miss2 <= 6."


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
