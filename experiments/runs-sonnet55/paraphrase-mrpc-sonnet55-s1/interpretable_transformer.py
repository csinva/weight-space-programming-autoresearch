"""
Interpretable transformer for character-level sequence tasks (task: paraphrase-mrpc).

Prompt layout (122 chars): sentence A (60 chars) '|' sentence B (60 chars) '=' ; answer is '0'/'1'.
Hand-written circuit (no training):
  L1  head0 / head1 : previous-token / previous-previous-token heads (copy token into PREV1/PREV2)
  L2  head0         : at each B position, find an A position with the same character trigram;
                      FLAG = attention mass on A keys (a sink at '|' absorbs the no-match case)
  L3  head0         : at '=', uniformly average FLAG over B positions -> FAVG (fraction of B trigrams in A)
  head              : logit('1') - logit('0') = 20 * (FAVG - THR)
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
        # no layer norms: the residual stream is an explicit set of named feature slots
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 256,
        n_heads: int = 2,
        n_layers: int = 3,
        d_ff: int = 1,
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
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

# NG: n-gram order of the match head; THR: fraction of B n-grams found in A above which we answer '1';
# S_MATCH: attention score per matching feature block; EXCL_SPACE: don't count B positions holding ' '.
NG, THR, S_MATCH, EXCL_SPACE = 3, 0.14, 20.0, True
DH = 128       # head dim (d_model 256 / 2 heads)


def write_weights(model: SimpleTransformer, task) -> None:
    V = task.vocab_size
    L = 60                      # sentence length
    SEP, EQ = L, 2 * L + 1      # positions of '|' and '='
    # residual-stream slots
    TOK, POS = 0, V             # one-hot token / one-hot position
    P1 = POS + model.max_seq_len
    P2 = P1 + V
    FLAG = P2 + V
    FAVG = FLAG + 1
    CONST = FAVG + 1
    assert CONST < model.d_model
    sq = math.sqrt(DH)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for t in range(V):
            model.token_emb.weight[t, TOK + t] = 1.0
        model.token_emb.weight[:, CONST] = 1.0
        for i in range(model.max_seq_len):
            model.pos_emb.weight[i, POS + i] = 1.0

        # ---- Layer 1: previous-token heads (head h attends to position i-(h+1)) ----
        a = model.blocks[0].attn
        for h in range(2):
            off = h * DH
            for i in range(model.max_seq_len):
                if i - (h + 1) >= 0:
                    a.W_q.weight[off + i - (h + 1), POS + i] = 30.0 * sq  # query "wants" pos i-1-h
                a.W_k.weight[off + i, POS + i] = 1.0
            for t in range(V):
                a.W_v.weight[off + t, TOK + t] = 1.0
                a.W_o.weight[(P1 if h == 0 else P2) + t, off + t] = 1.0

        # ---- Layer 2: trigram lookup from B positions into A ----
        a = model.blocks[1].attn
        # query/key layout inside head 0: [0:V) tok, [V:2V) prev1, [2V:3V) prev2, 3V a-bonus, 3V+1 sink
        for t in range(V):
            for k, src in enumerate((TOK, P1, P2)[:NG]):
                a.W_q.weight[k * V + t, src + t] = S_MATCH * sq
                a.W_k.weight[k * V + t, src + t] = 1.0
        K_A = NG * S_MATCH          # bonus for being an A key (kills B-self matches)
        C_SINK = (NG - 0.5) * S_MATCH + K_A  # sink score at '|': between "NG-1 of NG match" and "NG of NG match"
        a.W_q.weight[3 * V, CONST] = K_A * sq
        a.W_q.weight[3 * V + 1, CONST] = C_SINK * sq
        for j in range(L):
            a.W_k.weight[3 * V, POS + j] = 1.0
            a.W_v.weight[0, POS + j] = 1.0        # value = "this key is in A"
        a.W_k.weight[3 * V + 1, POS + SEP] = 1.0
        a.W_o.weight[FLAG, 0] = 1.0

        # ---- Layer 3: at '=', average FLAG over B positions with a full trigram ----
        a = model.blocks[2].attn
        a.W_q.weight[0, CONST] = 10.0 * sq
        for j in range(SEP + NG, EQ):
            a.W_k.weight[0, POS + j] = 1.0
        if EXCL_SPACE:   # B positions whose current char is a space are not counted
            a.W_k.weight[0, TOK + task.stoi[" "]] = -100.0
        a.W_v.weight[0, FLAG] = 1.0
        a.W_o.weight[FAVG, 0] = 1.0

        # ---- Readout ----
        zero, one = task.stoi["0"], task.stoi["1"]
        for tok in (zero, one):
            model.head.weight[tok, CONST] = 5.0
        model.head.weight[one, FAVG] = 20.0
        model.head.weight[zero, CONST] = 5.0 + 20.0 * THR


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
VARIANT = f"{NG}gram" + ("NoSp" if EXCL_SPACE else "")
model_shorthand_name = f"{VARIANT}Thr{round(THR * 100):02d}_s{round(S_MATCH)}"
model_description = (f"3-layer circuit: prev-token heads, B->A {VARIANT} match head with sink, mean over B at '=', "
                     f"threshold {THR}, match score {S_MATCH}.")


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
