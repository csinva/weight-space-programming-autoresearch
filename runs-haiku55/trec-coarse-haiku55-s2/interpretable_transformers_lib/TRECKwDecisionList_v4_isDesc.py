"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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
        # LayerNorm is disabled (identity) so every residual-stream channel stays linear.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """2-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 96,
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

# Trigram decision list (v1): (trigram, class). Earlier = higher priority.
# Classes: 0 ABBR, 1 ENTY, 2 DESC, 3 HUM, 4 LOC, 5 NUM.
TREC_KEYWORDS = [
    ("abb", 0), ("acr", 0),
    ("any", 5), ("muc", 5), ("hen", 5), ("yea", 5), ("dat", 5), ("old", 5), ("far", 5),
    ("pop", 5), ("umb", 5), ("erc", 5), ("tur", 5),
    ("her", 4), ("cou", 4), ("cit", 4), ("tat", 4), ("riv", 4), ("nta", 4), ("isl", 4),
    ("ont", 4), ("cea", 4), ("pit", 4), ("lak", 4), ("loc", 4), ("lac", 4), ("orl", 4),
    ("who", 3), ("rso", 3), ("peo", 3), ("aut", 3), ("org", 3),
    ("why", 2), ("how", 2), ("mea", 2), ("def", 2), ("des", 2), ("rea", 2), ("cau", 2),
    ("pur", 2), ("dif", 2),
    ("ani", 1), ("olo", 1), ("foo", 1), ("gam", 1), ("ord", 1), ("car", 1), ("boo", 1),
    ("ilm", 1), ("eli", 1), ("urr", 1), ("eta", 1), ("nam", 1), ("spo", 1), ("sym", 1),
    ("dog", 1), ("cat", 1), ("bir", 1), ("lan", 1), ("mus", 1), ("hap", 1), ("let", 1),
    ("eme", 1), ("ubs", 1),
    # lowest priority: "what is ..." with nothing else -> DESC
    ("is ", 2),
]
TREC_DEFAULT = 1  # class when no keyword fires (ENTY)
ALPHA = 15.0      # attention sharpness: score = ALPHA * priority (softmax ~ hard max)


def _pos_code(p: int, dim: int = 19) -> torch.Tensor:
    """Two-hot position code: one-hot(p % 9) ++ one-hot(p // 9). Zero for p < 0."""
    v = torch.zeros(dim)
    if p >= 0:
        v[p % 9] = 1.0
        v[9 + p // 9] = 1.0
    return v


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training.

    Layer 1: previous-character heads (offsets -1, -2) + one MLP unit per trigram
    (char t-2, t-1, t). Layer 2: priority attention from the final position picks
    the highest-priority trigram present; no trigram -> default class.
    """
    D, H = model.d_model, model.n_heads
    dh = D // H
    V = model.vocab_size
    NCLS = 6
    K = len(TREC_KEYWORDS)
    stoi = task.stoi

    TOK, POSK, POSQ1, POSQ2, CONST = 0, 36, 55, 74, 93
    P1, P2, FEAT = 94, 130, 166
    VOTE = FEAT + K
    assert VOTE + NCLS <= D and V <= dh

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        for _, i in stoi.items():
            model.token_emb.weight[i, TOK + i] = 1.0
            model.token_emb.weight[i, CONST] = 1.0
        for t in range(model.max_seq_len):
            model.pos_emb.weight[t, POSK:POSK + 19] = _pos_code(t)
            model.pos_emb.weight[t, POSQ1:POSQ1 + 19] = _pos_code(t - 1)
            model.pos_emb.weight[t, POSQ2:POSQ2 + 19] = _pos_code(t - 2)

        L1 = model.blocks[0]
        s = ALPHA * math.sqrt(dh)
        for j in range(19):
            L1.attn.W_q.weight[j, POSQ1 + j] = s
            L1.attn.W_k.weight[j, POSK + j] = 1.0
            L1.attn.W_q.weight[dh + j, POSQ2 + j] = s
            L1.attn.W_k.weight[dh + j, POSK + j] = 1.0
        for j in range(V):
            L1.attn.W_v.weight[j, TOK + j] = 1.0
            L1.attn.W_v.weight[dh + j, TOK + j] = 1.0
            L1.attn.W_o.weight[P1 + j, j] = 1.0
            L1.attn.W_o.weight[P2 + j, dh + j] = 1.0

        for k, (tri, _cls) in enumerate(TREC_KEYWORDS):
            a, b, c = tri
            L1.mlp.fc1.weight[k, P2 + stoi[a]] = 1.0
            L1.mlp.fc1.weight[k, P1 + stoi[b]] = 1.0
            L1.mlp.fc1.weight[k, TOK + stoi[c]] = 1.0
            L1.mlp.fc1.bias[k] = -2.5
            L1.mlp.fc2.weight[FEAT + k, k] = 2.0

        L2 = model.blocks[1]
        L2.attn.W_q.weight[0, CONST] = s
        for k, (tri, cls) in enumerate(TREC_KEYWORDS):
            L2.attn.W_k.weight[0, FEAT + k] = float(K - k)
            for c in range(NCLS):
                L2.attn.W_v.weight[c, FEAT + k] = float(c == cls) - float(c == TREC_DEFAULT)
        for c in range(NCLS):
            L2.attn.W_v.weight[c, CONST] = float(c == TREC_DEFAULT)
            L2.attn.W_o.weight[VOTE + c, c] = 1.0

        for c in range(NCLS):
            model.head.weight[stoi[str(c)], VOTE + c] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "TRECKwDecisionList_v4_isDesc"
model_description = (
    "v3 plus a lowest-priority DESC trigram 'is ' so 'what is ...' questions with no other keyword are DESC."
)


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

    accuracy, _details = evaluate(
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
