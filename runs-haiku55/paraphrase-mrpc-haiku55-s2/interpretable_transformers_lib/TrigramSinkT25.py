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

# Number of characters per match key (K=1 unigram, 2 bigram, 3 trigram, ...).
# Architecture depth/width depend on it: K+1 layers, d_model = 164 + 33*(K-1).
K = 3


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
    """Causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 164 + 33 * (K - 1),
        n_heads: int = 1,
        n_layers: int = K + 1,
        d_ff: int = 4,
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

T_UNIT = 65.0    # attention score of one matched character (after 1/sqrt(D) scaling)
PEN = 20.0       # penalty making disallowed keys unattractive
CM = 0.1         # scale of the match channels (keeps LayerNorm ~constant)
THETA = 0.25     # fraction of matched s2 k-grams needed to predict '1'  (re-run with fixed readout)


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Residual-stream layout (d_model = 164 + 33*(K-1)):
      CH     one-hot of the current character
      POS    one-hot of the absolute position
      REG    region flags: sentence-1 / separator '|' / sentence-2 / '='
      PREV_k one-hot of the character k positions back (k = 1..K-1)
      C      constant 1 at every token (used as the query/key "bias" channel)
      M      CM * (k-gram of this s2 position occurs in s1)
      N      CM * (1 - M/CM): mass the separator sink gets; M + N = CM everywhere
      FR     F = mean over s2 positions of M (the matched fraction, linear)

    Layers 0..K-2 shift the previous character(s) into PREV_k with a sharp
    "attend to position p-1" head. Layer K-1 scores each s2 query against each
    s1 key by the number of matching characters (CH, PREV_1..PREV_{K-1}); a
    separator "sink" key with score (K-0.5)*T_UNIT is only beaten by a full
    k-gram match, so the attention output is 1 iff the k-gram occurs in s1.
    Layer K averages M over the non-pad s2 positions. The readout compares F
    with a threshold.
    """
    D = model.d_model
    V = task.vocab_size
    S = model.max_seq_len
    L = task.SENT_LEN
    stoi = task.stoi
    assert len(model.blocks) == K + 1

    CH0, POS0, REG0 = 0, V, V + S
    S1, SEP, S2, EQ = REG0, REG0 + 1, REG0 + 2, REG0 + 3
    PREV0 = {k: REG0 + 4 + (k - 1) * V for k in range(1, K)}
    C = REG0 + 4 + (K - 1) * V
    M, N, FR = C + 1, C + 2, C + 3
    assert FR + 1 == D
    UND = stoi["_"]

    def attn_scale(n):
        """Scale so one matching one-hot pair contributes T_UNIT, given n hot entries."""
        mu = n / D
        sig = math.sqrt(n / D - mu * mu)
        h1 = (1 - mu) / sig
        return T_UNIT * math.sqrt(D) / h1 ** 2, h1

    with torch.no_grad():
        # Start from zeros so every weight is set explicitly.
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)

        # Embeddings
        for c in range(V):
            model.token_emb.weight[c, CH0 + c] = 1.0
        model.token_emb.weight[:, C] = 1.0
        for p in range(S):
            model.pos_emb.weight[p, POS0 + p] = 1.0
            if p < L:
                model.pos_emb.weight[p, S1] = 1.0
            elif p == L:
                model.pos_emb.weight[p, SEP] = 1.0
            elif p <= 2 * L:
                model.pos_emb.weight[p, S2] = 1.0
            elif p == 2 * L + 1:
                model.pos_emb.weight[p, EQ] = 1.0

        # Layers 0..K-2: shift heads. Position p attends to p-1 and copies the
        # previous block into PREV_{i+1}. The separator attends to itself so its
        # previous characters are '|' (no bogus match across the boundary).
        for i in range(K - 1):
            A = model.blocks[i].attn
            a, h1 = attn_scale(4 + i)
            src = CH0 if i == 0 else PREV0[i]
            dst = PREV0[i + 1]
            for m in range(S - 1):
                A.W_q.weight[POS0 + m, POS0 + m + 1] = a
            for m in range(S):
                A.W_k.weight[POS0 + m, POS0 + m] = 1.0
            A.W_q.weight[POS0 + L, SEP] = 2 * a
            for j in range(V):
                A.W_v.weight[dst + j, src + j] = 1.0 / h1
                A.W_o.weight[dst + j, dst + j] = 1.0

        # Layer K-1: k-gram match of each s2 query against s1 keys.
        n_match = 4 + (K - 1)
        a, h1 = attn_scale(n_match)
        A = model.blocks[K - 1].attn
        for start in [CH0] + [PREV0[k] for k in range(1, K)]:
            for j in range(V):
                A.W_q.weight[start + j, start + j] = a
                A.W_k.weight[start + j, start + j] = 1.0
        A.W_q.weight[C, C] = a
        A.W_k.weight[C, SEP] = K - 0.5          # sink: beats (K-1) matches, loses to K
        A.W_k.weight[C, S2] = -PEN              # never attend to s2 keys
        A.W_k.weight[C, CH0 + UND] = -PEN       # never attend to padding
        A.W_v.weight[M, S1] = 1.0 / h1          # value 1 on s1 keys
        A.W_o.weight[M, M] = CM
        A.W_v.weight[N, SEP] = 1.0 / h1         # value 1 on the sink
        A.W_o.weight[N, N] = CM

        # Layer K: '=' attends uniformly to non-pad s2 positions and averages M.
        A = model.blocks[K].attn
        a, _ = attn_scale(n_match)
        A.W_q.weight[C, C] = a
        A.W_k.weight[C, S1] = -PEN
        A.W_k.weight[C, SEP] = -PEN
        A.W_k.weight[C, EQ] = -PEN
        A.W_k.weight[C, CH0 + UND] = -PEN       # never average over padding
        A.W_v.weight[FR, M] = 1.0
        A.W_o.weight[FR, FR] = 1.0

        # Readout: '1' iff F > T_h. Both answer logits get +1 * LN(C) so they are
        # positive (beats the zero logits of other characters); the difference
        # logit('1') - logit('0') = LN(F) - T_h * LN(C) is unchanged.
        mu = n_match / D
        sig = math.sqrt(n_match / D - mu * mu)
        T_h = THETA * (CM / sig) - mu / sig
        model.head.weight[stoi["0"], C] = T_h + 1.0
        model.head.weight[stoi["1"], C] = 1.0
        model.head.weight[stoi["1"], FR] = 1.0
    model.FR_IDX = FR


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "TrigramSinkT25"
model_description = "K=3 trigram circuit with THETA=0.25 (re-run after fixing the negative-'0'-logit readout bug): predict 1 iff >=25% of s2 trigrams occur in s1."


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
    # DEBUG_START
    exs = task.generate_examples(400, seed=1)
    ids = torch.tensor([task.encode(e.prompt) for e in exs])
    ys = [int(e.answer) for e in exs]
    with torch.no_grad():
        h = model.token_emb(ids) + model.pos_emb(torch.arange(ids.shape[1]))[None]
        for blk in model.blocks:
            h = blk(h)
        Fv = h[:, -1, model.FR_IDX].tolist()
        pred = model(ids)[:, -1, :].argmax(-1).tolist()
    best = max(((sum((f > t) == (y == 1) for f, y in zip(Fv, ys)) / len(ys)), t)
               for t in [i / 200 for i in range(-20, 150)])
    print("seed1 net acc:", round(sum((p == 32) == (y == 1) for p, y in zip(pred, ys)) / len(ys), 4),
          "| seed1 best F-threshold:", best)
    # DEBUG_END

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
