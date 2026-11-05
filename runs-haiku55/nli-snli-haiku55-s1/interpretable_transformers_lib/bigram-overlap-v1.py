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
        # LayerNorm removed (Identity) so residual-stream arithmetic is exact and hand-designable.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
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
        d_model: int = 256,
        n_heads: int = 2,
        n_layers: int = 3,
        d_ff: int = 64,
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-built NLI circuit: character-bigram overlap + negation -> label.

    Residual layout (d_model=256):
      T [0,34)    current char one-hot          PV [34,68)  previous char one-hot
      P [68,191)  position one-hot              PREM,SEP,HYP,EQ flags at 191..194
      ONE 195     constant 1                    VALID 196   hyp bigram is letter-letter
      NEG 197     hyp bigram is "no" or "n'"    PRES 198    premise contains this hyp bigram
      FF 199      mean PRES over valid hyp      FN 200      mean NEG over valid hyp
    Layers:
      L1 head0: attend to position p-1 -> PV.  L1 MLP: zero PV at separator, VALID, NEG.
      L2 head0: match (cur,prev) bigram of hyp query against premise keys; a separator
                key acts as a null (score 1.5B) so PRES ~ 1 iff a 2-char match exists.
      L3 head0: average PRES and NEG over VALID hyp positions -> FF, FN.
      Head: logit0 = A*FF, logit1 = A*(1-FF), logit2 = K*FN (others suppressed).
    """
    D, H = model.d_model, model.n_heads
    dh = D // H
    sq = math.sqrt(dh)
    V = model.vocab_size
    T0, PV0, P0 = 0, 34, 68
    PREM, SEP, HYP, EQ, ONE, VALID, NEG, PRES, FF, FN = range(191, 201)
    B1, B2, B3, A, K = 30.0, 20.0, 20.0, 10.0, 10.0
    st = task.stoi
    letters = [st[c] for c in "abcdefghijklmnopqrstuvwxyz"]
    cls_ids = [st["0"], st["1"], st["2"]]

    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()

        # ---- embeddings ----
        for t in range(V):
            model.token_emb.weight[t, T0 + t] = 1.0
            model.token_emb.weight[t, ONE] = 1.0
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, P0 + p] = 1.0
            if p < 60:
                flag = PREM
            elif p == 60:
                flag = SEP
            elif p <= 120:
                flag = HYP
            else:
                flag = EQ
            model.pos_emb.weight[p, flag] = 1.0

        # ---- layer 1: previous-token attention ----
        a1 = model.blocks[0].attn
        for d in range(122):
            a1.W_q.weight[d, P0 + d + 1] = B1 * sq
        for d in range(123):
            a1.W_k.weight[d, P0 + d] = 1.0
        for c in range(34):
            a1.W_v.weight[c, T0 + c] = 1.0
            a1.W_o.weight[PV0 + c, c] = 1.0

        # ---- layer 1 MLP ----
        m1 = model.blocks[0].mlp
        u = 0
        for c in range(34):  # zero the prev-char at the separator (so it can serve as null key)
            m1.fc1.weight[u, PV0 + c] = 1.0
            m1.fc1.weight[u, SEP] = 1.0
            m1.fc1.bias[u] = -1.0
            m1.fc2.weight[PV0 + c, u] = -1.0
            u += 1
        # VALID = HYP & letter(cur) & letter(prev)
        m1.fc1.weight[u, HYP] = 1.0
        for l in letters:
            m1.fc1.weight[u, T0 + l] = 1.0
            m1.fc1.weight[u, PV0 + l] = 1.0
        m1.fc1.bias[u] = -2.0
        m1.fc2.weight[VALID, u] = 1.0
        u += 1
        # NEG = HYP & bigram in {"no", "n'"}
        for cur, prev in [("o", "n"), ("'", "n")]:
            m1.fc1.weight[u, HYP] = 1.0
            m1.fc1.weight[u, T0 + st[cur]] = 1.0
            m1.fc1.weight[u, PV0 + st[prev]] = 1.0
            m1.fc1.bias[u] = -2.0
            m1.fc2.weight[NEG, u] = 1.0
            u += 1

        # ---- layer 2: bigram match of hyp query against premise keys (sep = null) ----
        a2 = model.blocks[1].attn
        for c in range(34):
            a2.W_q.weight[c, T0 + c] = B2 * sq
            a2.W_q.weight[34 + c, PV0 + c] = B2 * sq
            a2.W_k.weight[c, T0 + c] = 1.0
            a2.W_k.weight[34 + c, PV0 + c] = 1.0
        a2.W_q.weight[68, ONE] = 1.5 * B2 * sq
        a2.W_k.weight[68, SEP] = 1.0
        # keys in the hypothesis / '=' get a -100 penalty so they can't absorb attention mass
        a2.W_q.weight[69, ONE] = -100.0 * sq
        a2.W_k.weight[69, HYP] = 1.0
        a2.W_k.weight[69, EQ] = 1.0
        a2.W_v.weight[0, PREM] = 1.0
        a2.W_o.weight[PRES, 0] = 1.0

        # ---- layer 3: average PRES and NEG over valid hypothesis positions ----
        a3 = model.blocks[2].attn
        a3.W_q.weight[0, ONE] = B3 * sq
        a3.W_k.weight[0, VALID] = 1.0
        a3.W_v.weight[1, PRES] = 1.0
        a3.W_v.weight[2, NEG] = 1.0
        a3.W_o.weight[FF, 1] = 1.0
        a3.W_o.weight[FN, 2] = 1.0

        # ---- readout (only the three label tokens can win) ----
        for t in range(V):
            if t not in cls_ids:
                model.head.weight[t, ONE] = -1000.0
        model.head.weight[cls_ids[0], FF] = A
        model.head.weight[cls_ids[1], FF] = -A
        model.head.weight[cls_ids[1], ONE] = A
        model.head.weight[cls_ids[2], FN] = K


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "bigram-overlap-v1"
model_description = ("No-LN 3-layer circuit: prev-char attention builds char bigrams; hyp bigrams matched against "
                     "premise bigrams (sep as null) give overlap F; 'no'/\"n'\" bigram gives negation N; "
                     "readout argmax(A*F, A*(1-F), K*N).")


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
