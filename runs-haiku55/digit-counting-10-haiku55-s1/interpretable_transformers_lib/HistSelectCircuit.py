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
    # LayerNorm removed: the hand-built circuit relies on exact linear arithmetic.
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
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
        d_model: int = 72,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
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
    """Hand-built digit-counting circuit (no training).

    Residual stream layout (d_model = 72):
      T_t   dims  0..10  token one-hot (digits 0-9, '=' = 10)
      P_p   dims 11..26  position one-hot (positions 0-15)
      C     dim  27      constant 1 (present at every position)
      Q_k   dims 28..37  copy of the query digit q (one-hot), filled at '=' / answer positions
      N_k   dims 38..47  histogram n_k = #{i<10 : d_i == k}
      S_t   dims 48..57  S_t = [n_q >= t] for t = 1..10
      L_k   dims 58..68  logits written by the final MLP

    Positions: d_0..d_9 at 0..9, query q at 10, '=' at 11, answer[0] at 12.
      pos 11 predicts the tens digit, pos 12 predicts the units digit.

    Layer 1 (attn head 0): positions >= 10 attend to position 10 and copy q into Q.
    Layer 2 (attn head 1): uniform attention over digit positions 0..9 (positions
            10, 11, 12 masked) gives n_k = 10 * mean(onehot(d_i)).
            MLP gates on Q_k select n_q and produce step features S_t = [n_q >= t].
    Layer 3 (MLP): I_c = S_c - S_{c+1} = [n_q == c]; gated by position flags
            to write tens digit (pos 11) and units digit (pos 12) into logits.
    """
    torch.manual_seed(0)
    BIG = 50.0
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        D = model.d_model
        dh = D // model.n_heads
        sq = math.sqrt(dh)

        T = lambda t: t
        P = lambda p: 11 + p
        C = 27
        Q = lambda k: 28 + k
        N = lambda k: 38 + k
        S = lambda t: 47 + t          # t in 1..10
        L = lambda k: 58 + k

        # --- embeddings ---
        for t in range(11):
            model.token_emb.weight[t, T(t)] = 1.0
        model.token_emb.weight[:, C] = 1.0
        for p in range(model.pos_emb.weight.shape[0]):
            model.pos_emb.weight[p, P(p)] = 1.0

        # --- layer 1: copy query digit into Q (head 0) ---
        b1 = model.blocks[0]
        alpha = 30.0
        b1.attn.W_q.weight[0, C] = sq * alpha          # score channel: query flag
        b1.attn.W_k.weight[0, P(10)] = 1.0             # key flag: position 10 only
        for k in range(10):
            b1.attn.W_v.weight[k, T(k)] = 1.0          # value: token one-hot
            b1.attn.W_o.weight[Q(k), k] = 1.0          # write into Q

        # --- layer 2: histogram via uniform attention (head 1) + select n_q ---
        b2 = model.blocks[1]
        h1 = dh                                          # channel offset of head 1
        b2.attn.W_q.weight[h1, C] = sq                   # query flag
        for p in (10, 11, 12):                           # mask query/'=' /answer keys
            b2.attn.W_k.weight[h1, P(p)] = -BIG
        for k in range(10):
            b2.attn.W_v.weight[h1 + k, T(k)] = 10.0      # value = 10*onehot -> counts
            b2.attn.W_o.weight[N(k), h1 + k] = 1.0

        # MLP: for each k, t: A = relu(2n_k - 2t + 1 - 100 + 100 Q_k),
        #                    B = relu(2n_k - 2t - 100 + 100 Q_k)
        # sum_k (A - B) = [n_q >= t]   (only k = q survives gating)
        idx = 0
        for k in range(10):
            for t in range(1, 11):
                ia, ib = idx, idx + 1
                idx += 2
                b2.mlp.fc1.weight[ia, N(k)] = 2.0
                b2.mlp.fc1.weight[ia, Q(k)] = 100.0
                b2.mlp.fc1.bias[ia] = -2.0 * t + 1.0 - 100.0
                b2.mlp.fc1.weight[ib, N(k)] = 2.0
                b2.mlp.fc1.weight[ib, Q(k)] = 100.0
                b2.mlp.fc1.bias[ib] = -2.0 * t - 100.0
                b2.mlp.fc2.weight[S(t), ia] = 1.0
                b2.mlp.fc2.weight[S(t), ib] = -1.0

        # --- layer 3: position-gated indicator decoding into logits ---
        b3 = model.blocks[2]
        # hidden a_c (units, pos 12) at index c; b_c (tens, pos 11) at index 11 + c
        for c in range(11):
            for hid, pos_p in ((c, 12), (11 + c, 11)):
                src = C if c == 0 else S(c)
                b3.mlp.fc1.weight[hid, src] = 1.0
                if c + 1 <= 10:
                    b3.mlp.fc1.weight[hid, S(c + 1)] = -1.0
                b3.mlp.fc1.weight[hid, P(pos_p)] = 1.0
                b3.mlp.fc1.bias[hid] = -1.0
        # units digit (pos 12): token k for count k (k<=9); count 10 -> token 0
        for k in range(1, 10):
            b3.mlp.fc2.weight[L(k), k] = 10.0
        b3.mlp.fc2.weight[L(0), 0] = 10.0
        b3.mlp.fc2.weight[L(0), 10] = 10.0
        # tens digit (pos 11): token 1 iff count == 10, else token 0
        for c in range(10):
            b3.mlp.fc2.weight[L(0), 11 + c] = 10.0
        b3.mlp.fc2.weight[L(1), 11 + 10] = 10.0

        # --- readout ---
        for k in range(11):
            model.head.weight[k, L(k)] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "HistSelectCircuit"
model_description = (
    "Hand-built 3-layer circuit without LayerNorm: copy query digit via prev-pos attention, "
    "uniform-attention digit histogram (masked to positions 0-9), ReLU gating to select n_q, "
    "position-gated step decoding into tens/units digits."
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
