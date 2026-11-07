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
# Design knobs (the experiments change these)
# ---------------------------------------------------------------------------
N_LAYERS = 2        # 3: copy -> histogram+select -> decode ; 2: copy -> histogram+select+position-gated readout
N_HEADS = 2
D_MODEL = 68
D_FF = 220
ALPHA = 30.0        # strength of the copy-query attention (layer 1)
BIG = 5.0           # mask strength for keys at positions 10, 11, 12
SOFT_K = None       # None: uniform attention -> digit counts; k: query-match attention, weight e^S = k on matches
K_SCALE = 3.0       # steepness of threshold steps relu(K * (N - theta))
G_MARGIN = 20.0     # gating margin that zeroes every digit bin except the query digit
DECODE = "steps"    # "steps": threshold indicators; "bump": scalar count + second-difference bumps (3 layers, counts only)
POS_GATE = True     # position flags gate the readout (False = ablation)


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
    """Causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
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

    Input: d_0..d_9 at positions 0..9, query digit q at 10, '=' at 11, answer[0] at 12.
    Output: tens digit predicted at position 11, units digit at position 12.

    Residual stream layout (d_model = 68 for the 2-layer variant):
      T_t  dims  0..10  token one-hot (digits 0-9, '=' = 10)
      P_p  dims 11..26  position one-hot (positions 0-15)
      C    dim  27      constant 1
      Q_k  dims 28..37  copy of the query digit q (one-hot)
      N_k  dims 38..47  digit statistic: n_k = #{i<10 : d_i == k}  (or fraction if SOFT_K)
      SU_t dims 48..57  [n_q >= t] (gated by position 12 in the 2-layer variant)
      ST_t dims 58..67  [n_q >= t] gated by position 11 (2-layer variant, t = 10 only)
      L_k  dims 68..78  output logits (3-layer variant)
      X    dim  79      scalar n_q (bump decoding)

    Layer 1 (attn head 0): positions >= 10 attend to position 10 and copy q into Q.
    Layer 2 (attn head 1): attention over digit positions 0..9 (10, 11, 12 masked)
            builds the digit histogram N. The MLP gates on Q_k so only bin q survives,
            producing threshold indicators [n_q >= t].
    Layer 3 (MLP): indicators [n_q == c] = SU_c - SU_{c+1}, gated by position flags,
            are written to logits (tens at 11, units at 12).
    """
    torch.manual_seed(0)
    if N_LAYERS == 2 and DECODE != "steps":
        raise ValueError("2-layer variant only supports DECODE='steps'")
    if SOFT_K is not None and DECODE != "steps":
        raise ValueError("bump decoding needs integer counts (SOFT_K=None)")

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        D = model.d_model
        dh = D // model.n_heads
        sq = math.sqrt(dh)
        K = K_SCALE
        G = G_MARGIN

        T = lambda t: t
        P = lambda pos: 11 + pos
        C = 27
        Q = lambda k: 28 + k
        N = lambda k: 38 + k
        SU = lambda t: 47 + t      # t = 1..10
        ST = lambda t: 57 + t      # t = 1..10
        L = lambda k: 68 + k
        X = 79

        def theta(t: int) -> float:
            if SOFT_K is None:
                return t - 0.5
            f = lambda c: c * SOFT_K / (c * SOFT_K + 10 - c)
            return 0.5 * (f(t - 1) + f(t))

        def add_threshold(mlp, out_dim: int, t: int, gate_pos, row: int) -> int:
            """Writes sum_k (relu(K(N_k-theta)+G(Q_k-1)) - relu(... - 1)) = [n_q >= t] into out_dim."""
            th = theta(t)
            for k in range(10):
                u, v = row, row + 1
                row += 2
                for h in (u, v):
                    mlp.fc1.weight[h, N(k)] = K
                    mlp.fc1.weight[h, Q(k)] = G
                    mlp.fc1.bias[h] = -K * th - G
                    if gate_pos is not None:
                        mlp.fc1.weight[h, P(gate_pos)] = G
                        mlp.fc1.bias[h] -= G
                mlp.fc1.bias[v] -= 1.0
                mlp.fc2.weight[out_dim, u] = 1.0
                mlp.fc2.weight[out_dim, v] = -1.0
            return row

        # --- embeddings ---
        for t in range(11):
            model.token_emb.weight[t, T(t)] = 1.0
        model.token_emb.weight[:, C] = 1.0
        for pos in range(model.pos_emb.weight.shape[0]):
            model.pos_emb.weight[pos, P(pos)] = 1.0

        # --- layer 1: copy query digit into Q (head 0) ---
        a1 = model.blocks[0].attn
        a1.W_q.weight[0, C] = sq * ALPHA
        a1.W_k.weight[0, P(10)] = 1.0
        for k in range(10):
            a1.W_v.weight[k, T(k)] = 1.0
            a1.W_o.weight[Q(k), k] = 1.0

        # --- layer 2: digit statistic via attention (head 1) ---
        a2 = model.blocks[1].attn
        h1 = dh
        if BIG > 0:
            a2.W_q.weight[h1 + 10, C] = sq
            for pos in (10, 11, 12):
                a2.W_k.weight[h1 + 10, P(pos)] = -BIG
        for k in range(10):
            if SOFT_K is None:
                a2.W_v.weight[h1 + k, T(k)] = 10.0
            else:
                a2.W_v.weight[h1 + k, T(k)] = 1.0
                a2.W_q.weight[h1 + k, Q(k)] = sq * math.log(SOFT_K)
                a2.W_k.weight[h1 + k, T(k)] = 1.0
            a2.W_o.weight[N(k), h1 + k] = 1.0

        # --- layer 2 MLP: selection of bin q and threshold indicators ---
        mlp2 = model.blocks[1].mlp
        row = 0
        if DECODE == "bump":
            for k in range(10):
                mlp2.fc1.weight[row, N(k)] = 1.0
                mlp2.fc1.weight[row, Q(k)] = G
                mlp2.fc1.bias[row] = -G
                mlp2.fc2.weight[X, row] = 1.0
                row += 1
        else:
            gate2 = (lambda pos: pos) if (N_LAYERS == 2 and POS_GATE) else (lambda pos: None)
            for t in range(1, 11):
                row = add_threshold(mlp2, SU(t), t, gate2(12), row)
            if N_LAYERS == 2:
                row = add_threshold(mlp2, ST(10), 10, 11 if POS_GATE else None, row)

        if N_LAYERS == 3:
            # --- layer 3 MLP: position-gated indicators [n_q == c] -> logits ---
            mlp3 = model.blocks[2].mlp
            row = 0
            terms = {}
            for pos in (12, 11):
                gp = P(pos) if POS_GATE else None
                for c in range(11):
                    if DECODE == "steps":
                        src = C if c == 0 else SU(c)
                        mlp3.fc1.weight[row, src] = 1.0
                        if c + 1 <= 10:
                            mlp3.fc1.weight[row, SU(c + 1)] = -1.0
                        if gp is not None:
                            mlp3.fc1.weight[row, gp] = 1.0
                            mlp3.fc1.bias[row] = -1.0
                        terms[(pos, c)] = [(row, 1.0)]
                        row += 1
                    else:  # bump: [X == c] = relu(X-c+1) - 2 relu(X-c) + relu(X-c-1)
                        lst = []
                        for s, coef in ((1, 1.0), (0, -2.0), (-1, 1.0)):
                            mlp3.fc1.weight[row, X] = 1.0
                            if gp is not None:
                                mlp3.fc1.weight[row, gp] = G
                                mlp3.fc1.bias[row] = -c + s - G
                            else:
                                mlp3.fc1.bias[row] = -c + s
                            lst.append((row, coef))
                            row += 1
                        terms[(pos, c)] = lst
            for pos in (12, 11):
                for c in range(11):
                    if pos == 12:       # units digit: count c -> token c (count 10 -> token 0)
                        tok = c if c <= 9 else 0
                    else:               # tens digit: count 10 -> token 1, else token 0
                        tok = 0 if c <= 9 else 1
                    for hid, coef in terms[(pos, c)]:
                        mlp3.fc2.weight[L(tok), hid] += 10.0 * coef
            for k in range(11):
                model.head.weight[k, L(k)] = 1.0
        else:
            # --- 2-layer readout: head reads the position-gated indicators directly ---
            hd = model.head.weight
            for c in range(11):
                if c <= 9:
                    src = P(12) if c == 0 else SU(c)
                    hd[c, src] += 10.0
                    if c + 1 <= 10:
                        hd[c, SU(c + 1)] -= 10.0
                else:
                    hd[0, SU(10)] += 10.0
            hd[0, P(11)] += 10.0               # tens digit: [n_q <= 9] at position 11
            hd[0, ST(10)] -= 10.0
            hd[1, ST(10)] += 10.0              # tens digit 1 iff n_q == 10


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "TwoLayerBig5"
model_description = (
    "Sensitivity test of the 2-layer circuit with a weak key mask (BIG 20 -> 5) to find where the mask "
    "stops mattering."
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
