"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. Task: digit-counting-10.
Prompt "d0 d1 ... d9 q =" (10 digits, a query digit q, then '=') -> the answer is
the count of digits d_i equal to q, written as two digits (e.g. "03").

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py --task digit-counting-10
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
# Knobs for the hand-built circuit (architecture-level, no training)
# ---------------------------------------------------------------------------

USE_LN = False        # LayerNorm off: keeps the hand-built arithmetic exact and linear
N_LAYERS = 2          # layer 1: histogram + query fetch + count; layer 2: count -> one-hot logits
D_FF = 36             # MLP width (layer 2 needs 36 units)
HAND_BUILT = True     # False -> leave random init (baseline)
BIG = 20.0            # attention sharpness: score gap between selected and unselected positions
GATE = 20.0           # gating margin that silences units outside their output position
USE_HIST = True       # head 0: uniform attention over the 10 digit positions -> histogram
USE_QUERY = True      # head 1: attend to the query position -> one-hot query digit
USE_TENS = True       # tens digit = [count == 10]
USE_ONES_TENS = False # ones digit also needs the count==10 case (maps 10 -> 0)


def _norm(d: int) -> nn.Module:
    return nn.LayerNorm(d) if USE_LN else nn.Identity()


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
        self.ln1 = _norm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = _norm(d_model)
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
        d_model: int = 64,
        n_heads: int = 4,
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
        self.final_ln = _norm(d_model)
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

# Residual-stream layout (d_model = 64), one-hot where noted:
#   dims 0..10   token one-hot (digits 0-9, '=' = 10)
#   dims 11..26  position one-hot (positions 0-15)
#   dims 27..36  H_k = count of digit k among the 10 input digits (head 0)
#   dims 37..46  Q_k = one-hot of the query digit (head 1)
#   dim  47      C   = count of the query digit (MLP 1)
#   dim  48      constant 1
#   dims 49..58  L_k = output logit for digit k (MLP 2)
# Positions: 11 is '=' (predicts answer[0], the tens digit);
#            12 is answer[0] (predicts answer[1], the ones digit).

_TOK = lambda t: t
_POS = lambda p: 11 + p
_H = lambda k: 27 + k
_Q = lambda k: 37 + k
_C = 47
_ONE = 48
_L = lambda k: 49 + k


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed."""
    torch.manual_seed(0)
    if not HAND_BUILT:
        return  # random init baseline

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: token one-hot, a constant 1 on every token, position one-hot.
        for t in range(11):
            model.token_emb.weight[t, _TOK(t)] = 1.0
        model.token_emb.weight[:, _ONE] = 1.0
        for p in range(model.pos_emb.weight.shape[0]):
            model.pos_emb.weight[p, _POS(p)] = 1.0

        dh = model.d_model // model.n_heads
        s = math.sqrt(dh)  # attention divides scores by sqrt(dh); pre-multiply to get score = BIG
        attn = model.blocks[0].attn

        # Head 0 (dims 0..15 of concat): uniform attention over the 10 digit positions
        # (score BIG on positions 0..9, 0 elsewhere), value = 10 * onehot(digit)
        # -> H_k = count_k exactly.
        if USE_HIST:
            attn.W_q.weight[0, _ONE] = 1.0
            for p in range(10):
                attn.W_k.weight[0, _POS(p)] = BIG * s
            for k in range(10):
                attn.W_v.weight[k, _TOK(k)] = 10.0
                attn.W_o.weight[_H(k), k] = 1.0

        # Head 1 (dims 16..31 of concat): attend to position 10 (the query digit)
        # -> Q_k = onehot(query).
        if USE_QUERY:
            attn.W_q.weight[dh, _ONE] = 1.0
            attn.W_k.weight[dh, _POS(10)] = BIG * s
            for k in range(10):
                attn.W_v.weight[dh + k, _TOK(k)] = 1.0
                attn.W_o.weight[_Q(k), dh + k] = 1.0

        # MLP 1: C = sum_k ReLU(H_k + 11*Q_k - 11) = count_k for the query k.
        # When Q_k = 0 the unit is ReLU(H_k - 11) = 0 (since H_k <= 10).
        mlp1 = model.blocks[0].mlp
        for k in range(10):
            mlp1.fc1.weight[k, _H(k)] = 1.0
            mlp1.fc1.weight[k, _Q(k)] = 11.0
            mlp1.fc1.bias[k] = -11.0
            mlp1.fc2.weight[_C, k] = 1.0

        if N_LAYERS < 2:
            # No second MLP: logits can only be the constant head bias below (no count info).
            pass
        else:
            mlp2 = model.blocks[1].mlp
            u = 0

            def unit(terms: dict, bias: float, outs: dict) -> None:
                nonlocal u
                for dim, coef in terms.items():
                    mlp2.fc1.weight[u, dim] = coef
                mlp2.fc1.bias[u] = bias
                for dim, coef in outs.items():
                    mlp2.fc2.weight[dim, u] = coef
                u += 1

            # Tens digit at position 11: v = [C == 10] via the triangle
            # ReLU(c-9) - 2ReLU(c-10) + ReLU(c-11); gated to position 11 by POS(11).
            if USE_TENS:
                for t, coef in [(9, 10.0), (10, -20.0), (11, 10.0)]:
                    unit({_C: 1.0, _POS(11): GATE}, -GATE - t, {_L(1): coef})

            # Ones digit at position 12: one-hot of C mod 10 (C = 10 maps to digit 0).
            # [C == k] = ReLU(c-k+1) - 2ReLU(c-k) + ReLU(c-k-1), gated by POS(12).
            for k in range(10):
                for t, coef in [(k - 1, 10.0), (k, -20.0), (k + 1, 10.0)]:
                    unit({_C: 1.0, _POS(12): GATE}, -GATE - t, {_L(k): coef})
            if USE_ONES_TENS:
                # [C == 10] also maps to ones digit 0.
                for t, coef in [(9, 10.0), (10, -20.0), (11, 10.0)]:
                    unit({_C: 1.0, _POS(12): GATE}, -GATE - t, {_L(0): coef})

            assert u <= model.d_ff, f"need {u} MLP units, have {model.d_ff}"

        # Output head: logit k = L_k (positions 11/12 both read L_k); at position 11 the
        # constant 5 keeps token '0' ahead of '1' when the tens digit is 0.
        head = model.head.weight
        if N_LAYERS >= 2:
            for k in range(10):
                head[k, _L(k)] = 1.0
        head[0, _POS(11)] = 5.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = (
    f"knobs-BIG{BIG:g}-G{GATE:g}-T{int(USE_TENS)}O{int(USE_ONES_TENS)}"
    f"-H{int(USE_HIST)}Q{int(USE_QUERY)}-L{N_LAYERS}-LN{int(USE_LN)}-B{int(HAND_BUILT)}-F{D_FF}"
)
model_description = (
    f"Hand-built 2-layer count circuit (histogram + query fetch + ReLU-triangle outputs) with knobs "
    f"BIG={BIG:g} GATE={GATE:g} TENS={USE_TENS} ONES_TENS={USE_ONES_TENS} HIST={USE_HIST} QUERY={USE_QUERY} "
    f"LAYERS={N_LAYERS} LN={USE_LN} HAND_BUILT={HAND_BUILT} D_FF={D_FF}."
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
