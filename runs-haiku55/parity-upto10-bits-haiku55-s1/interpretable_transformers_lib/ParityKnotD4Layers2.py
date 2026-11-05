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
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
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
        d_model: int = 64,
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

ARCH = dict(d_model=4, n_heads=1, n_layers=2, d_ff=10)   # architecture hyperparameters
ALPHA = 1000.0      # norm of one-hot token codes (large, so LayerNorm is ~linear)
KNOT = True         # knot-interpolation MLP (10 units) instead of the ReLU-pair staircase
SUPPRESS_SPECIAL = True   # give '=' and '_' a large negative logit in the head
N_STEPS = 10        # staircase thresholds j = 1..N_STEPS (staircase mode only)
M_SCALE = 10.0      # step sharpness: transition width = gap/M_SCALE (staircase mode only)
USE_NPAR = True     # also write 1-p into a residual dim (needed for a sign-only head)


def _layer_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm with weight=1, bias=0 (same formula as the model's LayerNorms)."""
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    return (x - mu) / torch.sqrt(var + eps)


def write_weights(model: SimpleTransformer, task) -> None:
    """Closed-form weights for parity of the bits in the prompt.

    Algorithm (all positions are read at the '=' position, index 10):
      1. Layer 0 attention is uniform over the 11 prompt positions and its value
         reads the "is this a '1'" feature of each token. The head output at '='
         is therefore proportional to k = (number of '1' bits), the count.
      2. Layer 0 MLP turns k into parity p = k mod 2 with a staircase:
         H_j = [k >= j] for j = 1..10 (one ReLU pair per threshold), and
         p = sum_j (-1)^(j+1) H_j  (1 for odd k, 0 for even k).
         It also writes 1 - p into a second residual dimension.
      3. Layers 1-2 are switched off (all weights zero).
      4. The head outputs '1' if p = 1 else '0' (logit diff = LN(p) - LN(1-p),
         whose sign is the parity; the LayerNorm mean/scale cancel).
    """
    torch.manual_seed(0)
    d = model.d_model
    EQ = task.stoi["="]
    ONE = task.stoi["1"]
    FEAT = ONE          # embedding dim holding the "is '1'" feature
    # Residual dims are the token dims. At the '=' position the token dims for
    # '0', '1' and '_' are otherwise zero, so the three computed features can reuse them.
    CNT_DIM = 0         # reuses token '0' dim (only written at '=' by attention)
    PAR_DIM = 3         # reuses token '_' dim (only written at '=' by MLP)
    NPAR_DIM = 1        # reuses token '1' dim (only written at '=' by MLP); FEAT=1 is read only by layer-0 values at input
    N_POS = 11          # prompt length including '='

    with torch.no_grad():
        # Token embedding: scaled one-hot per token id; no positional signal.
        model.token_emb.weight.zero_()
        for tok in range(task.vocab_size):
            model.token_emb.weight[tok, tok] = ALPHA
        model.pos_emb.weight.zero_()

        # Zero every linear layer in every block; set only what we need below.
        for blk in model.blocks:
            for name, p in blk.named_parameters():
                if "ln" not in name:
                    p.zero_()

        # Exact numbers that LayerNorm 1 produces for a single token.
        def token_ln(tok: int) -> torch.Tensor:
            x = torch.zeros(d)
            x[tok] = ALPHA
            return _layer_norm(x)

        v_one = float(token_ln(ONE)[FEAT])    # LN(embedding('1'))[FEAT]
        v_non = float(token_ln(EQ)[FEAT])     # same for any non-'1' token

        # Layer 0 attention: uniform (Q=K=0). Head 0, dim 0 averages the feature.
        attn = model.blocks[0].attn
        attn.W_v.weight[0, FEAT] = 1.0        # value dim 0 <- LN(x)[FEAT]
        attn.W_o.weight[CNT_DIM, 0] = 1.0     # residual CNT_DIM <- head-0 dim 0

        # Residual value at '=' after attention, for each possible count k.
        # Mean over the 11 uniform positions: (k * v_one + (11-k) * v_non) / 11.
        def u_of_k(k: int) -> float:
            mean_v = (k * v_one + (N_POS - k) * v_non) / N_POS
            x = torch.zeros(d)
            x[EQ] = ALPHA                      # '=' embedding stays in the residual
            x[CNT_DIM] = mean_v                # attention output
            return float(_layer_norm(x)[CNT_DIM])   # what MLP sees (LayerNorm 2)

        us = [u_of_k(k) for k in range(N_POS)]   # increasing in k
        mlp = model.blocks[0].mlp

        if KNOT:
            # Knot construction: the only inputs are the 11 exact values us[k].
            # Target f(k) = k mod 2. A piecewise-linear interpolant through the
            # knots is sum_j c_j relu(u - us[j]) with c_0 = s_0,
            # c_j = s_j - s_{j-1} (s_j = slope on [us[j], us[j+1]]). Uses N_POS-1 units.
            y = [float(k % 2) for k in range(N_POS)]
            s = [(y[k + 1] - y[k]) / (us[k + 1] - us[k]) for k in range(N_POS - 1)]
            for j in range(min(N_POS - 1, mlp.fc1.out_features)):
                c = s[j] - (s[j - 1] if j > 0 else 0.0)
                mlp.fc1.weight[j, CNT_DIM] = 1.0
                mlp.fc1.bias[j] = -us[j]
                mlp.fc2.weight[PAR_DIM, j] = c
                if USE_NPAR:
                    mlp.fc2.weight[NPAR_DIM, j] = -c
            if USE_NPAR:
                mlp.fc2.bias[NPAR_DIM] = 1.0    # residual NPAR_DIM = 1 - p
        else:
            gap = min(us[k + 1] - us[k] for k in range(N_POS - 1))
            M = M_SCALE / gap                   # sharp enough that each step is exact

            # Layer 0 MLP: staircase H_j = [k >= j], j = 1..N_STEPS, one ReLU pair each.
            for j in range(1, N_STEPS + 1):
                z = (us[j - 1] + us[j]) / 2     # threshold between k=j-1 and k=j
                sgn = 1.0 if j % 2 == 1 else -1.0   # (-1)^(j+1)
                hp, hm = 2 * (j - 1), 2 * (j - 1) + 1
                mlp.fc1.weight[hp, CNT_DIM] = M
                mlp.fc1.bias[hp] = -M * z
                mlp.fc1.weight[hm, CNT_DIM] = M
                mlp.fc1.bias[hm] = -M * z - 1.0
                # H_j = relu(M(u-z)) - relu(M(u-z)-1) = clamp(M(u-z), 0, 1)
                mlp.fc2.weight[PAR_DIM, hp] = sgn
                mlp.fc2.weight[PAR_DIM, hm] = -sgn
                if USE_NPAR:
                    mlp.fc2.weight[NPAR_DIM, hp] = -sgn
                    mlp.fc2.weight[NPAR_DIM, hm] = sgn
            if USE_NPAR:
                mlp.fc2.bias[NPAR_DIM] = 1.0    # residual NPAR_DIM = 1 - p

        # Head: logit('1') - logit('0') = LN(p) - LN(1-p) = (2p-1)/sigma.
        # '=' and '_' get a large negative logit so they are never output.
        model.head.weight.zero_()
        model.head.weight[1, PAR_DIM] = 1.0
        if USE_NPAR:
            model.head.weight[0, NPAR_DIM] = 1.0
        else:
            model.head.weight[0, PAR_DIM] = 0.0  # ablation: sign-only head on p alone
        if SUPPRESS_SPECIAL:
            model.head.weight[2, EQ] = -1.0
            model.head.weight[3, EQ] = -1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ParityKnotD4Layers2"
model_description = (
    "ParityKnotD4Shared with n_layers=2: block 1 is all-zero (identity). Checks that a single "
    "extra zero block leaves the circuit unchanged."
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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len, **ARCH)
    write_weights(model, task)
    model.eval()
    # TEMP EXHAUSTIVE CHECK (remove after use)
    import itertools
    bad, tot = [], 0
    for L in range(1, 11):
        for bits in itertools.product([0, 1], repeat=L):
            prompt = "".join(map(str, bits)).rjust(10, "_") + "="
            ans = str(sum(bits) % 2)
            with torch.no_grad():
                pred = model(torch.tensor([task.encode(prompt)]))[0, -1].argmax().item()
            tot += 1
            if task.itos[pred] != ans:
                bad.append(prompt)
    print("EXHAUSTIVE total", tot, "mismatches", len(bad), bad[:5])
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
