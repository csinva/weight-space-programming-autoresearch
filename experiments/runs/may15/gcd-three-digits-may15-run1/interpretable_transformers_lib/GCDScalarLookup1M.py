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

# Hyperparameter overrides used when build_model constructs SimpleTransformer.
# d_model = 96 = 6 heads * 16, n_heads = 6 (one per digit position), 2 layers.
# d_ff_layer1 ≈ 10^6 holds the full (a,b) -> gcd lookup as ReLU(s - K) units.
MODEL_KWARGS = dict(d_model=96, n_heads=6, n_layers=2, d_ff=1_000_000)


def _scalar_index(a: int, b: int) -> int:
    return 1000 * a + b


def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-build a GCD circuit.

    Circuit overview (vocab = "0123456789,=", prompt = "AAA,BBB=", answer = "GGG"):

      Channel layout in residual stream (d_model = 96):
        [0:10]   - digit one-hot from token_emb (digits '0'..'9').
        [10:21]  - position one-hot from pos_emb (positions 0..10).
        [21]     - scalar s = 1000*a + b, written by Layer 1 attention.
        [25:55]  - 3 one-hots (10-d each) of the 3 digits of gcd(a,b),
                   written by Layer 1 MLP.
        [55:67]  - 12-d "logit boost" channels (one per vocab token),
                   written by Layer 2 MLP; the final head reads these.

      Layer 1 attention: 6 heads, head i routes input position src_i in
        [0,1,2,4,5,6] (the 6 digit positions of "AAA,BBB=") to channel 21
        with multiplier coef_i in [100000, 10000, 1000, 100, 10, 1]. Their
        sum is s = 1000*a + b.

      Layer 1 MLP: d_ff = 10^6 hidden units h_K = ReLU(s - K), for K in
        0..d_ff-1. fc2 forms the "triangular indicator" 1[s == K'] as
        (h_{K'-1} - 2*h_{K'} + h_{K'+1}) and uses it to read out a
        precomputed table of (a, b) -> 3-digit one-hots of gcd.

      Layer 2 MLP: 30 hidden units h_(p, d) for p in {7,8,9}, d in {0..9};
        h_(p, d) = ReLU(pos_channel_p + digit_(p-7)_one_hot_d - 1.5),
        firing iff position == p AND that digit equals d. fc2 writes +10
        to channel 55+d. The head copies channels [55:67] to vocab logits.

      All LayerNorms are swapped for nn.Identity() to keep the channel
      values exactly as designed.
    """
    from math import gcd

    torch.manual_seed(0)

    D = model.d_model            # 96
    H = model.n_heads            # 6
    DH = D // H                  # 16
    V = model.vocab_size         # 12
    S = model.max_seq_len        # 11
    DFF = model.d_ff             # 10^6
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Disable LayerNorms — replace each with nn.Identity().
    for block in model.blocks:
        block.ln1 = nn.Identity()
        block.ln2 = nn.Identity()
    model.final_ln = nn.Identity()

    # --- Token embedding: digit '0'..'9' -> one-hot at channels 0..9. ---
    tok = torch.zeros(V, D, device=device, dtype=dtype)
    for d in range(10):
        tok[d, d] = 1.0
    # ',' (id 10) and '=' (id 11): all zeros.
    model.token_emb.weight.data.copy_(tok)

    # --- Position embedding: position t -> one-hot at channel 10+t. ---
    pos = torch.zeros(S, D, device=device, dtype=dtype)
    for t in range(S):
        pos[t, 10 + t] = 1.0
    model.pos_emb.weight.data.copy_(pos)

    # --- Layer 1 attention: 6 heads, one per digit position. ---
    src_positions = [0, 1, 2, 4, 5, 6]              # digit positions in prompt
    coefs = [100000, 10000, 1000, 100, 10, 1]       # so that sum_i coef_i * d_{src_i} == 1000*a + b
    attn = model.blocks[0].attn

    Wq = torch.zeros(D, D, device=device, dtype=dtype)
    Wk = torch.zeros(D, D, device=device, dtype=dtype)
    Wv = torch.zeros(D, D, device=device, dtype=dtype)
    Wo = torch.zeros(D, D, device=device, dtype=dtype)

    # K: for every head, K_t encodes position t as one-hot at index t inside
    #    the head's DH-dim slice.  W_k reads pos channels [10:10+S] and writes
    #    e_t into head h's slice. Scale by 100 so softmax is sharp.
    Q_SCALE = 100.0
    for h in range(H):
        for t in range(S):
            # head h's slice starts at row h*DH in the W_k output.
            Wk[h * DH + t, 10 + t] = 1.0

    # Q: for head i, at every position, Q[head_i] = Q_SCALE * e_{src_i}.
    #    We don't depend on the input position, so we use the position 21
    #    channel which is *always* zero ... no — Q must be derived from x.
    #    Trick: use a constant Q. Add a "constant 1" via the bias of W_q.
    # W_q has no bias (Linear(..., bias=False)), so we cannot inject a
    # constant. Instead, derive Q from the pos one-hot using the fact that
    # for ANY position t, pos_channel[10+t] = 1, so summing over all 11 pos
    # channels gives 1 (independent of t). Set W_q[head_i, src_i, 10+t] =
    # Q_SCALE for all t; that yields Q[head_i, src_i] = Q_SCALE * sum_t
    # pos_channel[10+t] = Q_SCALE.  All other Q dims are 0.
    for h in range(H):
        for t in range(S):
            Wq[h * DH + src_positions[h], 10 + t] = Q_SCALE

    # V: for head i, V_t = coef_i * digit_value(t) at channel 0 of head i.
    #    digit_value(t) = sum_d d * x_t[d]  (since channels 0..9 hold the
    #    digit one-hot). For non-digit tokens these channels are 0, so V_t=0.
    for h in range(H):
        for d in range(10):
            Wv[h * DH + 0, d] = coefs[h] * d

    # W_o: sum head i's channel-0 contributions into d_model channel 21.
    for h in range(H):
        Wo[21, h * DH + 0] = 1.0

    attn.W_q.weight.data.copy_(Wq)
    attn.W_k.weight.data.copy_(Wk)
    attn.W_v.weight.data.copy_(Wv)
    attn.W_o.weight.data.copy_(Wo)

    # --- Layer 1 MLP: lookup table over s = 1000*a + b. ---
    mlp1 = model.blocks[0].mlp

    # fc1: h_K = ReLU(channel_21 - K). Only column 21 has nonzeros.
    fc1_W = torch.zeros(DFF, D, device=device, dtype=dtype)
    fc1_W[:, 21] = 1.0
    fc1_b = -torch.arange(DFF, device=device, dtype=dtype)
    mlp1.fc1.weight.data.copy_(fc1_W)
    mlp1.fc1.bias.data.copy_(fc1_b)

    # Precompute target_value[c, K] for output channels c in [25:55] for
    # K in [0, DFF). target = 1 iff (a, b) = decoded(K) is valid AND the
    # corresponding digit of gcd(a, b) equals the indicated value.
    # We then form the discrete second difference along K to get fc2 weights:
    #   fc2_W[c, K] = target[c, K+1] - 2*target[c, K] + target[c, K-1]
    # so that sum_K fc2_W[c, K] * ReLU(s - K) = target[c, s] for integer s.
    target = torch.zeros(30, DFF + 2, device=device, dtype=dtype)
    # target index along axis 1 is K, shifted by +1 so we can index K-1, K+1.
    # i.e., target[c, K+1] corresponds to K.
    import numpy as np
    A_grid, B_grid = np.meshgrid(np.arange(1, 1000), np.arange(1, 1000), indexing='ij')
    G = np.gcd(A_grid, B_grid).reshape(-1)
    K_flat = (1000 * A_grid + B_grid).reshape(-1)
    K_idx = torch.from_numpy(K_flat.astype(np.int64) + 1).to(device)
    d0 = torch.from_numpy(((G // 100) % 10).astype(np.int64)).to(device)
    d1 = torch.from_numpy(((G // 10) % 10).astype(np.int64)).to(device)
    d2 = torch.from_numpy((G % 10).astype(np.int64)).to(device)
    target[d0, K_idx] = 1.0
    target[10 + d1, K_idx] = 1.0
    target[20 + d2, K_idx] = 1.0
    # Discrete second difference along K axis.
    # diff_K = target[c, K+1] - 2*target[c, K] + target[c, K-1]
    # Indexing: target axis-1 length = DFF + 2 (K in -1..DFF).
    # We need fc2_W of shape (30, DFF), index K in 0..DFF-1.
    t_plus = target[:, 2:]            # K+1
    t_zero = target[:, 1:-1]          # K
    t_minus = target[:, :-2]          # K-1
    diff2 = t_plus - 2 * t_zero + t_minus  # shape (30, DFF)
    del target, t_plus, t_zero, t_minus

    fc2_W = torch.zeros(D, DFF, device=device, dtype=dtype)
    # Output channels 25..54 hold the 3 digit one-hots.
    fc2_W[25:55, :] = diff2
    del diff2
    fc2_b = torch.zeros(D, device=device, dtype=dtype)
    mlp1.fc2.weight.data.copy_(fc2_W)
    mlp1.fc2.bias.data.copy_(fc2_b)
    del fc2_W

    # --- Layer 2 attention: zero out (identity). ---
    attn2 = model.blocks[1].attn
    for w in (attn2.W_q.weight, attn2.W_k.weight, attn2.W_v.weight, attn2.W_o.weight):
        w.data.zero_()

    # --- Layer 2 MLP: position-gated digit readout. ---
    mlp2 = model.blocks[1].mlp
    D_ff2_full = mlp2.fc1.out_features    # default = d_ff = DFF; but for layer 2 we'd want only 30.
    # We have to keep d_ff = DFF for both layers (architecture symmetric).
    # Allocate weights of full size but only use first 30 hidden units.
    fc1_2_W = torch.zeros(D_ff2_full, D, device=device, dtype=dtype)
    fc1_2_b = torch.zeros(D_ff2_full, device=device, dtype=dtype)
    fc2_2_W = torch.zeros(D, D_ff2_full, device=device, dtype=dtype)
    fc2_2_b = torch.zeros(D, device=device, dtype=dtype)

    LOGIT_GAIN = 10.0
    idx = 0
    for p in (7, 8, 9):
        digit_index = p - 7
        for d in range(10):
            # hidden_idx fires iff pos==p AND digit_p_channel_d == 1
            fc1_2_W[idx, 10 + p] = 1.0                     # pos channel for position p
            fc1_2_W[idx, 25 + digit_index * 10 + d] = 1.0  # digit_p one-hot channel for value d
            fc1_2_b[idx] = -1.5                            # threshold: fires only if both 1s present
            # Output: boost logit for token id d at channel 55+d.
            fc2_2_W[55 + d, idx] = LOGIT_GAIN
            idx += 1
    # Important: also make sure the rest of the d_ff hidden units don't fire.
    # They have all-zero weights, so h_k = ReLU(0 + 0) = 0. Good. (No bias.)
    mlp2.fc1.weight.data.copy_(fc1_2_W)
    mlp2.fc1.bias.data.copy_(fc1_2_b)
    mlp2.fc2.weight.data.copy_(fc2_2_W)
    mlp2.fc2.bias.data.copy_(fc2_2_b)

    # --- Final head: copy logit-boost channels 55..66 to vocab logits 0..11. ---
    head_W = torch.zeros(V, D, device=device, dtype=dtype)
    for v in range(V):
        head_W[v, 55 + v] = 1.0
    model.head.weight.data.copy_(head_W)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "GCDScalarLookup1M"
model_description = (
    "Hand-built GCD circuit: 6-head attention computes s=1000a+b, a 10^6-wide "
    "MLP forms triangular indicators 1[s==K] via ReLU(s-K) second-differences "
    "to look up the 3 digits of gcd(a,b), then a position-gated MLP emits each "
    "digit at output positions 7,8,9."
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
    max_seq_len = task.seq_len  # exactly prompt_len + answer_len
    kwargs = dict(MODEL_KWARGS) if "MODEL_KWARGS" in globals() else {}
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len, **kwargs)
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
