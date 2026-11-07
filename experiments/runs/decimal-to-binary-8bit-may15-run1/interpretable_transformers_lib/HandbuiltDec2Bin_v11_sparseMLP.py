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


class _SparseBitMLP(nn.Module):
    """Hand-built MLP for one D&Q bit block.

    Reads only two residual channels (CH_N and pos_onehot[p_out]) and writes
    to three (CH_N, CH_LOGIT0, CH_LOGIT1) with 4 hidden ReLU neurons. Stores
    only the 27 non-trivial parameters instead of the full ``2*d_model*d_ff +
    d_model + d_ff`` of a vanilla MLP.

    Mathematically identical to the original `MLP(d_model, 4)` writes used in
    v10: at every position it updates curN -= T*bit, and at the matching
    output position it adds ``+bit`` to LOGIT1 and ``-bit`` to LOGIT0.
    """

    def __init__(self, ch_n: int, ch_pos: int, ch_l0: int, ch_l1: int,
                 threshold: int, l_gate: float = 1000.0):
        super().__init__()
        self.ch_n, self.ch_pos = ch_n, ch_pos
        self.ch_l0, self.ch_l1 = ch_l0, ch_l1
        # fc1: 4 hidden neurons, each reads (curN, pos_onehot[p_out]).
        # Rows: [clip_a, clip_b, gate_a, gate_b].
        self.fc1_w = nn.Parameter(torch.zeros(4, 2))
        self.fc1_b = nn.Parameter(torch.zeros(4))
        # fc2: 3 outputs (delta_N, delta_logit0, delta_logit1) <- 4 hidden.
        self.fc2_w = nn.Parameter(torch.zeros(3, 4))
        T = float(threshold)
        with torch.no_grad():
            # clipped_step(curN, T) at every position
            self.fc1_w[0, 0] = 1.0;  self.fc1_b[0] = -(T - 1.0)
            self.fc1_w[1, 0] = 1.0;  self.fc1_b[1] = -T
            # AND(pos==p_out, clipped_step(curN, T))
            self.fc1_w[2, 0] = 1.0;  self.fc1_w[2, 1] = l_gate
            self.fc1_b[2] = -(T - 1.0) - l_gate
            self.fc1_w[3, 0] = 1.0;  self.fc1_w[3, 1] = l_gate
            self.fc1_b[3] = -T - l_gate
            # output 0 -> CH_N: curN -= T*bit  =  -T*(h0-h1)
            self.fc2_w[0, 0] = -T
            self.fc2_w[0, 1] = +T
            # output 1 -> CH_LOGIT0: -gated_bit
            self.fc2_w[1, 2] = -1.0
            self.fc2_w[1, 3] = +1.0
            # output 2 -> CH_LOGIT1: +gated_bit
            self.fc2_w[2, 2] = +1.0
            self.fc2_w[2, 3] = -1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = torch.stack([x[..., self.ch_n], x[..., self.ch_pos]], dim=-1)
        h = F.relu(inp @ self.fc1_w.t() + self.fc1_b)
        outs = h @ self.fc2_w.t()
        delta = torch.zeros_like(x)
        delta[..., self.ch_n] = outs[..., 0]
        delta[..., self.ch_l0] = outs[..., 1]
        delta[..., self.ch_l1] = outs[..., 2]
        return delta


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

class _ZeroAttn(nn.Module):
    """Drop-in replacement for CausalSelfAttention that contributes nothing.

    Used in blocks where the residual-stream update should be identity
    (i.e. the attention pathway is unused). Carries zero parameters so the
    overall parameter count reflects only the live computation.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class _ScalarLaneAttn(nn.Module):
    """Causal attention specialized to a single scalar query/key/value lane.

    Mathematically equivalent to a standard 1-head CausalSelfAttention whose
    W_q / W_k / W_v / W_o each have only one non-zero row (the "lane"). The
    full module uses 4 · d_model parameters (one weight vector per role)
    instead of 4 · d_model^2, which removes ~4.5k dead parameters.

    Computation per query position p:
        q_p       = w_q · x_p
        k_j       = w_k · x_j
        v_j       = w_v · x_j
        score(p,j) = q_p · k_j          (no sqrt(d_head) scaling: d_head = 1)
        attn      = softmax(causal_mask(score))
        attended  = Σ_j attn(p, j) · v_j
        out_p     = attended · w_o      (vector-valued residual update)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.w_q = nn.Parameter(torch.zeros(d_model))
        self.w_k = nn.Parameter(torch.zeros(d_model))
        self.w_v = nn.Parameter(torch.zeros(d_model))
        self.w_o = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = x @ self.w_q                     # (B, T)
        k = x @ self.w_k                     # (B, T)
        v = x @ self.w_v                     # (B, T)
        scores = q.unsqueeze(-1) * k.unsqueeze(-2)  # (B, T_q, T_k)
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        attended = (attn * v.unsqueeze(-2)).sum(dim=-1)   # (B, T)
        return attended.unsqueeze(-1) * self.w_o          # (B, T, D)


def write_weights(model: SimpleTransformer, task) -> None:
    """Fully unrolled divide-and-conquer: 8 blocks, one per bit, end-to-end.

    Each block b ∈ [0..7]:
      * bit_index = 7 - b  (MSB first)
      * threshold = 2 ** bit_index
      * extracts bit via clipped_step(curN, threshold) and decrements curN
      * SIMULTANEOUSLY routes the extracted bit to the logit channels IFF
        the current position equals the matching output position p = 3 + b.

    The position gating is fused into the same MLP using a 2-ReLU "AND"
    construction:
        ReLU(curN  + L*pos_onehot[p_b] - (T-1) - L)
      - ReLU(curN  + L*pos_onehot[p_b] -  T    - L)
       =  clipped_step(curN, T)   when at position p_b
       =  0                       at every other position
    (L is a large constant; outside p_b the inputs are ≤ ~ -L so ReLU=0.)
    That means we get the gated bit "for free" alongside the bit extraction,
    so we can also write its complement into the '0' logit and the bit
    itself into the '1' logit, without a separate gating block.

    Channels (d_model = 26):
      0..10   token one-hot
      11..22  position one-hot
      23      curN (mutates over the 8 blocks: N, N mod 128, N mod 64, ...)
      24      logit-for-'0'
      25      logit-for-'1'  (also reused as the attention lane in block 0)
    """
    import math as _math
    torch.manual_seed(0)
    D = model.d_model
    V = model.vocab_size
    S = model.max_seq_len
    assert D >= 26
    assert V == 11
    assert model.n_layers == 8

    for blk in model.blocks:
        blk.ln1 = nn.Identity()
        blk.ln2 = nn.Identity()
    model.final_ln = nn.Identity()

    # Block 0 only uses a single attention lane -> replace with slim module.
    model.blocks[0].attn = _ScalarLaneAttn(D)
    # Blocks 1..7's attention pathways are unused; strip their parameters.
    for b in range(1, 8):
        model.blocks[b].attn = _ZeroAttn()

    LANE = 25      # overlaps with CH_LOGIT1: attention reads/writes neither
    CH_N = 23
    CH_LOGIT0 = 24
    CH_LOGIT1 = 25

    with torch.no_grad():
        for p in model.parameters():
            p.data.zero_()

        # Bit-extraction blocks: built after the global zero so their hard-coded
        # weights are NOT clobbered.
        for b in range(8):
            bit_index = 7 - b
            T = 1 << bit_index
            p_out = 3 + b
            model.blocks[b].mlp = _SparseBitMLP(
                ch_n=CH_N, ch_pos=11 + p_out,
                ch_l0=CH_LOGIT0, ch_l1=CH_LOGIT1,
                threshold=T, l_gate=1000.0,
            )

        # --- Embeddings ---
        for t in range(V):
            model.token_emb.weight.data[t, t] = 1.0
        for p in range(min(S, 12)):
            model.pos_emb.weight.data[p, 11 + p] = 1.0

        # --- Block 0 attention: log-place weighted pool of digits into curN. ---
        # _ScalarLaneAttn parameterizes attention by 4 vectors (no d_head scaling).
        attn0 = model.blocks[0].attn
        LARGE_ATTN = 1000.0
        log_place = [_math.log(100.0), _math.log(10.0), _math.log(1.0)]
        # q_p = w_q · x_p = sum of token one-hot at p = 1 (every position has a token).
        for c in range(V):
            attn0.w_q.data[c] = 1.0
        # k_j = w_k · x_j ; suppress non-digit positions, write log_place on digit pos.
        for c in range(V):
            attn0.w_k.data[c] = -LARGE_ATTN
        for p in (0, 1, 2):
            attn0.w_k.data[11 + p] = LARGE_ATTN + log_place[p]
        # v_j = w_v · x_j = digit value (= sum over c=0..9 of c * token_onehot[c]).
        for d in range(10):
            attn0.w_v.data[d] = float(d)
        # Attended scalar = (100·d0 + 10·d1 + d2)/111 = N/111; route ×111 into CH_N.
        attn0.w_o.data[CH_N] = 111.0

        # --- 8 bit-extraction blocks are self-configured by _SparseBitMLP. ---

        # --- Head: ch 33 -> '0', ch 34 -> '1', plus a "1" baseline routed via
        #     position one-hots so that at position p the '0' logit is always
        #     at least +baseline, and the gated bit then flips winner if bit=1.
        # We want at output position p, logit_1 >= bit, logit_0 >= 1 - bit.
        # We've set logit_1 += bit (gated), logit_0 -= bit (gated). With a
        # constant +1 added to logit_0 (only at output positions), this works:
        #     bit=0 -> logit_0 = +1, logit_1 = 0  -> argmax '0'
        #     bit=1 -> logit_0 =  0, logit_1 = 1  -> argmax '1'
        # We add the +1 baseline via head.weight (head reads pos one-hots).
        LOGIT_SCALE = 10.0
        model.head.weight.data[0, CH_LOGIT0] = LOGIT_SCALE
        model.head.weight.data[1, CH_LOGIT1] = LOGIT_SCALE
        # Baseline: at output positions p ∈ {3..10}, push '0' logit up by
        # LOGIT_SCALE * 1 (relative to '1') so that ties go to '0' when bit=0
        # and bit=1 still wins because we also added LOGIT_SCALE to logit_1.
        for p in range(3, 11):
            model.head.weight.data[0, 11 + p] = LOGIT_SCALE  # '0' baseline


model_shorthand_name = "HandbuiltDec2Bin_v11_sparseMLP"
model_description = (
    "Replaces each D&Q bit-block's vanilla MLP(d_model, 4) with a "
    "custom _SparseBitMLP that reads only 2 residual channels (curN + "
    "the matching pos one-hot) and writes only 3 (curN + 2 logits), "
    "storing 27 params instead of 238. Behaves identically to v10."
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
    # Sequence length is exactly prompt + answer = 12 tokens, no slack needed.
    max_seq_len = task.seq_len
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=26,
        n_heads=1,
        n_layers=8,
        d_ff=4,  # overridden per-block inside write_weights
    )
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
