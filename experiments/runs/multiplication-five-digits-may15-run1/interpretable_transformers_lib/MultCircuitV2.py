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
    """Causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 320,
        n_heads: int = 10,
        n_layers: int = 11,
        d_ff: int = 3072,
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-built multiplication circuit for 5-digit * 5-digit.

    Algorithm:
      * Block 0 attention: each of 10 heads attends to one fixed prompt digit
        position and copies the digit one-hot into A_i / B_j channels.
      * Block 0 MLP: enumerate all (i,j,u,v) pairs to build neurons that fire
        when a_i=u AND b_j=v, value u*v summed into column-sum C_{i+j}.
        Result: scalars c_0..c_8 (LSD-first column sums, no inter-column carry).
      * Blocks 1..9: each performs one carry step. Block m+1 reads C_m and
        CARRY_m (scalar carry in), computes s_m = c_m + carry_in_m, then
        produces the one-hot digit D_m = s_m mod 10 and CARRY_{m+1} = floor(s_m/10).
        Block 9 additionally writes D_9 = carry_9.
      * Block 10: position-conditioned selection. For each (last_position p,
        digit value v) fires when POS_p AND D_{20-p}[v]=1, routing to SELECTED[v].
      * Head: SELECTED[v] -> logit for digit token v.

    LayerNorm handling: we set each LN's weight & bias per layer so that
    LN(x) ≈ x for the expected residual stream contents at that layer (the
    residual has a CONST=10 channel everywhere which dominates variance,
    keeping std nearly position-independent).
    """
    torch.manual_seed(0)
    D = model.d_model
    H = model.n_heads
    dh = D // H
    L = model.n_layers
    df = model.d_ff
    V = model.vocab_size
    Tmax = model.max_seq_len

    assert D >= 320, f"d_model={D} too small"
    assert H == 10, f"n_heads must be 10, got {H}"
    assert L == 11, f"n_layers must be 11, got {L}"
    assert df >= 2500, f"d_ff must be >= 2500, got {df}"
    assert V == 12

    # ----- Channel layout -----
    TOK0 = 0                      # 0..11 (token one-hot, 12 dims)
    POS0 = 12                     # 12..33 (position one-hot, 22 dims)
    A0 = POS0 + 22                # 34..83 (A_i: 5 digits x 10 one-hot = 50)
    B0 = A0 + 50                  # 84..133
    C0 = B0 + 50                  # 134..142 (column sums, 9 scalars, scaled by 1/C_SCALE)
    DIG0 = C0 + 9                 # 143..242 (D_k: 10 digits x 10 one-hot = 100)
    CARRY0 = DIG0 + 100           # 243..252 (carry scalars, scaled by 1/C_SCALE)
    SEL0 = CARRY0 + 10            # 253..262 (selected one-hot, 10 dims)
    CONST_CH = SEL0 + 10          # 263
    ZERO_CH = 310                 # guaranteed-zero channel for row-balancing
    # Total used: 264 (plus ZERO_CH at 310)

    def A_ch(i, v):  return A0 + 10 * i + v
    def B_ch(j, v):  return B0 + 10 * j + v
    def C_ch(m):     return C0 + m
    def D_ch(k, v):  return DIG0 + 10 * k + v
    def Cy_ch(m):    return CARRY0 + m
    def Sel_ch(v):   return SEL0 + v

    CONST_VAL = 100.0  # CONST channel value - large to dominate LN std
    C_SCALE   = 100.0  # column sums stored as c_m / C_SCALE to keep them small
    PROD_MAX  = 405    # max c_m = 5*81

    # Expected c_m magnitudes (LSB-first), for stat estimation.
    # E[c_m] = (#pairs contributing) * E[a*b] = count_m * 20.25; here we use
    # the actual scaled value c_m / C_SCALE.
    E_CM = [(min(5, m + 1, 9 - m) * 20.25) / C_SCALE for m in range(9)]

    # ----- Helper: LN ≈ identity for expected mean/std -----
    def estimate_stats(active_values):
        """active_values: list of channel values present in residual (rest are 0)."""
        s  = sum(active_values) + CONST_VAL
        s2 = sum(v * v for v in active_values) + CONST_VAL * CONST_VAL
        mean = s / D
        var = s2 / D - mean * mean
        std = max(var, 1e-6) ** 0.5
        return mean, std

    def set_ln_identity(ln, mean, std):
        ln.weight.data.fill_(std)
        ln.bias.data.fill_(mean)

    def balance_rows(linear):
        """Set ZERO_CH column so each row sums to 0. This cancels constant
        post-LN offsets (since LN of zero channel = offset c, and offset *
        sum(row weights) = 0 if row sum is 0)."""
        w = linear.weight.data
        w[:, ZERO_CH] = 0
        w[:, ZERO_CH] = -w.sum(dim=1)

    # ----- Zero everything -----
    with torch.no_grad():
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        for blk in model.blocks:
            blk.attn.W_q.weight.zero_()
            blk.attn.W_k.weight.zero_()
            blk.attn.W_v.weight.zero_()
            blk.attn.W_o.weight.zero_()
            blk.mlp.fc1.weight.zero_()
            blk.mlp.fc1.bias.zero_()
            blk.mlp.fc2.weight.zero_()
            blk.mlp.fc2.bias.zero_()
        model.head.weight.zero_()

        # ----- Embeddings -----
        # token_emb: TOK[t] = 1 for token t
        for t in range(V):
            model.token_emb.weight[t, TOK0 + t] = 1.0
        # pos_emb: POS[p] = 1, CONST = CONST_VAL
        for p in range(Tmax):
            if p < 22:
                model.pos_emb.weight[p, POS0 + p] = 1.0
            model.pos_emb.weight[p, CONST_CH] = CONST_VAL

        # ============================================================
        # Block 0: ATTENTION gathers a's & b's digits to every output position.
        # ============================================================
        # Residual at block 0 input has 2 ones (TOK, POS) + CONST_VAL=10. 
        mean0, std0 = estimate_stats([1.0, 1.0])
        set_ln_identity(model.blocks[0].ln1, mean0, std0)

        attn0 = model.blocks[0].attn
        # 10 heads, each gathers one digit.
        # Head h ∈ {0..4}: target prompt position = 4 - h, route to A_h.
        # Head h ∈ {5..9}: target prompt position = 10 - (h-5) = 15 - h, route to B_{h-5}.
        ATTN_SCALE = 12.0  # large multiplicative scale for sharp softmax
        for h in range(10):
            if h < 5:
                p_target = 4 - h
                out_start = A0 + 10 * h
            else:
                j = h - 5
                p_target = 10 - j
                out_start = B0 + 10 * j
            q_start = h * dh
            k_start = h * dh
            v_start = h * dh

            # Query: head h's local-dim 0 sourced from CONST channel
            # post-LN value at CONST is approximately CONST_VAL.
            # q[h][0] = ATTN_SCALE * CONST_VAL via this weight
            attn0.W_q.weight[q_start + 0, CONST_CH] = ATTN_SCALE
            # Key: head h's local-dim 0 is "is this position p_target?"
            # post-LN value at POS_{p_target} is 1; so k_p[0] = 1 iff p==p_target.
            attn0.W_k.weight[k_start + 0, POS0 + p_target] = ATTN_SCALE
            # Score at match: ATTN_SCALE * CONST_VAL * ATTN_SCALE / sqrt(dh) = 12*10*12/sqrt(32) ≈ 255
            # very sharp.
            # Value: token-one-hot dims 0..9 (digit tokens) copied to head v-dims 0..9
            for d in range(10):
                attn0.W_v.weight[v_start + d, TOK0 + d] = 1.0
            # Output: head v-dim d -> residual A_h[d] or B_j[d]
            for d in range(10):
                attn0.W_o.weight[out_start + d, v_start + d] = 1.0
        balance_rows(attn0.W_q)
        balance_rows(attn0.W_k)
        balance_rows(attn0.W_v)
        # After block 0 attn at any position p >= 11, A_i[a_i]=1 and B_j[b_j]=1.
        # At earlier positions, attention only partially gathers (irrelevant).

        # ----- Block 0 MLP: compute c_m = sum_{i+j=m} a_i*b_j -----
        # Residual now has TOK(1) + POS(1) + CONST(10) + A's 5 ones + B's 5 ones = 12 ones + 10.
        mean0m, std0m = estimate_stats([1.0] * 12)
        set_ln_identity(model.blocks[0].ln2, mean0m, std0m)
        mlp0 = model.blocks[0].mlp
        # For each (i,j,u,v): neuron = ReLU(A_i[u] + B_j[v] - 1) is 1 iff both =1.
        # Then fc2 weight u*v at C_{i+j} (scaled).
        n = 0
        for i in range(5):
            for j in range(5):
                for u in range(10):
                    for v in range(10):
                        if n >= df:
                            raise RuntimeError("d_ff too small for block 0 MLP")
                        mlp0.fc1.weight[n, A_ch(i, u)] = 1.0
                        mlp0.fc1.weight[n, B_ch(j, v)] = 1.0
                        mlp0.fc1.bias[n] = -1.0
                        # On match: ReLU(1+1-1) = 1. Off: at most 1-1=0 or 0+0-1=-1 → 0.
                        mlp0.fc2.weight[C_ch(i + j), n] = (u * v) / C_SCALE
                        n += 1
        balance_rows(mlp0.fc1)
        # After block 0: C_m channels hold c_m / C_SCALE (max 405/100 = 4.05)

        # ============================================================
        # Blocks 1..9: carry chain. Block m+1 processes column m.
        # ============================================================
        # Max c_m per column:
        col_max = [81, 162, 243, 324, 405, 324, 243, 162, 81]
        # Carry in max per column (computed iteratively).
        carry_max = [0]  # carry into column 0 is 0
        for m in range(9):
            cap = col_max[m] + carry_max[m]
            carry_max.append(cap // 10)
        # s_m max
        s_max = [col_max[m] + carry_max[m] for m in range(9)]

        for b in range(1, 10):
            m = b - 1  # column index handled by this block
            # Residual stream contents at this block's input:
            # 12 ones (TOK,POS,A's,B's) + 9 column scalars (c_m / C_SCALE in [0, ~4])
            # + previous digit one-hots D_0..D_{m-1} (m ones, each 1)
            # + CARRY_m scalar (in [0, ~5] / C_SCALE).
            active = [1.0] * 12 + [E_CM[k] for k in range(9)] + \
                     [1.0] * m + [carry_max[m] / C_SCALE / 2]
            mean_in, std_in = estimate_stats(active)
            set_ln_identity(model.blocks[b].ln1, mean_in, std_in)
            # We don't use attention in carry blocks: set output zero by leaving
            # W_q/W_k/W_v/W_o all zero. The attention contributes ~0 (softmax of zeros
            # gives uniform, value zero → output zero).

            # After (zero) attention, residual unchanged. LN2 stats same as LN1.
            set_ln_identity(model.blocks[b].ln2, mean_in, std_in)
            mlp = model.blocks[b].mlp
            # We need to compute s_m = c_m + carry_m as an integer, then enumerate.
            # Inputs are scaled by 1/C_SCALE; multiply by C_SCALE in fc1 weights.
            # Indicator(s_m == k) for k in [0, s_max[m]]:
            #   I_k(s) = ReLU(s - k + 1) - 2*ReLU(s - k) + ReLU(s - k - 1)
            # We construct 3 ReLU units per k value.
            n = 0
            base_per_k = []  # list of (idx_plus1, idx_zero, idx_minus1)
            for k in range(s_max[m] + 1):
                if n + 3 > df:
                    raise RuntimeError(f"d_ff too small for block {b}, col {m}, k={k}")
                # neuron a: ReLU(C_SCALE*(c_m + carry_m) - k + 1)
                # neuron b: ReLU(... - k)
                # neuron c: ReLU(... - k - 1)
                for offset_idx, offset in enumerate([1.0, 0.0, -1.0]):
                    mlp.fc1.weight[n, C_ch(m)] = C_SCALE
                    mlp.fc1.weight[n, Cy_ch(m)] = C_SCALE
                    mlp.fc1.bias[n] = -k + offset
                    n += 1
                base_per_k.append((n - 3, n - 2, n - 1))
            # fc2: combine 3 ReLUs into I_k(s) = +1*ReLU_a -2*ReLU_b +1*ReLU_c
            # Then sum into D_m[v] for v = k%10, and CARRY_{m+1} += (k//10) * I_k.
            # Also D_9 = carry_9 if this is the last column (m == 8).
            for k in range(s_max[m] + 1):
                a_idx, b_idx, c_idx = base_per_k[k]
                # D_m one-hot
                v_digit = k % 10
                c_out = k // 10
                # Contributions (each multiplied by indicator value 1):
                for idx, coef in [(a_idx, 1.0), (b_idx, -2.0), (c_idx, 1.0)]:
                    mlp.fc2.weight[D_ch(m, v_digit), idx] += coef
                    if m < 8:
                        mlp.fc2.weight[Cy_ch(m + 1), idx] += coef * (c_out / C_SCALE)
                    else:
                        # m == 8: write d_9 = carry_9
                        mlp.fc2.weight[D_ch(9, c_out), idx] += coef
            balance_rows(mlp.fc1)

        # ============================================================
        # Block 10: position-conditioned selection of output digit.
        # ============================================================
        # Need to select D_{20-p}[v] into SELECTED[v] when last-position == p.
        # Residual at block 10 input has 12 ones (TOK,POS,A,B) + 9 C scalars +
        # 10 D one-hots (each 1 magnitude in their selected v).
        active10 = [1.0] * 12 + [E_CM[k] for k in range(9)] + [1.0] * 10
        mean10, std10 = estimate_stats(active10)
        set_ln_identity(model.blocks[10].ln1, mean10, std10)
        set_ln_identity(model.blocks[10].ln2, mean10, std10)
        mlp10 = model.blocks[10].mlp
        # For each (p in 11..20, v in 0..9):
        # neuron = ReLU(POS[p] + D_{20-p}[v] - 1) -> SELECTED[v]
        n = 0
        for p in range(11, 21):
            k_sel = 20 - p  # which digit to output
            for v in range(10):
                if n >= df:
                    raise RuntimeError("d_ff too small for block 10 selection")
                mlp10.fc1.weight[n, POS0 + p] = 1.0
                mlp10.fc1.weight[n, D_ch(k_sel, v)] = 1.0
                mlp10.fc1.bias[n] = -1.0
                mlp10.fc2.weight[Sel_ch(v), n] = 1.0
                n += 1
        balance_rows(mlp10.fc1)

        # ----- Final LN + head -----
        # After block 10, SELECTED[v]=1 for the right v, others 0.
        # active channels: 12 ones + 9 C scalars + 10 D one-hots + 1 SEL one-hot.
        active_final = [1.0] * 12 + [E_CM[k] for k in range(9)] + \
                       [1.0] * 10 + [1.0]
        mean_f, std_f = estimate_stats(active_final)
        set_ln_identity(model.final_ln, mean_f, std_f)

        # Head: SELECTED[v] -> logit for token v (digit token v has id v).
        HEAD_SCALE = 20.0
        for v in range(10):
            model.head.weight[v, Sel_ch(v)] = HEAD_SCALE
        balance_rows(model.head)


model_shorthand_name = "MultCircuitV2"
model_description = (
    "Hand-built 11-layer transformer multiplication circuit: "
    "block 0 gathers digits via per-head positional attention and computes "
    "column sums c_m = sum a_i*b_j; blocks 1-9 do sequential decimal carry "
    "(s_m = c_m + carry_in, d_m = s_m%10, carry_out = s_m//10) via integer "
    "indicator MLPs; block 10 selects the right digit by last position. "
    "LN parameters set per-layer to approximate identity."
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
