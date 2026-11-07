"""
Interpretable transformer for character-level sequence tasks (edit-distance-8).

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * Weights are written directly in `write_weights()`.

Usage:
    uv run interpretable_transformer.py --task edit-distance-8
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
    # Heads have their own (small) query/key width d_qk and value width d_v.
    def __init__(self, d_model: int, n_heads: int, d_qk: int = 2, d_v: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.d_qk, self.d_v = d_qk, d_v
        # head 0 query is shared by all layers (W_q); head 1's query (the read pointer) differs per layer (W_qc)
        self.W_q = nn.Linear(d_model, d_qk, bias=True)
        self.W_qc = nn.Linear(d_model, d_qk, bias=True)     # weight tied across layers; only the bias (pointer offset) is per layer
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        q = torch.cat([self.W_q(x), self.W_qc(x)], dim=-1).view(B, T, H, self.d_qk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.d_qk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.d_v).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_qk)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * self.d_v)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    # No LayerNorm: the circuit uses exact linear residual-stream arithmetic.
    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_mlp: bool = True):
        super().__init__()
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff) if use_mlp else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 11,
        n_heads: int = 2,
        n_layers: int = 17,
        d_ff: int = 16,
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
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, use_mlp=(i < n_layers - 1)) for i in range(n_layers)])   # last block = readout attention only
        self.final_ln = nn.Identity()
        # weight tying: all blocks share one set of weights, except the per-layer read pointer W_qc
        b0 = self.blocks[0]
        for blk in self.blocks[1:]:
            blk.attn.W_q, blk.attn.W_k, blk.attn.W_v, blk.attn.W_o = b0.attn.W_q, b0.attn.W_k, b0.attn.W_v, b0.attn.W_o
            blk.attn.W_qc.weight = b0.attn.W_qc.weight
            if blk.mlp is not None:
                blk.mlp = b0.mlp
        self.head =nn.Linear(d_model, vocab_size, bias=True)    # bias replaces a constant-1 residual dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Levenshtein DP as an anti-diagonal wavefront, ONE block per diagonal.

    Prompt "s1(8),s2(8)=": position 8+j (j=1..8) plays DP column j; the comma is
    column 0.  Block t-1 computes anti-diagonal t, S_t(j) = D[t-j][j], and writes
    it into its own residual slot S_t.  Inputs of the cell:
        A = D[i-1][j]   = own slot S_{t-1}
        B = D[i][j-1]   = slot S_{t-1} of position p-1   (attention head 0)
        C = D[i-1][j-1] = slot S_{t-2} of position p-1   (attention head 0)
        match = [s1[i-1] == s2[j-1]]  from head 1, which attends to position t+7-p
    Adjacent DP cells differ by <=1, so min(A+1,B+1,C+1-match) collapses to
        C + (1-match) * (1 - relu(C-A) - relu(C-B) + relu(2C-A-B-1))
    which one relu layer can do, gating each unit by the query letter / key
    letter (match) and by "active column" (positions 9..16).
    Cells outside the table are pre-filled (in the position embedding) with the
    mirrored extension D[i][j]=j-i for i<0, so the same rule yields D[0][j]=j;
    the comma holds D[t][0]=t.  The answer D[8][8]=S_16 at position 16 is read
    by position 17 and decoded with logit_d = K*(2 d v - d^2).
    """
    # Pointers use quadratic scores  -beta*(k - target)^2 = beta*(2*k*target - k^2) + const,
    # so a key only needs the features (k, k^2) and a query the scalar `target`.
    G2, G, K = 20.0, 16.0 * math.sqrt(2), 1.0   # G = beta*sqrt(d_qk)
    HB = 2           # head-1 offset in q/k space
    HV = 2           # head-1 offset in value space
    # residual layout: letters 0-3, position features PS=p, PS2=p^2, ONE=1, shift registers R1=S_{t-1}, R2=S_{t-2},
    # scratch NB1/NB2 (neighbour's R1/R2), TOK (fetched letter), flags ACT (active column), M (match), CM (comma)
    L0, PS, PS2, R1, R2, NB1, NB2, TOK, ACT, M, CM = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10   # letters are a scalar code a,b,c,d = 0..3
    stoi = task.stoi
    T = 19
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()
        te, pe = model.token_emb.weight, model.pos_emb.weight
        for ch, idx in stoi.items():
            if ch in "abcd":
                te[idx, L0] = float("abcd".index(ch))
        for p in range(pe.shape[0]):
            pe[p, PS] = float(p)
            pe[p, PS2] = float(p * p)
        for p in range(9, 17):                       # active columns j=p-8
            j = p - 8
            pe[p, ACT] = 1.0
            pe[p, R1] = 2 * j                        # S_0(j)  = D[-j][j]  = 2j   (virtual cell)
            pe[p, R2] = 2 * j + 1                    # S_-1(j) = D[-1-j][j] = 2j+1
        pe[8, R1] = 0.0                              # comma: S_0 = D[0][0] = 0
        pe[8, R2] = 1.0                              # virtual D[-1][0] = 1
        pe[8, CM] = 1.0

        def attn_nbr(blk, t):
            a = blk.attn
            # head 0 points at target = p-1:  q = G*(2*(p-1), -1),  k = (k, k^2)
            a.W_q.weight[0, PS] = 2 * G; a.W_q.bias[0] = -2 * G
            a.W_q.bias[1] = -G
            a.W_k.weight[0, PS] = 1.0; a.W_k.weight[1, PS2] = 1.0
            a.W_v.weight[0, R1] = 1.0
            a.W_v.weight[1, R2] = 1.0
            a.W_o.weight[NB1, 0] = 1.0
            a.W_o.weight[NB2, 1] = 1.0

        def attn_cost(blk, t):
            a = blk.attn
            # head 1 points at target = (t+7) - p  (anti-diagonal); the ONLY per-layer number is the 2*G*(t+7) below
            a.W_qc.weight[0, PS] = -2 * G; a.W_qc.bias[0] = 2 * G * (t + 7)
            a.W_qc.bias[1] = -G
            a.W_k.weight[HB, PS] = 1.0; a.W_k.weight[HB + 1, PS2] = 1.0
            a.W_v.weight[HV, L0] = 1.0                     # fetch the letter code of the target position
            a.W_o.weight[TOK, HV] = 1.0

        for t in range(1, 17):
            blk = model.blocks[t - 1]
            attn_nbr(blk, t); attn_cost(blk, t + 1)      # head 1 fetches s1 letter for the NEXT step
            w1, b1, w2, b2 = blk.mlp.fc1.weight, blk.mlp.fc1.bias, blk.mlp.fc2.weight, blk.mlp.fc2.bias
            A = R1
            u = 0
            # mismatch-gated units for THIS step (M = match flag computed in the previous block):
            # nonzero only when column is active and M == 0
            terms = [  # (weights dict, const, sign into S_t)
                ({}, 1.0, +1),                                   # 1
                ({NB2: 1, A: -1}, 0.0, -1),                      # relu(C-A)
                ({NB2: 1, NB1: -1}, 0.0, -1),                    # relu(C-B)
                ({NB2: 2, A: -1, NB1: -1}, -1.0, +1),            # relu(2C-A-B-1)
            ]
            for wd, c0, sgn in terms:
                for d_, v in wd.items():
                    w1[u, d_] = v
                w1[u, ACT] = G2; w1[u, M] = -G2
                b1[u] = c0 - G2
                w2[R1, u] = sgn
                u += 1
            # match flag for the NEXT step: M = sum_k [own letter k] * [fetched letter k]
            # M = 1 - min(1, |TOK - L|) = 1 - [relu(d) - relu(d-1) + relu(-d) - relu(-d-1)],  d = TOK - L
            for sgn_d, off, sgn_out in [(1, 0, -1), (1, -1, +1), (-1, 0, -1), (-1, -1, +1)]:
                w1[u, TOK] = sgn_d; w1[u, L0] = -sgn_d; b1[u] = off; w2[M, u] = sgn_out; u += 1
            b2[M] = 1.0
            w1[u, M] = 1; w2[M, u] = -1; u += 1                   # clear old M
            # R1 <- S_t = C + terms: add gated C, remove gated old R1; comma just counts (R1 += 1)
            w1[u, NB2] = 1; w2[R1, u] = 1; w2[NB2, u] = -1; u += 1   # R1 += C (ungated: only ever read at active columns), clear NB2
            w1[u, R1] = 1; w1[u, ACT] = G2; b1[u] = -G2; w2[R1, u] = -1; u += 1   # remove old R1 (active columns only)
            w1[u, CM] = 1; w2[R1, u] = 1; u += 1      # comma counts up: D[t][0] = t
            # R2 <- old R1 (shift register)
            w1[u, R1] = 1; w2[R2, u] = 1; u += 1     # R2 <- old R1 (shift register)
            w1[u, R2] = 1; w2[R2, u] = -1; u += 1
            # scratch cleanup
            for d_ in [NB1, TOK]:
                w1[u, d_] = 1; w2[d_, u] = -1; u += 1
        attn_nbr(model.blocks[16], 17)                            # NB1 <- S_16 of position 16
        hd, hb = model.head.weight, model.head.bias
        for ch, idx in stoi.items():
            if ch.isdigit() and int(ch) <= 8:
                d = int(ch)
                hd[idx, NB1] = 2 * d * K
                hb[idx] = -(d ** 2) * K
            else:
                hb[idx] = -100.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "final_compact"
model_description = "Clean compact model: 17 tied blocks, d_model 11, d_ff 16, d_qk 2; DP wavefront with shift registers, quadratic pointers (beta=16), gate G2=20, scalar letters, head bias."


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
