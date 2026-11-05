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
        # LayerNorm is replaced by identity: the hand-built circuit stores a
        # scalar (x) in one residual dimension, which LN would rescale.
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
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 170,
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

# Residual-stream layout (d_model = 64):
TOK = 0      # digit value of the current token (0..9; '=' -> 0)
X = 1        # x = 100a + 10b + c                       (layer-1 attention)
BIT = 2      # output bit for the current position      (layer-2 MLP)
ONE = 3      # constant 1 (set in every position embedding)
R0 = 4       # c            = x mod 2 residue source     (layer-1 attention)
R1 = 5       # 2b + c       = x mod 4 residue source     (layer-1 attention)
R2 = 6       # 4a + 2b + c  = x mod 8 residue source     (layer-1 attention)
POS = 8      # one-hot position indicator e_p lives in dims POS..POS+11
BITS = 24    # dims BITS..BITS+7: bit_j features           (layer-1 MLP)

N_PROMPT = 4    # "abc="
N_ANSWER = 8    # 8 output bits, MSB first

ATTN_SHARPNESS = 40.0    # q.k before 1/sqrt(d_head); ~10 after scaling -> near-hard attention
LOGIT_SCALE = 10.0


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Two-layer circuit (all closed form):
      Layer 1 attention: each digit a,b,c is gathered by a fixed head and written,
         with place-value weights, into several residual 'residue' dims:
           X  = 100a + 10b + c,   R0 = c,   R1 = 2b + c,   R2 = 4a + 2b + c.
         Key fact: bit j of x depends only on x mod 2^(j+1), and
         x ≡ R0 (mod 2), x ≡ R1 (mod 4), x ≡ R2 (mod 8) since 100 ≡ 4 (mod 8),
         10 ≡ 2 (mod 8). So bit 0, 1, 2 can be read from small residues.
      Layer 1 MLP (thermometer comparators): for threshold v, two ReLUs give
         [R >= v] = relu(R - v + 1) - relu(R - v) (exact for integer R).
         bit_j(R) = sum_v sign_v [R >= v], v multiple of 2^j, sign = +1 for odd v/2^j.
           bit 0 from R0 (v = 1..9), bit 1 from R1 (v = 2..26 even),
           bit 2 from R2 (v = 4..32 mult of 4), bits 3..7 from X (v = 8..248 mult of 8).
         Total 122 comparator units.  All bit sums go to BITS+j.
      Layer 2 MLP (position select): unit_j = relu(bit_j + p_t(j) - 1), equal to
         bit_j at the output position t(j) and 0 elsewhere. Sum over j -> BIT.
      Output head: logit('1') = L*BIT, logit('0') = L*(1 - BIT).
    """
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        D = model.d_model
        H = model.n_heads
        dh = D // H

        # --- embeddings -----------------------------------------------------
        for d in range(10):
            model.token_emb.weight[d, TOK] = float(d)
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, ONE] = 1.0
            if p < N_PROMPT + N_ANSWER:
                model.pos_emb.weight[p, POS + p] = 1.0

        # --- layer 1 attention: gather digits into the residue dims ---------
        # head h copies one prompt digit (0 = hundreds a, 1 = tens b, 2 = units c);
        # the head's value is the digit, and W_o writes it into several dims.
        head_writes = {
            0: {X: 100.0, R2: 4.0},                  # a
            1: {X: 10.0, R1: 2.0, R2: 2.0},          # b
            2: {X: 1.0, R0: 1.0, R1: 1.0, R2: 1.0},  # c
        }
        attn = model.blocks[0].attn
        for h, writes in head_writes.items():
            base = h * dh
            attn.W_q.weight[base + h, ONE] = ATTN_SHARPNESS   # query: target prompt index h
            for i in range(N_PROMPT + N_ANSWER):
                attn.W_k.weight[base + i, POS + i] = 1.0      # key: one-hot of own position
            attn.W_v.weight[base + 0, TOK] = 1.0              # value: the digit
            for dim, w in writes.items():
                attn.W_o.weight[dim, base + 0] = w

        # --- layer 1 MLP: thermometer comparators on each residue -----------
        # (source dim, max value, bits it serves)
        residues = [
            (R0, 9, [0]),
            (R1, 27, [1]),
            (X, 255, [2, 3, 4, 5, 6, 7]),
        ]
        mlp1 = model.blocks[0].mlp
        W1, b1, W2 = mlp1.fc1.weight, mlp1.fc1.bias, mlp1.fc2.weight
        u = 0
        for dim, vmax, bits in residues:
            j0 = min(bits)
            for v in range(2 ** j0, vmax + 1, 2 ** j0):
                ia, ib = u, u + 1
                u += 2
                W1[ia, dim] = 1.0
                b1[ia] = 1.0 - v          # relu(R - v + 1)
                W1[ib, dim] = 1.0
                b1[ib] = -float(v)        # relu(R - v)
                for j in bits:
                    if v % (2 ** j) == 0:
                        sign = 1.0 if (v // (2 ** j)) % 2 == 1 else -1.0
                        W2[BITS + j, ia] = sign
                        W2[BITS + j, ib] = -sign
        assert u <= model.d_ff, f"layer 1 needs {u} units, have {model.d_ff}"

        # --- layer 2 MLP: select the bit for the current output position ----
        mlp2 = model.blocks[1].mlp
        W1b, b1b, W2b = mlp2.fc1.weight, mlp2.fc1.bias, mlp2.fc2.weight
        for j in range(N_ANSWER):
            # output bit j is emitted from position t = N_PROMPT - 1 + (N_ANSWER - 1 - j)
            t = N_PROMPT - 1 + (N_ANSWER - 1 - j)
            W1b[j, BITS + j] = 1.0
            W1b[j, POS + t] = 1.0
            b1b[j] = -1.0               # relu(bit_j + p_t - 1)
            W2b[BIT, j] = 1.0

        # --- output head ----------------------------------------------------
        model.head.weight[1, BIT] = LOGIT_SCALE            # '1'
        model.head.weight[0, ONE] = LOGIT_SCALE            # '0' = L*(1 - BIT)
        model.head.weight[0, BIT] = -LOGIT_SCALE


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ResidueThermo2L_noR2"
model_description = (
    "Ablation of ResidueThermo2L without the R2 = 4a+2b+c residue: bit 2 is read from x directly, which "
    "needs thresholds at all multiples of 4 on x (d_ff=170)."
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
