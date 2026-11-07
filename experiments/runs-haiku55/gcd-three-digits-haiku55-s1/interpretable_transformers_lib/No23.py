"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. Current task: gcd-three-digits.
Prompt "865,395=" -> answer "005" (3-digit zero-padded gcd).

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py --task gcd-three-digits
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
# LayerNorm is replaced by identity so that the residual stream stays exactly
# readable as one-hot / indicator features.
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
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task.

    d_ff is a per-layer tuple: layer 0 holds the divisibility lookup units,
    layer 1 the AND units, layer 2 the output lookup units.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 144,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: tuple = (2560, 32, 1536),
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
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff[i]) for i in range(n_layers)])
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

# Prime-power divisibility flags: each (p, k) means "divisible by p**k".
ALL_FLAGS = [
    (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 1), (3, 2), (3, 3),
    (5, 1), (5, 2), (5, 3),
    (7, 1), (11, 1), (13, 1), (17, 1), (19, 1), (23, 1),
]
NF = len(ALL_FLAGS)
# Which flags the circuit uses (must be nested: (p,k) requires (p,k-1)).
ENABLED_FLAGS = [f for f in ALL_FLAGS if f != (23, 1)]
# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "No23"
model_description = "AllFlags16 minus the 23 flag: ablation to see the cost of not handling prime 23."

# Residual-stream layout (d_model = 144)
TOK = 0                 # dims 0..11 : token one-hot
POS = 12                # dims 12..22: position one-hot (positions 0..10)
CONST = 23              # dim 23     : constant 1 (bias-like input to attention queries)
SLOT = 24               # dims 24..83: six digit slots, slot j = 10 one-hot dims
FLAG_A = SLOT + 60      # NF dims: "a divisible by p^k" flags
FLAG_B = FLAG_A + NF    # NF dims: "b divisible by p^k" flags
BOTH = FLAG_B + NF      # NF dims: "both divisible by p^k" flags
LOG = BOTH + NF         # 10 dims: output digit scores (read by the head)
D_RES = LOG + 10

SLOT_POS = [0, 1, 2, 4, 5, 6]   # prompt positions of a1 a2 a3 b1 b2 b3 ("aaa,bbb=")
ATTN_SCALE = 400.0              # large -> hard attention selection


def _suffix_len(m: int) -> int:
    """Number of trailing digits that determine divisibility by m (<= 3 for our 3-digit inputs)."""
    for L in (1, 2, 3):
        if (10 ** L) % m == 0:
            return L
    return 3


def _divisibility_units(m: int, number_index: int):
    """Enumerate digit tuples (over the trailing L digits of one 3-digit number) divisible by m.

    Returns (list of (slot_list, digit_list), L). Each tuple becomes one ReLU unit that
    fires iff the number's digits match that tuple (an exact AND of one-hot inputs).
    """
    L = _suffix_len(m)
    base = 3 * number_index  # 0 for a, 3 for b
    units = []
    for s in range(10 ** L):
        if s % m == 0:
            digits = [int(c) for c in str(s).zfill(L)]
            slots = [base + (3 - L) + i for i in range(L)]
            units.append((slots, digits))
    return units, L


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place with a hand-wired gcd circuit. No training.

    Circuit:
      L0 attention : six heads copy the digits a1 a2 a3 b1 b2 b3 into digit slots.
      L0 MLP       : one unit per digit tuple divisible by p^k -> FLAG_A / FLAG_B.
      L1 MLP       : BOTH[f] = FLAG_A[f] AND FLAG_B[f].
      L2 MLP       : for each candidate common factor P = prod p^e_p (e_p = largest k
                     with BOTH flag set, capped), and each output position, a unit that
                     fires iff the exponents match AND the position matches; it writes
                     the corresponding digit of P into LOG.
    The result is the gcd restricted to the enabled prime powers (exact whenever the
    true gcd's prime factors are all enabled and within the caps).
    """
    assert model.d_model >= D_RES, (model.d_model, D_RES)
    DH = model.blocks[0].attn.d_head
    assert DH >= 11
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # --- embeddings -------------------------------------------------------
        for t in range(12):
            model.token_emb.weight[t, TOK + t] = 1.0
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, CONST] = 1.0
            if p < 11:
                model.pos_emb.weight[p, POS + p] = 1.0

        # --- output head: read digit scores from LOG ----------------------------
        for d in range(10):
            model.head.weight[d, LOG + d] = 1.0

        # --- layer 0 attention: copy each digit into its slot -------------------
        attn = model.blocks[0].attn
        for j, src in enumerate(SLOT_POS):
            h0 = DH * j  # head j owns dims [DH*j, DH*j+DH) of q/k/v
            attn.W_q.weight[h0 + src, CONST] = ATTN_SCALE          # query = "attend to src"
            for i in range(11):
                attn.W_k.weight[h0 + i, POS + i] = 1.0             # key = own position
            for t in range(10):
                attn.W_v.weight[h0 + t, TOK + t] = 1.0             # value = digit one-hot
                attn.W_o.weight[SLOT + 10 * j + t, h0 + t] = 1.0   # write into slot j

        # --- layer 0 MLP: divisibility lookup units ----------------------------
        mlp0 = model.blocks[0].mlp
        u = 0
        for f, (p, k) in enumerate(ALL_FLAGS):
            if (p, k) not in ENABLED_FLAGS:
                continue
            for number_index in (0, 1):
                units, L = _divisibility_units(p ** k, number_index)
                flag_dim = (FLAG_A if number_index == 0 else FLAG_B) + f
                for slots, digits in units:
                    for slot, dig in zip(slots, digits):
                        mlp0.fc1.weight[u, SLOT + 10 * slot + dig] = 1.0
                    mlp0.fc1.bias[u] = -(L - 1)           # fires iff all L digits match
                    mlp0.fc2.weight[flag_dim, u] = 1.0
                    u += 1
        assert u <= model.d_ff[0], f"layer-0 needs {u} units, have {model.d_ff[0]}"

        # --- layer 1 MLP: BOTH = FLAG_A AND FLAG_B ------------------------------
        mlp1 = model.blocks[1].mlp
        u = 0
        for f in range(NF):
            if ALL_FLAGS[f] not in ENABLED_FLAGS:
                continue
            mlp1.fc1.weight[u, FLAG_A + f] = 1.0
            mlp1.fc1.weight[u, FLAG_B + f] = 1.0
            mlp1.fc1.bias[u] = -1.0
            mlp1.fc2.weight[BOTH + f, u] = 1.0
            u += 1
        assert u <= model.d_ff[1]

        # --- layer 2 MLP: exponent-combo lookup -> digits of P at positions 7,8,9
        primes = sorted({p for p, _ in ENABLED_FLAGS})
        caps = {p: max(k for pp, k in ENABLED_FLAGS if pp == p) for p in primes}
        mlp2 = model.blocks[2].mlp
        u = 0
        for e_vals in itertools.product(*[range(caps[p] + 1) for p in primes]):
            e = dict(zip(primes, e_vals))
            P = 1
            for p in primes:
                P *= p ** e[p]
            if P > 999:          # cannot divide two 3-digit numbers
                continue
            for out_pos in (7, 8, 9):   # 7 -> hundreds, 8 -> tens, 9 -> units
                for f, (p, k) in enumerate(ALL_FLAGS):
                    if (p, k) not in ENABLED_FLAGS:
                        continue
                    # +1 for flags up to the exponent, -1 for flags above it.
                    mlp2.fc1.weight[u, BOTH + f] = 1.0 if k <= e[p] else -1.0
                mlp2.fc1.weight[u, POS + out_pos] = 1.0
                mlp2.fc1.bias[u] = -float(sum(e.values()))
                digit = [P // 100, (P // 10) % 10, P % 10][out_pos - 7]
                mlp2.fc2.weight[LOG + digit, u] = 1.0
                u += 1
        assert u <= model.d_ff[2], f"layer-2 needs {u} units, have {model.d_ff[2]}"


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
