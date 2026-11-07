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

# Knobs (sweep these): architecture size and circuit parameters.
D_MODEL = 380           # residual width (must hold the slot layout below, >= 254; divisible by N_HEADS)
N_HEADS = 10            # one head per (operand, place)
N_LAYERS = 2            # 2 = attention gather + MLP digit circuit
D_FF = 512              # hidden width of every MLP (must be >= 500 for MLP 0)
ATTN_SHARPNESS = 3.5    # attention logit gap: matching key vs. everything else
CARRY_WINDOW = 5        # how many lower places the carry lookahead may inspect (5 = all)


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
    # LayerNorm removed: it rescales the residual stream nonlinearly, which
    # breaks exact hand-written features. Blocks are pure residual attn + MLP.
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
    """2-layer causal transformer, vocab/seq-len configured from the task."""

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
# Circuit (2 layers, 10 attention heads, no LayerNorm):
#   Attention (block 0): head r gathers the a-digit of place r, head 5+r gathers
#     the b-digit of place r. Each head has a single matching key, so it copies
#     that digit's one-hot into GA(r,.) / GB(r,.).
#   MLP 0: pair features per place r: NN(r) = [a_r+b_r != 9], CY_G(r) = [a_r+b_r >= 10],
#     SUM(r,s) = [a_r+b_r == s].
#   MLP 1: output digit for answer position j (place p = 5-j) as an OR of mutually
#     exclusive AND-events. The carry into p comes from the nearest lower place k
#     with NN(k)=1: carry = CY_G(k) (and 0 if no such k). Each event is one hidden unit
#     that writes (a_p+b_p+carry) mod 10 into DIG(.).

CONST = 0                    # constant 1 on every token
def DTOK(d): return 1 + d    # 1..10   token is digit d
def POS(t): return 11 + t    # 11..28  absolute position t one-hot
def PA(r): return 29 + r     # 29..33  input position holds the a-digit of place r
def PB(r): return 34 + r     # 34..38  input position holds the b-digit of place r
def GA(r, d): return 39 + 10 * r + d    # 39..88   a_r == d (after attn 0)
def GB(r, d): return 89 + 10 * r + d    # 89..138  b_r == d (after attn 0)
def NN(r): return 139 + r               # 139..143 a_r + b_r != 9
def CY_G(r): return 144 + r             # 144..148 a_r + b_r >= 10
def SUM(r, s): return 149 + 19 * r + s  # 149..243 a_r + b_r == s
def DIG(v): return 244 + v              # 244..253 output digit is v


def _and_unit(m: nn.Module, u: int, pos_lits: list[int], neg_lits: list[int], out_slot: int) -> None:
    """Hidden unit u = AND(pos_lits all 1, neg_lits all 0) via a ReLU threshold."""
    n = len(pos_lits) + len(neg_lits)
    for f in pos_lits:
        m.fc1.weight[u, f] += 1.0
    for f in neg_lits:
        m.fc1.weight[u, f] -= 1.0
    m.fc1.bias[u] = len(neg_lits) - (n - 1)
    m.fc2.weight[out_slot, u] = 1.0


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed."""
    tok = task.stoi
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()

        # --- embeddings ----------------------------------------------------
        E = model.token_emb.weight
        for v in range(task.vocab_size):
            E[v, CONST] = 1.0
        for d in range(10):
            E[tok[str(d)], DTOK(d)] = 1.0
        P = model.pos_emb.weight
        for t in range(model.max_seq_len):
            P[t, POS(t)] = 1.0
        for i in range(5):
            P[i, PA(4 - i)] = 1.0         # a-digit index i is place 4-i
            P[6 + i, PB(4 - i)] = 1.0     # b-digit index i sits at position 6+i

        # --- block 0 attention: 10 single-key gather heads -----------------
        at0 = model.blocks[0].attn
        dh = at0.d_head
        c = ATTN_SHARPNESS * math.sqrt(dh)   # score = c/sqrt(dh) = ATTN_SHARPNESS
        for h in range(10):
            base = h * dh
            if h < 5:
                key_slot, out_slot = PA(h), GA
                r = h
            else:
                key_slot, out_slot = PB(h - 5), GB
                r = h - 5
            at0.W_q.weight[base, CONST] = c
            at0.W_k.weight[base, key_slot] = 1.0
            for d in range(10):
                at0.W_v.weight[base + 1 + d, DTOK(d)] = 1.0
                at0.W_o.weight[out_slot(r, d), base + 1 + d] = 1.0

        # --- block 0 MLP: pair features per place ---------------------------
        m0 = model.blocks[0].mlp
        u = 0
        for r in range(5):
            for a in range(10):
                for b in range(10):
                    m0.fc1.weight[u, GA(r, a)] = 1.0
                    m0.fc1.weight[u, GB(r, b)] = 1.0
                    m0.fc1.bias[u] = -1.0          # relu(a_ind + b_ind - 1) = AND
                    s = a + b
                    if s != 9:
                        m0.fc2.weight[NN(r), u] = 1.0
                    if s >= 10:
                        m0.fc2.weight[CY_G(r), u] = 1.0
                    m0.fc2.weight[SUM(r, s), u] = 1.0
                    u += 1

        # --- block 1 MLP: output digit with carry lookahead ------------------
        # For answer position j (place p = 5-j), the nearest lower place k with
        # NN(k)=1 decides the carry: cin=CY_G(k). Events are exclusive/exhaustive:
        #   cin=1 : CY_G(k)=1,  NN(r)=0 for k<r<p
        #   cin=0 : NN(k)=1 and CY_G(k)=0, NN(r)=0 for k<r<p
        #   none  : NN(r)=0 for all r in the window
        m1 = model.blocks[1].mlp
        u = 0
        for j in range(6):
            p = 5 - j
            lo = max(0, p - CARRY_WINDOW)
            sums = [0] if p == 5 else range(19)
            for s in sums:
                s_lit = CONST if p == 5 else SUM(p, s)   # a_5 = b_5 = 0
                base_pos = [POS(11 + j), s_lit]
                for k in range(lo, p):
                    between_neg = [NN(r) for r in range(k + 1, p)]
                    _and_unit(m1, u, base_pos + [CY_G(k)], between_neg, DIG((s + 1) % 10))
                    u += 1
                    _and_unit(m1, u, base_pos + [NN(k)], [CY_G(k)] + between_neg, DIG(s % 10))
                    u += 1
                _and_unit(m1, u, base_pos, [NN(r) for r in range(lo, p)], DIG(s % 10))
                u += 1
        assert u <= model.d_ff, f"need {u} hidden units"

        # --- unembedding -------------------------------------------------------
        for d in range(10):
            model.head.weight[tok[str(d)], DIG(d)] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = f"HandCircuit2L-win{CARRY_WINDOW}-sh{ATTN_SHARPNESS:g}-d{D_MODEL}-ff{D_FF}-L{N_LAYERS}"
model_description = (
    f"2-layer hand-built circuit (no LayerNorm): 10 single-key attention heads gather a_r, b_r per place; "
    f"MLP0 computes per-place sum/generate/non-9 features; MLP1 outputs (a_p+b_p+carry) mod 10, carry = "
    f"generate bit of nearest lower non-9 place within window {CARRY_WINDOW}. "
    f"Knobs: d_model={D_MODEL}, d_ff={D_FF}, layers={N_LAYERS}, attn_sharpness={ATTN_SHARPNESS:g}."
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
