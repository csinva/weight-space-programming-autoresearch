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
    """4-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 384,
        n_heads: int = 1,
        n_layers: int = 4,
        d_ff: int = 512,
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
# Residual-stream layout (one named slot per feature):
#   CONST         constant 1 on every token (used as the attention query bias)
#   DTOK(d)       token is digit d
#   POS(t)        absolute position t one-hot
#   INDIG         position holds an input digit (prompt positions 0-4, 6-10)
#   PA(r), PB(r)  the input digit at this position is the a / b digit of place r
#   PROD_A/B(r,d) = DTOK(d) AND PA/PB(r)                      (MLP 0)
#   GRID_A/B(r,d) uniform average over the 10 input digits of PROD (attn 1)
#                 = (1/10) * [a_r == d] / [b_r == d]
#   NN(r)         a_r + b_r != 9         (place r "propagates" nothing special)
#   CY_G(r)       a_r + b_r >= 10        (place r generates a carry)
#   SUM(r,s)      a_r + b_r == s
#   CARRY(p)      carry INTO place p = CY_G of the nearest lower place k<p
#                 with NN(k)=1 (0 if none); lookahead via AND-chains (MLP 2)
#   DIG(v)        output digit v = (a_p + b_p + c_p) mod 10  (MLP 3)
# Output position for answer digit j (place p = 5 - j) is the position 11 + j.

D_MODEL = 384
N_SLOT_END = 361  # highest used slot + 1 (rest of d_model is zero padding)

CONST = 0
def DTOK(d): return 1 + d            # 1..10
def POS(t): return 11 + t            # 11..28
INDIG = 29
def PA(r): return 30 + r             # 30..34
def PB(r): return 35 + r             # 35..39
def PROD_A(r, d): return 40 + 10 * r + d    # 40..89
def PROD_B(r, d): return 90 + 10 * r + d    # 90..139
def GRID_A(r, d): return 140 + 10 * r + d   # 140..189
def GRID_B(r, d): return 190 + 10 * r + d   # 190..239
def NN(r): return 240 + r            # 240..244
def CY_G(r): return 245 + r          # 245..249
def SUM(r, s): return 250 + 19 * r + s      # 250..344
def CARRY(p): return 345 + p         # 345..350
def DIG(v): return 351 + v           # 351..360


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed."""
    tok = task.stoi
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()

        # --- embeddings -----------------------------------------------------
        E = model.token_emb.weight
        for v in range(task.vocab_size):
            E[v, CONST] = 1.0
        for d in range(10):
            E[tok[str(d)], DTOK(d)] = 1.0

        P = model.pos_emb.weight
        for t in range(model.max_seq_len):
            P[t, POS(t)] = 1.0
        for t in range(model.max_seq_len):
            if t < 5:                       # a digit, index i=t, place 4-i
                P[t, INDIG] = 1.0
                P[t, PA(4 - t)] = 1.0
            elif 6 <= t <= 10:              # b digit, index i=t-6
                P[t, INDIG] = 1.0
                P[t, PB(4 - (t - 6))] = 1.0

        # --- block 0 MLP: products DTOK(d) AND PA/PB(r) -----------------------
        m0 = model.blocks[0].mlp
        for r in range(5):
            for d in range(10):
                hA = 10 * r + d
                m0.fc1.weight[hA, DTOK(d)] = 1.0
                m0.fc1.weight[hA, PA(r)] = 1.0
                m0.fc1.bias[hA] = -1.0
                m0.fc2.weight[PROD_A(r, d), hA] = 1.0

                hB = 50 + 10 * r + d
                m0.fc1.weight[hB, DTOK(d)] = 1.0
                m0.fc1.weight[hB, PB(r)] = 1.0
                m0.fc1.bias[hB] = -1.0
                m0.fc2.weight[PROD_B(r, d), hB] = 1.0

        # --- block 1 attention: uniform average over the 10 input digits -----
        # query = c * CONST (dim 0), key = INDIG (dim 0): score = c/sqrt(D) on
        # input digit positions, 0 elsewhere -> ~uniform over the 10 digits.
        at1 = model.blocks[1].attn
        c = 30.0 * math.sqrt(D_MODEL)
        at1.W_q.weight[0, CONST] = c
        at1.W_k.weight[0, INDIG] = 1.0
        for i in range(40, 140):             # PROD slots -> value -> GRID slots
            at1.W_v.weight[i, i] = 1.0
            at1.W_o.weight[i + 100, i] = 1.0

        # --- block 1 MLP: per-place pair features -----------------------------
        m1 = model.blocks[1].mlp
        for r in range(5):
            for a in range(10):
                for b in range(10):
                    u = 100 * r + 10 * a + b
                    # 10 * GRID = indicator, so relu(ind_a + ind_b - 1) = AND
                    m1.fc1.weight[u, GRID_A(r, a)] = 10.0
                    m1.fc1.weight[u, GRID_B(r, b)] = 10.0
                    m1.fc1.bias[u] = -1.0
                    s = a + b
                    if s != 9:
                        m1.fc2.weight[NN(r), u] = 1.0
                    if s >= 10:
                        m1.fc2.weight[CY_G(r), u] = 1.0
                    m1.fc2.weight[SUM(r, s), u] = 1.0

        # --- block 2 MLP: carry lookahead -------------------------------------
        # c_p = sum_{k<p} relu( CY_G(k) - sum_{k<r<p} NN(r) )
        # (exactly the nearest lower place with NN=1 contributes, and only if
        #  it generates a carry)
        m2 = model.blocks[2].mlp
        u = 0
        for p in range(1, 6):
            for k in range(p):
                m2.fc1.weight[u, CY_G(k)] = 1.0
                for r in range(k + 1, p):
                    m2.fc1.weight[u, NN(r)] = -1.0
                m2.fc2.weight[CARRY(p), u] = 1.0
                u += 1

        # --- block 3 MLP: digit = (a_p + b_p + c_p) mod 10 selected by position
        m3 = model.blocks[3].mlp
        u = 0
        for j in range(6):
            p = 5 - j
            for s in ([0] if p == 5 else range(19)):
                for cin in (0, 1):
                    m3.fc1.weight[u, POS(11 + j)] = 1.0
                    if p == 5:
                        m3.fc1.weight[u, CONST] = 1.0    # a_5 = b_5 = 0
                    else:
                        m3.fc1.weight[u, SUM(p, s)] = 1.0
                    if cin == 1:
                        m3.fc1.weight[u, CARRY(p)] = 1.0
                        m3.fc1.bias[u] = -2.0
                    else:
                        m3.fc1.weight[u, CARRY(p)] = -1.0
                        m3.fc1.bias[u] = -1.0
                    m3.fc2.weight[DIG((s + cin) % 10), u] = 1.0
                    u += 1

        # --- unembedding -------------------------------------------------------
        for d in range(10):
            model.head.weight[tok[str(d)], DIG(d)] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "HandCircuitCarryLookahead-v1"
model_description = (
    "4-layer hand-built circuit, no LayerNorm: MLP0 forms digit-place products, attention 1 "
    "averages them over the 10 input digits, MLP1 computes per-place sum/generate/propagate, "
    "MLP2 resolves each carry as the generate bit of the nearest lower non-9 place, MLP3 outputs "
    "(a+b+carry) mod 10 for the current output position."
)
D_MODEL = 384


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
