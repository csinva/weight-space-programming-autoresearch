"""
Interpretable transformer for the `boolean-circuit-5-bits` task (hand-set weights, no training).

Task format (learned by probing src/task.py):
  prompt = b0 op1 b1 op2 b2 op3 b3 op4 b4 '='   (10 chars, ops in {&,|,^}, bits in {0,1})
  answer = one char, the value of the expression evaluated LEFT TO RIGHT (verified 2592/2592
           exhaustively; standard &>^>| precedence matches only ~75% of generated examples).

Circuit (exact over {0,1}):
  Each step t=1..4 is the unary map v -> a_t * v XOR c_t on GF(2):
      &0 -> (a,c)=(0,0)   &1 -> (1,0)   |0 -> (1,0)   |1 -> (0,1)   ^0 -> (1,0)   ^1 -> (1,1)
  Unrolling from v_0 = b0:
      answer = (a1 a2 a3 a4) b0  XOR  sum_t c_t * prod_{s>t} a_s
  Layers (computed at the answer position, index 9, the '=' token):
    L1  attention: head k (k=0..8) attends to position k and copies its token one-hot.
        MLP: per-step lookup (op_t, b_t) -> (a_t, c_t) via the indicators needed for a_t and c_t.
    L2  MLP: the five AND-products T_0..T_4 of the formula above.
    L3  MLP: S = sum_j T_j in {0..5}; S mod 2 = S - 2[S>=2] - 2[S>=4] via 5 ReLU units.
    Head: logit('1') = 1*y, logit('0') = 1*(1-y).
  LayerNorm is replaced by identity so the residual stream stays exactly linear between ReLUs.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task boolean-circuit-5-bits --n-samples 500
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
        # LayerNorm replaced by identity: keeps the residual stream linear (exact hand-built routing).
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
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
        max_seq_len: int = 16,
        d_model: int = 96,
        n_heads: int = 12,
        n_layers: int = 3,
        d_ff: int = 32,
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

# Hyperparameters of the hand-built circuit (derived from the layout below).
N_POS_USED = 10        # positions 0..9 (the prompt length)
N_GATHER = 9           # positions 0..8 are gathered at the answer position
ANS_POS = 9            # index of '=' in the prompt
N_HEADS = 14           # >= N_GATHER; extra heads stay zero
LUT_KEYS = [("&", 0), ("|", 1), ("^", 1)]  # (op, operand) indicators actually used: &0, |1, ^1
# (the full 6-key LUT is LUT_KEYS = [(op, b) for op in "&|^" for b in (0, 1)])

TOKEN_ID = {"0": 0, "1": 1, "&": 2, "|": 3, "^": 4, "=": 5}

# Residual-stream layout (packed):
TOK_SLOT = 0                              # 0..5   one-hot of the token at this position
POS_SLOT = 6                              # 6..6+N_POS_USED-1   one-hot of this position index
GATH = POS_SLOT + N_POS_USED              # 6 slots per gathered position k
A_SLOT = GATH + 6 * N_GATHER              # a_1..a_4 (slope of step t)
C_SLOT = A_SLOT + 4                       # c_1..c_4 (intercept of step t)
T_SLOT = C_SLOT + 4                       # T_0..T_4 (AND products)
Y_SLOT = T_SLOT + 5                       # parity of sum_j T_j = the answer bit
D_MODEL = Y_SLOT + 1
D_HEAD = D_MODEL // N_HEADS
N_LUT = len(LUT_KEYS)
CONST_UNIT = 4 * N_LUT                    # layer-1 constant unit (ReLU(1) at answer position)
D_FF = CONST_UNIT + 1
EQ_SLOT = TOK_SLOT + TOKEN_ID["="]        # '=' one-hot at the answer position is 1


def _gslot(k: int, tok: int) -> int:
    """Residual slot holding 'token at position k is tok' after layer 1."""
    return GATH + 6 * k + tok


def _unit(t: int, op: str, b: int) -> int:
    return (t - 1) * N_LUT + LUT_KEYS.index((op, b))


def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-set every parameter (no training).

    Layer 1 attention: head k (k=0..8) at the answer position attends to position k and copies
        its token one-hot into residual slots GATH+6k .. GATH+6k+5. The score margin on the matching
        key is set by the constant in `s` below (sensitivity sweep: margin 2 and 3 fail, 4 nearly
        passes at 96%, 5 and up pass).
    Layer 1 MLP: unit ReLU(tok[op_t] + tok[b_t] - 1) detects each (op,b) pair; then
        a_t = 1 - [&0] - [|1],  c_t = [|1] + [^1].
    Layer 2 MLP: T_j = AND of its inputs via ReLU(sum - (n-1)).
    Layer 3 MLP: S mod 2 = S - 2[S>=2] - 2[S>=4] via 5 ReLU units R_j = ReLU(S - j), j=0..4.
    Head: logit[1] = 1*y, logit[0] = 1*(1-y).
    """
    d, H, dh = model.d_model, model.n_heads, model.blocks[0].attn.d_head
    assert model.vocab_size == 6 and model.n_layers == 3
    assert H >= N_GATHER and dh >= 6 and d == D_MODEL and model.d_ff == D_FF
    assert model.max_seq_len > ANS_POS

    s = math.sqrt(5.0 * math.sqrt(dh))  # s*s/sqrt(dh) = 5: score on the matching key
    blk = model.blocks
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: token one-hot in slots 0..5, position one-hot in POS slots.
        for tok in range(6):
            model.token_emb.weight[tok, TOK_SLOT + tok] = 1.0
        for p in range(N_POS_USED):
            model.pos_emb.weight[p, POS_SLOT + p] = 1.0

        # ---- Layer 1 attention: gather tokens 0..8 at the answer position ----
        attn = blk[0].attn
        for k in range(N_GATHER):
            r = k * dh
            attn.W_q.weight[r, POS_SLOT + ANS_POS] = s   # query: 'I am the answer position'
            attn.W_k.weight[r, POS_SLOT + k] = s         # key: 'I am position k'
            for i in range(6):
                attn.W_v.weight[r + i, TOK_SLOT + i] = 1.0   # value: token one-hot
                attn.W_o.weight[_gslot(k, i), r + i] = 1.0   # write to gathered slot
        # heads N_GATHER..N_HEADS-1 stay zero.

        # ---- Layer 1 MLP: per-step lookup of (op, operand) ----
        mlp1 = blk[0].mlp
        for t in range(1, 5):
            for op, b in LUT_KEYS:
                u = _unit(t, op, b)
                mlp1.fc1.weight[u, _gslot(2 * t - 1, TOKEN_ID[op])] = 1.0  # op at position 2t-1
                mlp1.fc1.weight[u, _gslot(2 * t, TOKEN_ID[str(b)])] = 1.0  # operand at 2t
                mlp1.fc1.bias[u] = -1.0
        mlp1.fc1.weight[CONST_UNIT, POS_SLOT + ANS_POS] = 1.0  # ReLU(1) = 1 at answer position (from position one-hot)
        for t in range(1, 5):
            a, c = A_SLOT + t - 1, C_SLOT + t - 1
            mlp1.fc2.weight[a, CONST_UNIT] = 1.0
            mlp1.fc2.weight[a, _unit(t, "&", 0)] = -1.0   # &0 -> const 0 (kills slope)
            mlp1.fc2.weight[a, _unit(t, "|", 1)] = -1.0   # |1 -> const 1 (kills slope)
            mlp1.fc2.weight[c, _unit(t, "|", 1)] = 1.0    # |1 -> intercept 1
            mlp1.fc2.weight[c, _unit(t, "^", 1)] = 1.0    # ^1 -> NOT (intercept 1)

        # ---- Layer 2 MLP: AND products T_0..T_4 ----
        mlp2 = blk[1].mlp
        A = lambda t: A_SLOT + t - 1
        C = lambda t: C_SLOT + t - 1
        term_inputs = [
            [_gslot(0, TOKEN_ID["1"]), A(1), A(2), A(3), A(4)],  # T0 = b0 a1 a2 a3 a4
            [C(1), A(2), A(3), A(4)],                            # T1 = c1 a2 a3 a4
            [C(2), A(3), A(4)],                                  # T2 = c2 a3 a4
            [C(3), A(4)],                                        # T3 = c3 a4
            [C(4)],                                              # T4 = c4
        ]
        for j, ins in enumerate(term_inputs):
            for i in ins:
                mlp2.fc1.weight[j, i] = 1.0
            mlp2.fc1.bias[j] = -(len(ins) - 1)
            mlp2.fc2.weight[T_SLOT + j, j] = 1.0

        # ---- Layer 3 MLP: S mod 2 = S - 2[S>=2] - 2[S>=4] (floor(S/2) = [S>=2] + [S>=4]) ----
        # R_j = ReLU(S - j). Then S = R_0, [S>=2] = R_1 - R_2, [S>=4] = R_3 - R_4.
        mlp3 = blk[2].mlp
        for j in range(5):
            for i in range(5):
                mlp3.fc1.weight[j, T_SLOT + i] = 1.0
            mlp3.fc1.bias[j] = -float(j)
        mlp3.fc2.weight[Y_SLOT, 0] = 1.0                     # + S
        mlp3.fc2.weight[Y_SLOT, 1] = -2.0                    # - 2 [S>=2]
        mlp3.fc2.weight[Y_SLOT, 2] = 2.0
        mlp3.fc2.weight[Y_SLOT, 3] = -2.0                    # - 2 [S>=4]
        mlp3.fc2.weight[Y_SLOT, 4] = 2.0

        # ---- Output head ----
        model.head.weight[TOKEN_ID["1"], Y_SLOT] = 1.0
        model.head.weight[TOKEN_ID["0"], Y_SLOT] = -1.0
        model.head.weight[TOKEN_ID["0"], POS_SLOT + ANS_POS] = 1.0  # position-based constant for the '0' logit


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ConstFromPos"
model_description = (
    "Floor5Parity circuit with score margin 5, but the constant unit and the '0' logit read the "
    "answer-position one-hot (position 9) instead of the '=' token embedding."
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
    model = SimpleTransformer(
        vocab_size=task.vocab_size, max_seq_len=max_seq_len,
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
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
