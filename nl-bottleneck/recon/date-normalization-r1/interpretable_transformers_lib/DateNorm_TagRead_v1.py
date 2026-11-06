"""
Interpretable transformer for character-level sequence tasks.

Hand-written (closed-form) 2-layer transformer for `date-normalization`.

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py --task date-normalization
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
    """Multi-head causal attention with separate (small) q/k and v head dims."""

    def __init__(self, d_model: int, n_heads: int, d_qk: int, d_v: int):
        super().__init__()
        self.n_heads, self.d_qk, self.d_v = n_heads, d_qk, d_v
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, self.d_qk).transpose(1, 2)
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
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# residual-stream layout
C, DV, DG, LC, LS, DOT, SL, PAD, EQ = range(9)          # token features
PC, PS, P01 = 9, 10, 11                                  # position circle (+ pos0/pos1 marker)
DV1, DG1, LC1, LS1, DV2, DV3 = 12, 13, 14, 15, 16, 17    # layer-1 attention copies
TY, TM, TD, S, HY = 18, 19, 20, 21, 22                   # tags, answer scalar, hyphen flag
QH, QTY, QOY, QTM, QOM, QTD, QOD = 23, 24, 25, 26, 27, 28, 29  # query-field flags
D_MODEL = 30
D_FF = 28


class SimpleTransformer(nn.Module):
    """2 layers, 3 heads each, no LayerNorm. Layer 1: attn + MLP; layer 2: attn only."""

    def __init__(self, vocab_size: int, max_seq_len: int = 42, d_model: int = D_MODEL):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.attn1 = CausalSelfAttention(d_model, 3, d_qk=3, d_v=4)
        self.mlp1 = MLP(d_model, D_FF)
        self.attn2 = CausalSelfAttention(d_model, 3, d_qk=4, d_v=1)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        h = h + self.attn1(h)
        h = h + self.mlp1(h)
        h = h + self.attn2(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

PERIOD = 48
MONTH_KEYS = ["an", "eb", "ar", "pr", "ay", "un", "ul", "ug", "ep", "ct", "ov", "ec"]  # letters 2-3


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    st = task.stoi
    al = 2 * math.pi / PERIOD
    ang = lambda p: al * p

    # ---- embeddings -------------------------------------------------------
    E = model.token_emb.weight
    for ch, i in st.items():
        E[i, C] = 1.0
        if ch.isdigit():
            E[i, DV] = float(ch)
            E[i, DG] = 1.0
        elif ch.isalpha():
            a = 2 * math.pi * (ord(ch) - ord("a")) / 26
            E[i, LC], E[i, LS] = math.cos(a), math.sin(a)
    E[st["."], DOT] = 1.0
    E[st["/"], SL] = 1.0
    E[st["_"], PAD] = 1.0
    E[st["="], EQ] = 1.0
    P = model.pos_emb.weight
    for p in range(model.max_seq_len):
        P[p, PC], P[p, PS] = math.cos(ang(p)), math.sin(ang(p))
    P[0, P01], P[1, P01] = 1.0, -1.0

    # ---- layer-1 attention: rotary-style k-back heads -------------------------
    a1 = model.attn1
    SC = 2000.0 * math.sqrt(a1.d_qk)
    copies = {1: [(DV, DV1), (DG, DG1), (LC, LC1), (LS, LS1)], 2: [(DV, DV2)], 3: [(DV, DV3)]}
    for h, kb in enumerate([1, 2, 3]):
        c, s = math.cos(kb * al), math.sin(kb * al)
        r = h * a1.d_qk
        # q = R(-k*alpha) (cos th_p, sin th_p)
        a1.W_q.weight[r + 0, PC], a1.W_q.weight[r + 0, PS] = SC * c, SC * s
        a1.W_q.weight[r + 1, PC], a1.W_q.weight[r + 1, PS] = -SC * s, SC * c
        a1.W_k.weight[r + 0, PC] = 1.0
        a1.W_k.weight[r + 1, PS] = 1.0
        if kb == 2:  # at position 1 the 2-back head reads itself
            a1.W_q.weight[r + 2, P01] = 100.0 * math.sqrt(a1.d_qk)
            a1.W_k.weight[r + 2, P01] = 1.0
        for j, (src_d, dst_d) in enumerate(copies[kb]):
            a1.W_v.weight[h * a1.d_v + j, src_d] = 1.0
            a1.W_o.weight[dst_d, h * a1.d_v + j] = 1.0

    # ---- layer-1 MLP ------------------------------------------------------
    W1, b1, W2 = model.mlp1.fc1.weight, model.mlp1.fc1.bias, model.mlp1.fc2.weight
    u = iter(range(D_FF))

    def bump(center, width_neigh):
        """ReLU(K(cos(th_p - th_c) - c)): 1 at target position(s), -1 at the neighbours."""
        cin, cout = width_neigh
        K = 2.0 / (cin - cout)
        cc = (cin + cout) / 2
        return K * math.cos(ang(center)), K * math.sin(ang(center)), -K * cc, K * (cin - cc)

    single = (1.0, math.cos(al))
    flags = {}
    for t in range(31, 41):
        i = next(u)
        kc, ks, kb_, _ = bump(t, single)
        W1[i, PC], W1[i, PS], b1[i] = kc, ks, kb_
        flags[t] = i
    W2[S, flags[31]] = 2.0          # first year digit: 2 - (digit 3 back)/9
    W2[QH, flags[31]] = 1.0
    W2[QTY, flags[33]] = 1.0
    W2[QOY, flags[34]] = 1.0
    W2[HY, flags[35]] = 1.0
    W2[QTM, flags[36]] = 1.0
    W2[QOM, flags[37]] = 1.0
    W2[HY, flags[38]] = 1.0
    W2[QTD, flags[39]] = 1.0
    W2[QOD, flags[40]] = 1.0
    W2[TD, flags[36]] = -1.0        # generated hyphens are not DAY separators
    W2[TD, flags[39]] = -1.0

    # second year digit: ReLU(18 - 9*first) at position 32
    i = next(u); G = 2500.0
    W1[i, DV] = -9.0
    W1[i, PC], W1[i, PS] = G * math.cos(ang(32)), G * math.sin(ang(32))
    b1[i] = 18.0 - G
    W2[S, i] = 1.0

    # run end (non-digit after digit)
    RE = next(u); W1[RE, DG1] = 1.0; W1[RE, DG] = -1.0
    DOTE = next(u); W1[DOTE, DG1] = 1.0; W1[DOTE, DOT] = 1.0; b1[DOTE] = -1.0
    PADE = next(u); W1[PADE, DG1] = 1.0; W1[PADE, PAD] = 1.0; b1[PADE] = -1.0
    M = 10.0
    Y4 = next(u)
    kc, ks, kb_, _ = bump(4, single)
    W1[Y4, DG1] = M; W1[Y4, DOT] = M; W1[Y4, PC] = kc; W1[Y4, PS] = ks; b1[Y4] = kb_ - 2 * M
    SLL = next(u)
    kc, ks, kb_, _ = bump(1.5, (math.cos(0.5 * al), math.cos(1.5 * al)))
    W1[SLL, DG1] = M; W1[SLL, SL] = M; W1[SLL, PC] = kc; W1[SLL, PS] = ks; b1[SLL] = kb_ - 2 * M

    W2[TY, Y4] = 1.0; W2[TY, PADE] = 0.5
    W2[TM, DOTE] = 1.0; W2[TM, Y4] = -1.0; W2[TM, SLL] = 1.0
    W2[TD, RE] = 1.0; W2[TD, DOTE] = -1.0; W2[TD, SLL] = -1.0; W2[TD, PADE] = -0.5

    # month units: match letters 2-3 (previous letter, current letter)
    eps = (1 - math.cos(2 * math.pi / 26)) / 2
    A = 1.0 / eps
    for m, key in enumerate(MONTH_KEYS, start=1):
        i = next(u)
        a2 = 2 * math.pi * (ord(key[0]) - 97) / 26
        a3 = 2 * math.pi * (ord(key[1]) - 97) / 26
        W1[i, LC1], W1[i, LS1] = A * math.cos(a2), A * math.sin(a2)
        W1[i, LC], W1[i, LS] = A * math.cos(a3), A * math.sin(a3)
        b1[i] = -A * (2 - eps)
        W2[TM, i] = 0.5 if key == "ar" else 1.0
        W2[DV2, i] = float(m // 10)
        W2[DV1, i] = float(m % 10)

    # ---- layer-2 attention: field readers --------------------------------
    a2_ = model.attn2
    L = 20.0 * math.sqrt(a2_.d_qk)
    beta = 0.3
    heads = [  # (query flag -> tag) pairs, value source, value scale
        ([(QH, TY)], DV3, -1.0 / 9.0),
        ([(QTY, TY), (QTM, TM), (QTD, TD)], DV2, 1.0),
        ([(QOY, TY), (QOM, TM), (QOD, TD)], DV1, 1.0),
    ]
    tag_slot = {TY: 0, TM: 1, TD: 2}
    for h, (pairs, vsrc, vs) in enumerate(heads):
        r = h * a2_.d_qk
        for qf, tg in pairs:
            a2_.W_q.weight[r + tag_slot[tg], qf] = L
            a2_.W_k.weight[r + tag_slot[tg], tg] = 1.0
        a2_.W_q.weight[r + 3, C] = L * beta     # fallback: '='
        a2_.W_k.weight[r + 3, EQ] = 1.0
        a2_.W_v.weight[h, vsrc] = vs
        a2_.W_o.weight[S, h] = 1.0

    # ---- readout: digit nearest the scalar S; hyphen from position ----------
    Wh = model.head.weight
    for ch, i in st.items():
        if ch.isdigit():
            d = float(ch)
            Wh[i, S] = 2 * d
            Wh[i, C] = -d * d
        else:
            Wh[i, C] = -100.0
    Wh[st["-"], HY] = 1000.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "DateNorm_TagRead_v1"
model_description = ("2-layer no-LN transformer: rotary k-back copy heads, MLP AND-gate field tags "
                     "(DAY default, dot@4 YEAR, slash<=2/dot MONTH, pad half YEAR/DAY, month letter-pair units, "
                     "march half), layer-2 sharp tag readers for hundreds/tens/ones, nearest-digit readout.")


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
