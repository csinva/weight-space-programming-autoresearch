"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) 2-layer transformer for `date-normalization`.
Snapshot rotcopy_tags_v2_sharp40 (200/200 on seed 0).

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

# Named residual-stream channels.
CHANNELS = [
    "ONE", "DV", "ISD", "LC", "LS", "DOT", "SL", "PAD", "HY", "EQ", "S",   # token features
    "PC", "PS",                                                           # position circle
    "DV1", "ISD1", "LC1", "LS1",                                          # copied from 1 back
    "DV2", "ISD2", "LC2", "LS2",                                          # copied from 2 back
    "DV3", "ISD3", "LC3", "LS3",                                          # copied from 3 back
    "TY", "TM", "TD",                                                     # field tags
    "B31", "B33", "B34", "B36", "B37", "B39", "B40", "HYP",               # position selectors
]
C = {n: i for i, n in enumerate(CHANNELS)}
D_MODEL = len(CHANNELS)

PERIOD = 48
STEP = 17          # position p sits at angle 2*pi*STEP*p/48 on the period-48 circle
D_FF = 30


class MultiHeadAttention(nn.Module):
    """Causal attention with per-head projections (no LayerNorm anywhere)."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_v: int):
        super().__init__()
        self.n_heads, self.d_head, self.d_v = n_heads, d_head, d_v
        self.W_q = nn.Parameter(torch.zeros(n_heads, d_model, d_head))
        self.W_k = nn.Parameter(torch.zeros(n_heads, d_model, d_head))
        self.W_v = nn.Parameter(torch.zeros(n_heads, d_model, d_v))
        self.W_o = nn.Parameter(torch.zeros(n_heads, d_v, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = torch.einsum("btd,hde->bhte", x, self.W_q)
        k = torch.einsum("btd,hde->bhte", x, self.W_k)
        v = torch.einsum("btd,hde->bhte", x, self.W_v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = attn @ v                                       # (B,H,T,dv)
        return torch.einsum("bhte,hed->btd", out, self.W_o)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    """Layer 1: 3 positional copy heads + ReLU MLP. Layer 2: 3 heads, no MLP. No LayerNorm."""

    def __init__(self, vocab_size: int, max_seq_len: int = 42, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = D_MODEL
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        self.attn1 = MultiHeadAttention(D_MODEL, 3, 2, 4)
        self.mlp1 = MLP(D_MODEL, D_FF)
        self.attn2 = MultiHeadAttention(D_MODEL, 3, 4, 1)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        h = h + self.attn1(h)
        h = h + self.mlp1(h)
        h = h + self.attn2(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

def _theta(p: float) -> float:
    return 2 * math.pi * STEP * p / PERIOD


def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        stoi = task.stoi

        # ---------------- embeddings ----------------
        E = model.token_emb.weight
        for ch, i in stoi.items():
            E[i, C["ONE"]] = 1.0
            if ch.isdigit():
                E[i, C["DV"]] = float(ch)
                E[i, C["ISD"]] = 1.0
            elif ch.isalpha():
                a = 2 * math.pi * (ord(ch) - ord("a")) / 26
                E[i, C["LC"]] = math.cos(a)
                E[i, C["LS"]] = math.sin(a)
        E[stoi["."], C["DOT"]] = 1.0
        E[stoi["/"], C["SL"]] = 1.0
        E[stoi["_"], C["PAD"]] = 1.0
        E[stoi["-"], C["HY"]] = 1.0
        E[stoi["="], C["EQ"]] = 1.0
        E[stoi["="], C["S"]] = 2.0           # constant "2" of the first year digit

        P = model.pos_emb.weight
        for p in range(model.max_seq_len):
            P[p, C["PC"]] = math.cos(_theta(p))
            P[p, C["PS"]] = math.sin(_theta(p))

        # ---------------- layer 1 attention: copy from k back ----------------
        S1 = 3500.0 * math.sqrt(2)
        A1 = model.attn1
        feats = ["DV", "ISD", "LC", "LS"]
        for h, kback in enumerate([1, 2, 3]):
            phi = STEP * kback * 2 * math.pi / PERIOD
            # query = S1 * rotate(pos, -k steps); key = pos
            A1.W_q[h, C["PC"], 0] = S1 * math.cos(phi)
            A1.W_q[h, C["PS"], 0] = S1 * math.sin(phi)
            A1.W_q[h, C["PC"], 1] = -S1 * math.sin(phi)
            A1.W_q[h, C["PS"], 1] = S1 * math.cos(phi)
            A1.W_k[h, C["PC"], 0] = 1.0
            A1.W_k[h, C["PS"], 1] = 1.0
            for e, f in enumerate(feats):
                A1.W_v[h, C[f], e] = 1.0
                A1.W_o[h, e, C[f + str(kback)]] = 1.0

        # ---------------- layer 1 MLP ----------------
        W1, b1, W2 = model.mlp1.fc1.weight, model.mlp1.fc1.bias, model.mlp1.fc2.weight
        u = [0]

        def unit(inputs: dict, bias: float, outputs: dict):
            j = u[0]; u[0] += 1
            for n, w in inputs.items():
                W1[j, C[n]] += w
            b1[j] = bias
            for n, w in outputs.items():
                W2[C[n], j] += w

        def at_pos(p: int, K: float) -> tuple[dict, float]:
            """K*(cos(theta_i - theta_p) - 1): 0 at position p, <= -K*0.0086 elsewhere."""
            return {"PC": K * math.cos(_theta(p)), "PS": K * math.sin(_theta(p))}, -K

        def merge(*ds):
            out = {}
            for d in ds:
                for k, v in d.items():
                    out[k] = out.get(k, 0.0) + v
            return out

        KP = 200.0
        # token ending a digit run: default DAY
        unit({"ISD1": 1, "ISD": -1}, 0.0, {"TD": 1})
        # padding after digits: half YEAR, half DAY
        unit({"PAD": 1, "ISD1": 1}, -1.0, {"TY": 0.5, "TD": -0.5})
        # dot after a digit: MONTH
        unit({"DOT": 1, "ISD1": 1}, -1.0, {"TM": 1, "TD": -1})
        # dot (after digit) at position 4: YEAR
        pw, pb = at_pos(4, KP)
        unit(merge({"DOT": 1, "ISD1": 1}, pw), -1.0 + pb, {"TY": 1, "TM": -1})
        # slash at position <= 2: MONTH
        for p in (1, 2):
            pw, pb = at_pos(p, KP)
            unit(merge({"SL": 1, "ISD1": 1}, pw), -1.0 + pb, {"TM": 1, "TD": -1})
        # answer hyphens (positions 35+1, 38+1 of the stream) are not date fields
        for p in (36, 39):
            pw, pb = at_pos(p, KP)
            unit(merge({"HY": 1, "ISD1": 1}, pw), -1.0 + pb, {"TD": -1})
        # month-name units: letters 2-3 of each month
        eps = 0.5 * (1 - math.cos(2 * math.pi / 26))
        for m, name in enumerate(src.task.DateNormalizationTask.MONTHS, start=1):
            l2, l3 = name[1], name[2]
            a2 = 2 * math.pi * (ord(l2) - ord("a")) / 26
            a3 = 2 * math.pi * (ord(l3) - ord("a")) / 26
            inputs = {"LC": math.cos(a3) / eps, "LS": math.sin(a3) / eps,
                      "LC1": math.cos(a2) / eps, "LS1": math.sin(a2) / eps}
            tag = 0.5 if name == "march" else 1.0
            unit(inputs, -(2 - eps) / eps, {"TM": tag, "DV2": m // 10, "DV1": m % 10})
        # second year digit: ReLU(18 - 9*first), first = token at position 32
        pw, pb = at_pos(32, 2500.0)
        unit(merge({"DV": -9.0}, pw), 18.0 + pb, {"S": 1.0})
        # position bumps for layer-2 field selection and hyphens
        c0 = math.cos(math.pi / PERIOD)          # half the minimum angular gap
        Kb = 1.0 / (1.0 - c0)
        for p, out in [(31, "B31"), (33, "B33"), (34, "B34"), (36, "B36"), (37, "B37"),
                       (39, "B39"), (40, "B40"), (35, "HYP"), (38, "HYP")]:
            unit({"PC": Kb * math.cos(_theta(p)), "PS": Kb * math.sin(_theta(p))}, -Kb * c0, {out: 1.0})
        assert u[0] == D_FF, u[0]

        # ---------------- layer 2 attention ----------------
        A2 = model.attn2
        S2 = 20.0 * math.sqrt(4) * math.sqrt(4)   # effective logit 20*sqrt(d_head) per unit tag
        # heads: tens (DV2), ones (DV1), year-high (-DV3/9)
        sels = [("B33", "B36", "B39"), ("B34", "B37", "B40"), ("B31", None, None)]
        vals = [{"DV2": 1.0}, {"DV1": 1.0}, {"DV3": -1.0 / 9}]
        for h in range(3):
            for e, (sel, tag) in enumerate(zip(sels[h], ["TY", "TM", "TD"])):
                if sel is not None:
                    A2.W_q[h, C[sel], e] = S2
                A2.W_k[h, C[tag], e] = 1.0
            A2.W_q[h, C["ONE"], 3] = 0.25 * S2       # fallback to "=" (zeros)
            A2.W_k[h, C["EQ"], 3] = 1.0
            for n, w in vals[h].items():
                A2.W_v[h, C[n], 0] = w
            A2.W_o[h, 0, C["S"]] = 1.0

        # ---------------- readout: digit nearest scalar S ----------------
        U = model.head.weight
        U[:, C["ONE"]] = -100.0
        for k in range(10):
            i = stoi[str(k)]
            U[i, C["ONE"]] = -float(k * k)
            U[i, C["S"]] = 2.0 * k
        U[stoi["-"], C["ONE"]] = -10.0
        U[stoi["-"], C["HYP"]] = 20.0


model_shorthand_name = "rotcopy_tags_v2_sharp40"
model_description = ("2-layer no-LN transformer: L1 rotary-style copy heads (1/2/3 back, period-48 circle, step 17) "
                     "+ ReLU AND-gate tags (DAY default, dot@4 YEAR, slash<=2/dot MONTH, pad half Y/D, month letter units); "
                     "L2 tens/ones/high heads attend sharply to field tag or '='; nearest-digit readout.")


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
