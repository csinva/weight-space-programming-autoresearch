"""
Interpretable transformer for character-level sequence tasks (hand-wired, no training).

Task date-normalization: prompt (31 chars padded with '_' then '=') -> "YYYY-MM-DD".
Answer char k is predicted at position 31+k.

Circuit overview (see write_weights):
  L0  prev-char heads; START = digit with no digit before it; month-name detector (3-letter prefix)
  L1  prev-START heads; E1..E3 = offset-within-run; P1 = "some run has started so far"
  L2  prev P1; START2 = run start that has an earlier run; flags for '/', '.', month name
  L3  P2 = at least 2 runs so far
  L4  prev P2; START3
  L5  P3 = at least 3 runs; RUN1..RUN3 = digit belongs to run r; slot queries per output char
  L6  K(r,t) = digit at offset t of run r (prompt only)
  L7  fetch digit A / B for the chosen pair; ReLU logic emits the output character
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
    """Residual block without LayerNorm, so the residual stream stays linear and readable."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """8-layer causal transformer, no LayerNorm, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 42,
        d_model: int = 280,
        n_heads: int = 4,  # d_head = 70
        n_layers: int = 8,
        d_ff: int = 323,
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
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

BIG = 10.0                       # logit gap for "attend to the matching token" heads
MONTH_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
N_PAIRS = 12                     # (run r in 1..3) x (offset t in 0..3); index = (r-1)*4 + t
N_OUT = 10                       # answer length


def _pidx(r: int, t: int) -> int:
    return (r - 1) * 4 + t


def _fterm(f: str) -> dict:
    """Format indicator as a sum of residual slots (value is 0/1)."""
    if f == "N":   # named month present
        return {f"MONV{m}": 1.0 for m in range(12)}
    if f == "S":   # numeric with '/'  (M/D/Y)
        return {"FS": 1.0}
    return {"FD": 1.0}  # numeric with '.' and no month name (Y.M.D)


# (output index k, format, rule, query pair A, query pair B, constant)
# format N = named month (run1 day, run2 year), D = dotted numeric (run1 year, run2 month, run3 day),
# S = slash numeric (run1 month, run2 day, run3 year, 2-digit year -> 20YY)
CASES = []
for _k in range(4):
    CASES.append((_k, "N", "copyA", (2, _k), None, None))
    CASES.append((_k, "D", "copyA", (1, _k), None, None))
CASES += [
    (0, "S", "AifBex", (3, 0), (3, 2), "2"),
    (1, "S", "AifBex", (3, 1), (3, 2), "0"),
    (2, "S", "AifAex", (3, 2), (3, 0), None),
    (3, "S", "AifAex", (3, 3), (3, 1), None),
    (5, "N", "monT", None, None, None),
    (6, "N", "monU", None, None, None),
]
for _f, _mo, _dy in [("D", 2, 3), ("S", 1, 2), ("N", None, 1)]:
    if _mo is not None:
        CASES.append((5, _f, "Bzero", (_mo, 0), (_mo, 1), None))
        CASES.append((6, _f, "Bunit", (_mo, 0), (_mo, 1), None))
    CASES.append((8, _f, "Bzero", (_dy, 0), (_dy, 1), None))
    CASES.append((9, _f, "Bunit", (_dy, 0), (_dy, 1), None))
ABLATE = set()          # ablation switch: format letters whose rules are removed (empty = full circuit)
ABLATE_RULES = set()    # ablation switch: rule names removed
CASES = [c for c in CASES if c[1] not in ABLATE and c[2] not in ABLATE_RULES]


class _Weights:
    """Symbolic residual layout + per-layer attention heads and ReLU units, written into the torch model."""

    def __init__(self, task, model):
        self.stoi = task.stoi
        self.V = model.head.weight.shape[0]
        self.P = model.max_seq_len
        self.D = model.d_model
        self.H = model.n_heads
        self.DH = self.D // self.H
        self.L = len(model.blocks)
        self.DFF = model.blocks[0].mlp.fc1.out_features
        names = [f"CH{c}" for c in range(self.V)] + [f"POS{p}" for p in range(self.P)]
        names += ["ISP", "ONE", "DIG", "P1DIG", "P2DIG", "START", "S1", "S2", "S3",
                  "P1", "P2", "P3", "PP1", "PP2", "START2", "START3",
                  "E1", "E2", "E3", "FDOT", "FS", "FD", "RUN1", "RUN2", "RUN3"]
        names += [f"P1CH{c}" for c in self.letter_ids()] + [f"P2CH{c}" for c in self.letter_ids()]
        names += [f"MON{m}" for m in range(12)] + [f"MONV{m}" for m in range(12)]
        for pre in ["K", "QA", "QB", "AP", "BP"]:
            names += [f"{pre}{p}" for p in range(N_PAIRS)]
        names += [f"AD{d}" for d in range(10)] + [f"BD{d}" for d in range(10)]
        names += [f"LOG{c}" for c in self.out_ids()]
        assert len(names) <= self.D, (len(names), self.D)
        self.ix = {n: i for i, n in enumerate(names)}
        self.heads = [[] for _ in range(self.L)]
        self.units = [[] for _ in range(self.L)]

    # ---- helpers -------------------------------------------------------
    def ch(self, s: str) -> str:
        return f"CH{self.stoi[s]}"

    def out_ids(self):
        return [self.stoi[x] for x in "0123456789-"]

    def letter_ids(self):
        return [self.stoi[x] for x in "abcdefghijklmnopqrstuvwxyz"]

    def dig(self, d: int) -> str:
        return f"CH{self.stoi[str(d)]}"

    def logit_digit(self, d: int) -> str:
        return f"LOG{self.stoi[str(d)]}"

    def unit(self, l, inputs, bias, outs):
        self.units[l].append((dict(inputs), float(bias), dict(outs)))

    def prev_head(self, shift, pairs):
        """Attend exactly to position i-shift; copy the listed source slots."""
        q = {d: {f"POS{d}": BIG} for d in range(self.P)}
        k = {d: {f"POS{d - shift}": 1.0} for d in range(shift, self.P)}
        v = {j: {src: 1.0} for j, (src, _) in enumerate(pairs)}
        o = {out: {j: 1.0} for j, (_, out) in enumerate(pairs)}
        return dict(q=q, k=k, v=v, o=o)

    def agg_head(self, key, pairs):
        """Attend (softmax) to all causal positions whose key is on; output = OR / one-hot pick."""
        q = {0: {"ONE": BIG}}
        k = {0: dict(key)}
        v = {j: {src: 1.0} for j, (src, _) in enumerate(pairs)}
        o = {out: {j: 1.0} for j, (_, out) in enumerate(pairs)}
        return dict(q=q, k=k, v=v, o=o)

    def fetch_head(self, qpre, dpre, ppre):
        """Query = one-hot run/offset pair; return the matched digit and the pair one-hot (existence)."""
        q = {p: {f"{qpre}{p}": BIG} for p in range(N_PAIRS)}
        k = {p: {f"K{p}": 1.0} for p in range(N_PAIRS)}
        v = {d: {self.dig(d): 1.0} for d in range(10)}
        v.update({10 + p: {f"K{p}": 1.0} for p in range(N_PAIRS)})
        o = {f"{dpre}{d}": {d: 1.0} for d in range(10)}
        o.update({f"{ppre}{p}": {10 + p: 1.0} for p in range(N_PAIRS)})
        return dict(q=q, k=k, v=v, o=o)

    # ---- the circuit ---------------------------------------------------
    def build(self):
        V = self.V
        # L0: previous-char heads; START; month-name detector
        self.heads[0] = [
            self.prev_head(1, [(f"CH{c}", f"P1CH{c}") for c in self.letter_ids()] + [("DIG", "P1DIG")]),
            self.prev_head(2, [(f"CH{c}", f"P2CH{c}") for c in self.letter_ids()] + [("DIG", "P2DIG")]),
        ]
        self.unit(0, {"DIG": 1, "P1DIG": -1, "POS0": 1}, 0, {"START": 1})
        self.unit(0, {"POS0": 1, "DIG": -1}, 0, {"START": -1})
        for m, pre in enumerate(MONTH_ABBR):
            n1, n2, n3 = (self.stoi[c] for c in pre)
            self.unit(0, {f"CH{n3}": 1, f"P1CH{n2}": 1, f"P2CH{n1}": 1},
                      -2, {f"MON{m}": 1})

        # L1: earlier-START features; offset-in-run; P1 = some run has started
        self.heads[1] = [
            self.prev_head(1, [("START", "S1")]),
            self.prev_head(2, [("START", "S2")]),
            self.prev_head(3, [("START", "S3")]),
            self.agg_head({"START": 1}, [("START", "P1")]),
        ]
        self.unit(1, {"DIG": 1, "S1": 1, "POS0": -1}, -1, {"E1": 1})
        self.unit(1, {"DIG": 1, "P1DIG": 1, "S2": 1, "POS0": -1, "POS1": -1}, -2, {"E2": 1})
        self.unit(1, {"DIG": 1, "P1DIG": 1, "P2DIG": 1, "S3": 1, "POS0": -1, "POS1": -1, "POS2": -1},
                  -3, {"E3": 1})

        # L2: START2; month value; format flags
        self.heads[2] = [
            self.prev_head(1, [("P1", "PP1")]),
            self.agg_head({f"MON{m}": 1 for m in range(12)}, [(f"MON{m}", f"MONV{m}") for m in range(12)]),
            self.agg_head({self.ch("/"): 1}, [(self.ch("/"), "FS")]),
            self.agg_head({self.ch("."): 1}, [(self.ch("."), "FDOT")]),
        ]
        self.unit(2, {"START": 1, "PP1": 1, "POS0": -1}, -1, {"START2": 1})
        fd_in = {"FDOT": 1.0}
        fd_in.update({f"MONV{m}": -1.0 for m in range(12)})
        self.unit(2, fd_in, 0, {"FD": 1})

        # L3: P2 = at least two runs so far
        self.heads[3] = [self.agg_head({"START2": 1}, [("START2", "P2")])]

        # L4: START3
        self.heads[4] = [self.prev_head(1, [("P2", "PP2")])]
        self.unit(4, {"START": 1, "PP2": 1, "POS0": -1}, -1, {"START3": 1})

        # L5: P3, RUN1..3; slot queries (which run/offset each output char needs)
        self.heads[5] = [self.agg_head({"START3": 1}, [("START3", "P3")])]
        self.unit(5, {"DIG": 1, "P1": 1, "P2": -1}, -1, {"RUN1": 1})
        self.unit(5, {"DIG": 1, "P2": 1, "P3": -1}, -1, {"RUN2": 1})
        self.unit(5, {"DIG": 1, "P3": 1}, -1, {"RUN3": 1})
        for (k, f, rule, qA, qB, _c) in CASES:
            if qA is not None:
                self.unit(5, {f"POS{31 + k}": 1, **_fterm(f)}, -1, {f"QA{_pidx(*qA)}": 1})
            if qB is not None:
                self.unit(5, {f"POS{31 + k}": 1, **_fterm(f)}, -1, {f"QB{_pidx(*qB)}": 1})

        # L6: K(r,t) for prompt digits
        E = ["START", "E1", "E2", "E3"]
        for r in (1, 2, 3):
            for t in range(4):
                self.unit(6, {f"RUN{r}": 1, E[t]: 1, "ISP": 1}, -2, {f"K{_pidx(r, t)}": 1})

        # L7: fetch A/B digits, then emit logits
        self.heads[7] = [
            self.fetch_head("QA", "AD", "AP"),
            self.fetch_head("QB", "BD", "BP"),
        ]
        for k in (4, 7):
            self.unit(7, {f"POS{31 + k}": 1}, 0, {f"LOG{self.stoi['-']}": 1})
        for (k, f, rule, qA, qB, const) in CASES:
            P = {f"POS{31 + k}": 1.0}
            Fd = _fterm(f)
            if rule == "copyA":
                for d in range(10):
                    self.unit(7, {**P, **Fd, f"AD{d}": 1}, -2, {self.logit_digit(d): 1})
            elif rule == "AifBex":
                pb = _pidx(*qB)
                for d in range(10):
                    self.unit(7, {**P, **Fd, f"BP{pb}": 1, f"AD{d}": 1}, -3, {self.logit_digit(d): 1})
                self.unit(7, {**P, **Fd, f"BP{pb}": -1}, -1, {self.logit_digit(int(const)): 1})
            elif rule == "AifAex":
                pa = _pidx(*qA)
                for d in range(10):
                    self.unit(7, {**P, **Fd, f"AP{pa}": 1, f"AD{d}": 1}, -3, {self.logit_digit(d): 1})
                    self.unit(7, {**P, **Fd, f"AP{pa}": -1, f"BD{d}": 1}, -2, {self.logit_digit(d): 1})
            elif rule == "Bzero":
                pb = _pidx(*qB)
                for d in range(10):
                    self.unit(7, {**P, **Fd, f"BP{pb}": 1, f"AD{d}": 1}, -3, {self.logit_digit(d): 1})
                self.unit(7, {**P, **Fd, f"BP{pb}": -1}, -1, {self.logit_digit(0): 1})
            elif rule == "Bunit":
                pb = _pidx(*qB)
                for d in range(10):
                    self.unit(7, {**P, **Fd, f"BP{pb}": 1, f"BD{d}": 1}, -3, {self.logit_digit(d): 1})
                    self.unit(7, {**P, **Fd, f"BP{pb}": -1, f"AD{d}": 1}, -2, {self.logit_digit(d): 1})
            elif rule == "monT":
                for m in range(12):
                    self.unit(7, {"POS36": 1, f"MONV{m}": 1}, -1,
                              {self.logit_digit(1 if m >= 9 else 0): 1})
            elif rule == "monU":
                for m in range(12):
                    self.unit(7, {"POS37": 1, f"MONV{m}": 1}, -1, {self.logit_digit((m + 1) % 10): 1})

    # ---- write into torch ---------------------------------------------
    def apply(self, model):
        ix = self.ix
        sq = math.sqrt(self.DH)
        with torch.no_grad():
            for prm in model.parameters():
                prm.zero_()
            for c in range(self.V):
                model.token_emb.weight[c, ix[f"CH{c}"]] = 1.0
                model.token_emb.weight[c, ix["ONE"]] = 1.0
            for d in range(10):
                model.token_emb.weight[self.stoi[str(d)], ix["DIG"]] = 1.0
            for p in range(self.P):
                model.pos_emb.weight[p, ix[f"POS{p}"]] = 1.0
                if p <= 31:
                    model.pos_emb.weight[p, ix["ISP"]] = 1.0
            for c in self.out_ids():
                model.head.weight[c, ix[f"LOG{c}"]] = 1.0
            for l, blk in enumerate(model.blocks):
                assert len(self.heads[l]) <= self.H
                att = blk.attn
                for h, sp in enumerate(self.heads[l]):
                    base = h * self.DH
                    for d, terms in sp["q"].items():
                        for n, c in terms.items():
                            att.W_q.weight[base + d, ix[n]] += c * sq
                    for d, terms in sp["k"].items():
                        for n, c in terms.items():
                            att.W_k.weight[base + d, ix[n]] += c
                    for d, terms in sp["v"].items():
                        for n, c in terms.items():
                            att.W_v.weight[base + d, ix[n]] += c
                    for out, terms in sp["o"].items():
                        for d, c in terms.items():
                            att.W_o.weight[ix[out], base + d] += c
                assert len(self.units[l]) <= self.DFF, (l, len(self.units[l]))
                for u, (inp, bias, outs) in enumerate(self.units[l]):
                    for n, c in inp.items():
                        blk.mlp.fc1.weight[u, ix[n]] += c
                    blk.mlp.fc1.bias[u] = bias
                    for n, c in outs.items():
                        blk.mlp.fc2.weight[ix[n], u] += c


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place by hand. No training."""
    spec = _Weights(task, model)
    spec.build()
    spec.apply(model)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "HandWiredDateCircuit-logitSlim"
model_description = ("8-layer hand-wired circuit, width 280, d_ff 323: run-start/offset detection, "
                     "run-count OR chains, slot queries fetch digits; logits only for output chars.")


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
