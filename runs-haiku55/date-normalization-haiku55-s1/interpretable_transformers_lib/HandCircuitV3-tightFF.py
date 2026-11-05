"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. Task: `date-normalization`.
Prompt: "<date text, left-aligned, padded with '_' to 31 chars>=" (32 tokens)
Answer: "YYYY-MM-DD" (10 tokens), greedy autoregressive decoding.

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Circuit (3 layers, no LayerNorm, all weights set in closed form):
  L1 attn: four "previous-token" gathers (offsets 1..4) with an attention sink.
     MLP1: digit-run ends (run of length 1/2/4 terminated by a non-digit,
           prompt only) and month-name trigram detectors.
  L2 attn: uniform count of short (1-2 digit) runs so far.
     MLP2: thresholded one-hot count (run ordinal) and canonical digit values
           of the 4-digit run (LD) and of the short runs (SDT/SDU).
  L3 attn: retrieval of the long (4-digit) run, the short runs with ordinal
           1/2/3, and the month name.
     MLP3: output logits gated by output step (position) and by format
           (named-month vs numeric).

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

DH = 43            # per-head dimension (>= vocab size 43)
BIG = 30.0         # attention score for a match (post-scaling units)
SINK = 15.0        # attention-sink logit: wins when nothing matches
GE_SCALE = 1000.0  # sharpness for thresholded counts
SQ = math.sqrt(DH)


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Causal attention with an optional per-head sink logit.

    The sink is an extra softmax entry with a constant logit and zero value.
    It absorbs attention when no key matches, so a "gather" head returns
    exactly zero instead of a uniform average. A very negative sink logit
    disables it, giving plain uniform averaging over the causal prefix.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)
        self.sink = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        sink = self.sink.view(1, H, 1, 1).expand(B, H, T, 1)
        attn = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :-1]
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """Pre-residual block without LayerNorm (so the residual slots stay exact)."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# (n_heads, d_ff) per layer
LAYER_SPEC = [(4, 15), (1, 69), (5, 136)]


class SimpleTransformer(nn.Module):
    """3-layer causal transformer. The residual stream is split into named slots (see make_layout)."""

    def __init__(self, vocab_size: int, max_seq_len: int = 42, d_model: int = 496):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, h, DH, ff) for h, ff in LAYER_SPEC])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Residual-stream layout (named slots)
# ---------------------------------------------------------------------------

def make_layout(V: int) -> dict:
    """Slots in the residual stream. Each slot is written by exactly one component."""
    names = [
        ("TOK", V),        # one-hot of the current token
        ("POS", 42),       # one-hot of the position
        ("ONE", 1),        # constant 1
        ("PROMPT", 1),     # 1 on prompt positions (0..31), 0 on answer positions
        ("PT1", V), ("PT2", V), ("PT3", V), ("PT4", V),   # one-hot of token at t-1..t-4
        ("MON", 12),       # month-name trigram ends here (month 1..12)
        ("E1", 1), ("E2", 1), ("E4", 1),                  # digit run of len 1/2/4 ends at t-1
        ("CNT", 1),        # mean of short-run flags over prefix (= count/(t+1))
        ("GE", 4),         # thresholds: GE_c = [count >= c]
        ("LD", 40),        # 4-digit run digits, MSB first: LD[j*10+d]
        ("SDT", 10), ("SDU", 10),                         # short run digits (tens, units), zero-padded
        ("RLD", 40), ("RONE", 1),                         # retrieved long run + existence
        ("RS1T", 10), ("RS1U", 10),                       # retrieved short run ordinal 1
        ("RS2T", 10), ("RS2U", 10),                       # retrieved short run ordinal 2
        ("RS3T", 10), ("RS3U", 10),                       # retrieved short run ordinal 3
        ("RNAME", 12), ("RHAS", 1),                       # retrieved month name, existence
        ("OUT", V),        # output logits (one-hot target char)
    ]
    L, off = {}, 0
    for n, s in names:
        L[n] = off
        off += s
    L["_size"] = off
    return L


def _set_head(attn: CausalSelfAttention, h: int, q=(), k=(), v=(), o=(), sink: float = SINK):
    """q/k/v: (dim_in_head, src_slot, weight); o: (dst_slot, dim_in_head, weight)."""
    with torch.no_grad():
        for i, s, w in q:
            attn.W_q.weight[h * DH + i, s] = w
        for i, s, w in k:
            attn.W_k.weight[h * DH + i, s] = w
        for i, s, w in v:
            attn.W_v.weight[h * DH + i, s] = w
        for d, i, w in o:
            attn.W_o.weight[d, h * DH + i] += w
        attn.sink[h] = sink


# Neuron literal helpers. Each literal returns (terms, const) with value = const + sum(w * x).
def _p(s):
    return [(s, 1.0)], 0.0          # x


def _n(s):
    return [(s, -1.0)], 1.0         # 1 - x


def _g(ss):
    return [(s, 1.0) for s in ss], 0.0     # sum of exclusive 0/1 slots


def _ng(ss):
    return [(s, -1.0) for s in ss], 1.0    # 1 - sum


def _and(lits, outs):
    """Neuron firing (output 1 per out-weight) iff every literal is 1."""
    terms, const = [], 0.0
    for t, c in lits:
        terms.extend(t)
        const += c
    bias = const - (len(lits) - 0.5)
    return terms, bias, [(d, 2.0 * w) for d, w in outs]


def _set_mlp(mlp: MLP, neurons):
    assert len(neurons) <= mlp.fc1.out_features, (len(neurons), mlp.fc1.out_features)
    with torch.no_grad():
        for n, (terms, bias, outs) in enumerate(neurons):
            for s, w in terms:
                mlp.fc1.weight[n, s] += w
            mlp.fc1.bias[n] = bias
            for d, w in outs:
                mlp.fc2.weight[d, n] += w


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed."""
    V = task.vocab_size
    L = make_layout(V)
    stoi = task.stoi
    P = task.prompt_len                 # 32: index of '=' is P-1
    A = task.answer_len                 # 10
    dig = [stoi[str(d)] for d in range(10)]
    DASH = stoi["-"]

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # --- embeddings and output head -----------------------------------
        for c in range(V):
            model.token_emb.weight[c, L["TOK"] + c] = 1.0
        for t in range(model.pos_emb.weight.shape[0]):
            model.pos_emb.weight[t, L["POS"] + t] = 1.0
            model.pos_emb.weight[t, L["ONE"]] = 1.0
            if t < P:
                model.pos_emb.weight[t, L["PROMPT"]] = 1.0
        for c in range(V):
            model.head.weight[c, L["OUT"] + c] = 1.0

        # --- Layer 1: previous-token gathers -------------------------------
        a1 = model.blocks[0].attn
        for o in range(1, 5):
            _set_head(
                a1, o - 1,
                q=[(j, L["POS"] + j + o, BIG * SQ) for j in range(42 - o)],
                k=[(j, L["POS"] + j, 1.0) for j in range(42)],
                v=[(i, L["TOK"] + i, 1.0) for i in range(V)],
                o=[(L[f"PT{o}"] + i, i, 1.0) for i in range(V)],
                sink=SINK,
            )

        D = {o: _g([L[f"PT{o}"] + d for d in dig]) for o in range(1, 5)}
        isd = [L["TOK"] + d for d in dig]
        pt = {o: [L[f"PT{o}"] + d for d in dig] for o in range(1, 5)}
        PR = _p(L["PROMPT"])
        n1 = []
        # digit run of length 1 / 2 / 4 ending at t-1, with t a non-digit prompt position
        n1.append(_and([PR, D[1], _ng(pt[2]), _ng(isd)], [(L["E1"], 1.0)]))
        n1.append(_and([PR, D[1], D[2], _ng(pt[3]), _ng(isd)], [(L["E2"], 1.0)]))
        n1.append(_and([PR, D[1], D[2], D[3], D[4], _ng(isd)], [(L["E4"], 1.0)]))
        # month-name trigram ends at t (3-letter abbreviation)
        for m in range(12):
            ab = task.MONTHS[m][:3]
            c1, c2, c3 = stoi[ab[0]], stoi[ab[1]], stoi[ab[2]]
            n1.append(_and([PR, _p(L["TOK"] + c3), _p(L["PT1"] + c2), _p(L["PT2"] + c1)],
                           [(L["MON"] + m, 1.0)]))
        _set_mlp(model.blocks[0].mlp, n1)

        # --- Layer 2: uniform count of short runs --------------------------
        a2 = model.blocks[1].attn
        _set_head(a2, 0, v=[(0, L["E1"], 1.0), (0, L["E2"], 1.0)],
                  o=[(L["CNT"], 0, 1.0)], sink=-1e4)

        n2 = []
        for c in range(1, 5):
            terms = [(L["CNT"], GE_SCALE)] + [(L["POS"] + j, -GE_SCALE * (c - 0.5) / (j + 1))
                                              for j in range(42)]
            n2.append((terms, 0.0, [(L["GE"] + c - 1, 1.0)]))         # ReLU(X)
            n2.append((terms, -1.0, [(L["GE"] + c - 1, -1.0)]))       # -ReLU(X-1)  => clip to [0,1]
        for j in range(4):                                             # long-run digits, MSB first
            for d in range(10):
                n2.append(_and([_p(L["E4"]), _p(pt[4 - j][d])], [(L["LD"] + 10 * j + d, 1.0)]))
        for d in range(10):                                            # short run tens (2-digit run)
            n2.append(_and([_p(L["E2"]), _p(pt[2][d])], [(L["SDT"] + d, 1.0)]))
        n2.append(_and([_p(L["E1"])], [(L["SDT"], 1.0)]))              # 1-digit run -> tens '0'
        for d in range(10):                                            # short run units
            n2.append(_and([_g([L["E1"], L["E2"]]), _p(pt[1][d])], [(L["SDU"] + d, 1.0)]))
        _set_mlp(model.blocks[1].mlp, n2)

        # --- Layer 3: retrieval + output assembly --------------------------
        a3 = model.blocks[2].attn
        # head 0: long (4-digit) run
        _set_head(
            a3, 0,
            q=[(0, L["ONE"], BIG * SQ)],
            k=[(0, L["E4"], 1.0)],
            v=[(i, L["LD"] + i, 1.0) for i in range(40)] + [(40, L["ONE"], 1.0)],
            o=[(L["RLD"] + i, i, 1.0) for i in range(40)] + [(L["RONE"], 40, 1.0)],
            sink=SINK,
        )
        # heads 1..3: short run with ordinal k (eq_k = GE_k - GE_{k+1}, key matched iff SHORT and eq_k)
        for k in range(1, 4):
            _set_head(
                a3, k,
                q=[(0, L["ONE"], BIG * SQ), (1, L["ONE"], BIG * SQ)],
                k=[(0, L["E1"], 1.0), (0, L["E2"], 1.0),
                   (0, L["GE"] + k - 1, 1.0), (0, L["GE"] + k, -1.0),
                   (1, L["ONE"], -1.0)],
                v=[(i, L["SDT"] + i, 1.0) for i in range(10)] + [(10 + i, L["SDU"] + i, 1.0) for i in range(10)],
                o=[(L[f"RS{k}T"] + i, i, 1.0) for i in range(10)] + [(L[f"RS{k}U"] + i, 10 + i, 1.0) for i in range(10)],
                sink=SINK,
            )
        # head 4: month name (and existence flag via ONE)
        _set_head(
            a3, 4,
            q=[(0, L["ONE"], BIG * SQ)],
            k=[(0, L["MON"] + m, 1.0) for m in range(12)],
            v=[(m, L["MON"] + m, 1.0) for m in range(12)] + [(12, L["ONE"], 1.0)],
            o=[(L["RNAME"] + m, m, 1.0) for m in range(12)] + [(L["RHAS"], 12, 1.0)],
            sink=SINK,
        )

        def posk(k):
            return _p(L["POS"] + (P - 1 + k))

        NR1 = _n(L["RONE"])     # no long run
        NH = _n(L["RHAS"])      # numeric format (no month name)
        PH = _p(L["RHAS"])      # named-month format
        O = lambda d: L["OUT"] + d
        n3 = []
        # year digits 0..3
        for j in range(4):
            for d in range(10):
                n3.append(_and([posk(j), _p(L["RLD"] + 10 * j + d)], [(O(dig[d]), 1.0)]))
        n3.append(_and([posk(0), NR1], [(O(dig[2]), 1.0)]))            # 2-digit year: '2'
        n3.append(_and([posk(1), NR1], [(O(dig[0]), 1.0)]))            # '0'
        for d in range(10):
            n3.append(_and([posk(2), NR1, _p(L["RS3T"] + d)], [(O(dig[d]), 1.0)]))
            n3.append(_and([posk(3), NR1, _p(L["RS3U"] + d)], [(O(dig[d]), 1.0)]))
        # separators
        for k in (4, 7):
            n3.append(_and([posk(k)], [(O(DASH), 1.0)]))
        # month tens (step 5)
        n3.append(_and([posk(5), _g([L["RNAME"] + m for m in (9, 10, 11)])], [(O(dig[1]), 1.0)]))
        n3.append(_and([posk(5), _g([L["RNAME"] + m for m in range(9)])], [(O(dig[0]), 1.0)]))
        for d in range(10):
            n3.append(_and([posk(5), NH, _p(L["RS1T"] + d)], [(O(dig[d]), 1.0)]))
        # month units (step 6): named month m -> units digit
        mon_units = {0: [9], 1: [0, 10], 2: [1, 11]}                   # month 10 | 1,11 | 2,12
        for d in range(10):
            idxs = mon_units.get(d, [d - 1])
            n3.append(_and([posk(6), _g([L["RNAME"] + m for m in idxs])], [(O(dig[d]), 1.0)]))
        for d in range(10):
            n3.append(_and([posk(6), NH, _p(L["RS1U"] + d)], [(O(dig[d]), 1.0)]))
        # day (steps 8, 9): named -> ordinal-1 short run; numeric -> ordinal-2 short run
        for d in range(10):
            n3.append(_and([posk(8), PH, _p(L["RS1T"] + d)], [(O(dig[d]), 1.0)]))
            n3.append(_and([posk(8), NH, _p(L["RS2T"] + d)], [(O(dig[d]), 1.0)]))
            n3.append(_and([posk(9), PH, _p(L["RS1U"] + d)], [(O(dig[d]), 1.0)]))
            n3.append(_and([posk(9), NH, _p(L["RS2U"] + d)], [(O(dig[d]), 1.0)]))
        _set_mlp(model.blocks[2].mlp, n3)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "HandCircuitV3-tightFF"
model_description = ("DH=43 plus MLP widths set to exactly the neuron counts used (15/69/136) "
                     "instead of padded 16/80/160.")


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
    d_model = make_layout(task.vocab_size)["_size"]
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len, d_model=d_model)
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
