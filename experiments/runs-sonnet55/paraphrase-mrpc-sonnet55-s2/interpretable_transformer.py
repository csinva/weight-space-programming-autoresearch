"""
Interpretable transformer for character-level sequence tasks (task: paraphrase-mrpc).

Prompt layout (122 chars): sentence A (60 chars) '|' sentence B (60 chars) '='
Answer: a single character '0' / '1' (paraphrase or not). ~66% of labels are '1'.
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
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
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
        n_layers: int = 3,
        d_ff: int = 256,
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
        self.final_ln = nn.LayerNorm(d_model)
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Baseline: majority-class predictor. Every parameter is zero except a
    constant 'bias' direction: final_ln.bias = e0 and head[token '1'] = e0,
    so the logit of '1' is always +1 and everything else 0 -> always answer '1'
    (accuracy ~ label prior, ~0.66)."""
    if MODE == "const":
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.final_ln.bias[0] = 1.0
            model.head.weight[task.stoi["1"], 0] = 1.0
        return
    V = task.vocab_size
    P = model.max_seq_len
    K = NGRAM
    C = [k * V for k in range(K)]          # C[k] = one-hot channel holding the char k steps back (C[0] = current char)
    TOK, POS = 0, K * V
    ISB, ISA, SNK, F, G, BIAS = POS + P, POS + P + 1, POS + P + 2, POS + P + 3, POS + P + 4, POS + P + 5
    D = model.d_model
    st = task.stoi
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in [model.final_ln] + [b.ln1 for b in model.blocks] + [b.ln2 for b in model.blocks]:
            m.weight.fill_(1.0)
        e = model.token_emb.weight
        pe = model.pos_emb.weight
        for t in range(V):
            e[t, TOK + t] = 1.0
        e[st["|"], SNK] = 1.0
        for p in range(P):
            pe[p, POS + p] = 1.0
            if 61 <= p <= 120:
                pe[p, ISB] = 1.0
            elif p < 60:
                pe[p, ISA] = 1.0
        # layers 0..K-2: "shift register". Layer l is a previous-token head that copies channel C[l]
        # of position i-1 into channel C[l+1] of position i, so after K-1 layers every position
        # knows its last K characters.
        for l in range(K - 1):
            a0 = model.blocks[l].attn
            for p in range(P):
                a0.W_q.weight[POS + p, POS + p] = PREV_ALPHA
                if p + 1 < P:
                    a0.W_k.weight[POS + p + 1, POS + p] = PREV_ALPHA
            ln_scale = math.sqrt(D / (3 + l))   # LayerNorm blow-up of a one-hot among ~(3+l) active dims
            for t in range(V):
                a0.W_v.weight[C[l + 1] + t, C[l] + t] = 1.0 / ln_scale
                a0.W_o.weight[C[l + 1] + t, C[l + 1] + t] = 1.0
        # layer K-1: B position looks for A positions with the same K-gram; sink at '|'
        a1 = model.blocks[K - 1].attn
        for t in range(V):
            for off in C:
                a1.W_q.weight[off + t, off + t] = MATCH_BETA
                a1.W_k.weight[off + t, off + t] = MATCH_BETA
        a1.W_q.weight[SNK, ISB] = SINK_Q
        a1.W_k.weight[SNK, SNK] = 1.0
        a1.W_q.weight[ISA, ISB] = 1.0       # reuse ISA dim as "penalty channel"
        a1.W_k.weight[ISA, ISB] = -BPEN     # keys in B are not allowed
        a1.W_v.weight[F, SNK] = 1.0 / math.sqrt(D / (2 + K))
        a1.W_o.weight[F, F] = 1.0
        # layer K: '=' averages F over B positions -> G = fraction of B K-grams NOT found in A
        a2 = model.blocks[K].attn
        a2.W_q.weight[ISB, st["="]] = POOL_Q
        a2.W_k.weight[ISB, ISB] = 1.0
        for p in range(61, 121):   # optional position tilt of the pooling (slope>0 favours late B chars)
            a2.W_k.weight[ISB, POS + p] = POOL_SLOPE * (p - 90) / 30
        a2.W_v.weight[G, F] = 1.0 / math.sqrt(D / (2 + K))
        a2.W_o.weight[G, G] = 1.0
        # readout: answer '1' iff G small
        model.final_ln.bias[BIAS] = 1.0
        model.head.weight[st["0"], BIAS] = 1.0
        model.head.weight[st["1"], BIAS] = 1.0 + THR
        model.head.weight[st["1"], G] = -1.0


PREV_ALPHA, MATCH_BETA, SINK_Q, BPEN, POOL_Q, THR = 2.0, 1.6, 3.8, 6.0, 3.0, 2.6


DIAG = False
MODE = "const"       # "const" (majority class) or "ngram"
NGRAM = 2
POOL_SLOPE = 0.0
D_MODEL, N_LAYERS, D_FF = 1, 0, 1# only used by MODE == "const"


def sweep(task):
    global MATCH_BETA, SINK_Q
    for mb in (1.0, 1.4, 1.8):
        for sq in [x * NGRAM / 2 for x in (2.0, 3.0, 3.8, 4.6)]:
            MATCH_BETA, SINK_Q = mb, sq
            m = build_model(task)
            print("mb", mb, "sq", sq, end=" ")
            diagnose(m, task)


def diagnose(model, task):
    exs = list(task.generate_examples(2000, 7))
    ids = torch.tensor([[task.stoi[c] for c in e.prompt] for e in exs])
    with torch.no_grad():
        d = model(ids)[:, -1, :]
    s = d[:, task.stoi["0"]] - d[:, task.stoi["1"]]   # = G_ln (+ -THR); larger => less similar
    y = torch.tensor([int(e.answer) for e in exs])
    s = s + THR  # undo THR offset
    dm = (s[y == 0].mean() - s[y == 1].mean()).item() / s.std().item()
    best = max(((((s < t) == (y == 1)).float().mean().item(), t.item()) for t in s.sort().values[::10]))
    print("d'=%.3f best acc %.3f thr %.3f" % (dm, best[0], best[1]))


# A unique shorthand name + 1-2 sentence description of what this attempt does.
if MODE == "const":
    model_shorthand_name = f"const_one_d{D_MODEL}_L{N_LAYERS}_ff{D_FF}"
else:
    model_shorthand_name = f"ngram{NGRAM}_mb{MATCH_BETA}_sq{SINK_Q}_thr{THR}_ps{POOL_SLOPE}_pq{POOL_Q}"
if MODE == "const":
    model_description = (f"Majority-class: all weights zero except final_ln bias -> head['1']; always answers 1. "
                         f"Shrunk to d_model={D_MODEL}, layers={N_LAYERS}, d_ff={D_FF}.")
else:
  model_description = (f"{NGRAM-1} shift-register prev-token layers -> B->A {NGRAM}-gram-match head with sink at '|' -> mean-pool at '=' -> threshold readout "
                     f"(match_beta={MATCH_BETA}, sink_q={SINK_Q}, thr={THR}).")


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
    d_model, n_layers, d_ff = D_MODEL, N_LAYERS, D_FF
    if MODE == "ngram":
        d_model, n_layers, d_ff = NGRAM * task.vocab_size + max_seq_len + 6, NGRAM + 1, 8
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len, d_model=d_model, n_heads=1, n_layers=n_layers, d_ff=d_ff)
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
    if False:  # EXPLORE
        exs = list(task.generate_examples(3000, 5))
        def sp(p):
            a, b = p[:-1].split("|")
            return a.replace("_", " ").split(), b.replace("_", " ").split()
        def ngr(s, n):
            return set(s[i:i + n] for i in range(len(s) - n + 1))
        rows = []
        for e in exs:
            a, b = sp(e.prompt)
            A, B = set(a), set(b)
            ra = " ".join(a); rb = " ".join(b)
            c = len(A & B)
            f = dict(
                prec=c / max(1, len(B)), rec=c / max(1, len(A)), c=c,
                lena=len(ra), lenb=len(rb), nb=len(b), na=len(a),
                c4=len(ngr(ra, 4) & ngr(rb, 4)) / max(1, len(ngr(rb, 4))),
                c4a=len(ngr(ra, 4) & ngr(rb, 4)) / max(1, len(ngr(ra, 4))),
                lcp=next((i for i in range(60) if i >= min(len(ra), len(rb)) or ra[i] != rb[i]), 60),
                uniq_b=len(B - A), uniq_a=len(A - B),
            )
            rows.append((f, int(e.answer)))
        n = len(rows)
        for k, edges in [("lcp", [0, 1, 2, 4, 8, 16, 32, 59, 61]), ("c4", [0, .05, .1, .2, .3, .5, .7, .9, 1.01]), ("uniq_b", [0, 1, 2, 3, 5, 8, 20])]:
            out = []
            for lo, hi in zip(edges, edges[1:]):
                sel = [r[1] for r in rows if lo <= r[0][k] < hi]
                out.append((lo, len(sel), round(sum(sel) / max(1, len(sel)), 2)))
            print(k, out)
        for k in rows[0][0]:
            vals = sorted(set(r[0][k] for r in rows))
            best = max(((sum((r[0][k] >= t) == bool(r[1]) for r in rows) / n, t, 1) for t in vals[1:]) , default=(0, 0, 0))
            bestn = max(((sum((r[0][k] < t) == bool(r[1]) for r in rows) / n, t, -1) for t in vals[1:]), default=(0, 0, 0))
            print(k, max(best, bestn))
        sys.exit()
    model = build_model(task).to(args.device)
    if DIAG:
        sweep(task)

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
