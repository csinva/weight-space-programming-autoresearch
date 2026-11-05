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
        self.ln1 = nn.Identity()  # no LayerNorm: keeps hand-built features exact
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
        max_seq_len: int = 32,
        d_model: int = 276,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = (19, 84),
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
        ffs = list(d_ff) if isinstance(d_ff, (tuple, list)) else [d_ff] * n_layers   # per-layer MLP width
        self.blocks = nn.ModuleList([Block(d_model, n_heads, ffs[i]) for i in range(n_layers)])
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

ABL = "none"  # which single component to leave out ("none" = full model)
A_SHARP = 8.0   # attention sharpness (score margin of the intended key)


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Default implementation: leave the random initialization untouched.
    The agent should replace this with hand-designed weights that implement
    the task (e.g. 5-digit addition).
    """
    S = task.stoi
    D = [S[str(d)] for d in range(10)]
    P = 42           # positions
    A = A_SHARP      # attention sharpness
    DH = model.d_model // model.n_heads
    sq = math.sqrt(DH)

    # ---- residual stream layout -------------------------------------
    names = {}
    cur = [0]
    def alloc(name, n):
        names[name] = cur[0]; cur[0] += n
    alloc("TOK", 43); alloc("POS", P); alloc("ONE", 1)
    alloc("C1", 36); alloc("C2", 36)              # letter/digit identity 1 / 2 chars back (token ids 0..35; punctuation -> all zero)
    alloc("C3", 12); alloc("C4", 10)              # digit (+ '/' and '.' flags for C3) 3 / 4 chars back
    alloc("RY", 1); alloc("RM", 1); alloc("RD", 1)  # role flags (year/month/day source)
    for k in range(1, 5): alloc(f"YC{k}", 10)     # year closer: digit k chars back
    alloc("YS", 1)                                # year closer has '/' 3 back (2-digit year)
    alloc("MT", 10); alloc("MU", 10); alloc("DT", 10); alloc("DU", 10)  # month/day digits (MT/MU also get name lookup)
    alloc("OUT", 10)                              # digit logits
    assert cur[0] <= model.d_model, cur[0]
    n = lambda k: names[k]
    TOK = lambda t: n("TOK") + t
    POS = lambda p: n("POS") + p
    ONE = n("ONE")

    for p_ in model.parameters():
        p_.data.zero_()

    with torch.no_grad():
        for t in range(43):
            model.token_emb.weight[t, TOK(t)] = 1
        for p in range(P):
            model.pos_emb.weight[p, POS(p)] = 1
            model.pos_emb.weight[p, ONE] = 1

        # ===== Layer 1 attention =====
        a = model.blocks[0].attn
        # heads 0..3: "look k tokens back" -> copy raw token into C_k
        for k in range(1, 5):
            h = (k - 1) * DH
            for p in range(P):
                j = p - k if p >= k else (0 if ABL == "no_fallback" else p)  # fallback: self
                a.W_q.weight[h + j, POS(p)] = A * sq
            for j in range(P):
                a.W_k.weight[h + j, POS(j)] = 1
            if k <= 2:      # raw token copy
                for t in range(36):
                    a.W_v.weight[h + t, TOK(t)] = 1
                    a.W_o.weight[n(f"C{k}") + t, h + t] = 1
            else:           # digit one-hot (+ slash flag for C3)
                for d in range(10):
                    a.W_v.weight[h + d, TOK(D[d])] = 1
                    a.W_o.weight[n(f"C{k}") + d, h + d] = 1
                if k == 3:
                    a.W_v.weight[h + 10, TOK(S['/'])] = 1
                    a.W_o.weight[n("C3") + 10, h + 10] = 1
                    a.W_v.weight[h + 11, TOK(S['.'])] = 1       # '.' 3 back <=> YYYY.MM.DD format at its end
                    a.W_o.weight[n("C3") + 11, h + 11] = 1
        B = n("C3") + 11

        # ===== Layer 1 MLP: closer detection / role flags / month-name lookup =====
        m = model.blocks[0].mlp
        u = [0]
        def unit(terms, bias, outs, tag=None):
            if tag is not None and tag == ABL:
                return                                   # ablation: leave this unit out
            i = u[0]; u[0] += 1
            for idx, w in terms:
                m.fc1.weight[i, idx] = w
            m.fc1.bias[i] = bias
            for idx, w in outs:
                m.fc2.weight[idx, i] = w
        pd = [(n("C1") + d, 1.0) for d in D]            # previous char is a digit
        early = [(POS(1), 1.0), (POS(2), 1.0)]
        ans = [(POS(p), 1.0) for p in range(32, P)]
        RY, RM, RD = n("RY"), n("RM"), n("RD")
        unit([(TOK(S['_']), 1)] + pd + [(B, -1)], -1, [(RY, 1)], "no_endY")  # end of text, not B -> year
        unit([(TOK(S['_']), 1), (B, 1)] + pd, -2, [(RD, 1)], "no_endB")          # end of text, B -> day
        unit([(TOK(S['/']), 1)] + pd + early, -2, [(RM, 1)], "no_slashE")      # early slash -> month
        unit([(TOK(S['/']), 1)] + pd + [(i, -w) for i, w in early], -1, [(RD, 1)], "no_slashL")  # late slash -> day
        unit([(TOK(S['.']), 1), (POS(4), 1)] + pd, -2, [(RY, 1)], "no_dot4")  # YYYY. -> year
        unit([(TOK(S['.']), 1), (POS(7), 1)] + pd, -2, [(RM, 1)], "no_dot7")  # YYYY.MM. -> month
        other = [(TOK(S[c]), 1) for c in (" ,-" if ABL == "no_suffix" else " ,-snrt")]
        unit(other + pd + [(i, -2 * w) for i, w in ans], -1, [(RD, 1)], "no_other")  # named-format day closer
        months = task.MONTHS
        for mi, name in enumerate(months):
            l1, l2, l3 = name[0], name[1], name[2]
            mo = mi + 1
            outs = [(RM, 1), (n("MT") + mo // 10, 1), (n("MU") + mo % 10, 1)]
            unit([(n("C2") + S[l1], 1), (n("C1") + S[l2], 1), (TOK(S[l3]), 1)], -2, outs, "no_" + name[:3])

        # ===== Layer 2 attention: output positions fetch the flagged sources =====
        a = model.blocks[1].attn
        def role_head(hi, role, vals):
            h = hi * DH
            a.W_q.weight[h, ONE] = A * sq
            a.W_k.weight[h, n(role)] = 1
            for vi, (srcs, dst) in enumerate(vals):
                # srcs: list of (kind, base): 'tok' = raw token slot (digit d at base+D[d]); 'dig' = digit slot (base+d)
                for d in range(10):
                    for kind, base in srcs:
                        a.W_v.weight[h + vi * 10 + d, base + (D[d] if kind == 'tok' else d)] = 1
                    a.W_o.weight[n(dst) + d, h + vi * 10 + d] = 1
        t = lambda k: ('tok', n(f"C{k}"))
        g = lambda k: ('dig', n(f"C{k}"))
        role_head(0, "RY", [([t(1)], "YC1"), ([t(2)], "YC2"), ([g(3)], "YC3"), ([g(4)], "YC4")])
        a.W_v.weight[0 * DH + 40, n("C3") + 10] = 1
        a.W_o.weight[n("YS"), 0 * DH + 40] = 1
        role_head(1, "RM", [([t(2), ('dig', n("MT"))], "MT"), ([t(1), ('dig', n("MU"))], "MU")])
        role_head(2, "RD", [([t(2)], "DT"), ([t(1)], "DU")])

        # ===== Layer 2 MLP: position-gated digit routing =====
        m = model.blocks[1].mlp
        u = [0]
        OUT = n("OUT")
        def route(i, src, extra=()):
            p = 31 + i
            for d in range(10):
                unit([(n(src) + d, 1), (POS(p), 1)] + list(extra), -1, [(OUT + d, 1)])
        YS = n("YS")
        route(0, "YC4", [(YS, -1)])
        unit([(YS, 1), (POS(31), 1)], -1, [(OUT + 2, 1)], "no_ys")
        route(1, "YC3")
        unit([(YS, 1), (POS(32), 1)], -1, [(OUT + 0, 1)], "no_ys")
        route(2, "YC2"); route(3, "YC1")
        route(5, "MT"); route(6, "MU"); route(8, "DT"); route(9, "DU")
        for i, src in ((5, "MT"), (8, "DT")):   # zero-pad a missing tens digit
            unit([(POS(31 + i), 1)] + [(n(src) + d, -1) for d in range(10)], 0, [(OUT + 0, 1)], "no_pad")

        # ===== Output head =====
        for d in range(10):
            model.head.weight[D[d], OUT + d] = 1
        model.head.weight[S['-'], POS(35)] = 10
        model.head.weight[S['-'], POS(38)] = 10


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = ("noBhead_290_5h" if A_SHARP == 12.0 else f"noBhead_A{A_SHARP:g}_276_4h") if ABL == "none" else "abl2_" + ABL
model_description = (f"noBhead model, A={A_SHARP:g}, MLP widths (19, 84), d_model 276 with 4 heads; C1/C2 slots shrunk to 36 letter/digit dims."
                     if ABL == "none" else f"noBhead_290_5h with component '{ABL}' left out (ablation).")


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
