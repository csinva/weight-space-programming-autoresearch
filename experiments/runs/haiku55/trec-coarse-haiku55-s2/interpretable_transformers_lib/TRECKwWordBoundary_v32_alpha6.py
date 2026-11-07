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
        # LayerNorm is disabled (identity) so every residual-stream channel stays linear.
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
        d_model: int = 384,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 192,
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

# TREC coarse keyword decision list: (pattern, class). Earlier = higher priority.
# Classes: 0 ABBR, 1 ENTY, 2 DESC, 3 HUM, 4 LOC, 5 NUM.
# Pattern forms: "xyz" (substring, 3 chars), "wxyz" (substring, 4 chars),
# " xyz" (word-start prefix: a word beginning with xyz).
TREC_KEYWORDS = [
    (" abb", 0), (" acr", 0), (" ful", 0),
    ("woma", 3), ("huma", 3),
    # HUM (moved above NUM)
    ("who", 3), ("pers", 3), ("peop", 3), ("auth", 3), ("orga", 3), ("actr", 3),
    ("acto", 3), ("lawy", 3), ("writ", 3), ("athl", 3), ("team", 3), ("sing", 3), ("foun", 3),
    (" son", 3), (" boy", 3), (" gir", 3), (" kid", 3), ("fath", 3), ("wife", 3), ("frie", 3),
    ("fami", 3), ("lead", 3), ("king", 3), ("quee", 3), ("empe", 3), ("doct", 3), ("nurs", 3),
    ("teac", 3), ("poet", 3), ("coac", 3), ("army", 3), ("club", 3), ("gove", 3), ("cian", 3),
    ("dent", 3), ("memb", 3), ("owne", 3), ("stud", 3), ("empl", 3), ("scie", 3), ("inve", 3),
    ("dire", 3),
    # LOC (moved above NUM)
    ("wher", 4), ("coun", 4), (" cit", 4), ("tate", 4), (" riv", 4), ("moun", 4),
    (" isl", 4), ("ntin", 4), ("ocea", 4), ("apit", 4), (" lak", 4), ("loca", 4),
    ("plac", 4), ("worl", 4), ("bord", 4), ("regi", 4), ("nati", 4),
    ("road", 4), ("stre", 4), ("park", 4), ("buil", 4), ("brid", 4), ("airp", 4),
    # NUM
    ("many", 5), ("much", 5), ("when", 5), ("year", 5), ("date", 5), (" old", 5),
    (" far", 5), (" pop", 5), ("numb", 5), ("perc", 5), ("tury", 5),
    (" age", 5), ("mone", 5), ("pric", 5), ("spee", 5), ("heig", 5), ("leng", 5), ("mile", 5),
    ("feet", 5), ("hour", 5), ("week", 5), ("birt", 5), ("doll", 5), ("minu", 5),
    # DESC
    ("why", 2), ("mean", 2), ("defi", 2), ("desc", 2), ("reas", 2), ("caus", 2),
    ("purp", 2), ("diff", 2),
    (" way", 2), ("meth", 2), ("proc", 2), ("step", 2), ("effe", 2), ("exam", 2), ("orig", 2),
    ("hist", 2), ("expl", 2), (" use", 2),
    # ENTY
    ("anim", 1), ("colo", 1), ("food", 1), ("game", 1), ("word", 1), (" car", 1), ("book", 1),
    ("film", 1), ("reli", 1), ("curr", 1), ("meta", 1), ("name", 1), ("spor", 1), ("symb", 1),
    (" dog", 1), ("bird", 1), ("lang", 1), ("musi", 1), ("shap", 1), ("lett", 1), ("elem", 1),
    ("subs", 1),
    ("plan", 1), ("frui", 1), ("vege", 1), ("tree", 1), ("fish", 1), ("hors", 1), ("snak", 1),
    ("drug", 1), ("chem", 1), ("song", 1), ("drin", 1), ("coin", 1), ("ship", 1), ("dise", 1),
    # lowest priority: "how ..." and "what is ..." with nothing else -> DESC
    ("how", 2), ("is ", 2),
]
TREC_DEFAULT = 1  # class when no keyword fires (ENTY)
ALPHA = 6.0       # attention sharpness: score = ALPHA * priority (softmax ~ hard max)


def _pos_code(p: int, dim: int = 19) -> torch.Tensor:
    """Two-hot position code: one-hot(p % 9) ++ one-hot(p // 9). Zero for p < 0.

    Dot product of codes of positions a and b = [a%9==b%9] + [a//9==b//9],
    which equals 2 only when a == b, so attention can match exact offsets.
    """
    v = torch.zeros(dim)
    if p >= 0:
        v[p % 9] = 1.0
        v[9 + p // 9] = 1.0
    return v


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training.

    Residual stream layout (d_model = 384):
      TOK [0,36) current char one-hot | POSK [36,55) own position code
      POSQ1/2/3 [55,74)/[74,93)/[93,112) codes of positions t-1/t-2/t-3
      CONST [112] | P1/P2/P3 [113,149)/[149,185)/[185,221) chars at t-1/t-2/t-3
      FEAT [221, 221+K) keyword indicators | VOTE next 6 class scores
    Layer 1: three previous-character heads (offsets -1, -2, -3) + MLP units that
      detect each keyword window (3-char substring, 4-char substring, or ' xyz' word start).
    Layer 2: one head from the final position; keys = keyword priorities, so the
      softmax picks the highest-priority keyword present; no keyword -> default class.
    """
    D, H = model.d_model, model.n_heads
    dh = D // H
    V = model.vocab_size
    NCLS = 6
    K = len(TREC_KEYWORDS)
    stoi = task.stoi

    TOK, POSK, POSQ1, POSQ2, POSQ3, CONST = 0, 36, 55, 74, 93, 112
    P1, P2, P3, FEAT = 113, 149, 185, 221
    VOTE = FEAT + K
    assert VOTE + NCLS <= D and V <= dh

    # One MLP unit per matchable pattern.
    units = []  # (keyword index, kind, chars)
    for k, (pat, _cls) in enumerate(TREC_KEYWORDS):
        if len(pat) == 3:
            units.append((k, "tri", pat))
        elif len(pat) == 4 and pat[0] == " ":
            units.append((k, "anch", pat[1:]))
            units.append((k, "start", pat[1:]))  # word starting at position 0 (t == 2)
        elif len(pat) == 4:
            units.append((k, "quad", pat))
        else:
            raise ValueError(pat)
    assert len(units) <= model.d_ff, (len(units), model.d_ff)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # --- embeddings
        for _, i in stoi.items():
            model.token_emb.weight[i, TOK + i] = 1.0
            model.token_emb.weight[i, CONST] = 1.0
        for t in range(model.max_seq_len):
            model.pos_emb.weight[t, POSK:POSK + 19] = _pos_code(t)
            model.pos_emb.weight[t, POSQ1:POSQ1 + 19] = _pos_code(t - 1)
            model.pos_emb.weight[t, POSQ2:POSQ2 + 19] = _pos_code(t - 2)
            model.pos_emb.weight[t, POSQ3:POSQ3 + 19] = _pos_code(t - 3)

        # --- layer 1 attention: previous-character heads (offset -1, -2, -3)
        L1 = model.blocks[0]
        s = ALPHA * math.sqrt(dh)  # cancels 1/sqrt(dh) -> score = ALPHA * (#matching codes)
        for h, (qslot, out_slot) in enumerate([(POSQ1, P1), (POSQ2, P2), (POSQ3, P3)]):
            base = h * dh
            for j in range(19):
                L1.attn.W_q.weight[base + j, qslot + j] = s
                L1.attn.W_k.weight[base + j, POSK + j] = 1.0
            for j in range(V):
                L1.attn.W_v.weight[base + j, TOK + j] = 1.0
                L1.attn.W_o.weight[out_slot + j, base + j] = 1.0

        # --- layer 1 MLP: one unit per pattern -> FEAT[keyword]
        fc1, fc2 = L1.mlp.fc1, L1.mlp.fc2
        for u, (k, kind, chars) in enumerate(units):
            c = [stoi[ch] for ch in chars]
            if kind == "tri":        # chars at t-2, t-1, t
                fc1.weight[u, P2 + c[0]] = 1.0
                fc1.weight[u, P1 + c[1]] = 1.0
                fc1.weight[u, TOK + c[2]] = 1.0
                fc1.bias[u] = -2.5
            elif kind == "quad":     # chars at t-3..t
                fc1.weight[u, P3 + c[0]] = 1.0
                fc1.weight[u, P2 + c[1]] = 1.0
                fc1.weight[u, P1 + c[2]] = 1.0
                fc1.weight[u, TOK + c[3]] = 1.0
                fc1.bias[u] = -3.5
            elif kind == "anch":     # space at t-3, then 3 letters at t-2..t
                fc1.weight[u, P3 + stoi[" "]] = 1.0
                fc1.weight[u, P2 + c[0]] = 1.0
                fc1.weight[u, P1 + c[1]] = 1.0
                fc1.weight[u, TOK + c[2]] = 1.0
                fc1.bias[u] = -3.5
            elif kind == "start":    # word starts at position 0 (so t == 2 ends the prefix)
                fc1.weight[u, P2 + c[0]] = 1.0
                fc1.weight[u, P1 + c[1]] = 1.0
                fc1.weight[u, TOK + c[2]] = 1.0
                fc1.weight[u, POSK + 2] = 1.0   # position 2 = (mod 9 -> 2, div 9 -> 0)
                fc1.weight[u, POSK + 9] = 1.0
                fc1.bias[u] = -4.5
            fc2.weight[FEAT + k, u] = 2.0       # ReLU output is 0.5 on exact match -> 1

        # --- layer 2 attention: priority decision list -> VOTE
        L2 = model.blocks[1]
        L2.attn.W_q.weight[0, CONST] = s                 # constant query
        for k, (pat, cls) in enumerate(TREC_KEYWORDS):
            L2.attn.W_k.weight[0, FEAT + k] = float(K - k)  # priority (earlier = larger)
            for c in range(NCLS):
                # value = default vote + indicator * (class vote - default vote)
                L2.attn.W_v.weight[c, FEAT + k] = float(c == cls) - float(c == TREC_DEFAULT)
        for c in range(NCLS):
            L2.attn.W_v.weight[c, CONST] = float(c == TREC_DEFAULT)
            L2.attn.W_o.weight[VOTE + c, c] = 1.0

        # --- output head: digit token '0'+c reads VOTE[c]
        for c in range(NCLS):
            model.head.weight[stoi[str(c)], VOTE + c] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "TRECKwWordBoundary_v32_alpha6"
model_description = (
    "v31 with attention sharpness ALPHA=6 instead of 15 (weaker priority separation; a sensitivity check)."
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

    accuracy, _details = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )
    # DEBUG (temporary): full-pool accuracy (2000 examples, seed 1) and error listing
    _full, _ = evaluate(model, task, n_samples=2000, seed=1, device=args.device)
    print("DEBUG full_pool_acc_2000_seed1", round(_full, 4))
    _errs = [d for d in _details if not d.correct]
    print("DEBUG n_errors", len(_errs))
    from collections import Counter as _C
    print("DEBUG confusion (target,pred):", _C((d.target, d.prediction) for d in _errs).most_common(12))

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
