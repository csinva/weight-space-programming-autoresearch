"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) question classifier for `trec-coarse`.

Architecture (2 layers, no LayerNorm, every weight set in closed form):
  * Embedding: one-hot token | one-hot position | constant 1.
  * Layer 1 attention: K "look-back" heads. Head j attends (hard, via a
    position-match score) to position i-j and copies that token's one-hot into
    slot j of the residual stream -> every position sees the last K characters.
  * Layer 1 MLP: one ReLU unit per hand-written phrase pattern (word lists
    from my own knowledge of the TREC question taxonomy). A unit fires (=1)
    exactly when the pattern ends at this position; it writes class votes.
  * Layer 2 attention: one uniform head (all scores 0) that averages the class
    votes over the whole prompt into the final ('=') position.
  * Head: logits for '0'..'5' = 81 * mean votes + small prior bias.
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
# Lexicon (written by hand; classes: 0 ABBR, 1 ENTY, 2 DESC, 3 HUM, 4 LOC, 5 NUM)
# Pattern syntax: '^' = start of question, '#' = word boundary
# (leading '#' -> space or start of question; trailing '#' -> space/_/'/=).
# ---------------------------------------------------------------------------
A, E, D, H, L, N = range(6)
WINDOW = 40

PRIOR = [0.0, 0.5, 0.3, 0.2, 0.1, 0.15]

PHRASES: list[tuple[str, dict]] = [
    # ---- ABBR
    ("#stand for#", {A: 12}), ("#stands for#", {A: 12}), ("#abbreviation#", {A: 12}),
    ("#abbreviations#", {A: 12}), ("#acronym#", {A: 12}), ("#abbreviated#", {A: 10}),
    ("#abbreviate#", {A: 10}), ("#full form#", {A: 10}), ("#short for#", {A: 8}),
    ("#initials#", {A: 6}),
    # ---- DESC
    ("^why#", {D: 10}), ("#why#", {D: 4}),
    ("^how#", {D: 4}), ("^how come#", {D: 5}),
    ("^what is#", {D: 2.5}), ("^what are#", {D: 2}), ("^what's#", {D: 2.5}),
    ("^what is the#", {D: -2.5, E: 0.5}), ("^what are the#", {D: -2, E: 0.5}),
    ("^what's the#", {D: -2.5, E: 0.5}),
    ("^what is a#", {D: 2}), ("^what is an#", {D: 2}), ("^what are some#", {D: -2}),
    ("^what does#", {D: 1}),
    ("#mean#", {D: 5}), ("#meaning#", {D: 6}), ("#means#", {D: 4}),
    ("#definition#", {D: 8}), ("#define#", {D: 8}), ("#describe#", {D: 8}),
    ("#explain#", {D: 6}), ("#difference between#", {D: 8}), ("#difference#", {D: 3}),
    ("#origin of#", {D: 5}), ("#origins of#", {D: 5}), ("#history of#", {D: 3}),
    ("^what causes#", {D: 6}), ("^what caused#", {D: 5}), ("#cause of#", {D: 4}),
    ("^what happens#", {D: 6}), ("^what happened#", {D: 4}),
    ("#purpose of#", {D: 5}), ("#function of#", {D: 4}), ("#how to#", {D: 4}),
    ("#used for#", {D: 4}), ("#effect of#", {D: 4}), ("#effects of#", {D: 4}),
    ("#significance#", {D: 5}), ("#reason#", {D: 5}), ("#reasons#", {D: 5}),
    ("#come from#", {D: 1}), ("#get its name#", {D: 4}), ("#get their name#", {D: 4}),
    # ---- HUM
    ("^who#", {H: 10}), ("#who#", {H: 5}), ("#whom#", {H: 8}), ("#whose#", {H: 4}),
    ("#name of the man#", {H: 4}), ("#name of the woman#", {H: 4}),
    # ---- LOC
    ("^where#", {L: 10}), ("#where#", {L: 5}),
    ("#located#", {L: 4}), ("#location#", {L: 4}), ("#border#", {L: 2}), ("#borders#", {L: 2}),
    ("#united states#", {L: -1.5}), ("#capital#", {L: 1.5}),
    # ---- NUM
    ("^when#", {N: 10}), ("#when#", {N: 3}),
    ("#how many#", {N: 10}), ("#how much#", {N: 10}), ("#how long#", {N: 10}),
    ("#how far#", {N: 10}), ("#how old#", {N: 10}), ("#how big#", {N: 10}),
    ("#how tall#", {N: 10}), ("#how high#", {N: 10}), ("#how fast#", {N: 10}),
    ("#how large#", {N: 10}), ("#how wide#", {N: 10}), ("#how deep#", {N: 10}),
    ("#how heavy#", {N: 10}), ("#how hot#", {N: 10}), ("#how cold#", {N: 10}),
    ("#how often#", {N: 10}), ("#how short#", {N: 10}), ("#how small#", {N: 10}),
    ("#how close#", {N: 10}), ("#how warm#", {N: 10}), ("#how thick#", {N: 10}),
    ("#how quickly#", {N: 10}), ("#how soon#", {N: 10}), ("#how frequently#", {N: 10}),
    ("#population#", {N: 4}), ("#percentage#", {N: 4}), ("#percent#", {N: 4}),
    ("#temperature#", {N: 3}), ("#salary#", {N: 4}), ("#cost#", {N: 3}), ("#costs#", {N: 3}),
    ("#number of#", {N: 3}), ("#date#", {N: 3}), ("#birthday#", {N: 3}),
    ("#boiling point#", {N: 5}), ("#freezing point#", {N: 5}), ("#melting point#", {N: 5}),
    ("#zip code#", {N: 6}), ("#area code#", {N: 6}), ("#phone number#", {N: 6}),
    ("#telephone number#", {N: 6}), ("#weigh#", {N: 4}), ("#year#", {N: 1.5}),
    ("#speed#", {N: 2}), ("#distance#", {N: 2}), ("#height#", {N: 2}), ("#price#", {N: 2}),
    ("#average#", {N: 2}), ("#age#", {N: 2}),
    # ---- ENTY
    ("#kind of#", {E: 3}), ("#type of#", {E: 3}), ("#sort of#", {E: 3}), ("#called#", {E: 2}),
    ("#do you call#", {E: 5}), ("#name of#", {E: 1}),
    # ---- v2 additions
    (" do_", {D: 3}), ("#made of#", {E: 3}), ("#made from#", {E: 3}),
    ("#term for#", {E: 4}), ("#word for#", {E: 4}), ("#name for#", {E: 3}),
    ("#real name#", {H: 5}), ("#best way#", {D: 4}), ("#way to#", {D: 3}),
    ("#famous for#", {D: 4}), ("#known for#", {D: 3}), ("^what famous#", {H: 3}),
    ("#what part of#", {L: 3}), ("#calories#", {N: 3}), ("#dollars#", {N: 2}),
    ("#miles#", {N: 2}), ("#degrees#", {N: 2}), ("#worth#", {N: 3}),
    ("^how do you say#", {E: 6}), ("#nationality#", {L: 1}),
]

HEAD_PREFIXES = [
    "what ", "which ", "what is the ", "what was the ", "what are the ", "what were the ",
    "what's the ", "name of the ", "name of a ", "name the ", "name a ", "kind of ", "type of ",
    "sort of ", "the largest ", "the biggest ", "the longest ", "the highest ", "the smallest ",
    "the oldest ", "the tallest ", "the first ", "the most famous ", "the most popular ",
    "the most common ", "what u s ", "which u s ", "what famous ", "the deepest ", "the hottest ",
    "the coldest ", "the fastest ", "the richest ", "the best ", "the last ", "the only ",
    "the second ", "the world's ", "the world's largest ", "the most ", "what two ",
]

HEAD_NOUNS = {
    H: """person people man woman men women actor actress actors president presidents author authors
          writer novelist poet playwright singer singers composer painter artist sculptor architect
          inventor director producer player players emperor king kings queen prince princess pope
          leader ruler scientist explorer astronaut senator governor general admiral comedian
          character characters hero villain company companies corporation firm team teams group band
          organization manufacturer airline tribe dictator monarch prophet saint philosopher chemist
          physicist mathematician doctor lawyer pilot coach quarterback pitcher boxer golfer wrestler
          jockey musician guitarist drummer vocalist rapper cartoonist journalist reporter host
          businessman billionaire millionaire criminal assassin killer spy detective son daughter
          wife husband father mother brother sister superhero deity designer founder creator owner
          builder partner girlfriend boyfriend manager chairman ceo editor minister chancellor mayor
          ambassador secretary judge sheriff soldier sailor captain commander warrior knight magician
          dancer chef cartoonist hostess anchorman""",
    L: """country countries city cities state states continent continents river rivers mountain
          mountains island islands ocean oceans lake lakes sea seas capital capitals nation nations
          province provinces county counties town towns village region place places desert bay
          canal park street avenue address planet hemisphere peninsula volcano waterfall airport
          port harbor hotel stadium museum location continent territory colony kingdom range valley""",
    N: """year years date day month time century decade percentage percent population age
          temperature number speed distance length height weight cost price size amount average
          rate salary value total depth width diameter radius circumference volume frequency score
          odds birthday anniversary duration lifespan density mass ratio""",
    E: """color colour colors animal animals kind type sort breed language languages sport sports
          game games instrument instruments disease diseases illness food foods drink drinks book
          books novel movie movies film films song songs album play opera show series program
          religion currency money word words term bird birds dog dogs cat cats fish flower flowers
          plant plants tree trees fruit fruits vegetable metal element chemical drug medicine vitamin
          product products brand car cars ship boat plane aircraft vehicle weapon gun award prize
          trophy event war battle holiday festival sign symbol letter invention machine device tool
          material substance gas liquid organ bone muscle style dish dance music genre star
          constellation mineral gem stone rock insect creature mammal reptile cocktail beer wine
          computer software toy candy cheese sauce spice ingredient virus title sound shape fabric
          meal dessert soup bread crop grain tea coffee juice soda dinosaur horse snake whale shark
          monkey spider bug bean nut seed perfume comic cartoon poem painting statue sculpture speech
          quote phrase slogan motto prayer law treaty amendment document philosophy satellite rocket
          spacecraft train engine fuel acid protein hormone gene magazine""",
}
HEAD_WEIGHT = 6.0
MID_NOUN_WEIGHT = {H: 2.0, L: 1.5, N: 0.0, E: 0.0}


def build_patterns() -> list[tuple[str, bool, dict]]:
    """Expand lexicon into (literal pattern, anchored_at_start, votes) triples.

    In literal patterns '#' (only allowed at the end) means a trailing boundary char.
    """
    raw: list[tuple[str, dict]] = list(PHRASES)
    for c, words in HEAD_NOUNS.items():
        forms = set(words.split())
        for w in list(forms):  # add regular plurals
            if w.endswith("y") and w[-2:-1] not in "aeiou":
                forms.add(w[:-1] + "ies")
            elif not w.endswith("s"):
                forms.add(w + ("es" if w.endswith(("ch", "sh", "x")) else "s"))
        for w in sorted(forms):
            if MID_NOUN_WEIGHT[c]:
                raw.append(("#" + w + "#", {c: MID_NOUN_WEIGHT[c]}))
            for p in HEAD_PREFIXES:
                raw.append(("#" + p + w + "#", {c: HEAD_WEIGHT}))
    out: dict[tuple[str, bool], dict] = {}

    def add(pat, anchored, votes):
        key = (pat, anchored)
        acc = out.setdefault(key, {})
        for c, w in votes.items():
            acc[c] = acc.get(c, 0.0) + w

    for pat, votes in raw:
        if pat.startswith("^"):
            add(pat[1:], True, votes)
        elif pat.startswith("#"):
            add(" " + pat[1:], False, votes)   # preceded by a space
            add(pat[1:], True, votes)          # or at the start of the question
        else:
            add(pat, False, votes)
    return [(p, a, v) for (p, a), v in out.items()]


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Causal attention with explicit per-head Q/K/V/O matrices."""

    def __init__(self, d_model: int, n_heads: int, d_qk: int, d_v: int):
        super().__init__()
        self.n_heads, self.d_qk, self.d_v = n_heads, d_qk, d_v
        self.W_q = nn.Parameter(torch.zeros(n_heads, d_model, d_qk))
        self.W_k = nn.Parameter(torch.zeros(n_heads, d_model, d_qk))
        self.W_v = nn.Parameter(torch.zeros(n_heads, d_model, d_v))
        self.W_o = nn.Parameter(torch.zeros(n_heads, d_v, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = torch.einsum("btd,hdk->bhtk", x, self.W_q)
        k = torch.einsum("btd,hdk->bhtk", x, self.W_k)
        v = torch.einsum("btd,hdv->bhtv", x, self.W_v)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        out = attn @ v
        return torch.einsum("bhtv,hvd->btd", out, self.W_o)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 82):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.patterns = build_patterns()
        V, Lp, K = vocab_size, max_seq_len, WINDOW
        # residual layout
        self.o_tok = 0
        self.o_pos = V
        self.o_one = V + Lp
        self.o_slot = self.o_one + 1
        self.o_vote = self.o_slot + K * V
        self.o_agg = self.o_vote + 6
        self.d_model = self.o_agg + 6
        D = self.d_model
        self.embed = nn.Embedding(V, D)
        self.pos_embed = nn.Embedding(Lp, D)
        self.attn1 = MultiHeadAttention(D, K, Lp, V)
        self.mlp1 = MLP(D, len(self.patterns))
        self.attn2 = MultiHeadAttention(D, 1, 1, 6)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.embed(ids) + self.pos_embed(pos)[None]
        h = h + self.attn1(h)
        h = h + self.mlp1(h)
        h = h + self.attn2(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    V, Lp, K, D = model.vocab_size, model.max_seq_len, WINDOW, model.d_model
    stoi = task.stoi
    boundary = [stoi[c] for c in " _'="]
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        # embeddings
        for t in range(V):
            model.embed.weight[t, model.o_tok + t] = 1.0
            model.embed.weight[t, model.o_one] = 1.0
        for i in range(Lp):
            model.pos_embed.weight[i, model.o_pos + i] = 1.0

        # layer 1: look-back heads. head j at query i matches key position i-j.
        beta = 60.0
        a1 = model.attn1
        for j in range(K):
            for i in range(Lp):
                if i - j >= 0:
                    a1.W_q[j, model.o_pos + i, i - j] = beta
            for p in range(Lp):
                a1.W_k[j, model.o_pos + p, p] = 1.0
            for t in range(V):
                a1.W_v[j, model.o_tok + t, t] = 1.0
                a1.W_o[j, t, model.o_slot + j * V + t] = 1.0

        # MLP 1: one unit per pattern
        fc1, fc2 = model.mlp1.fc1, model.mlp1.fc2
        for u, (pat, anchored, votes) in enumerate(model.patterns):
            m = len(pat)
            assert m <= K, pat
            for idx, ch in enumerate(pat):
                j = m - 1 - idx            # offset back from current position
                chars = boundary if ch == "#" else [stoi[ch]]
                for t in chars:
                    fc1.weight[u, model.o_slot + j * V + t] = 1.0
            if anchored:
                # must end exactly at position m-1 (pattern starts at position 0)
                fc1.weight[u, model.o_pos + m - 1] = 1.0
                fc1.bias[u] = -(m + 0.5)
            else:
                for p in range(min(m - 1, Lp)):
                    fc1.weight[u, model.o_pos + p] = -100.0
                fc1.bias[u] = -(m - 0.5)
            for c, w in votes.items():
                fc2.weight[model.o_vote + c, u] = 2.0 * w

        # layer 2: uniform attention averaging the votes
        a2 = model.attn2
        for c in range(6):
            a2.W_v[0, model.o_vote + c, c] = 1.0
            a2.W_o[0, c, model.o_agg + c] = 1.0

        # head
        T = task.prompt_len
        for t in range(V):
            model.head.weight[t, model.o_one] = -100.0
        for c in range(6):
            t = stoi[str(c)]
            model.head.weight[t, model.o_one] = PRIOR[c]
            model.head.weight[t, model.o_agg + c] = float(T)


model_shorthand_name = "LexiconVotes_v2"
model_description = ("v1 + expanded hand-written lexicon (auto plurals, more head nouns/prefixes, "
                     "extra DESC/ENTY/HUM/NUM cue phrases).")


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
