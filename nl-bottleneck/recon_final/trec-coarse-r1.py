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
    def __init__(self, d_model: int, n_heads: int, d_head: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)

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
    """Pre-norm block with LayerNorms replaced by identity (exact integer logic)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_head: int = 8):
        super().__init__()
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
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
        d_model: int | None = None,
        n_heads: int | None = None,
        n_layers: int = 2,
        d_ff: int | None = None,
    ):
        super().__init__()
        d_model = d_model or D_MODEL
        n_heads = n_heads or N_OFFSETS
        d_ff = d_ff or len(PATTERNS)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # Block 1: 20 copy heads (d_head 8) + keyword AND-gate MLP.
        # Block 2: 1 salience-argmax head (d_head 8) + trivial (zero) MLP of width 1.
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, d_head=8),
            Block(d_model, 1, 1, d_head=8),
        ])
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

# ---- Keyword decision list (written from general knowledge, not from data) ----
#
# Pattern mini-language (read left to right, last symbol = current position):
#   letter   : that exact letter
#   ' '      : a word-boundary char (space, '_' pad, or '=')
#   '^x'     : letter x AND it is the first token of the prompt (start flag)
#   '_'      : a padding char
#   '?'      : wildcard (no constraint)
#   '<'      : left word edge = boundary char here OR next letter has the start
#              flag (an exact OR inside the AND gate: bnd(+-1) + start(+-1) = 0 iff
#              exactly one holds, -2 if neither)
ABBR, ENTY, DESC, HUM, LOC, NUM = 0, 1, 2, 3, 4, 5


def _word(w):      # whole word, anywhere (incl. question-initial)
    return ["<" + w + " "]


def _wstart(w):    # word-initial stem (allows plurals / suffixes)
    return ["<" + w]


def _ws(*ws):      # many word-initial stems
    return ["<" + w for w in ws]


def _ww(*ws):      # many whole words
    return ["<" + w + " " for w in ws]


def _stem(w):      # raw substring anywhere
    return [w]


def _init(w):      # question-initial word
    return ["^" + w + " "]


_RULES = []  # (priority, class, [patterns])


def _add(prio, cls, *groups):
    pats = []
    for g in groups:
        pats.extend(g)
    _RULES.append((prio, cls, pats))


# 14: ABBR
_add(14, ABBR, _stem("stands for"), _stem("stand for"), _stem("abbreviat"))
# 13: DESC "comes from"
_add(13, DESC, _stem("comes from"))
# 12: initial who -> HUM
_add(12, HUM, ["^who"])   # initial who / whom / whose / who's
# 11: where -> LOC
_add(11, LOC, _word("where"))
# 10: initial when -> NUM
_add(10, NUM, _init("when"))
# 9: why -> DESC
_add(9, DESC, _word("why"))
# 8: definition / cause words -> DESC
_add(8, DESC, _wstart("mean"), _stem("origin"), _wstart("reason"), _wstart("defin"),
     _wstart("cause"), _wstart("purpose"), _wstart("explain"), _wstart("describe"),
     _wstart("differen"))
# 7: how did / does / is -> DESC
_add(7, DESC, _word("how did"), _word("how does"), _word("how is"), _word("how do"))
# 6: other how -> NUM
_add(6, NUM, _word("how"))
# 5: short "what is X" (< 17 chars) -> DESC
# (checked one step after the first pad so it never shares a position with a word match)
_add(5, DESC, ["^what is ????????_?"])
# 4: topical nouns
_add(4, HUM,
     _ww("man", "men", "king", "god", "spy", "hero", "band", "host", "pilot", "coach",
         "group", "tribe", "duke", "lord", "chef", "doctor", "lawyer", "judge", "boxer"),
     _ws("woman", "women", "person", "president", "author", "actor", "actress", "singer",
         "queen", "emperor", "inventor", "leader", "player", "writer", "company",
         "companies", "team", "composer", "painter", "artist", "scientist", "astronaut",
         "director", "founder", "poet", "senator", "governor", "pope", "minister",
         "comedian", "musician", "nickname", "explorer", "philosopher", "husband", "wife",
         "son of", "father", "mother", "character", "goddess", "prophet", "saint",
         "architect", "designer", "athlete", "golfer", "pitcher", "quarterback",
         "manufacturer", "corporation", "airline", "organization", "detective",
         "novelist", "playwright", "sculptor", "monarch", "dictator", "ruler", "prince",
         "princess", "admiral", "captain", "sailor", "assassin", "daughter", "brother",
         "sister", "dynasty", "superhero", "celebrit", "villain", "cartoonist",
         "politician", "chairman", "owner", "winner",
         "champion", "philanthrop", "economist", "physicist", "chemist",
         "astronomer", "mathematician", "biologist", "historian", "psychologist",
         "candidate", "hostess", "anchorman", "newscaster", "journalist", "editor",
         "publisher", "producer", "conductor", "guitarist", "drummer", "pianist",
         "dancer", "ballerina", "rapper", "magician", "clown", "cowboy", "outlaw",
         "pirate", "gangster", "criminal", "murderer", "killer"))
_add(4, LOC,
     _ww("sea", "park", "zoo", "bay", "gulf", "port", "moon", "cave", "falls",
         "road", "town", "towns", "state", "nation", "nations"),
     _ws("countr", "city", "cities", "capital", "river", "mountain", "island",
         "continent", "ocean", "lake", "located", "province", "county",
         "desert", "airport", "street", "region", "border", "volcano", "place",
         "peninsula", "canal", "hemisphere", "planet", "village", "territor", "bridge",
         "building", "museum", "address", "headquarter", "hometown", "birthplace",
         "hotel", "restaurant", "stadium", "church",
         "temple", "castle", "palace", "tower", "monument", "landmark", "galaxy",
         "forest", "valley", "canyon", "waterfall", "strait", "coast", "beach",
         "harbor", "avenue", "highway", "district", "neighborhood", "kingdom",
         "colony", "suburb", "prison", "resort", "glacier", "reef", "jungle", "swamp"))
_add(4, NUM,
     _ww("age", "tall", "day", "days", "size", "rate", "worth", "time", "area",
         "pay", "tax", "fee", "cost", "costs", "mass"),
     _ws("year", "date", "population", "number", "percent", "distance", "temperature",
         "speed", "price", "height", "weigh", "mile", "degree", "centur",
         "average", "length", "salary", "dollar", "amount", "depth", "diameter",
         "birthday", "decade", "hours", "minutes", "seconds", "months", "weeks", "feet",
         "pounds", "lifespan", "zip code", "boiling point",
         "freezing point", "melting point", "calorie", "score", "odds", "chance",
         "probability", "volume", "density", "frequency", "radius",
         "circumference", "capacity", "duration", "life expectancy", "budget",
         "revenue", "income", "wage", "acre", "kilometer", "inches", "yards", "tons",
         "gallons", "ounces", "grams", "ratio", "value", "anniversary", "earn",
         "gross", "sales", "percentage", "latitude", "longitude", "altitude",
         "elevation", "period"))
_add(4, ENTY,
     _ww("cat", "car", "cars", "play", "war", "term", "organ", "bee", "ant", "bear", "rat",
         "pig", "cow", "owl", "pet", "gun", "oil", "gold", "iron", "tv", "hat",
         "toy", "gas", "gem", "nut", "fuel", "tool", "cell", "gene", "bone", "law"),
     _ws("animal", "color", "colour", "food", "book", "movie", "film", "sport", "game",
         "language", "instrument", "disease", "drug", "bird", "dog", "plant", "flower",
         "song", "novel", "currency", "religion", "fruit", "vegetable", "fish", "tree",
         "breed", "product", "brand", "drink", "beer", "wine", "dish", "symbol", "award",
         "prize", "invention", "material", "metal", "element", "chemical", "substance",
         "weapon", "vehicle", "ship", "boat", "plane", "aircraft", "magazine",
         "newspaper", "show", "program", "word", "medicine", "virus", "insect", "horse",
         "snake", "mammal", "music", "dance", "cocktail", "sauce", "cheese", "mineral",
         "stone", "hobby", "technique", "method", "system", "computer", "software",
         "painting", "statue", "sculpture", "poem", "opera", "album", "series",
         "cartoon", "comic", "battle", "treaty", "dessert", "letter", "alphabet",
         "fabric", "cloth", "dress", "shoe", "costume", "uniform", "flag", "emblem",
         "logo", "slogan", "title", "phrase", "saying", "quote", "expression", "meal",
         "recipe", "ingredient", "spice", "candy", "cereal", "soup", "bread", "cake",
         "cookie", "meat", "breakfast", "crop", "grain", "seed", "machine", "device",
         "engine", "rocket", "satellite", "spacecraft", "train", "tank", "rifle", "bomb",
         "missile", "sword", "coin", "stamp", "card", "medal", "trophy", "muscle",
         "blood", "gland", "protein", "vitamin", "acid", "wood", "glass", "plastic",
         "paper", "crystal", "diamond", "silver", "steel", "constellation", "comet",
         "storm", "hurricane", "disaster", "event", "holiday", "festival", "ceremony",
         "tradition", "custom", "zodiac", "creature", "monster", "dinosaur", "whale",
         "shark", "spider", "worm", "lion", "tiger", "monkey", "sheep", "goat", "rabbit",
         "mouse", "duck", "chicken", "eagle", "species", "television", "sitcom",
         "channel", "website", "web site", "video", "lyric", "musical", "piano",
         "guitar", "drum", "style", "doll", "puzzle", "theory", "principle", "belief",
         "philosophy", "dialect", "liquor", "beverage", "juice", "item", "object",
         "shape", "illness", "symptom", "cancer", "syndrome", "disorder", "infection",
         "treatment", "cure", "remedy", "vaccine", "poison", "herb", "grass", "berry"))
# 3: "called" / "what kind" -> ENTY
_add(3, ENTY, _stem("called"), _wstart("what kind"), _wstart("what type"))
# 2: "what are the" -> ENTY
_add(2, ENTY, _word("what are the"))
# 1: initial "what is" -> DESC
_add(1, DESC, _init("what is"))


def _parse(p):
    """-> list of constraints, left to right: (kind, letter, start)."""
    toks, i = [], 0
    while i < len(p):
        c = p[i]
        if c == "^":
            toks.append(("L", p[i + 1], True)); i += 2; continue
        if c == "<":
            toks.append(("E", None, False))
        elif c == " ":
            toks.append(("B", None, False))
        elif c == "_":
            toks.append(("P", None, False))
        elif c == "?":
            toks.append(("W", None, False))
        else:
            toks.append(("L", c, False))
        i += 1
    return toks


PATTERNS = []  # (priority, class, constraints, source string)
_seen = set()
for _prio, _cls, _pats in _RULES:
    for _p in _pats:
        if _p in _seen:
            continue
        _seen.add(_p)
        PATTERNS.append((_prio, _cls, _parse(_p), _p))


def _cofire_possible(a, b):
    """Conservative check whether two patterns can fire at the same position."""
    a, b = a[::-1], b[::-1]   # offset 0 first
    for k in range(min(len(a), len(b))):
        x, y = a[k], b[k]
        kinds = {x[0], y[0]}
        if "W" in kinds:
            continue
        if "E" in kinds:
            other, seq = (y, b) if x[0] == "E" else (x, a)
            if other[0] == "L" and not (k >= 1 and seq[k - 1][2]):
                return False      # left edge vs a letter preceded by a non-start letter
            continue
        if x[2] != y[2] and "L" == x[0] == y[0]:
            return False          # start flag mismatch
        if x[0] == "L" and y[0] == "L" and x[1] != y[1]:
            return False
        if ("L" in kinds) and (kinds & {"B", "P"}):
            return False
    return True


COLLISIONS = [(PATTERNS[i][3], PATTERNS[j][3])
              for i in range(len(PATTERNS)) for j in range(i + 1, len(PATTERNS))
              if _cofire_possible(PATTERNS[i][2], PATTERNS[j][2])]
if COLLISIONS:
    print(f"[info] {len(PATTERNS)} patterns; possible same-position co-firing pairs:", COLLISIONS)

N_OFFSETS = 20
N_FEAT = 8                  # 5 code bits, boundary, start, pad  (all +-1)
F_BITS, F_BND, F_START, F_PAD = 0, 5, 6, 7
D_ONE, D_POS, D_POS2 = 8, 9, 10
D_COPY = 11                 # 20 offsets x 8 features
D_SAL = D_COPY + N_OFFSETS * N_FEAT      # salience
D_VOTE = D_SAL + 1                       # 6 local class votes
D_OUT = D_VOTE + 6                       # 6 copied class votes at '='
D_MODEL = D_OUT + 6

assert all(len(c) <= N_OFFSETS for _, _, c, _ in PATTERNS)


def _code(ch):
    if "a" <= ch <= "z":
        return ord(ch) - ord("a") + 1
    if ch == "'":
        return 27
    return 0


def write_weights(model: SimpleTransformer, task) -> None:
    """Closed-form weights for the keyword decision list."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        D = model.d_model
        # ---- embeddings ----
        E = model.token_emb.weight
        for tok, idx in task.stoi.items():
            c = _code(tok)
            for b in range(5):
                E[idx, F_BITS + b] = 1.0 if (c >> b) & 1 else -1.0
            E[idx, F_BND] = 1.0 if tok in " _=" else -1.0
            E[idx, F_PAD] = 1.0 if tok == "_" else -1.0
        P = model.pos_emb.weight
        for j in range(model.max_seq_len):
            P[j, F_START] = 1.0 if j == 0 else -1.0
            P[j, D_ONE] = 1.0
            P[j, D_POS] = float(j)
            P[j, D_POS2] = float(j * j)

        # ---- block 1: head h copies features from position i-h (Gaussian kernel) ----
        b1 = model.blocks[0]
        A = b1.attn
        dh = A.d_head
        beta = 20.0 * math.sqrt(dh)
        for h in range(N_OFFSETS):
            r = h * dh
            # score = beta*(2 j (i-h) - j^2) = -beta (j-(i-h))^2 + const(i)
            A.W_q.weight[r + 0, D_POS] = 2 * beta
            A.W_q.weight[r + 0, D_ONE] = -2 * beta * h
            A.W_q.weight[r + 1, D_ONE] = -beta
            A.W_k.weight[r + 0, D_POS] = 1.0
            A.W_k.weight[r + 1, D_POS2] = 1.0
            for f in range(N_FEAT):
                A.W_v.weight[r + f, f] = 1.0
                A.W_o.weight[D_COPY + h * N_FEAT + f, r + f] = 1.0

        # ---- block 1 MLP: one exact AND gate per keyword pattern ----
        W1, B1, W2 = b1.mlp.fc1.weight, b1.mlp.fc1.bias, b1.mlp.fc2.weight
        for n, (prio, cls, cons, _) in enumerate(PATTERNS):
            L = len(cons)
            M = 0
            for k, (kind, ch, start) in enumerate(cons):
                off = L - 1 - k
                base = D_COPY + off * N_FEAT
                if kind == "L":
                    c = _code(ch)
                    for b in range(5):
                        W1[n, base + F_BITS + b] = 1.0 if (c >> b) & 1 else -1.0
                    M += 5
                    if start:
                        W1[n, base + F_START] = 1.0
                        M += 1
                elif kind == "B":
                    W1[n, base + F_BND] = 1.0; M += 1
                elif kind == "E":
                    # left word edge: boundary here OR start flag on the next char
                    # (+1-1 = 0 if exactly one holds, -2 if neither); adds 0 to M
                    W1[n, base + F_BND] = 1.0
                    W1[n, base - N_FEAT + F_START] = 1.0
                elif kind == "P":
                    W1[n, base + F_PAD] = 1.0; M += 1
            B1[n] = -(M - 1)
            W2[D_SAL, n] = 300.0 * prio
            W2[D_VOTE + cls, n] = 1.0

        # ---- block 2: '=' attends to argmax of 300*prio - 3*pos, copies its vote ----
        A2 = model.blocks[1].attn
        lam = 10.0 * math.sqrt(A2.d_head)
        A2.W_q.weight[0, D_ONE] = lam
        A2.W_k.weight[0, D_SAL] = 1.0
        A2.W_k.weight[0, D_POS] = -3.0
        for c in range(6):
            A2.W_v.weight[c, D_VOTE + c] = 1.0
            A2.W_o.weight[D_OUT + c, c] = 1.0

        # ---- readout: copied vote -> label digit; ENTY default via bias on ONE ----
        Hd = model.head.weight
        for c in range(6):
            Hd[task.stoi[str(c)], D_OUT + c] = 1.0
        Hd[task.stoi[str(ENTY)], D_ONE] = 0.5


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "KeywordDecisionList_v4"
model_description = ("v4: v3 with question-initial 'who' as a stem (who/whom/whose); 20 Gaussian copy heads, "
                     "~580 AND-gate keyword neurons, '=' picks max 300*prio-3*pos and copies its vote; default ENTY.")


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
