"""
Interpretable transformer for TREC-coarse question classification.

Hand-written weights (no training). Prompt = question padded with '_' then '='; the answer
is one digit: 0 ABBR, 1 ENTY, 2 DESC, 3 HUM, 4 LOC, 5 NUM.

Circuit (2 blocks, no LayerNorm, all weights set in closed form):
  Block 1 attention : 6 "shift" heads; head k attends exactly to position t-k and copies that
                      letter into slot k.  Every position now sees its last 7 letters.
  Block 1 MLP       : keyword detectors. One ReLU unit per keyword = AND over its letters
                      (and, for first/second-word cues, the absolute position).
  Block 2 attention : uniform attention from '=' over the sentence, rescaled by 81, which turns
                      per-position detections into bag-of-keywords counts.
  Block 2 MLP       : class rules = AND of keyword features -> class score (priority weights).
  Head              : reads the 6 class scores into digit tokens '0'..'5'.

Usage:
    uv run interpretable_transformer.py --task trec-coarse
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
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer (no LayerNorm), vocab/seq-len configured from the task."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, n_heads: int, d_ffs: list[int]):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, f) for f in d_ffs])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Interpretable weight assignment
# ---------------------------------------------------------------------------

N_SHIFT = 6          # shift heads: letters at offsets 1..6
SEQ = 82             # positions (one-hot position code)
NCLS = 6

# Detectors for block 1 MLP: name -> (pattern, allowed end positions or None).
# pattern is a list of letter-sets (last element = the current position); a string is spelled out.
def spell(s):
    return list(s)

DETECTORS = {}

def det(name, pattern, positions=None):
    DETECTORS[name] = (pattern if isinstance(pattern, list) else spell(pattern), positions)

# --- first word (absolute position anchored)
det("F_what", "what", {3})
det("F_who", "who", {2})
det("F_when", "when", {3})
det("F_where", "where", {4})
det("F_why", "why", {2})
det("F_which", "which", {4})
det("F_how", "how", {2})
# --- second word cues after "what " (starts at position 5) / "how " (position 4)
det("wh_is", "is", {6})
det("wh_are", "are", {7})
det("wh_does", "does", {8})
det("wh_do", "do ", {7})
det("wh_s", "'s", {6})
det("how_desc", ["dcisaw"], {4})
det("how_num", ["mlofhtbpg"], {4})        # how many / much / long / old / far / fast / high / tall ...
# --- "what is xyz" where the whole rest of the question is a 3-letter word: likely an acronym (ABBR)
ALPHA = "abcdefghijklmnopqrstuvwxyz"
det("wh_is_acro", [" ", ALPHA, ALPHA, ALPHA, "_"], {11})
# --- keywords anywhere (substring match)
KEYWORDS = [
    # (name, pattern, class, priority weight)
    ("year", "year", 5, 3), ("percent", "perce", 5, 3), ("popul", "ulati", 5, 3), ("cost", "cost", 5, 2),
    ("price", "price", 5, 2),
    ("city", "city", 4, 4.5), ("country", "ountr", 4, 4.5), ("state", " state", 4, 3.5), ("capital", "apita", 4, 4),
    ("island", "islan", 4, 3), ("river", " river", 4, 2), ("mount", " mount", 4, 3), ("ocean", "ocean", 4, 3),
    ("contin", "ntine", 4, 4),
    ("president", "resid", 3, 2), ("actor", "actor", 3, 3), ("actress", "actre", 3, 3), ("author", "autho", 3, 3),
    ("compos", "ompos", 3, 3), ("invent", "inven", 3, 3), ("singer", "singe", 3, 3), ("wrote", "wrote", 3, 2),
    ("person", "perso", 3, 3),
    ("stand", "stand", 0, 5), ("abbre", "abbre", 0, 3), ("acron", "acron", 0, 3), ("full", "full ", 0, 3),
    ("mean", " mean", 2, 2), ("differ", "diffe", 2, 3), ("cause", " cause", 2, 2), ("reason", "reaso", 2, 3),
    ("origin", " origin", 2, 2), ("defin", "defin", 2, 3),
    ("kind", "kind", 1, 2), ("type", "type", 1, 2), ("color", "color", 1, 3), ("sport", "sport", 1, 3),
    ("animal", "anima", 1, 3), ("food", "food", 1, 3), ("langu", "langu", 1, 3),
    # v4 additions
    ("number", "numbe", 5, 3), ("average", "avera", 5, 2), ("speed", "speed", 5, 3), ("wage", "wage", 5, 3),
    ("temp", "tempe", 5, 3), ("weight", "weigh", 5, 3), ("height", "heigh", 5, 3), ("expect", "expec", 5, 3),
    ("fract", "fract", 5, 3), ("quant", "quant", 5, 3), ("birthday", "birth", 5, 3), ("length", "lengt", 5, 3),
    ("distance", "dista", 5, 3),
    ("nickname", "nickn", 3, 3), ("pseudo", "pseud", 3, 3), ("company", "compa", 3, 3), ("team", "team ", 3, 2),
    ("player", "playe", 3, 3), ("band", " band ", 3, 2), ("leader", "leade", 3, 3), ("king", " king ", 3, 1),
    ("writer", "writer", 3, 3), ("star", "star ", 3, 2), ("poet", "poet ", 3, 3), ("artist", "artis", 3, 3),
    ("painter", "painte", 3, 3), ("charact", "chara", 3, 2), ("organ", "organ", 3, 2),
    ("term", "term ", 1, 1), ("called", "calle", 1, 1), ("word", "word ", 1, 2),
    # v5 additions
    ("est", "est ", 1, 1.0), ("fear", "fear ", 1, 3), ("name", "name ", 1, 2.5), ("book", "book ", 1, 3.5),
    ("lake", "lake ", 4, 4), ("town", "town ", 4, 3), ("cities", "citie", 4, 3), ("her", " her ", 3, 1.5),
    ("queen", "queen", 3, 2.5), ("judge", "judge", 3, 3), ("host", "host ", 3, 3), ("effects", "effec", 2, 3),
    ("month", "month", 5, 2), ("types", "types", 1, 3),
    # v9 additions: whose-name cues
    ("poss_name", "'s name", 3, 3.5), ("real_name", "al name", 3, 3.5), ("first_name", "st name", 3, 3.5),
    ("nd_name", "nd name", 3, 3.5), ("mid_name", "le name", 3, 3.5),
    ("astro", "astro", 3, 3), ("villain", "villa", 3, 3),
    # v12 additions
    ("made", "made of", 1, 3), ("call", " call ", 1, 2), ("most", "most ", 1, 1.5), 
("common", "commo", 1, 2), ("eat", " eat ", 1, 2), ("used", " used ", 1, 2), ("breed", "breed", 1, 2),
    ("bird", "bird", 1, 2), ("plant", "plant", 1, 3), ("drug", "drug ", 1, 3), ("disease", "disea", 1, 3),
    ("instrument", "instru", 1, 3), ("exchange", "excha", 5, 3), ("dollar", "dolla", 5, 2),
    ("mother", "mothe", 3, 3), ("father", "fathe", 3, 3), ("wife", "wife ", 3, 3), ("husband", "husba", 3, 3),
    ("son", " son ", 3, 3), ("daughter", "daugh", 3, 3), ("brother", "broth", 3, 3), ("sister", "siste", 3, 3),
    ("locat", "locat", 4, 3), ("build", "build", 4, 2), ("street", "stree", 4, 3), ("airport", "airpo", 4, 3),
    ("bridge", "bridg", 4, 3), ("provin", "provi", 4, 3), ("address", "addre", 4, 2),
    # v20 additions
    ("date", " date", 5, 3), ("rate", " rate", 5, 2), ("latitude", "latit", 5, 3), ("toll", "toll ", 5, 3),
    ("age", " age ", 5, 3), ("occupation", "occup", 3, 3), ("profession", "profe", 3, 3), ("army", " army", 3, 3),
    ("police", "polic", 3, 2), ("depart", "depar", 3, 2), ("club", "club ", 3, 2), ("group", "group", 3, 1),
    ("univ", "unive", 3, 2), ("agency", "agenc", 3, 3),
    ("seaport", "seapo", 4, 3), ("valley", "valle", 4, 3), ("desert", "deser", 4, 3), ("area", " area ", 4, 2),
    ("region", "regio", 4, 3), ("bay", " bay ", 4, 3), ("gulf", "gulf ", 4, 3), ("beach", "beach", 4, 3),
    ("happen", "happen", 2, 3), ("purpose", "purpo", 2, 3), ("say", " say ", 2, 1.5),
    ("drink", "drink", 1, 3), ("system", "syste", 1, 1), ("novel", "novel", 1, 2), ("currency", "curre", 1, 3),
    ("metal", "metal", 1, 3), ("element", "eleme", 1, 3), ("vehicle", "vehic", 1, 3), ("religion", "relig", 1, 3),
    ("citizen", "citiz", 3, 3), ("party", " party", 3, 2), ("college", "colle", 3, 2), ("soldier", "soldi", 3, 3),
    ("designer", "desig", 3, 3), ("boxer", "boxer", 3, 3), ("producer", "ducer ", 3, 2), ("hunter", "hunte", 3, 3),
    ("stadium", "stadi", 4, 3), ("cathedral", "cathe", 4, 3), ("nation", " nati", 4, 1.5), ("touris", "touri", 4, 3),
    ("day", " day ", 5, 3), ("season", "seaso", 5, 2), ("fare", " fare", 5, 3), ("income", "incom", 5, 3),
    ("paid", " paid", 5, 3), ("makes", " makes", 2, 2), ("plural", "plura", 1, 3), ("symbol", "symbo", 1, 3),
    ("manufacturer", "manuf", 3, 3), ("comedian", "comed", 3, 3), ("lawyer", "lawye", 3, 3),
    ("film", "film ", 1, 3), ("movie", "movie", 1, 3), ("product", "produc", 1, 2.5),
    ("congress", "congr", 3, 3), ("woman", "woman", 3, 3), ("model", " model", 3, 2), ("command", "comma", 3, 3),
    ("county", "ounty", 4, 3), ("come_from", "me from", 2, 5), ("history", "histo", 2, 2),
    ("hotel", "hotel", 4, 3), ("library", "libra", 4, 3), ("home", " home ", 4, 1.5),
    ("brand", "brand ", 1, 3), ("tribe", "tribe", 3, 3), ("superstar", "super", 3, 2), ("novelist", "novelis", 3, 4),
    ("director", "direc", 3, 3), ("men", " men ", 3, 2), ("century", "centur", 5, 3), ("frequency", "frequ", 5, 3),
    ("sea", " sea ", 4, 3),
    ("how_say", "ou say ", 1, 10), ("mall", " mall", 4, 3), ("brewery", "brewe", 4, 3), ("chairman", "chair", 3, 3),
    ("explorer", "explo", 3, 3), ("scholar", "schol", 3, 3), ("firm", " firm ", 3, 3),
    ("event", "event", 1, 4), ("role", " role ", 2, 3), ("function", "funct", 2, 3), ("excuse", "excus", 2, 3),
    ("girlfriend", "girlf", 3, 3), ("terrorist", "terro", 3, 3), ("witch", "witch", 3, 3),
    ("flower", "flowe", 1, 3), ("fish", "fish ", 1, 3), ("insect", "insec", 1, 3), ("weapon", "weapo", 1, 3),
]
for name, pat, _, _ in KEYWORDS:
    det("K_" + name, pat)

# Block 2 rules: (required features, class, priority weight). class: 0 ABBR 1 ENTY 2 DESC 3 HUM 4 LOC 5 NUM
RULES = [
    (["F_who"], 3, 5), (["F_when"], 5, 4), (["F_where"], 4, 4), (["F_why"], 2, 3), (["F_which"], 1, 0.6),
    (["F_how", "how_desc"], 2, 9), (["F_how", "how_num"], 5, 6), (["F_how"], 5, 3),
    (["F_what", "wh_is"], 2, 0.8), (["F_what", "wh_is", "wh_is_acro"], 0, 2.5), (["F_what", "wh_are"], 2, 1.0), (["F_what", "wh_does"], 2, 1.0),
    (["F_what", "wh_do"], 2, 1.0), (["F_what", "wh_s"], 2, 1.0),
] + [(["K_" + n], c, w) for n, _, c, w in KEYWORDS]
DEFAULT_CLASS = 1    # ENTY gets a small constant bias
DEFAULT_BIAS = 0.3


def layout(V):
    names = list(DETECTORS)
    L = {}
    L["TOK"] = 0
    L["POS"] = V
    L["SH"] = lambda k: V + SEQ + V * (k - 1)           # shift slot k = 1..N_SHIFT
    L["FEAT"] = V + SEQ + V * N_SHIFT                    # feature dims
    L["FIDX"] = {n: i for i, n in enumerate(names)}
    L["CLS"] = L["FEAT"] + len(names)
    L["D"] = L["CLS"] + NCLS
    return L


def write_weights(model: SimpleTransformer, task) -> None:
    V = task.vocab_size
    L = layout(V)
    H, dh = model.n_heads, model.d_model // model.n_heads
    stoi = task.stoi
    for prm in model.parameters():
        prm.data.zero_()

    for c in range(V):
        model.token_emb.weight.data[c, L["TOK"] + c] = 1.0
    for p in range(SEQ):
        model.pos_emb.weight.data[p, L["POS"] + p] = 1.0

    # ---- Block 1 attention: head k-1 attends to position t-k, copies its letter to slot k
    a1 = model.blocks[0].attn
    S = 150.0
    for k in range(1, N_SHIFT + 1):
        h = k - 1
        for t in range(k, SEQ):
            a1.W_q.weight.data[h * dh + (t - k), L["POS"] + t] = S
        for s in range(SEQ):
            a1.W_k.weight.data[h * dh + s, L["POS"] + s] = 1.0
        for c in range(V):
            a1.W_v.weight.data[h * dh + c, L["TOK"] + c] = 1.0
            a1.W_o.weight.data[L["SH"](k) + c, h * dh + c] = 1.0

    # ---- Block 1 MLP: keyword detectors
    m1 = model.blocks[0].mlp
    for name, (pattern, positions) in DETECTORS.items():
        u = L["FIDX"][name]
        n = len(pattern)
        for i, letters in enumerate(pattern):
            off = n - 1 - i
            for ch in letters:
                col = L["TOK"] + stoi[ch] if off == 0 else L["SH"](off) + stoi[ch]
                m1.fc1.weight.data[u, col] = 1.0
        nc = n
        if positions is not None:
            for p in positions:
                m1.fc1.weight.data[u, L["POS"] + p] = 1.0
            nc += 1
        m1.fc1.bias.data[u] = -(nc - 1)
        m1.fc2.weight.data[L["FEAT"] + u, u] = 1.0

    # ---- Block 2 attention: '=' attends uniformly to all 81 positions; x81 turns average into count
    a2 = model.blocks[1].attn
    nF = len(DETECTORS)
    for j in range(nF):
        h, d = divmod(j, dh)
        a2.W_v.weight.data[h * dh + d, L["FEAT"] + j] = 1.0
        a2.W_o.weight.data[L["FEAT"] + j, h * dh + d] = float(SEQ - 1)   # 81 = number of keys seen at '='

    # ---- Block 2 MLP: class rules
    m2 = model.blocks[1].mlp
    for u, (feats, cls, w) in enumerate(RULES):
        for f in feats:
            m2.fc1.weight.data[u, L["FEAT"] + L["FIDX"][f]] = 1.0
        m2.fc1.bias.data[u] = -(len(feats) - 1)
        m2.fc2.weight.data[L["CLS"] + cls, u] = w
    m2.fc2.bias.data[L["CLS"] + DEFAULT_CLASS] = DEFAULT_BIAS

    for k in range(NCLS):
        model.head.weight.data[stoi[str(k)], L["CLS"] + k] = 1.0


DEBUG = False


def debug_report(model, task):
    from collections import Counter
    ex = task.generate_examples(1000, seed=9)
    ids = torch.tensor([[task.stoi[c] for c in e.prompt] for e in ex])
    with torch.no_grad():
        pred = model(ids)[:, -1].argmax(-1)
    pred = [task.itos[int(i)] for i in pred]
    ok = sum(p == e.answer for p, e in zip(pred, ex))
    print("DEBUG acc1000", ok / len(ex))
    L = layout(task.vocab_size)
    for s in ["what city has the world 's longest subway system", "name the largest country in south america"]:
        s = s.ljust(80, "_") + "="
        x = torch.tensor([[task.stoi[c] for c in s]])
        with torch.no_grad():
            h = model.token_emb(x) + model.pos_emb(torch.arange(x.shape[1]))[None]
            h = model.blocks[0](h)
            feats = h[0, -1, L["FEAT"]:L["FEAT"] + len(DETECTORS)]
            h = model.blocks[1](h)
        print("DEBUG active", s[:40], [n for n, v in zip(DETECTORS, feats) if v > 0.5],
              "scores", h[0, -1, L["CLS"]:L["CLS"] + 6].tolist())
    conf = Counter((e.answer, p) for p, e in zip(pred, ex))
    print("DEBUG conf (true,pred):", sorted(conf.items()))
    n = 0
    for p, e in zip(pred, ex):
        if p != e.answer and n < 70:
            print("DEBUG ERR true", e.answer, "pred", p, e.prompt.rstrip("=").rstrip("_"))
            n += 1


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ShiftKeywordBag_final"
model_description = "2 blocks no LN: 6 shift heads give 7-letter context, MLP1 keyword/first-word detectors, uniform attention from '=' counts keywords, MLP2 AND-rules -> class."


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
    V = task.vocab_size
    L = layout(V)
    H = N_SHIFT + 2                            # 6 shift heads + 2 spare heads (room for feature dims)
    dh = SEQ                                  # query vectors need one dim per position
    d_model = H * dh
    assert d_model >= L["D"], (d_model, L["D"])
    assert len(DETECTORS) <= H * dh
    model = SimpleTransformer(vocab_size=V, max_seq_len=SEQ, d_model=d_model, n_heads=H,
                              d_ffs=[len(DETECTORS), len(RULES)])
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
    if DEBUG:
        debug_report(model, task)

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
