"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) rule-based NLI model for `nli-snli`.

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py --task nli-snli
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
# No LayerNorms; attention heads have per-layer head counts / head dims and no
# 1/sqrt(d) scaling (all scales are written explicitly into W_q).

SENT_LEN = 60
P_END = SENT_LEN            # premise positions 0..59, '|' at 60
H_START = SENT_LEN + 1      # hypothesis positions 61..120, '=' at 121
EQ_POS = 2 * SENT_LEN + 1
LETTERS = "abcdefghijklmnopqrstuvwxyz"

# ---- hypothesis-only lexicon (votes per class: E, N, C); '*' = prefix match
VOTES: dict[str, list[float]] = {}


def _vote(words: str, cls: int, val: float) -> None:
    for w in words.split():
        VOTES.setdefault(w, [0.0, 0.0, 0.0])[cls] = val


_vote("no not nobody never alone empty cat tv asleep sleep* nap naps napping", 2, 6)
_vote("naked bed couch home swimming flying", 2, 3)
# neutral: emotion, family, attribute (3); purpose / competition (4)
_vote("sad happy angry upset excited scared afraid tired bored proud nervous lonely mad "
      "mother father mom dad son daughter wife husband brother sister family parents "
      "grandmother grandfather girlfriend boyfriend friends friend tall", 1, 3)
_vote("trying winning wins win won competition competing contest tournament championship "
      "because waiting practicing birthday vacation first best", 1, 4)
_vote("people someone outside animal", 0, 4)

# ---- exclusive groups: group -> classes (synonym sets)
GROUPS = {
    "gender": ["man men guy guys gentleman male males", "woman women lady ladies female females",
               "boy boys", "girl girls"],
    "verb": ["sitting sits", "standing stands", "running runs", "walking walks",
             "sleeping sleeps", "eating eats", "drinking drinks", "swimming swims",
             "dancing dances", "singing sings", "riding rides", "driving drives",
             "jumping jumps", "climbing climbs", "laying lying lays", "reading reads",
             "cooking cooks"],
    "animal": ["dog dogs puppy puppies", "cat cats kitten", "horse horses", "bird birds",
               "cow cows", "sheep", "fish", "elephant elephants", "camel camels"],
    "place": ["beach", "park", "street road", "mountain mountains", "kitchen", "store shop",
              "restaurant", "city", "ocean sea", "lake", "river", "field", "school", "church",
              "office", "desert", "forest woods", "pool", "stage", "library"],
    "vehicle": ["car cars", "bike bikes bicycle bicycles", "motorcycle motorcycles", "bus",
                "truck trucks", "boat boats", "train", "plane airplane", "skateboard"],
    "colour": ["red", "blue", "green", "yellow", "black", "white", "orange", "pink",
               "purple", "brown", "gray grey"],
    "inout": ["inside indoors", "outside outdoors"],
}
CLASSES: list[tuple[str, list[str]]] = []
WORD2CLASS: dict[str, int] = {}
for _g, _cls in GROUPS.items():
    for _c in _cls:
        for _w in _c.split():
            WORD2CLASS[_w] = len(CLASSES)
        CLASSES.append((_g, _c.split()))
NC = len(CLASSES)
HYP_WORDS = sorted(set(VOTES) | set(WORD2CLASS))
PREM_WORDS = sorted(WORD2CLASS)
N_OFF = 13
VOCAB_N = 34


class Layout:
    def __init__(self):
        self.n = 0

    def add(self, size=1):
        r = list(range(self.n, self.n + size)) if size > 1 else self.n
        self.n += size
        return r


L = Layout()
TOK = L.add(VOCAB_N)
CONST = L.add(); POS = L.add(); POS2 = L.add()
IS_PREM = L.add(); IS_PREM2 = L.add(); IS_SEP = L.add(); IS_HYP = L.add()
OFF = [None] + [L.add(VOCAB_N) for _ in range(N_OFF)]
PT = L.add(3 * 26)
TRI_VALID = L.add()
VOTE = L.add(3)
HCLS = L.add(NC)
PCLS = L.add(NC)
FOUND_TRI = L.add(); CONF_FOUND = L.add(); FOUND_VALID = L.add()
SVOTE = L.add(3)
NCONF = L.add(); LEN_H = L.add(); LEN_P = L.add(); COV = L.add()
D_MODEL = L.n

N_FF1 = len(HYP_WORDS) + len(PREM_WORDS) + 78 + 1
LAYER_CFG = [  # (n_heads, d_head, d_ff)
    (N_OFF, VOCAB_N, N_FF1),
    (2, max(79, NC + 1), 1),
    (3, 5, 1),
]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
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
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer without LayerNorm; residual layout defined above."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, **_):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = D_MODEL
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList([Block(D_MODEL, h, dh, ff) for h, dh, ff in LAYER_CFG])
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

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

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    assert task.SENT_LEN == SENT_LEN and task.vocab_size == VOCAB_N
    S = task.stoi
    for p in model.parameters():
        p.zero_()

    def ov(i, ch):  # residual dim holding one-hot of char `ch` at offset i
        return TOK[S[ch]] if i == 0 else OFF[i][S[ch]]

    # ---- embeddings
    for i in range(VOCAB_N):
        model.token_emb.weight[i, TOK[i]] = 1.0
    pe = model.pos_emb.weight
    for p in range(model.max_seq_len):
        pe[p, CONST] = 1.0
        pe[p, POS] = float(p)
        pe[p, POS2] = float(p * p)
        pe[p, IS_PREM] = float(p < P_END)
        pe[p, IS_PREM2] = float(2 <= p < P_END)
        pe[p, IS_SEP] = float(p == P_END)
        pe[p, IS_HYP] = float(H_START <= p <= EQ_POS)

    # ---- layer 1 attention: head k reads the char at offset -k
    b1 = 20.0
    a = model.blocks[0].attn
    dh = a.d_head
    for k in range(1, N_OFF + 1):
        h0 = (k - 1) * dh
        a.W_q.weight[h0 + 0, POS] = 1.0
        a.W_q.weight[h0 + 0, CONST] = -float(k)
        a.W_q.weight[h0 + 1, CONST] = 1.0
        a.W_k.weight[h0 + 0, POS] = 2 * b1
        a.W_k.weight[h0 + 1, POS2] = -b1          # score = b1*(2j(t-k) - j^2)
        for i in range(VOCAB_N):
            a.W_v.weight[h0 + i, TOK[i]] = 1.0
            a.W_o.weight[OFF[k][i], h0 + i] = 1.0

    # ---- layer 1 MLP: word detectors + trigram features
    m = model.blocks[0].mlp
    u = 0

    def word_unit(w, gate):
        nonlocal u
        if w.endswith("*"):
            pfx = w[:-1]; n = len(pfx); assert n <= N_OFF
            for i in range(n):
                m.fc1.weight[u, ov(i, pfx[n - 1 - i])] += 1
            for ch in " |":
                m.fc1.weight[u, ov(n, ch)] += 1
            m.fc1.bias[u] = -(n + 1)
        else:
            n = len(w); assert n + 1 <= N_OFF
            for i in range(1, n + 1):
                m.fc1.weight[u, ov(i, w[n - i])] += 1
            for ch in " _=":
                m.fc1.weight[u, ov(0, ch)] += 1
            for ch in " |":
                m.fc1.weight[u, ov(n + 1, ch)] += 1
            m.fc1.bias[u] = -(n + 2)
        m.fc1.weight[u, gate] += 1
        u += 1
        return u - 1

    for w in HYP_WORDS:
        j = word_unit(w, IS_HYP)
        if w in VOTES:
            for c in range(3):
                m.fc2.weight[VOTE[c], j] = VOTES[w][c]
        if w in WORD2CLASS:
            m.fc2.weight[HCLS[WORD2CLASS[w]], j] = 1.0
    for w in PREM_WORDS:
        j = word_unit(w, IS_PREM)
        m.fc2.weight[PCLS[WORD2CLASS[w]], j] = 1.0
    for s in range(3):
        for ci, c in enumerate(LETTERS):
            m.fc1.weight[u, ov(s, c)] = 1.0
            m.fc1.weight[u, IS_PREM2] = 1.0
            m.fc1.bias[u] = -1.0
            m.fc2.weight[PT[s * 26 + ci], u] = 1.0
            u += 1
    for s in range(3):
        for c in LETTERS:
            m.fc1.weight[u, ov(s, c)] = 1.0
    m.fc1.weight[u, IS_HYP] = 1.0
    m.fc1.bias[u] = -3.0
    m.fc2.weight[TRI_VALID, u] = 1.0
    u += 1
    assert u == N_FF1

    # ---- layer 2 attention
    b2 = 40.0
    a = model.blocks[1].attn
    dh = a.d_head
    # head 0: does this hypothesis trigram occur in the premise? (sink at '|')
    for s in range(3):
        for ci, c in enumerate(LETTERS):
            a.W_q.weight[s * 26 + ci, ov(s, c)] = b2
            a.W_k.weight[s * 26 + ci, PT[s * 26 + ci]] = 1.0
    a.W_q.weight[78, CONST] = 2.5 * b2
    a.W_k.weight[78, IS_SEP] = 1.0
    a.W_v.weight[0, IS_PREM] = 1.0
    a.W_o.weight[FOUND_TRI, 0] = 1.0
    # head 1: does the premise contain a *different* member of this word's group?
    h0 = dh
    for c, (g, _) in enumerate(CLASSES):
        for c2, (g2, _) in enumerate(CLASSES):
            if g2 == g and c2 != c:
                a.W_q.weight[h0 + c2, HCLS[c]] = b2
    for c2 in range(NC):
        a.W_k.weight[h0 + c2, PCLS[c2]] = 1.0
    a.W_q.weight[h0 + NC, CONST] = 0.5 * b2
    a.W_k.weight[h0 + NC, IS_SEP] = 1.0
    a.W_v.weight[h0, IS_PREM] = 1.0
    a.W_o.weight[CONF_FOUND, h0] = 1.0

    # ---- layer 2 MLP: found_valid = found AND valid trigram
    m = model.blocks[1].mlp
    m.fc1.weight[0, FOUND_TRI] = 1.0
    m.fc1.weight[0, TRI_VALID] = 1.0
    m.fc1.bias[0] = -1.0
    m.fc2.weight[FOUND_VALID, 0] = 1.0

    # ---- layer 3 attention: aggregation at '='
    b3 = 30.0
    a = model.blocks[2].attn
    dh = a.d_head
    n_h = EQ_POS - H_START + 1
    # head 0: sums over hypothesis positions
    a.W_q.weight[0, CONST] = b3
    a.W_k.weight[0, IS_HYP] = 1.0
    for c in range(3):
        a.W_v.weight[c, VOTE[c]] = 1.0
        a.W_o.weight[SVOTE[c], c] = float(n_h)
    a.W_v.weight[3, CONF_FOUND] = 1.0
    a.W_o.weight[NCONF, 3] = float(n_h)
    a.W_v.weight[4, IS_HYP] = 1.0
    a.W_v.weight[4, TOK[S['_']]] = -1.0
    a.W_v.weight[4, TOK[S['=']]] = -1.0
    a.W_o.weight[LEN_H, 4] = float(n_h)
    # head 1: premise length
    a.W_q.weight[dh, CONST] = b3
    a.W_k.weight[dh, IS_PREM] = 1.0
    a.W_v.weight[dh, IS_PREM] = 1.0
    a.W_v.weight[dh, TOK[S['_']]] = -1.0
    a.W_o.weight[LEN_P, dh] = float(SENT_LEN)
    # head 2: coverage = mean of found_valid over valid hypothesis trigrams
    a.W_q.weight[2 * dh, CONST] = b3
    a.W_k.weight[2 * dh, TRI_VALID] = 1.0
    a.W_k.weight[2 * dh, IS_SEP] = 0.5      # empty-set sink with value 0
    a.W_v.weight[2 * dh, FOUND_VALID] = 1.0
    a.W_o.weight[COV, 2 * dh] = 1.0
    # layer 3 MLP unused (zeros)

    # ---- unembedding: logits for '0','1','2'
    W = model.head.weight
    e, n_, c_ = S['0'], S['1'], S['2']
    for t in (e, n_, c_):
        W[t, CONST] = 100.0
    # entailment: 16cov - 6.4 - 0.08(D+12) + 2*votesE
    W[e, COV] += 16.0; W[e, LEN_H] += -0.08; W[e, LEN_P] += 0.08
    W[e, CONST] += -6.4 - 0.96; W[e, SVOTE[0]] += 2.0
    # neutral: 8cov - 1.2 + 0.08(D+12) + 2*votesN
    W[n_, COV] += 8.0; W[n_, LEN_H] += 0.08; W[n_, LEN_P] += -0.08
    W[n_, CONST] += -1.2 + 0.96; W[n_, SVOTE[1]] += 2.0
    # contradiction: 4*conflicts (0.5 each) + 2*votesC
    W[c_, NCONF] += 4.0 * 0.5; W[c_, SVOTE[2]] += 2.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "nli_rules_v2"
model_description = ("v1 with contradiction-6/3 and entailment-4 lexicons restricted to the words the "
                     "description lists literally (no person/outdoors/television/sofa extras).")


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
