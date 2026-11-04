"""
Hand-written transformer for SNLI: "premise(60)|hypothesis(60)=" -> 0 entail / 1 neutral / 2 contra.

Every activation is a "register file": each block reads and adds to named slices of x.
  Block 1 attention: one head per offset k copies the character at i-k into WIN[k].
  Block 1 MLP:       each hidden unit is an exact-match word detector voting for a class.
  Block 2 attention: a hypothesis trigram attends to itself unless the premise holds a copy.
  Block 3 attention: at '=', pool lexicon votes (sum) and trigram coverage (mean).
No training anywhere: all weights are hand-set below.
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
# Register layout
# ---------------------------------------------------------------------------
VOCAB, L, N_WIN = 34, 14, 3      # chars, window length, n-gram length for coverage

# Mutually exclusive concept groups. Members are separated by spaces; aliases of one
# member are joined by "/". Same group + different member in premise = conflict.
GROUPS = [
    "man/men/guy/guys/gentleman/male/males woman/women/lady/ladies/female/females boy/boys girl/girls",
    "sit*/sat/seated stand* walk* run/runs/running jump* lay/lays/laying/lying danc* sleep*/asleep/napping climb* swim* ride/rides/riding drive/drives/driving fly/flies/flying skat* ski/skis/skiing surf* hik*",
    "dog/dogs/puppy cat/cats/kitten horse/horses bird/birds cow/cows sheep elephant/elephants fish",
    "beach/ocean/sea park street/road/sidewalk city mountain*/hill kitchen restaurant store/shop/market school library church field/grass snow/snowy pool forest/woods desert office house/home bar stage gym lake/river hospital museum",
    "car/cars/truck bike/bikes/bicycle/bicycles bus boat/boats plane/airplane train/subway motorcycle/motorcycles",
    "red blue green black white yellow orange pink purple brown gray/grey",
    "one two three four five six",
    "indoors/inside outdoors/outside",
    "day/daytime night/nighttime",
    "guitar piano violin drums trumpet/trumpets",
    "soccer basketball baseball football tennis hockey golf",
    "eat*/ate/dinner/lunch drink*/drank cook*/baking read*/reads write*/writing paint* sing*/sang play*",
]
MEMBERS = [(g, tuple(m.split("/"))) for g, line in enumerate(GROUPS) for m in line.split()]
N_G, N_M = len(GROUPS), len(MEMBERS)
N_ALIAS = sum(len(a) for _, a in MEMBERS)

BIAS = 0
CHAR = 1                         # char at i (offset 0)
M11 = CHAR + VOCAB               # position mod 11
M13 = M11 + 11                   # position mod 13
PREM, HYP = M13 + 13, M13 + 14   # segment flags
WIN = HYP + 1                    # chars at offsets 1..L-1
VALID = WIN + VOCAB * (L - 1)    # this position ends a hypothesis letter trigram
MATCH = VALID + 1                # trigram found in premise
EV = MATCH + 1                   # per-position class votes (3)
SUM = EV + 3                     # pooled class votes (3)
COV = SUM + 3                    # pooled trigram coverage
GRP = COV + 1                    # this position ends a word of group g
MEM = GRP + N_G                  # this position ends a word of member m
HITM = MEM + N_M                 # the premise holds this same member
HITG = HITM + 1                  # the premise holds a member of this group
CONF = HITG + 1                  # pooled count of conflicting hypothesis words
LEN = CONF + 1                   # hypothesis length minus premise length, in chars
FRESH = LEN + 1                  # hypothesis concept word whose group the premise lacks
NEWS = FRESH + 1                 # pooled count of such new-topic words
D_MODEL = NEWS + 1
LETTERS = range(26)


def win_reg(offset: int) -> int:
    return CHAR if offset == 0 else WIN + VOCAB * (offset - 1)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        d = n_heads * d_head
        self.W_q = nn.Linear(d_model, d, bias=False)
        self.W_k = nn.Linear(d_model, d, bias=False)
        self.W_v = nn.Linear(d_model, d, bias=False)
        self.W_o = nn.Linear(d, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o((attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int = 0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff) if d_ff else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x) if self.mlp else x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList([
            Block(D_MODEL, L - 1, VOCAB, d_ff=len(LEXICON) + 1 + N_ALIAS),
            Block(D_MODEL, 3, max(N_WIN * VOCAB, N_M) + 1, d_ff=1),
            Block(D_MODEL, 6, 3),
        ])
        self.head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Hand-written knowledge: word -> class votes. A trailing * means prefix match.
# ---------------------------------------------------------------------------
E, N, C = 0, 1, 2
LEXICON: dict[str, tuple[int, float]] = {}
for words, cls, vote in [
    ("nobody no not none never nothing asleep sleep* napping naps empty alone cat cats tv television", C, 3.0),
    ("naked nude bed couch home moon pizza swimming flying driving", C, 1.5),
    ("tall first competition sad favorite friend* vacation tourist* brother sister mother father", N, 2.0),
    ("husband wife birthday family party trying because winning wins championship", N, 2.0),
    ("happy excited angry tired bored proud scared nervous beautiful funny busy professional", N, 1.5),
    ("waiting going about getting ready wants new old rich famous married practicing", N, 1.5),
    ("outdoors outside people person someone something some there least instrument animal* human*", E, 2.0),
]:
    for w in words.split():
        LEXICON[w] = (cls, vote)

BETA = 40.0
COV_LOW, COV_HIGH, COV_GAIN = 0.15, 0.65, 8.0
CONFLICT_VOTE = 4.0
LEN_VOTE, LEN_CENTER = 0.08, -22.0  # logits per extra char; LEN is minus the hypothesis padding
NEWS_VOTE = 0.0                     # disabled: e17 and e18 showed no dev gain


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    s = task.stoi
    tok, pos = model.token_emb.weight, model.pos_emb.weight

    # Embeddings: one-hot char plus position residues and segment flags.
    tok[:, BIAS] = 1.0
    tok[range(VOCAB), [CHAR + c for c in range(VOCAB)]] = 1.0
    for p in range(pos.shape[0]):
        pos[p, M11 + p % 11] = 1.0
        pos[p, M13 + p % 13] = 1.0
        if p < task.SENT_LEN:
            pos[p, PREM] = 1.0
        elif task.SENT_LEN < p < task.prompt_len:
            pos[p, HYP] = 1.0

    # Block 1 attention: head k fetches the char k steps back via position residues.
    b1, a1 = model.blocks[0], model.blocks[0].attn
    for k in range(1, L):
        h = (k - 1) * VOCAB
        for base, m in ((M11, 11), (M13, 13)):
            for r in range(m):
                a1.W_q.weight[h + r + (0 if base == M11 else 11), base + (r + k) % m] = BETA
                a1.W_k.weight[h + r + (0 if base == M11 else 11), base + r] = 1.0
        for c in range(VOCAB):
            a1.W_v.weight[h + c, CHAR + c] = 1.0
            a1.W_o.weight[win_reg(k) + c, h + c] = 1.0

    # Block 1 MLP: unit 0 flags valid hypothesis trigrams; the rest detect words.
    fc1, fc2 = b1.mlp.fc1, b1.mlp.fc2
    fc1.weight[0, HYP] = 1.0
    for off in range(N_WIN):
        for c in LETTERS:
            fc1.weight[0, win_reg(off) + c] = 1.0
    fc1.bias[0] = -N_WIN
    fc2.weight[VALID, 0] = 1.0

    starts, ends = " |", " _="

    def detect(u: int, word: str, extra: int = 0) -> None:
        """Make hidden unit u fire exactly when `word` ends at this position."""
        w = word.rstrip("*")
        pattern = [starts] + list(w) + ([] if word.endswith("*") else [ends])
        for off, options in enumerate(reversed(pattern)):
            for ch in options:
                fc1.weight[u, win_reg(off) + s[ch]] = 1.0
        fc1.bias[u] = -(len(pattern) - 1 + extra)

    for u, (word, (cls, vote)) in enumerate(LEXICON.items(), start=1):
        detect(u, word, extra=1)
        fc1.weight[u, HYP] = 1.0
        fc2.weight[EV + cls, u] = vote

    u = len(LEXICON) + 1
    for m, (g, aliases) in enumerate(MEMBERS):
        for alias in aliases:
            detect(u, alias)
            fc2.weight[GRP + g, u] = 1.0
            fc2.weight[MEM + m, u] = 1.0
            u += 1

    # Block 2 attention: compare trigrams. Hypothesis keys take a -0.5 penalty, so a
    # position attends to itself unless a full copy exists in the premise.
    a2 = model.blocks[1].attn
    for off in range(N_WIN):
        for c in range(VOCAB):
            a2.W_q.weight[off * VOCAB + c, win_reg(off) + c] = BETA
            a2.W_k.weight[off * VOCAB + c, win_reg(off) + c] = 1.0
    dh = a2.d_head
    pen = dh - 1                     # last dim of every head holds the self penalty
    for h in range(3):
        a2.W_q.weight[h * dh + pen, BIAS] = BETA
        a2.W_k.weight[h * dh + pen, HYP] = -0.5
        a2.W_v.weight[h * dh, PREM] = 1.0
    a2.W_o.weight[MATCH, 0] = 1.0

    # Heads 1 and 2 ask the same question of whole words: does the premise hold this
    # member (head 1), or any member of this group (head 2)? HITG - HITM = conflict.
    for m in range(N_M):
        a2.W_q.weight[dh + m, MEM + m] = BETA
        a2.W_k.weight[dh + m, MEM + m] = 1.0
    for g in range(N_G):
        a2.W_q.weight[2 * dh + g, GRP + g] = BETA
        a2.W_k.weight[2 * dh + g, GRP + g] = 1.0
    a2.W_o.weight[HITM, dh] = 1.0
    a2.W_o.weight[HITG, 2 * dh] = 1.0

    # Block 2 MLP: FRESH = hypothesis AND "a concept word ends here" AND NOT HITG.
    fc = model.blocks[1].mlp
    fc.fc1.weight[0, HITG] = -1.0
    fc.fc1.weight[0, HYP] = 1.0
    for g in range(N_G):
        fc.fc1.weight[0, GRP + g] = 1.0
    fc.fc1.bias[0] = -1.0
    fc.fc2.weight[FRESH, 0] = 1.0

    # Block 3 attention: head 0 sums votes over the hypothesis; head 1 averages MATCH
    # over valid trigrams; head 2 counts conflicts (HITG - HITM) over the hypothesis.
    a3 = model.blocks[2].attn
    n_hyp = task.SENT_LEN + 1
    for h in (0, 2):
        a3.W_q.weight[h * 3, BIAS] = BETA
        a3.W_k.weight[h * 3, HYP] = 1.0
    a3.W_q.weight[3, BIAS] = BETA
    a3.W_k.weight[3, VALID] = 1.0
    for c in range(3):
        a3.W_v.weight[c, EV + c] = 1.0
        a3.W_o.weight[SUM + c, c] = n_hyp
    a3.W_v.weight[3, MATCH] = 1.0
    a3.W_o.weight[COV, 3] = 1.0
    a3.W_v.weight[6, HITG] = 1.0
    a3.W_v.weight[6, HITM] = -1.0
    a3.W_o.weight[CONF, 6] = n_hyp

    # Heads 3 and 4 measure padding by averaging the "_" flag over each sentence; more
    # padding in the premise than in the hypothesis means a longer hypothesis.
    for h, seg, sign in ((3, PREM, 0.0), (4, HYP, -1.0)):
        a3.W_q.weight[h * 3, BIAS] = BETA
        a3.W_k.weight[h * 3, seg] = 1.0
        a3.W_v.weight[h * 3, CHAR + s["_"]] = 1.0
        a3.W_o.weight[LEN, h * 3] = sign * task.SENT_LEN

    # Head 5 counts new-topic concept words over the hypothesis.
    a3.W_q.weight[15, BIAS] = BETA
    a3.W_k.weight[15, HYP] = 1.0
    a3.W_v.weight[15, FRESH] = 1.0
    a3.W_o.weight[NEWS, 15] = n_hyp

    # Head: votes plus piecewise-linear coverage thresholds.
    hd = model.head
    hd.bias[:] = -1000.0
    labels = [s["0"], s["1"], s["2"]]
    for c, lab in enumerate(labels):
        hd.bias[lab] = 0.0
        hd.weight[lab, SUM + c] = 1.0
    hd.weight[labels[N], COV] = COV_GAIN
    hd.bias[labels[N]] = -COV_GAIN * COV_LOW
    hd.weight[labels[E], COV] = 2 * COV_GAIN
    hd.bias[labels[E]] = -COV_GAIN * (COV_LOW + COV_HIGH)
    hd.weight[labels[C], CONF] = CONFLICT_VOTE
    hd.weight[labels[N], LEN] = LEN_VOTE
    hd.weight[labels[E], LEN] = -LEN_VOTE
    hd.bias[labels[N]] -= LEN_VOTE * LEN_CENTER
    hd.bias[labels[E]] += LEN_VOTE * LEN_CENTER
    hd.weight[labels[C], NEWS] = NEWS_VOTE
    dev_report(model, task)


@torch.no_grad()
def dev_report(model: SimpleTransformer, task, n: int = 1500, seed: int = 4242) -> None:
    """Measure only (nothing here changes a weight): confusion matrix on a non-eval seed,
    so that design choices are not judged on the 200-example eval sample alone."""
    ex = task.generate_examples(n, seed=seed)
    ids = torch.tensor([task.encode(e.prompt) for e in ex])
    gold = torch.tensor([int(e.answer) for e in ex])
    labels = [task.stoi[c] for c in "012"]
    preds = torch.cat([model(ids[i:i + 200])[:, -1, labels].argmax(-1) for i in range(0, n, 200)])
    conf = torch.zeros(3, 3, dtype=torch.long)
    for g, p in zip(gold.tolist(), preds.tolist()):
        conf[g, p] += 1
    print(f"DEV acc={(preds == gold).float().mean():.4f} n={n}")
    print("DEV confusion (rows=gold E/N/C, cols=pred E/N/C):", conf.tolist())


model_shorthand_name = "e20_absolute_length_center_38"
model_description = "Hypothesis-length vote recentered at 38 chars because a 30-char center over-predicted neutral."


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
