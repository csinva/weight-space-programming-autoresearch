"""
Interpretable transformer for character-level NLI (SNLI), hand-written weights.
Snapshot LexMatchConflictStem_v2 (accuracy 0.5500 on nli-snli, 200 examples seed 0).

Architecture (no LayerNorm, explicit residual-stream layout, 4 blocks):

  Embedding: token one-hot, constant 1, position p and p^2, region flags
             (premise / hypothesis / pos0), "virtual boundary" flags VB_j = [p < j].

  Block 1 attention: L "look-back" heads. Head j attends (via the quadratic
             positional trick score = -beta*(s-(t-j))^2) to position t-j and copies
             its token one-hot into slot G_j.  Every position now sees its last L chars.
  Block 1 MLP: at word-boundary positions (space, '_', '|', '=') whose previous char
             is a letter ("word ends"):
               * premWE / hypWE flags,
               * masked word vector M_j[c] (chars of the just-finished word, aligned from
                 its end, terminated by a boundary marker MB_j),
               * one ReLU detector per lexicon word (exact whole-word match), writing to
                 role flags: stopword, contradiction/neutral/entailment cue, hypernym-class
                 query/key flags, and conflict-rule flags,
               * suffix flags ingF / sF.
  Block 2 MLP (stemming): S = M shifted by 3 slots if the word ends in -ing, by 1 if -s.
  Block 3 attention:
      head A ("matched"): hypothesis word ends attend to premise word ends whose stem
             vector equals theirs exactly (or that belong to a class the hypothesis
             word is a hypernym/synonym of). A sink at position 0 absorbs mass when nothing
             matches -> matched in {0,1}.
      head B ("conflict"): hypothesis word in some category (colour, animal species,
             gender, number, place, activity, indoor/outdoor ...) attends to premise
             words in a conflicting category -> hit in {0,1}.
  Block 3 MLP: per hypothesis word: U = unmatched content word, CP = conflict & not
             matched, CC/NC/EC = contradiction/neutral/entailment cue words not matched.
  Block 4 attention: uniform average over all positions at '=' -> counts.
  Head: linear logits for '0','1','2' from the counts.

All weights are written in closed form; lexicons are hand-written from general knowledge.
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
    """Multi-head causal attention with separate q/k and v head sizes, no 1/sqrt(d) scale."""

    def __init__(self, d_model: int, n_heads: int, d_qk: int, d_v: int):
        super().__init__()
        self.n_heads, self.d_qk, self.d_v = n_heads, d_qk, d_v
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, self.d_qk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.d_qk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.d_v).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * self.d_v)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_qk, d_v, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_qk, d_v)
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None

    def forward(self, x):
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Lexicons (hand-written from general knowledge of English / NLI phenomena)
# ---------------------------------------------------------------------------

L = 12                      # look-back window (max word length detectable = L-1)
BETA = 20.0                 # attention sharpness

STOP = """a an the is are was were be being been of in on at with and or his her their its it
some there this that these those by as from into onto up down out over while who which has
have had he she they them one another other each both very just next front""".split()

CONTRA_CUES = """not no nobody never nothing none noone nowhere sleeping asleep sleeps sleep nap
napping cat cats naked nude alone empty tv television couch bed dead""".split()

NEUTRAL_CUES = """for because first tall favorite sad friends friend competition trying waiting
wife husband mother father son daughter sister brother family girlfriend boyfriend vacation
birthday party won win winning championship new married about going getting ready professional
famous best date owner after before work job lunch dinner tired fun""".split()

ENTAIL_CUES = """outdoors outside people person someone somebody animal animals human""".split()

# Hypernym / synonym classes: if hypothesis word in HYP set and a premise word in PREM set,
# the hypothesis word counts as matched.
PERSONISH = """man men woman women boy boys girl girls child children kid kids guy guys lady ladies
person people player players worker workers adult adults baby toddler crowd group team couple
family biker bikers cyclist skier surfer rider riders dancer dancers musician musicians student
students someone gentleman female male females males teenager teen teens performer performers
worker athlete athletes swimmer runner runners climber skateboarder snowboarder audience""".split()
OUTDOORISH = """outside outdoors street streets park beach field grass snow water ocean lake river
mountain mountains road sidewalk city yard garden forest woods trail sand desert hill sun sunny
outdoor waves wave pool rocks rock dirt market crowd parade""".split()
INDOORISH = """inside indoors kitchen restaurant store office room bedroom house church mall
library bar gym classroom shop building indoor stage""".split()
ANIMALS = """dog dogs cat cats horse horses bird birds cow cows sheep goat goats puppy puppies fish
chicken chickens pig pigs bull bulls camel camels elephant animal animals pet pets""".split()
CHILDISH = """boy boys girl girls child children kid kids toddler toddlers baby babies infant youngster
little young""".split()

CLASSES = [
    ("person", "person people someone somebody human humans persons individual individuals".split(), PERSONISH),
    ("man", "man guy male gentleman he".split(), "man men guy guys gentleman male".split()),
    ("woman", "woman lady female she".split(), "woman women lady ladies female".split()),
    ("men", "men guys males".split(), "men guys males".split()),
    ("women", "women ladies females".split(), "women ladies females".split()),
    ("kid", "kid kids child children young youngster".split(), CHILDISH),
    ("animal", "animal animals pet pets creature".split(), ANIMALS),
    ("dog", "dog dogs puppy".split(), "dog dogs puppy puppies".split()),
    ("outside", "outside outdoors outdoor".split(), OUTDOORISH),
    ("inside", "inside indoors indoor".split(), INDOORISH),
    ("water", "water wet".split(), "ocean lake river pool sea beach swimming waves wave surf water swims swim".split()),
    ("music", "instrument instruments music musician playing".split(),
     "guitar piano drum drums violin cello saxophone trumpet flute band guitars".split()),
    ("vehicle", "vehicle".split(), "car truck bike bicycle motorcycle bus boat train cars".split()),
    ("bike", "bike bicycle bikes bicycles".split(), "bike bicycle bikes bicycles".split()),
    ("ball", "ball sport sports game".split(),
     "soccer basketball football volleyball baseball tennis hockey ball rugby".split()),
    ("happy", "happy".split(), "smiling smiles laughing laughs smile".split()),
    ("old", "old elderly older".split(), "old elderly older aged".split()),
    ("road", "road street".split(), "road street".split()),
    ("shirt", "shirt top".split(), "shirt top tshirt".split()),
    ("sit", "sitting sits sit".split(), "sitting sits sit seated".split()),
    ("stand", "standing stands stand".split(), "standing stands stand".split()),
    ("run", "running runs run".split(), "running runs run jogging".split()),
    ("walk", "walking walks walk".split(), "walking walks walk".split()),
    ("play", "playing plays play".split(), "playing plays play".split()),
    ("ride", "riding rides ride".split(), "riding rides ride".split()),
    ("moving", "moving".split(), "running walking jumping riding dancing swimming climbing".split()),
    ("group", "group".split(), "people crowd group team men women".split()),
    ("several", "several many".split(), "two three four five several many group crowd".split()),
    ("two", "two".split(), "two couple".split()),
]

COLORS = "red blue green yellow black white orange pink purple brown gray grey".split()
NUMBERS = "one two three four five six seven eight nine ten".split()
SPECIES = """dog dogs cat cats horse horses bird birds cow cows sheep goat goats fish chicken pig
elephant camel bull bear""".split()
PLACES = """beach park street kitchen restaurant store office bedroom church school mall pool ocean
lake river mountain desert forest field stage library bar city road house home gym classroom
snow airport""".split()
ACTIVITIES = """sitting standing lying laying sleeping running walking jumping swimming dancing eating
drinking reading cooking driving climbing riding singing skiing surfing biking skating kneeling
sits stands runs walks sleeps swims dances eats reads cooks drives climbs rides sings""".split()
MALE = "man men boy boys guy guys male males gentleman he".split()
FEMALE = "woman women girl girls lady ladies female females she".split()
ADULT = "man men woman women adult adults elderly lady ladies gentleman".split()
CHILD = "boy boys girl girls child children kid kids toddler baby babies".split()

CONFLICTS = [
    ("color", COLORS, COLORS),
    ("number", NUMBERS, NUMBERS),
    ("species", SPECIES, SPECIES),
    ("place", PLACES, PLACES),
    ("activity", ACTIVITIES, ACTIVITIES),
    ("m_f", MALE, FEMALE),
    ("f_m", FEMALE, MALE),
    ("a_c", ADULT, CHILD),
    ("c_a", CHILD, ADULT),
    ("in_out", "inside indoors indoor".split(), OUTDOORISH),
    ("out_in", "outside outdoors outdoor".split(), INDOORISH),
    ("alone", "alone".split(), "group crowd two three people men women couple team".split()),
    ("night_day", "night dark".split(), "day sunny sun daytime".split()),
    ("day_night", "day sunny daytime".split(), "night dark".split()),
    ("old_young", "old elderly older".split(), CHILDISH),
    ("young_old", "young little".split(), "old elderly older".split()),
    ("empty_crowd", "empty deserted".split(), "crowd crowded people busy".split()),
    ("dry_wet", "dry".split(), "water wet rain ocean pool lake river swimming".split()),
    ("vehicles", "car truck bike bicycle motorcycle bus boat train plane".split(),
     "car truck bike bicycle motorcycle bus boat train plane".split()),
    ("sports", "soccer basketball football volleyball baseball tennis hockey golf".split(),
     "soccer basketball football volleyball baseball tennis hockey golf".split()),
    ("instruments", "guitar piano drums violin cello saxophone trumpet flute".split(),
     "guitar piano drums violin cello saxophone trumpet flute".split()),
]

# Readout weights (hand-set heuristics)
W_ENT_BIAS = 1.6
W_ENT_EC = 0.5
W_NEU_U = 1.0
W_NEU_NC = 1.0
W_CON_CC = 3.0
W_CON_CP = 2.2
W_CON_U = 0.5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Layout:
    def __init__(self):
        self.n = 0
        self.d = {}

    def add(self, name, size=1):
        self.d[name] = list(range(self.n, self.n + size))
        self.n += size
        return self.d[name]

    def __getitem__(self, name):
        r = self.d[name]
        return r[0] if len(r) == 1 else r


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, task=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.build_layout(task)
        D = self.lay.n
        self.d_model = D
        self.token_emb = nn.Embedding(vocab_size, D)
        self.pos_emb = nn.Embedding(max_seq_len, D)
        self.blocks = nn.ModuleList([
            Block(D, n_heads=L, d_qk=2, d_v=vocab_size, d_ff=self.ff1),
            Block(D, n_heads=1, d_qk=1, d_v=1, d_ff=self.ffS),   # stemming (attention unused)
            Block(D, n_heads=2, d_qk=self.qk2, d_v=1, d_ff=self.ff2),
            Block(D, n_heads=1, d_qk=1, d_v=5, d_ff=0),
        ])
        self.head = nn.Linear(D, vocab_size, bias=False)

    def build_layout(self, task):
        V = task.vocab
        self.letters = [c for c in V if c.isalpha() or c == "'"]
        self.bounds = [" ", "_", "|", "="]
        lay = Layout()
        lay.add("tok", len(V))
        for n in ["one", "p", "p2", "prem", "hyp", "pos0"]:
            lay.add(n)
        lay.add("VB", L)
        for j in range(1, L + 1):
            lay.add(f"G{j}", len(V))
        lay.add("premWE"); lay.add("hypWE")
        for j in range(1, L + 1):
            lay.add(f"M{j}", len(self.letters))
        lay.add("MB", L)
        lay.add("ingF"); lay.add("sF")
        for j in range(1, L + 1):
            lay.add(f"S{j}", len(self.letters))
        lay.add("SB", L)
        for n in ["stop", "cc", "nc", "ec"]:
            lay.add(n)
        for name, _, _ in CLASSES:
            lay.add("qC_" + name); lay.add("kC_" + name)
        for name, _, _ in CONFLICTS:
            lay.add("qX_" + name); lay.add("kX_" + name)
        for n in ["matched", "hit", "U", "CP", "CC", "NC", "EC"]:
            lay.add(n)
        for n in ["aU", "aCP", "aCC", "aNC", "aEC"]:
            lay.add(n)
        self.lay = lay

        # lexicon: word -> list of (dim, weight)
        lex = {}
        def put(w, dim, wt=1.0):
            if len(w) > L - 1 or not all(c in self.letters for c in w):
                return
            lex.setdefault(w, {})
            lex[w][dim] = max(lex[w].get(dim, 0.0), wt)
        for w in STOP: put(w, lay["stop"])
        for w in CONTRA_CUES: put(w, lay["cc"])
        for w in NEUTRAL_CUES: put(w, lay["nc"])
        for w in ENTAIL_CUES: put(w, lay["ec"])
        for name, hs, ps in CLASSES:
            for w in hs: put(w, lay["qC_" + name], float(min(len(w), L - 1) + 1))
            for w in ps: put(w, lay["kC_" + name])
        for name, hs, ps in CONFLICTS:
            for w in hs: put(w, lay["qX_" + name])
            for w in ps: put(w, lay["kX_" + name])
        self.lex = lex
        self.ff1 = 2 + L * len(self.letters) + L + len(lex) + 2
        self.ffS = 3 * (L * len(self.letters) + L)
        self.qk2 = (L * len(self.letters) + L) + 2 + len(CLASSES) + 1
        self.qk2 = max(self.qk2, len(CONFLICTS) + 2)
        self.ff2 = 5

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    lay = model.lay
    V = task.vocab
    stoi = task.stoi
    S = task.SENT_LEN
    for p in model.parameters():
        p.zero_()

    # ---- embeddings
    E = model.token_emb.weight
    for i in range(len(V)):
        E[i, lay["tok"][i]] = 1.0
    P = model.pos_emb.weight
    for t in range(model.max_seq_len):
        P[t, lay["one"]] = 1.0
        P[t, lay["p"]] = float(t)
        P[t, lay["p2"]] = float(t * t)
        P[t, lay["prem"]] = 1.0 if t <= S else 0.0          # premise word ends: 1..S ('|' at S)
        P[t, lay["hyp"]] = 1.0 if S < t <= 2 * S + 1 else 0.0  # hyp word ends up to '='
        P[t, lay["pos0"]] = 1.0 if t == 0 else 0.0
        for j in range(1, L + 1):
            P[t, lay["VB"][j - 1]] = 1.0 if t < j else 0.0

    tokdim = {c: lay["tok"][stoi[c]] for c in V}
    bound_tok = [tokdim[c] for c in model.bounds]

    # ---- layer 1 attention: look-back heads
    a1 = model.blocks[0].attn
    nv = len(V)
    for j in range(1, L + 1):
        hh = j - 1
        # q = beta*[p - j, 1]; k = [2s, -s^2]  -> beta*(2s(t-j) - s^2) = -beta(s-(t-j))^2 + const
        a1.W_q.weight[hh * 2 + 0, lay["p"]] = BETA
        a1.W_q.weight[hh * 2 + 0, lay["one"]] = -BETA * j
        a1.W_q.weight[hh * 2 + 1, lay["one"]] = BETA
        a1.W_k.weight[hh * 2 + 0, lay["p"]] = 2.0
        a1.W_k.weight[hh * 2 + 1, lay["p2"]] = -1.0
        for i in range(nv):
            a1.W_v.weight[hh * nv + i, lay["tok"][i]] = 1.0
            a1.W_o.weight[lay[f"G{j}"][i], hh * nv + i] = 1.0

    def G(j, c):
        return lay[f"G{j}"][stoi[c]]

    # ---- layer 1 MLP
    m1 = model.blocks[0].mlp
    W1, b1, W2 = m1.fc1.weight, m1.fc1.bias, m1.fc2.weight
    n = 0

    def add_Bt(row, w=1.0):
        for d in bound_tok:
            W1[row, d] += w

    def add_Bslot(row, j, w=1.0):
        for c in model.bounds:
            W1[row, G(j, c)] += w
        W1[row, lay["VB"][j - 1]] += w

    def add_L1(row, w=1.0):
        for c in model.letters:
            W1[row, G(1, c)] += w

    # premWE / hypWE
    for region, out in [("prem", "premWE"), ("hyp", "hypWE")]:
        add_Bt(n); add_L1(n); W1[n, lay[region]] = 1.0; b1[n] = -2.0
        W2[lay[out], n] = 1.0
        n += 1
    # masked letters
    for j in range(1, L + 1):
        for ci, c in enumerate(model.letters):
            W1[n, G(j, c)] = 1.0
            add_Bt(n)
            for i in range(1, j + 1):
                add_Bslot(n, i, -1.0)
            b1[n] = -1.0
            W2[lay[f"M{j}"][ci], n] = 1.0
            n += 1
    # masked boundary markers
    for j in range(1, L + 1):
        add_Bslot(n, j, 1.0)
        add_Bt(n)
        for i in range(1, j):
            add_Bslot(n, i, -1.0)
        b1[n] = -1.0
        W2[lay["MB"][j - 1], n] = 1.0
        n += 1
    # word detectors
    for w, roles in model.lex.items():
        k = len(w)
        for j in range(1, k + 1):
            W1[n, G(j, w[k - j])] += 1.0
        add_Bslot(n, k + 1)
        add_Bt(n)
        b1[n] = -float(k + 1)
        for dim, wt in roles.items():
            W2[dim, n] = wt
        n += 1
    # stem flags: word ends in "ing" with >=2-letter stem; word ends in "s" (not "ss"), length >= 3
    for c, j in [("g", 1), ("n", 2), ("i", 3)]:
        W1[n, G(j, c)] += 1.0
    add_Bt(n); add_Bslot(n, 4, -1.0); add_Bslot(n, 5, -1.0)
    b1[n] = -3.0
    W2[lay["ingF"], n] = 1.0
    n += 1
    W1[n, G(1, "s")] += 1.0; W1[n, G(2, "s")] -= 1.0
    add_Bt(n); add_Bslot(n, 2, -1.0); add_Bslot(n, 3, -1.0)
    b1[n] = -1.0
    W2[lay["sF"], n] = 1.0
    n += 1
    assert n == model.ff1

    # ---- stemming MLP (block index 1): S = M shifted by 3 if ing, by 1 if s, else M
    ms = model.blocks[1].mlp
    W1, b1, W2 = ms.fc1.weight, ms.fc1.bias, ms.fc2.weight

    def mdim(j, ci):
        return lay[f"M{j}"][ci] if ci is not None else lay["MB"][j - 1]

    def sdim(j, ci):
        return lay[f"S{j}"][ci] if ci is not None else lay["SB"][j - 1]
    n = 0
    for j in range(1, L + 1):
        for ci in list(range(len(model.letters))) + [None]:
            # default (no suffix)
            W1[n, mdim(j, ci)] = 1.0; W1[n, lay["ingF"]] = -1.0; W1[n, lay["sF"]] = -1.0
            W2[sdim(j, ci), n] = 1.0
            n += 1
            # -s stripped
            if j + 1 <= L:
                W1[n, mdim(j + 1, ci)] = 1.0; W1[n, lay["sF"]] = 1.0; b1[n] = -1.0
                W2[sdim(j, ci), n] = 1.0
            n += 1
            # -ing stripped
            if j + 3 <= L:
                W1[n, mdim(j + 3, ci)] = 1.0; W1[n, lay["ingF"]] = 1.0; b1[n] = -1.0
                W2[sdim(j, ci), n] = 1.0
            n += 1
    assert n == model.ffS

    # ---- layer 2 attention
    a2 = model.blocks[2].attn
    dq = model.qk2
    Wq, Wk, Wv, Wo = a2.W_q.weight, a2.W_k.weight, a2.W_v.weight, a2.W_o.weight
    # head 0: exact / class match
    r = 0
    wv_dims = [d for j in range(1, L + 1) for d in lay[f"S{j}"]] + list(lay["SB"])
    for d in wv_dims:
        Wq[r, d] = BETA
        Wk[r, d] = 1.0
        r += 1
    # length term: q = -(sum M) + 0.5, k = premWE
    for d in wv_dims:
        Wq[r, d] = -BETA
    Wq[r, lay["one"]] = 0.5 * BETA
    Wk[r, lay["premWE"]] = 1.0
    r += 1
    # penalty: q = 50, k = premWE + pos0 - 1
    Wq[r, lay["one"]] = 50.0 * BETA
    Wk[r, lay["premWE"]] = 1.0; Wk[r, lay["pos0"]] = 1.0; Wk[r, lay["one"]] = -1.0
    r += 1
    for name, _, _ in CLASSES:
        Wq[r, lay["qC_" + name]] = BETA
        Wk[r, lay["kC_" + name]] = 1.0
        r += 1
    assert r <= dq
    Wv[0, lay["premWE"]] = 1.0
    Wo[lay["matched"], 0] = 1.0
    # head 1: conflict
    o = dq
    r = 0
    for name, _, _ in CONFLICTS:
        Wq[o + r, lay["qX_" + name]] = BETA
        Wk[o + r, lay["kX_" + name]] = 1.0
        r += 1
    Wq[o + r, lay["one"]] = -0.5 * BETA
    Wk[o + r, lay["premWE"]] = 1.0
    r += 1
    Wq[o + r, lay["one"]] = 50.0 * BETA
    Wk[o + r, lay["premWE"]] = 1.0; Wk[o + r, lay["pos0"]] = 1.0; Wk[o + r, lay["one"]] = -1.0
    Wv[1, lay["premWE"]] = 1.0
    Wo[lay["hit"], 1] = 1.0

    # ---- layer 2 MLP
    m2 = model.blocks[2].mlp
    W1, b1, W2 = m2.fc1.weight, m2.fc1.bias, m2.fc2.weight
    # U = relu(hypWE - stop - matched)
    W1[0, lay["hypWE"]] = 1.0; W1[0, lay["stop"]] = -1.0; W1[0, lay["matched"]] = -1.0
    W2[lay["U"], 0] = 1.0
    # CP = relu(hit + hypWE - matched - 1)
    W1[1, lay["hit"]] = 1.0; W1[1, lay["hypWE"]] = 1.0; W1[1, lay["matched"]] = -1.0; b1[1] = -1.0
    W2[lay["CP"], 1] = 1.0
    for row, (src_, dst) in enumerate([("cc", "CC"), ("nc", "NC"), ("ec", "EC")], start=2):
        W1[row, lay[src_]] = 1.0; W1[row, lay["hypWE"]] = 1.0; W1[row, lay["matched"]] = -1.0
        b1[row] = -1.0
        W2[lay[dst], row] = 1.0

    # ---- layer 3 attention: uniform average at '=' (position 2S+1 -> 2S+2 positions)
    a3 = model.blocks[3].attn
    navg = float(2 * S + 2)
    for i, (src_, dst) in enumerate([("U", "aU"), ("CP", "aCP"), ("CC", "aCC"), ("NC", "aNC"), ("EC", "aEC")]):
        a3.W_v.weight[i, lay[src_]] = 1.0
        a3.W_o.weight[lay[dst], i] = navg

    # ---- head: logits
    Hd = model.head.weight
    for c in V:
        Hd[stoi[c], lay["one"]] = -100.0
    e, nu, co = stoi["0"], stoi["1"], stoi["2"]
    Hd[e, lay["one"]] = W_ENT_BIAS
    Hd[e, lay["aEC"]] = W_ENT_EC
    Hd[nu, lay["one"]] = 0.0
    Hd[nu, lay["aU"]] = W_NEU_U
    Hd[nu, lay["aNC"]] = W_NEU_NC
    Hd[co, lay["one"]] = 0.0
    Hd[co, lay["aCC"]] = W_CON_CC
    Hd[co, lay["aCP"]] = W_CON_CP
    Hd[co, lay["aU"]] = W_CON_U

    if DEBUG:
        _debug_handmade(model, task)


DEBUG = True


@torch.no_grad()
def _debug_handmade(m, task):
    """Inspect internal features on hand-written sentence pairs (not dataset data)."""
    lay = m.lay
    pad = lambda s: s[:60].ljust(60, "_")
    pairs = [("a man in a red shirt is running in the park", "a man is outside"),
             ("a man in a red shirt is running in the park", "a woman is sleeping"),
             ("a man in a red shirt is running in the park", "a man in a blue shirt runs"),
             ("two dogs play in the snow", "the dogs are playing for their owner because they are friends"),
             ("a young boy rides a bike", "a kid is on a bicycle"),
             ("two dogs are playing with balls at night", "a dog plays with a ball during the day")]
    for p, h in pairs:
        s = pad(p) + "|" + pad(h) + "="
        ids = torch.tensor([task.encode(s)])
        x = m.token_emb(ids) + m.pos_emb(torch.arange(ids.shape[1]))[None]
        for b in m.blocks[:3]:
            x = b(x)
        for t in range(61, 122):
            if x[0, t, lay["hypWE"]] > 0.5:
                w = s[:t].split("|")[-1].split(" ")[-1]
                print(f"  {w:10s} " + " ".join(f"{k}={x[0,t,lay[k]].item():.2f}" for k in
                      ["matched", "hit", "stop", "U", "CP", "CC", "NC", "EC"]))
        x = m.blocks[3](x)
        print(h, "| agg", [round(x[0, -1, lay[k]].item(), 2) for k in ["aU", "aCP", "aCC", "aNC", "aEC"]],
              "logits", [round(v, 2) for v in m.head(x)[0, -1, [task.stoi[c] for c in "012"]].tolist()])


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "LexMatchConflictStem_v2"
model_description = ("v1 + stemming layer (strip -s/-ing before premise matching) and extra conflict "
                     "categories (day/night, old/young, vehicles, sports, instruments, ...).")


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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len, task=task)
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
