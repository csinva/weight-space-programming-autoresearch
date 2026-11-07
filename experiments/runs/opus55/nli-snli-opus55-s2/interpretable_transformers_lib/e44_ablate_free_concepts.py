"""
Hand-written character-level transformer for SNLI-style NLI (never trained).

Prompt: premise(60)|hypothesis(60)=   Answer: 0 entail, 1 neutral, 2 contradict.

Circuit:
  B1 attention : rotation-coded position attention copies the last 4 chars into a window.
  B1 MLP       : word-anchored stem token T = first 4 chars of a word (blank after word end),
                 emitted at the position 3 chars after the word start.
  B2 MLP       : stem detectors -> concept channels G, cue votes V, content flag.
  B3 attention : each hypothesis token queries premise tokens (sentinel at position 0):
                 MATCH = same stem/concept/hypernym, GROUP = same exclusive group.
  B3 MLP       : gates: hyp content word, matched, conflict (group present, not matched).
  B4 attention : at '=' average "matched" over content words; max over conflict and votes.
  head         : linear thresholds on those aggregates.
"""

from __future__ import annotations

import argparse
import csv
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
# Hand-written knowledge
# ---------------------------------------------------------------------------

USE_CONCEPTS, USE_CONFLICT, USE_VOTES = True, True, True
T_ENTAIL, T_CONTRA, GAIN = 0.6, 0.3, 8.0       # thresholds on fraction of matched content words
W_CONFLICT, W_VOTE = (6.0, (3.0, 3.0, 3.0))    # conflict weight, vote weight per class
W_UNMATCHED, N_HYP_POS = 1.5, 58               # entail penalty per unmatched word; positions averaged over
NEUTRAL_BIAS = -0.3                            # constant added to the neutral logit

STOP = ("a an the is are was were be being been in on at of to and with his her their its there "
        "some it this that as by from he she they them who while one two three four very has have "
        "for no not nobody noone none nothing never "
        "wear* dres* hold* look* doin* do does using use grou* coup* othe* each many lots lot next near over "
        "up down out into around through")

# group -> concept -> stems. Concepts inside a group are mutually exclusive.
# Stem syntax: "man" = exact word, "dog*" = prefix; only the first 4 chars are ever visible.
GROUPS = {
    "person": {"man": "man men guy* male* gent* dude* husb*", "woman": "woman wome* lady ladi* girl* fema* wife",
               "boy": "boy*", "child": "chil* kid*"},
    "action": {"sit": "sit* seat*", "stand": "stan*", "walk": "walk* stro* hik*", "run": "run* jog* spri*",
               "sleep": "slee* nap* asle*", "lay": "lay* lyi*", "eat": "eat* din*", "drink": "drin*",
               "swim": "swim*", "dance": "danc*", "ride": "rid*", "jump": "jump* leap*", "play": "play*",
               "read": "read*", "talk": "talk* chat* spea*", "cook": "cook*", "drive": "driv*"},
    "place": {"indoor": "indo* insi*", "outdoor": "outd* outs*", "beach": "beac*", "park": "park*",
              "street": "stre* road*", "mountain": "moun*", "forest": "fore* wood*", "water": "lake* rive* ocea* sea",
              "kitchen": "kitc*", "home": "hous* home*", "store": "stor* mall* shop* supe*", "school": "scho*",
              "office": "offi*", "church": "chur*", "restaurant": "rest* cafe*"},
    "animal": {"dog": "dog* pupp*", "cat": "cat cats kitt*", "bird": "bird*", "horse": "hors*"},
    "color": {"red": "red", "blue": "blue*", "green": "gree*", "yellow": "yell*", "black": "blac*", "white": "whit*",
              "orange": "oran*", "pink": "pink*", "purple": "purp*", "brown": "brow*"},
    "vehicle": {"car": "car cars", "truck": "truc*", "bus": "bus*", "bike": "bike* bicy*", "boat": "boat*",
                "plane": "plan*", "train": "trai*"},
    "sport": {"soccer": "socc*", "football": "foot*", "basketball": "bask*", "baseball": "base*", "tennis": "tenn*",
              "golf": "golf*", "hockey": "hock*"},
    "instrument": {"guitar": "guit*", "piano": "pian*", "violin": "viol*", "drums": "drum*", "flute": "flut*",
                   "trumpet": "trum*", "saxophone": "saxo*"},
    "activity": {"climb": "clim*", "ski": "ski*", "surf": "surf*", "skate": "skat*", "fish": "fish*", "paint": "pain*",
                 "write": "writ*", "clean": "clea* wash*", "fight": "figh*", "kiss": "kiss* hug*", "cry": "cry* wee*"},
}
FREE_CONCEPTS = {"person": "pers* peop* huma* some*", "old": "old elde* aged", "young": "youn*", "animal": "anim*",
                 "music": "musi* song* sing* band*", "vehicle": "vehi*", "ball": "ball*"}
# a premise concept also unlocks these more general hypothesis concepts
HYPERNYM = {"man": "person", "woman": "person", "boy": "person child", "child": "person", "dog": "animal",
            "cat": "animal", "bird": "animal", "horse": "animal", "beach": "outdoor", "park": "outdoor",
            "street": "outdoor", "mountain": "outdoor", "forest": "outdoor", "water": "outdoor", "kitchen": "indoor",
            "home": "indoor", "store": "indoor", "school": "indoor", "office": "indoor", "church": "indoor",
            "restaurant": "indoor", "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "boat": "vehicle",
            "plane": "vehicle", "train": "vehicle", "soccer": "ball", "football": "ball", "basketball": "ball",
            "baseball": "ball", "tennis": "ball", "guitar": "music", "piano": "music", "violin": "music",
            "drums": "music", "flute": "music", "trumpet": "music", "saxophone": "music"}
VOTES = [
    "outd* outs* huma* pers* peop* anim* inst*",                                          # entail cues
    "for beca* frie* vaca* comp* first sad tall tire* fami* birt* goin* tryi* abou* afte* moth* fath* wife husb* new part*",  # neutral
    "no nobo* noon* none noth* neve* not isn'* aren* slee* nap*",                         # contradiction
]


# ---------------------------------------------------------------------------
# Channel layout
# ---------------------------------------------------------------------------

S, BL, L = 28, 27, 4                                # symbols (a-z ' blank), blank index, stem length
ALPHA = "abcdefghijklmnopqrstuvwxyz'"
CONCEPTS = sorted({c for g in GROUPS.values() for c in g} | set(FREE_CONCEPTS))
CIDX = {c: i for i, c in enumerate(CONCEPTS)}
GNAMES = list(GROUPS)

_ch, _next = {}, 0
for _name, _n in [("SYM", S), ("POS", 2), ("ONE", 1), ("SENT", 1), ("START", 1), ("HYP", 1), ("W", L * S), ("T", L * S),
                  ("TOK", 1), ("G", len(CONCEPTS)), ("CONT", 1), ("V", 3), ("M", 1), ("GP", 1), ("HC", 1), ("MC", 1),
                  ("CF", 1), ("VH", 3), ("UC", 1), ("FM", 1), ("FC", 1), ("FV", 3), ("FU", 1)]:
    _ch[_name], _next = _next, _next + _n
C = type("C", (), _ch)
D_MODEL = _next


def stem_targets():
    """Map (slots, is_exact) stems to the output channels they drive."""
    out: dict[str, list[tuple[int, float]]] = {}

    def add(words, ch, w=1.0):
        for word in words.split():
            out.setdefault(word, []).append((ch, w))

    add(STOP, C.CONT, -1.0)
    if USE_CONCEPTS or USE_CONFLICT:
        for group in GROUPS.values():
            for c, words in group.items():
                add(words, C.G + CIDX[c])
        for c, words in FREE_CONCEPTS.items():
            add(words, C.G + CIDX[c])
    if USE_VOTES:
        for i, words in enumerate(VOTES):
            add(words, C.V + i)
    return out


STEMS = stem_targets()


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class Pruned(nn.Module):
    """Same function as a Linear or Embedding, storing only its nonzero rows and columns."""

    def __init__(self, source: nn.Module):
        super().__init__()
        weight = source.weight.detach()
        self.embedding = isinstance(source, nn.Embedding)
        bias = getattr(source, "bias", None)
        keep_rows = weight.abs().sum(1) + (0 if bias is None else bias.abs())
        rows, cols = keep_rows.nonzero().flatten(), weight.abs().sum(0).nonzero().flatten()
        rows = torch.arange(weight.shape[0]) if self.embedding else rows
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.out_features = weight.shape[1] if self.embedding else weight.shape[0]
        self.weight = nn.Parameter(weight[rows][:, cols].clone())
        self.bias = None if bias is None else nn.Parameter(bias[rows].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding:
            y = torch.zeros(*x.shape, self.out_features, device=x.device, dtype=self.weight.dtype)
            y[..., self.cols] = self.weight[x]
            return y
        y = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
        kept = x[..., self.cols] @ self.weight.T
        y[..., self.rows] = kept if self.bias is None else kept + self.bias
        return y


def prune(model: nn.Module) -> None:
    """Replace every Linear and Embedding by its Pruned equivalent."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.Linear, nn.Embedding)):
                setattr(module, name, Pruned(child))


class Head(nn.Module):
    """One causal attention head reading a d_key key space, writing d_v channels at `out`."""

    def __init__(self, d: int, d_key: int, d_v: int, out: int):
        super().__init__()
        self.W_q = nn.Linear(d, d_key, bias=False)
        self.W_k = nn.Linear(d, d_key, bias=False)
        self.W_v = nn.Linear(d, d_v, bias=False)
        self.out = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        scores = self.W_q(x) @ self.W_k(x).transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = torch.zeros_like(x)
        y[..., self.out:self.out + self.W_v.out_features] = attn @ self.W_v(x)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """No LayerNorm: each stage writes fresh channels, residuals keep everything earlier."""

    def __init__(self, heads: list[Head], mlp: MLP | None):
        super().__init__()
        self.heads = nn.ModuleList(heads)
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.heads):
            x = x + sum(h(x) for h in self.heads)
        return x if self.mlp is None else x + self.mlp(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        d = D_MODEL
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_seq_len, d)
        self.blocks = nn.ModuleList([
            Block([Head(d, 2, S, C.W + o * S) for o in range(L)], MLP(d, L * 27 + L - 1)),
            Block([], MLP(d, len(STEMS) + 1)),
            Block([Head(d, L * S + len(CONCEPTS) + 2, 1, C.M), Head(d, len(GNAMES) + 2, 1, C.GP)], MLP(d, 7)),
            Block([Head(d, 1, 1, C.FM), Head(d, 1, 1, C.FC)] + [Head(d, 1, 1, C.FV + i) for i in range(3)]
                  + [Head(d, 1, 1, C.FU)], None),
        ])
        self.head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(ids) + self.pos_emb(torch.arange(ids.shape[1], device=ids.device))[None]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

THETA, BETA_POS, BETA = 2 * torch.pi / 256, 1e5, 10.0
LAMBDA, TAU_MATCH, TAU_GROUP, A_CONCEPT = 10.0, L - 1.5, 0.5, 2.5   # A_CONCEPT**2 > TAU_MATCH


def slot_ch(k: int, c: int) -> int:
    """Channel of symbol c for stem slot k (slot k sits L-1-k chars back from the current one)."""
    o = L - 1 - k
    return C.SYM + c if o == 0 else C.W + (o - 1) * S + c


def write_stage_b1(block: Block) -> None:
    for o, head in enumerate(block.heads, start=1):      # copy the char `o` steps back
        co, si = torch.cos(torch.tensor(o * THETA)), torch.sin(torch.tensor(o * THETA))
        head.W_q.weight[0, C.POS], head.W_q.weight[0, C.POS + 1] = BETA_POS * co, BETA_POS * si
        head.W_q.weight[1, C.POS], head.W_q.weight[1, C.POS + 1] = -BETA_POS * si, BETA_POS * co
        head.W_k.weight[0, C.POS], head.W_k.weight[1, C.POS + 1] = 1.0, 1.0
        for c in range(S):
            head.W_v.weight[c, C.SYM + c] = 1.0

    fc1, fc2 = block.mlp.fc1, block.mlp.fc2
    anchor = lambda n: (fc1.weight.data.__setitem__((n, C.W + (L - 1) * S + BL), 1.0),
                        fc1.weight.data.__setitem__((n, C.START), 1.0))
    n = 0
    for k in range(L):                                    # letter c in slot k of a word-anchored stem
        for c in range(27):
            anchor(n)
            fc1.weight[n, slot_ch(k, c)] = 1.0
            for j in range(k):
                fc1.weight[n, slot_ch(j, BL)] = -1.0
            fc1.bias[n] = -1.0
            fc2.weight[C.T + k * S + c, n] = 1.0
            if k == 0:
                fc2.weight[C.TOK, n] = 1.0
            n += 1
    for j in range(1, L):                                 # the word ends exactly at slot j: blank from j on
        anchor(n)
        fc1.weight[n, slot_ch(j, BL)] = 1.0
        for i in range(j):
            fc1.weight[n, slot_ch(i, BL)] = -1.0
        fc1.bias[n] = -1.0
        for k in range(j, L):
            fc2.weight[C.T + k * S + BL, n] = 1.0
        n += 1


def write_stage_b2(block: Block) -> None:
    fc1, fc2 = block.mlp.fc1, block.mlp.fc2
    for n, (stem, targets) in enumerate(STEMS.items()):
        prefix = stem.endswith("*")
        word = stem.rstrip("*")[:L]
        slots = [(k, ALPHA.index(ch)) for k, ch in enumerate(word)]
        if not prefix and len(word) < L:                  # exact short word: require blanks after it
            slots += [(k, BL) for k in range(len(word), L)]
        for k, c in slots:
            fc1.weight[n, C.T + k * S + c] = 1.0
        fc1.bias[n] = 1.0 - len(slots)
        for ch, w in targets:
            fc2.weight[ch, n] = w
    n = len(STEMS)                                        # every word is content unless a stopword
    fc1.weight[n, C.TOK] = 1.0
    fc2.weight[C.CONT, n] = 1.0


def write_stage_b3(block: Block) -> None:
    match, group = block.heads
    last = L * S + len(CONCEPTS)
    for i in range(L * S):
        match.W_q.weight[i, C.T + i] = BETA
        match.W_k.weight[i, C.T + i] = 1.0
    if USE_CONCEPTS:
        for c, i in CIDX.items():
            match.W_q.weight[L * S + i, C.G + i] = A_CONCEPT * BETA
            match.W_k.weight[L * S + i, C.G + i] = A_CONCEPT
            for parent in HYPERNYM.get(c, "").split():
                match.W_k.weight[L * S + CIDX[parent], C.G + i] = A_CONCEPT
    match.W_q.weight[last, C.ONE], match.W_k.weight[last, C.HYP] = -LAMBDA * BETA, 1.0
    match.W_q.weight[last + 1, C.ONE], match.W_k.weight[last + 1, C.SENT] = TAU_MATCH * BETA, 1.0
    match.W_v.weight[0, C.TOK] = 1.0

    last = len(GNAMES)
    if USE_CONFLICT:
        for g, name in enumerate(GNAMES):
            for c in GROUPS[name]:
                group.W_q.weight[g, C.G + CIDX[c]] = BETA
                group.W_k.weight[g, C.G + CIDX[c]] = 1.0
    group.W_q.weight[last, C.ONE], group.W_k.weight[last, C.HYP] = -LAMBDA * BETA, 1.0
    group.W_q.weight[last + 1, C.ONE], group.W_k.weight[last + 1, C.SENT] = TAU_GROUP * BETA, 1.0
    group.W_v.weight[0, C.TOK] = 1.0

    fc1, fc2 = block.mlp.fc1, block.mlp.fc2
    gates = [
        (C.HC, {C.CONT: 1, C.HYP: 1}, -1.0),                    # hypothesis content word
        (C.MC, {C.CONT: 1, C.HYP: 1, C.M: 1}, -2.0),            # ... that is matched
        (C.CF, {C.CONT: 1, C.HYP: 1, C.GP: 1, C.M: -1}, -2.0),  # ... conflicting with the premise
        (C.UC, {C.CONT: 1, C.HYP: 1, C.M: -1}, -1.0),           # ... that is absent from the premise
    ] + [(C.VH + i, {C.V + i: 1, C.HYP: 1}, -1.0) for i in range(3)]   # hypothesis cue votes
    for n, (out, inputs, bias) in enumerate(gates):
        for ch, w in inputs.items():
            fc1.weight[n, ch] = float(w)
        fc1.bias[n] = bias
        fc2.weight[out, n] = 1.0


def write_stage_b4(block: Block) -> None:
    keys = [C.HC, C.CF, C.VH, C.VH + 1, C.VH + 2, C.HYP]       # last head: uniform over hypothesis positions
    values = [C.MC, C.CF, C.VH, C.VH + 1, C.VH + 2, C.UC]      # so its mean * N_HYP_POS is a word count
    for head, key, value in zip(block.heads, keys, values):
        head.W_q.weight[0, C.ONE] = BETA
        head.W_k.weight[0, key], head.W_k.weight[0, C.SENT] = 1.0, 0.5
        head.W_v.weight[0, value] = 1.0


def write_weights(model: SimpleTransformer, task) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for i, ch in task.itos.items():
            model.token_emb.weight[i, C.SYM + (ALPHA.index(ch) if ch in ALPHA else BL)] = 1.0
        for p in range(model.pos_emb.num_embeddings):
            row = model.pos_emb.weight[p]
            row[C.POS], row[C.POS + 1] = float(torch.cos(torch.tensor(p * THETA))), float(torch.sin(torch.tensor(p * THETA)))
            row[C.ONE], row[C.SENT], row[C.START], row[C.HYP] = 1.0, float(p == 0), float(p == L - 1), float(p >= 60 + L)
        write_stage_b1(model.blocks[0])
        write_stage_b2(model.blocks[1])
        write_stage_b3(model.blocks[2])
        write_stage_b4(model.blocks[3])
        W = model.head.weight
        W[:, C.ONE] = -100.0
        e, n, c = task.stoi["0"], task.stoi["1"], task.stoi["2"]
        W[e] = W[e] * 0
        W[n] = W[n] * 0
        W[c] = W[c] * 0
        W[e, C.FM], W[e, C.ONE], W[e, C.FV] = GAIN, -GAIN * T_ENTAIL, W_VOTE[0]
        W[e, C.FU] = -W_UNMATCHED * N_HYP_POS
        W[n, C.FV + 1], W[n, C.ONE] = W_VOTE[1], NEUTRAL_BIAS
        W[c, C.FM], W[c, C.ONE], W[c, C.FC], W[c, C.FV + 2] = -GAIN, GAIN * T_CONTRA, W_CONFLICT, W_VOTE[2]
        prune(model)
        diagnose(model, task)


def diagnose(model, task, n=1000, seed=987654, show=0):
    """Measure only: confusion matrix on a non-eval seed. Nothing here feeds back into weights."""
    examples = task.generate_examples(n, seed=seed)
    ids = torch.tensor([task.encode(e.prompt) for e in examples])
    with torch.no_grad():
        pred = model(ids)[:, -1].argmax(-1)
    gold = torch.tensor([task.stoi[e.answer] for e in examples])
    labels = [task.stoi[s] for s in "012"]
    confusion = [[int(((gold == a) & (pred == b)).sum()) for b in labels] for a in labels]
    print(f"dev acc {float((pred == gold).float().mean()):.4f} confusion (row gold, col pred) {confusion}")
    fresh = task.generate_examples(n, seed=seed + 1)
    with torch.no_grad():
        fresh_pred = model(torch.tensor([task.encode(e.prompt) for e in fresh]))[:, -1].argmax(-1)
    fresh_gold = torch.tensor([task.stoi[e.answer] for e in fresh])
    print(f"second-seed acc {float((fresh_pred == fresh_gold).float().mean()):.4f}")
    dev_prompts = {e.prompt for e in examples}
    eval_prompts = {e.prompt for e in task.generate_examples(200, seed=0)}
    print(f"overlap counts: {len(eval_prompts & dev_prompts)} of {len(eval_prompts)} seed-0 prompts are in dev")
    wrong = (pred != gold).nonzero().flatten().tolist()[:show]
    for i in wrong:
        print(f"gold {examples[i].answer} pred {task.itos[int(pred[i])]} | {examples[i].prompt.replace('_', '')}")


model_shorthand_name = "e44_ablate_free_concepts"
model_description = "Removes the photo, food, beverage, smile and work synonym concepts to test whether they earn their detectors."


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
