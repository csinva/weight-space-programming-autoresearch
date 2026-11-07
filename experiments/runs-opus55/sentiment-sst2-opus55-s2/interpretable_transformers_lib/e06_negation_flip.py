"""
Interpretable transformer for sentiment-sst2 (never trained; every weight is written by hand).

Circuit:
  Block 1 attention: one head per character offset k, attending exactly to position t-k
                     (Gaussian attention from quadratic position features), so each position
                     sees its last KW characters as one-hot slots.
  Block 1 MLP:       one hidden neuron per lexicon pattern, an exact AND over character
                     constraints; fc2 adds the pattern's sentiment weight to the SCORE channel.
  Block 2 attention: the same exact-offset heads sum the NEG channel over the previous
                     KW-1 characters; its MLP clamps that count to a 0/1 FLAG.
  Block 3 MLP:       gated ReLUs flip SCORE at positions where FLAG = 1.
  Block 4 attention: uniform attention at '=' averages SCORE over the sequence.
  Head:              compares the average with a threshold to pick '1' or '0'.

Usage:
    uv run interpretable_transformer.py --task sentiment-sst2
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
# Hand-written lexicon. "word" = whole word, "stem*" = prefix, "*tail" = suffix,
# "_" = a space inside a pattern.
# ---------------------------------------------------------------------------

LEXICON = {
    2: """masterpiece* masterful* brilliant* superb* excellent* wonderful* marvelous* marvellous* terrific*
          outstanding* extraordinary* magnificent* exhilarating* breathtaking* captivating* riveting* astonishing*
          stunning* dazzling* delightful* delight* enchanting* beautiful* beautifully gorgeous* perfect* perfectly
          best finest greatest great greatly amazing* fantastic* remarkable* remarkably compelling* powerful*
          moving touching heartwarming* charming* charm* engrossing* engaging* entertaining* enjoyable* hilarious*
          funny funniest fun love loved loves lovely triumph* gem wit witty smart intelligent* clever* inventive*
          imaginative* original refreshing* poignant* thrilling* thrill* treat winning* memorable* wise""",
    1: """good better well nice fine solid strong rich warm sweet tender gentle sincere honest heartfelt
          impressive* interesting* intriguing* satisfying* pleasant* pleasure* pleasing* appealing* likable likeable
          sharp fresh vivid lively spirited energetic* fascinating* insightful* thoughtful* thought_prov*
          beauty humor humour humorous* laugh laughs laughter wonder* joy* playful* inspired inspiring* uplifting*
          sensitive* subtle* elegant* graceful* grace stylish* style skillful* skilled* talented* talent* gifted
          confident* assured ambitious* tasty sexy cool nicely deft* crisp* affecting* absorbing* entertain*
          enjoy* success* succeeds* effective* hope* pretty real truth* beauty rewarding* surprising* surprisingly
          sweetly kind* unique* fun_ cute* sly* quirky* solid* worth worthy* tribute* compassion* intimate* excit*
          heart* soul* wonderfully* classic* pure* rare* true human* emotion* passion* intelligence spectacular*
          visual* entertainment* inventiveness* easy gem* sweet* smile* beautif* radiant* lovingly affection*""",
    -1: """too no not never n't lack* less little nothing without only yet however dark sad slow thin flat *less
           problem* trouble* mistake* wrong bland* mediocre* predictable* formulaic* contrived* cliche* familiar
           tired* silly* forced* overlong* long empty hollow* shallow* superficial* uneven* muddled* confus*
           disappoint* unfunny* weak* poor* poorly fail* failed fails failure* cheap* sloppy* clumsy* awkward*
           heavy_handed obvious* excessive* derivative* generic* stale* repetitive* routine* simplistic*
           pretentious* self_indulgent* sentimental* melodrama* manipulative* tedious* plodding* sluggish*
           unsatisfying* uninspired* unimaginative* uninteresting* unintentional* implausible* lifeless* lazy
           annoying* irritating* distasteful* ugly* crude* offensive* mean* boring* sour* sorry waste* wasted
           instead rather unless hardly barely neither nor tries trying attempt* seems cheesy sappy schmaltz*
           hackneyed bloated dreary grating gross* rip_off sitcom instead* merely mostly enough minutes""",
    -2: """bad badly worst awful* terrible* horrible* horrid* dreadful* appalling* atrocious* abysmal* lousy
           stupid* dumb* idiotic* moronic* inept* incoherent* incompetent* unwatchable* unbearable* insufferable*
           dull* bore bored bores boredom lame* mess* messy pathetic* pointless* senseless* worthless* mindless*
           garbage trash* junk crap* ridiculous* laughable* ludicrous* embarrass* disaster* disgust* dreck
           stinker stinks stink hate* hated nasty* tiresome* wearisome* witless humorless humourless charmless
           joyless* soulless* tasteless* useless* lifeless* flawed* vapid* banal* inane* insipid* drab*""",
}

KW = 16        # characters visible to each position (window length)
DH = 35        # head width: 32 char dims + 3 position dims
D_MODEL = KW * DH
SPACE = 26     # the task's ' ' token: virtual left padding so every window is defined
BETA = 30.0    # attention sharpness: a neighbour one position away gets weight e^-BETA
POS0 = 32      # position features (1, t, t^2) live in dims 32, 33, 34 of slot 0
SCORE = 67     # free dim in slot 1: per-position sentiment
OUT = 68       # free dim in slot 1: sequence-averaged sentiment
NEG = 69       # free dim in slot 1: a negator ends at this position
NEAR = 102     # free dim in slot 2: number of negators in the previous KW-1 characters
FLAG = 103     # free dim in slot 2: NEAR clamped to 0 or 1
TIE_POS = 0.5  # the head adds this many sentiment points toward '1' (ties go positive)
GATE = 8.0     # larger than any |SCORE| at one position; closes the flip neurons when FLAG = 0
FLIP_POS = 1.0  # a negated positive word scores -FLIP_POS times its weight
FLIP_NEG = 0.5  # a negated negative word scores +FLIP_NEG times its weight
NEGATORS = "no not never n't without hardly barely neither nor".split()


def parse_lexicon() -> tuple[list[str], list[float]]:
    """Flatten LEXICON; a repeated pattern keeps its first (strongest listed) weight."""
    weights: dict[str, float] = {}
    for w, text in LEXICON.items():
        for pat in text.split():
            if pat.strip("*_") and pat not in weights:
                weights[pat] = float(w)
    return list(weights), list(weights.values())


PATTERNS, PATTERN_WEIGHTS = parse_lexicon()


def pattern_constraints(pat: str, stoi: dict) -> list[tuple[int, list[int]]]:
    """Return (offset back from the firing position, allowed char ids) for each character."""
    left, right = not pat.startswith("*"), not pat.endswith("*")
    body = pat.strip("*").replace("_", " ")
    boundary = [stoi[c] for c in " _=01"]
    cons, off = [], 0
    if right:
        cons.append((0, boundary))
        off = 1
    for c in reversed(body):
        cons.append((off, [stoi[c]]))
        off += 1
    if left:
        cons.append((off, [stoi[" "]]))
    assert off < KW, f"pattern too long: {pat}"
    return cons


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
    """No LayerNorm: channels carry exact counts that normalisation would distort.

    d_ff=0 drops the MLP and use_attn=False drops the attention, so unused
    zero matrices are not counted as parameters.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_attn: bool = True):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads) if use_attn else None
        self.mlp = MLP(d_model, d_ff) if d_ff else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer: read words, find negators, flip scores, pool sentiment."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len + KW, D_MODEL)
        self.blocks = nn.ModuleList([
            Block(D_MODEL, KW, len(PATTERNS)),       # read words
            Block(D_MODEL, KW, 2),                   # count nearby negators, clamp to a flag
            Block(D_MODEL, KW, 2, use_attn=False),   # flip scores where the flag is set
            Block(D_MODEL, KW, 0),                   # pool sentiment at '='
        ])
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        ids = torch.cat([torch.full((B, KW), SPACE, device=ids.device), ids], dim=1)
        pos = torch.arange(T + KW, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)[:, KW:]


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter in closed form."""
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: one-hot character plus position features (1, t, t^2).
        model.token_emb.weight[:, :32] = torch.eye(32)
        t = torch.arange(model.pos_emb.weight.shape[0]).float()
        model.pos_emb.weight[:, POS0:POS0 + 3] = torch.stack([torch.ones_like(t), t, t * t], dim=1)

        # Blocks 1 and 2 attention: head k scores position i as -BETA * (i - (t - k))^2.
        a1, a2 = model.blocks[0].attn, model.blocks[1].attn
        b = BETA * math.sqrt(DH)
        for attn in (a1, a2):
            for k in range(1, KW):
                s = k * DH
                attn.W_q.weight[s + 32, POS0] = -b
                attn.W_q.weight[s + 33, POS0 + 1] = 2 * b
                attn.W_q.weight[s + 33, POS0] = -2 * k * b
                attn.W_q.weight[s + 34, POS0 + 2] = -b
                attn.W_q.weight[s + 34, POS0 + 1] = 2 * k * b
                attn.W_q.weight[s + 34, POS0] = -k * k * b
                attn.W_k.weight[s + 32, POS0 + 2] = 1.0
                attn.W_k.weight[s + 33, POS0 + 1] = 1.0
                attn.W_k.weight[s + 34, POS0] = 1.0
        for k in range(1, KW):
            s = k * DH
            for c in range(32):                 # block 1 copies the character at t-k
                a1.W_v.weight[s + c, c] = 1.0
                a1.W_o.weight[s + c, s + c] = 1.0
            a2.W_v.weight[s, NEG] = 1.0         # block 2 sums the negator flag over t-k
            a2.W_o.weight[NEAR, s] = 1.0

        # Block 1 MLP: each hidden neuron is an AND over its pattern's constraints.
        m1 = model.blocks[0].mlp
        for i, (pat, w) in enumerate(zip(PATTERNS, PATTERN_WEIGHTS)):
            cons = pattern_constraints(pat, task.stoi)
            for off, chars in cons:
                for c in chars:
                    m1.fc1.weight[i, off * DH + c] = 1.0
            m1.fc1.bias[i] = 1.0 - len(cons)
            m1.fc2.weight[SCORE, i] = w
            m1.fc2.weight[NEG, i] = float(pat in NEGATORS)

        # Block 2 MLP: FLAG = relu(NEAR) - relu(NEAR - 1), i.e. min(NEAR, 1).
        m2 = model.blocks[1].mlp
        m2.fc1.weight[0, NEAR] = 1.0
        m2.fc1.weight[1, NEAR] = 1.0
        m2.fc1.bias[1] = -1.0
        m2.fc2.weight[FLAG, 0] = 1.0
        m2.fc2.weight[FLAG, 1] = -1.0

        # Block 3 MLP: neuron 0 reads SCORE > 0 and neuron 1 reads SCORE < 0, both only
        # when FLAG = 1; they replace SCORE with its flipped value.
        m3 = model.blocks[2].mlp
        for i, sign in enumerate((1.0, -1.0)):
            m3.fc1.weight[i, SCORE] = sign
            m3.fc1.weight[i, FLAG] = GATE
            m3.fc1.bias[i] = -GATE
        m3.fc2.weight[SCORE, 0] = -(1.0 + FLIP_POS)
        m3.fc2.weight[SCORE, 1] = 1.0 + FLIP_NEG

        # Block 4 attention: zero q/k give uniform attention, so head 0 averages SCORE.
        a4 = model.blocks[3].attn
        a4.W_v.weight[0, SCORE] = 1.0
        a4.W_o.weight[OUT, 0] = 1.0

        # Head: '1' wins when (mean SCORE) * T + TIE_POS > 0; other symbols are silenced.
        T = model.pos_emb.weight.shape[0] - 1
        one, zero = task.stoi["1"], task.stoi["0"]
        model.head.weight[:, POS0] = -1e3
        model.head.weight[zero, POS0] = 0.0
        model.head.weight[one, POS0] = TIE_POS
        model.head.weight[one, OUT] = float(T)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "e06_negation_flip"
model_description = "e04 plus a negation circuit: offset heads count negators in the previous 15 chars, gated ReLUs flip SCORE (positive to -1x, negative to +0.5x)."


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
