"""
Interpretable transformer for character-level sentiment (sentiment-sst2).

Hand-written weights, no training.
  Block 1: attention shift register, then one AND-gate MLP unit per pattern.
  Block 2: attention finds a recent negator, MLP flips polarity after one.
  Block 3: uniform attention pools the evidence at the final token.
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
# Hand-written lexicon. "word" = whole word, "stem*" = word starting with stem.
# ---------------------------------------------------------------------------

POSITIVE = """
admir* amaz* amus* appeal* astonish* beautiful* beauty best better bold brilliant* captivat*
celebrat* charm* classic clever* comic compassion* compelling confident* convincing cool courag*
dazzl* delicate* delight* deserv* distinctive* effective* elegan* emotional* energy engag* engross*
enjoy* entertain* epic excellen* exceptional* excit* exhilarat* exquisite* extraordinar* fantastic
fascinat* fine finest fresh* fun funnier funny gem generous* genuine* gentle* glorious* good
gorgeous* graceful* grand great greatest grip* happy heartfelt heartwarming hilarious* honest*
hope* humane humor* imaginati* importan* impress* ingenious* insight* inspir* intelligen* intense
intimate intrigu* inventive joy* laugh* likable lively love* lovely lyric* magic* magnificen*
marvel* master* memorabl* moving observant original* passion* perfect* playful* pleasant*
pleasur* poetic* poignan* polish* powerful* pretty profound* provocative quirky rare refresh*
remarkabl* resonan* respect* rewarding rich* riveting romantic satisf* sensitiv* sharp* sincer*
skill* smart* solid special spectacular* spirit* splendid* stirring striking strong* stun* stylish*
sublime* subtle* succe* superb* surprising* suspense* sweet* talent* tender* terrific* thoughtful*
thrill* touch* tragic transcend* treat triumph* truly truth* unexpected* unforgettabl* unique*
uplift* vibran* vivid* warm* wise* wit witty wonder* worth*
""".split()

NEGATIVE = """
too+long too+much absurd* aimless* annoy* artificial awful* awkward* bad badly banal* bland* bloat*
bore* boring cheap* cheesy clich* clumsy* confus* contrived convoluted crass* crude* cynic* dated
derivative disappoint* disaster* disjointed dismal* drags dreadful* dreary dull* dumb* embarrass*
empty endless* exploit* fail* failure fake* flat* flaw* flimsy* forced forgettable formula* foolish*
frustrat* garbage generic gimmick* gross* hack* hate* hollow horribl* idiot* ill implausib* inane*
incoherent* inept* insipid* insult* irritat* junk* lack* lame* laughable lazy* lifeless limp*
listless* lousy* ludicrous* manipulat* mechanical mediocr* melodrama* mess* mildly mindless*
misguided miss* mistake* monotonous* muddl* mundane murky nasty* nothing numb* obnoxious* obvious*
offensive* ordinary overblown overdone overlong overly painful* pathetic* plodding pointless*
ponderous* poor* preachy predictab* pretentious* problem* rambl* recycled redundant repetiti*
ridicul* rote rotten rushed sad sadly sappy senseless* shallow silly simplistic sleep* slight
sloppy* slow* sluggish* smug* soggy soulless stale* stereotyp* stiff* stink* strain* stupid*
suffer* superficial* tacky tasteless* tedious* terribl* thin tired tiresome* trash* trite* ugly
unbearabl* unconvincing uneven unfortunate* unfunny uninspir* uninterest* unlikable unnecessar*
unoriginal unpleasant* useless* vapid* vulgar* waste* weak* wooden worn worse worst worthless*
wrong*
""".split()

# A negator flips the polarity of evidence up to NEG_REACH characters later.
NEGATORS = """
not n't no never none neither nor without nothing hardly barely scarcely isn't wasn't doesn't don't
didn't can't cannot lacks lacking lacked fails failed failing too less short missing rather+than
""".split()

# Weak priors: common review words that lean one way without being judgments.
WEAK_POS = """
film portrait performance performances cast drama life heart human family world culture tale
journey moments and with both own us
""".split()
WEAK_NEG = """
movie plot script screenplay sequel hollywood just only if thing anything something would could
should might seems tries trying even any another minutes hours video bit little enough instead
merely simply mere *less un* dis* mis* over* non* anti* pseudo* *ish
""".split()
WEAK = 0.4

POS_W, NEG_W = 1.0, 1.0
FLIP_POS = 2.0                  # a negated positive counts as this much negative
NEG_REACH = 15
NEGATOR_W = 1.0                 # a negator is itself mild negative evidence
TIE_BIAS = 0.01                 # no evidence at all -> answer positive


def unique_entries() -> list[tuple[str, float, float]]:
    """(pattern, polarity, negator-bit) with duplicates removed, positive wins."""
    seen: dict[str, list[float]] = {}
    for w in POSITIVE:
        seen.setdefault(w.replace("+", " "), [POS_W, 0.0])
    for w in NEGATIVE:
        seen.setdefault(w.replace("+", " "), [-NEG_W, 0.0])
    for w in NEGATORS:
        seen.setdefault(w.replace("+", " "), [0.0, 0.0])[1] = 1.0
    for w in WEAK_POS:
        seen.setdefault(w, [WEAK, 0.0])
    for w in WEAK_NEG:
        seen.setdefault(w, [-WEAK, 0.0])
    out = []
    for w, (pol, flag) in seen.items():
        body = w.strip("*")
        if len(body) + (1 if "*" in w else 2) > W_HEADS:
            w = body[: W_HEADS - 1] + "*"    # too long: truncate to a stem that fits
        out.append((w, pol, flag))
    return out


# Layout: heads of 9 dims. Head i < W_HEADS holds the character at shift i as a
# 6-bit +/-1 code (5 identity bits, 1 boundary bit); the last head is the scratch
# register of named channels below.
BITS, BOUNDARY, DH, W_HEADS = 6, 5, 9, 8
LEXICON = unique_entries()
H = W_HEADS + 1
D = DH * H
SCRATCH = DH * W_HEADS
POS0, POS1, PEV, NEV, NEGATOR, CONST, FLAG, SCORE, TOTAL = range(SCRATCH, SCRATCH + 9)
OMEGA = math.pi / 90
SHARP = 3e5
BIG = 10.0


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head attention with no output projection: heads write in place."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

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
        return (attn @ v).transpose(1, 2).contiguous().view(B, T, D)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm is dropped so that every channel keeps an exact logical meaning."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, D)
        self.pos_emb = nn.Embedding(max_seq_len, D)
        self.blocks = nn.ModuleList([Block(D, H, n) for n in (len(LEXICON), 3, 1)])
        self.head = nn.Linear(D, vocab_size, bias=True)

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

def compile_pattern(entry: str, stoi: dict) -> list[list[int]]:
    """Return, for each shift, the character ids accepted at that shift."""
    stem = entry.endswith("*")
    word = entry.strip("*")
    slots = [[stoi[c]] for c in reversed(word)]
    if not entry.startswith("*"):                 # "*suffix" has no left boundary
        slots = slots + [[stoi[" "]]]
    if not stem:
        slots = [[stoi[" "], stoi["_"]]] + slots  # right boundary
    assert len(slots) <= W_HEADS, entry
    return slots


@torch.no_grad()
def dev_report(model, task) -> None:
    """Accuracy on task examples that are not in the 200 eval examples."""
    model.eval()
    held = {e.prompt for e in task.generate_examples(200, seed=0)}
    dev = [e for e in task.generate_examples(2000, seed=1) if e.prompt not in held]
    ids = torch.tensor([task.encode(e.prompt) for e in dev])
    logits = torch.cat([model(b)[:, -1] for b in ids.split(256)])
    pred = [task.itos[i] for i in logits.argmax(-1).tolist()]
    right = [p == e.answer for p, e in zip(pred, dev)]
    pos = [r for r, e in zip(right, dev) if e.answer == "1"]
    neg = [r for r, e in zip(right, dev) if e.answer == "0"]
    print(f"DEV acc {sum(right)/len(dev):.4f} n={len(dev)} "
          f"tpr {sum(pos)/max(len(pos),1):.3f} tnr {sum(neg)/max(len(neg),1):.3f} "
          f"patterns {len(LEXICON)}")
    # dev[600:] sits past every error printed so far, so it is the clean score.
    clean = right[600:]
    print(f"CLEAN acc {sum(clean)/len(clean):.4f} n={len(clean)}")


def char_code(c: int) -> list[float]:
    """5 identity bits of token id c as +/-1."""
    return [1.0 if (c >> b) & 1 else -1.0 for b in range(5)]


def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    stoi = task.stoi
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: +/-1 bit code in head 0, rotating position in scratch.
        for c in range(task.vocab_size):
            model.token_emb.weight[c, :5] = torch.tensor(char_code(c))
            is_boundary = c in (stoi[" "], stoi["_"])
            model.token_emb.weight[c, BOUNDARY] = 1.0 if is_boundary else -1.0
        for t in range(model.max_seq_len):
            model.pos_emb.weight[t, POS0] = math.cos(OMEGA * t)
            model.pos_emb.weight[t, POS1] = math.sin(OMEGA * t)
            model.pos_emb.weight[t, CONST] = 1.0
        # Position 0 also sets the boundary bit, so a word can start the sentence.
        model.pos_emb.weight[0, BOUNDARY] = 2.0

        # Block 1 attention: head i reads the character i steps back.
        a1 = model.blocks[0].attn
        for i in range(1, W_HEADS):
            base = DH * i
            co, si = math.cos(OMEGA * i), math.sin(OMEGA * i)
            a1.W_q.weight[base, POS0] = SHARP * co
            a1.W_q.weight[base, POS1] = SHARP * si
            a1.W_q.weight[base + 1, POS0] = -SHARP * si
            a1.W_q.weight[base + 1, POS1] = SHARP * co
            a1.W_k.weight[base, POS0] = 1.0
            a1.W_k.weight[base + 1, POS1] = 1.0
            for b in range(BITS):
                a1.W_v.weight[base + b, b] = 1.0

        # Block 1 MLP: one AND gate per pattern, wired to its evidence channels.
        m1 = model.blocks[0].mlp
        for u, (entry, polarity, negator) in enumerate(LEXICON):
            slots = compile_pattern(entry, stoi)
            full_match = 0.0   # any wrong slot lowers the sum by at least 2
            for shift, chars in enumerate(slots):
                if stoi[" "] in chars:
                    m1.fc1.weight[u, DH * shift + BOUNDARY] = 1.0
                    full_match += 1.0
                else:
                    m1.fc1.weight[u, DH * shift : DH * shift + 5] = torch.tensor(char_code(chars[0]))
                    full_match += 5.0
            m1.fc1.bias[u] = -(full_match - 1.0)
            m1.fc2.weight[PEV, u] = max(polarity, 0.0)
            m1.fc2.weight[NEV, u] = max(-polarity, 0.0)
            m1.fc2.weight[NEGATOR, u] = negator
            m1.fc2.weight[SCORE, u] = -NEGATOR_W * negator  # bypasses the flip gate

        # Block 2 attention (scratch head): score(s) = S cos(w(t-s)) + B negator(s).
        # A negator within NEG_REACH beats self-attention, anything further loses.
        a2 = model.blocks[1].attn
        sq = math.sqrt(DH)
        S = 40.0 / (OMEGA ** 2 * NEG_REACH)
        B = S * (1 - math.cos(OMEGA * (NEG_REACH + 0.5)))
        a2.W_q.weight[POS0, POS0] = S * sq
        a2.W_q.weight[POS1, POS1] = S * sq
        a2.W_k.weight[POS0, POS0] = 1.0
        a2.W_k.weight[POS1, POS1] = 1.0
        a2.W_q.weight[CONST, CONST] = B * sq
        a2.W_k.weight[CONST, NEGATOR] = 1.0
        a2.W_v.weight[FLAG, NEGATOR] = 1.0

        # Block 2 MLP: pass evidence if not flagged; a flagged negative is dropped
        # and a flagged positive turns negative.
        m2 = model.blocks[1].mlp
        for u, (src_ch, sign) in enumerate(((PEV, -BIG), (NEV, 0.0), (PEV, BIG))):
            m2.fc1.weight[u, src_ch] = 1.0
            m2.fc1.weight[u, FLAG] = sign
            m2.fc1.bias[u] = 0.0 if sign <= 0 else -BIG
        m2.fc2.weight[SCORE, 0] = 1.0
        m2.fc2.weight[SCORE, 1] = -1.0
        m2.fc2.weight[SCORE, 2] = -FLIP_POS

        # Block 3 attention: q = k = 0 gives uniform attention, so sum = mean * T.
        a3 = model.blocks[2].attn
        a3.W_v.weight[TOTAL, SCORE] = 81.0

        # Output: sign of the total evidence picks '1' or '0'.
        model.head.bias.fill_(-1e3)
        model.head.bias[stoi["1"]] = TIE_BIAS
        model.head.bias[stoi["0"]] = 0.0
        model.head.weight[stoi["1"], TOTAL] = 1.0
        model.head.weight[stoi["0"], TOTAL] = -1.0

    dev_report(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "core_w8_negw1"
model_description = "core_w8_trunc with positive and negative words weighted equally, testing whether the separate negative weight can go."


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
