"""
Interpretable transformer for sentiment-sst2 (never trained; every weight is written by hand).

Circuit:
  Block 1 attention: one head per character offset k, attending exactly to position t-k
                     (Gaussian attention from quadratic position features), so each position
                     sees its last KW characters as one-hot slots.
  Block 1 MLP:       one hidden neuron per lexicon pattern, an exact AND over character
                     constraints; fc2 adds the pattern's sentiment weight to the SCORE channel.
  Block 2 attention: uniform attention at '=' averages SCORE over the sequence.
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
          stunning* dazzling* delight* enchanting* beautiful* gorgeous* perfect*
          best finest greatest great greatly amazing* fantastic* remarkable* remarkably compelling* powerful*
          moving touching heartwarming* charm* engrossing* engaging* entertaining* enjoyable* hilarious*
          funny funniest fun love loved loves lovely triumph* gem wit witty smart intelligent* clever* inventive*
          imaginative* original refreshing* poignant* thrill* treat winning* memorable* wise""",
    1: """good better nice fine solid strong rich warm tender gentle sincere honest heartfelt
          impressive* interesting* intriguing* satisfying* pleasant* pleasure* pleasing* appealing* likable likeable
          vivid spirited energetic* fascinating* insightful* thoughtful* thought_prov*
          beauty humor humour humorous* laugh laughs laughter joy* playful* inspired inspiring* uplifting*
          sensitive* subtle* elegant* graceful* stylish* style skillful* skilled* talented* talent* gifted
          confident* assured ambitious* tasty sexy cool nicely deft* crisp* affecting* absorbing*
          success* succeeds* effective* hope* truth* rewarding* surprising*
          kind* unique* cute* sly* quirky* worth worthy* tribute* compassion* intimate* excit*
          classic* pure* rare* passion* intelligence spectacular*
          entertainment* inventiveness* easy sweet* smile* radiant* lovingly affection*
          meaningful* authentic* genuine* believable* convincing* credible* accomplished* polished* sophisticated*
          nuanced* layered* textured* lyrical* poetic* haunting* mesmeriz* hypnotic* transcend* resonant*
          resonat* profound* provocative* illuminating* enlightening* bracing* invigorating*
          exuberant* infectious* irresistible* winning* endearing* winsome* sparkling* buoyant* cheerful* giddy*
          hearty* wry* droll* amusing* amuses amused gripping* suspense* taut*
          vibrant* colorful* lush* sumptuous* striking* inspire* courage* brave* bold* daring* audacious* fearless*
          celebrat* embrace* redeem* redemption* honesty* humanity* dignity* grace* warmth* tenderness* generous*
          terrifically nicely handsome* appeal* admire* admirable* respect* recommend* satisf* magic* miracle*
          glorious* splendid* sublime* exquisite* superior* first_rate top_notch tour_de_forc* wonderment
          fresh* lively* alive vital* strength* grand* epic* rousing* stirring* heroic* touching*
          friendly* sharp* savvy* perceptive* observant* eloquent* articulate* insight* wisdom
          must_see worth_seeing well_made well_acted well_done finely* expertly* skillfully* gracefully*
          deeply* richly* warmly* tenderly* movingly* hilariously* wickedly* delicately*
          artistry* artful* career_best crowd_pleas* feel_good good_natured""",
    -1: """less little sad slow thin flat
           problem* trouble* mistake* wrong bland* mediocre* predictable* formulaic* cliche*
           tired* silly* forced* overlong* empty hollow* shallow* superficial* uneven* muddled* confus*
           disappoint* unfunny* weak* poor* fail* cheap* sloppy* clumsy* awkward*
           heavy_handed obvious* excessive* derivative* generic* stale* repetitive* routine* simplistic*
           pretentious* self_indul* sentimental* melodrama* manipulative* tedious* plodding* sluggish*
           unsatisfying* uninspired* unimaginative* uninteresting* unintentional* implausible* lifeless* lazy
           annoying* irritating* distasteful* ugly* crude* offensive* mean mean_spirit* boring* sour* sorry waste* wasted
           unfortunately* sadly regrett* pain* suffer* struggl* stiff* wooden* flimsy* limp*
           numbing* exhaust* exasperat* frustrat* unpleasant* unlikable unconvincing* unsure* undone* unfocused*
           dated overdone* overwrought* overblown* overacted* overly overbearing* underdevelop* underwritten*
           misguided* misfire* miss* nonsens* preposterous* absurd* farce* hokey* corny* trite*
           pedestrian* amateur* sophomoric* juvenile* childish* gimmick* cynical* creepy* vulgar* tawdry* sleazy*
           meandering* rambling* aimless* bewildering* convoluted* murky* choppy* disjointed* slack* thinly
           kills* killed dead* lost losing loses drag* fizzle* sink* falls collapse* crass* grim*
           dismal* depress* gloomy* bleak* unnecessary* needless* shoddy* slapdash* ineffective* mediocrity
           instead unless hardly barely neither nor tries trying attempt* seems cheesy sappy schmaltz*
           hackneyed bloated dreary grating gross* rip_off sitcom merely
           nobody none nowhere fake* phony* artificial* stupidly oddly strangely
           sadistic* unwieldy* unable incapable* impossible* inadequate* inferior* insult* ill
           loser* cheat* lies lie liar* fool* silliness* sillier worse poorer weaker
           awkwardly* clumsily* lamely* tediously* dully plot_holes tv_movie straight_to*
           sequel* remake* formula* cartoonish* caricature* stereotyp* contriv* unoriginal* recycled* rehash*
           forgettable* forgotten* unmemorable* overcooked* undercooked* half_baked half_hearted*""",
    -2: """no not never n't nothing too lack* without bad badly worst awful* terrible* horrible* horrid* dreadful* appalling* atrocious* abysmal* lousy
           stupid* dumb* idiotic* moronic* inept* incoherent* incompetent* unwatchable* unbearable* insufferable*
           dull* bore bored bores boredom lame* mess* pathetic* pointless* senseless* worthless* mindless*
           garbage trash* junk crap* ridiculous* laughable* ludicrous* embarrass* disaster* disgust* dreck
           stinker stinks stink hate* nasty* tiresome* wearisome* witless humorless humourless charmless
           joyless* soulless* tasteless* useless* flawed* vapid* banal* inane* insipid* drab*""",
}

KW = 8         # characters visible to each position (window length)
NC = 32        # one-hot width of a character slot
SPACE = 26     # the task's ' ' token: one virtual space before the text; heads reaching past it land on it
BETA = 30.0    # attention sharpness: a neighbour one position away gets weight e^-BETA
POS0 = KW * NC # position features (1, t, t^2) follow the KW character slots
SCORE = POS0 + 3   # per-position sentiment
OUT = POS0 + 4     # sequence-averaged sentiment
D_MODEL = POS0 + 5
TIE_POS = 0.5  # the head adds this many sentiment points toward '1' (ties go positive)


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
    if len(body) + right + left > KW:   # too long for the window: match its opening as a prefix
        right, body = False, body[:KW - left]
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
    def __init__(self, d_model: int, n_heads: int, d_key: int, d_value: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value
        self.W_q = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_value, bias=False)
        self.W_o = nn.Linear(n_heads * d_value, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dk, dv = self.n_heads, self.d_key, self.d_value
        q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dv).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dv)
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

    d_ff=0 drops the MLP, so a pooling block carries no unused zero matrices.
    """

    def __init__(self, d_model: int, n_heads: int, d_key: int, d_value: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_key, d_value)
        self.mlp = MLP(d_model, d_ff) if d_ff else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer: block 1 reads words, block 2 pools their sentiment."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len + 1, D_MODEL)
        self.blocks = nn.ModuleList([
            Block(D_MODEL, KW - 1, 3, NC, len(PATTERNS)),   # read words (offsets 1..KW-1)
            Block(D_MODEL, 1, 1, 1, 0),                 # pool sentiment at '=' (one scalar head)
        ])
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        ids = torch.cat([torch.full((B, 1), SPACE, device=ids.device), ids], dim=1)
        pos = torch.arange(T + 1, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)[:, 1:]


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

        # Block 1 attention: head k scores position i as -BETA * (i - (t - k))^2 and
        # copies the character it finds there into slot k.
        a1 = model.blocks[0].attn
        b = BETA * math.sqrt(3)
        for k in range(1, KW):
            h = 3 * (k - 1)                     # this head's three key dims
            a1.W_q.weight[h, POS0] = -b
            a1.W_q.weight[h + 1, POS0 + 1] = 2 * b
            a1.W_q.weight[h + 1, POS0] = -2 * k * b
            a1.W_q.weight[h + 2, POS0 + 2] = -b
            a1.W_q.weight[h + 2, POS0 + 1] = 2 * k * b
            a1.W_q.weight[h + 2, POS0] = -k * k * b
            a1.W_k.weight[h, POS0 + 2] = 1.0
            a1.W_k.weight[h + 1, POS0 + 1] = 1.0
            a1.W_k.weight[h + 2, POS0] = 1.0
            for c in range(NC):
                a1.W_v.weight[NC * (k - 1) + c, c] = 1.0
                a1.W_o.weight[NC * k + c, NC * (k - 1) + c] = 1.0

        # Block 1 MLP: each hidden neuron is an AND over its pattern's constraints.
        m1 = model.blocks[0].mlp
        for i, (pat, w) in enumerate(zip(PATTERNS, PATTERN_WEIGHTS)):
            cons = pattern_constraints(pat, task.stoi)
            for off, chars in cons:
                for c in chars:
                    m1.fc1.weight[i, off * NC + c] = 1.0
            m1.fc1.bias[i] = 1.0 - len(cons)
            m1.fc2.weight[SCORE, i] = w

        # Block 2 attention: zero q/k give uniform attention, so head 0 averages SCORE.
        a2 = model.blocks[1].attn
        a2.W_v.weight[0, SCORE] = 1.0
        a2.W_o.weight[OUT, 0] = 1.0

        # Head: '1' wins when (mean SCORE) * T + TIE_POS > 0; other symbols are silenced.
        T = model.pos_emb.weight.shape[0] - 1
        one, zero = task.stoi["1"], task.stoi["0"]
        model.head.weight[:, POS0] = -1e3
        model.head.weight[zero, POS0] = 0.0
        model.head.weight[one, POS0] = TIE_POS
        model.head.weight[one, OUT] = float(T)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "e46_drop_unused_head"
model_description = "e45 with the never-used head 0 removed: exactly seven heads, one per character offset 1 to 7."


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
