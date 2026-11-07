"""
Hand-written (never trained) transformer for SST-2 phrase sentiment.

Circuit:
  L1 attention : head h attends to the character h+1 steps back (sin/cos position match) and copies it
                 into residual block h+1  -> every position sees its last 9 characters.
  L1 MLP       : one ReLU neuron per lexicon entry = AND over its characters (+ word boundary);
                 writes +-polarity into the (P+, P-) channel.
  L2 attention : head 0 averages the polarity over the prefix.
  head         : logit('1') - logit('0') ~ summed polarity (tie -> '1').
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
# Lexicon: "word" = whole word (space/pad must follow), "stem*" = any word starting with stem,
# entries with spaces are short phrases.  Weight = polarity written by the neuron.
# ---------------------------------------------------------------------------
POS_STRONG = """excellent wonderful* brilliant* superb terrific fantastic masterpiece hilarious beautiful*
gorgeous delightful stunning riveting moving powerful charming remarkable fascinating funniest best great
love loved captivating engrossing splendid magnificent spectacular extraordinary outstanding touching
irresistible entertain* enjoyable compelling heartwarming intelligent inventive thoughtful witty clever lovely
amazing amusing exhilarating flawless""".split()
POS_MILD = """good fun funny nice fine solid sweet warm rich fresh smart sharp strong engaging impressive effective
pleasing pleasant appealing lively refreshing genuine honest original memorable haunting stylish polished
graceful elegant accomplished gem treat joy delight* better enjoy* admire* inspir* beauty heartfelt sincere
tender poignant uplifting magical dazzling epic unforgettable evocative spirited likable lovable endearing
winning worth worthy recommend* ambitious richly crisp bold daring dynamic energetic laughs wow sensitive
subtle nuanced satisfying thrilling exciting absorbing intriguing gripping masterful deft vivid successful
pleasure hope exquisite mesmerizing nicely attractive grand exceptional insightful compassion* finesse zippy
amiable gloriously flair wit originality creative imaginative innovative ingenious authentic believable
rewarding worthwhile gentle affecting heartbreaking profound* wise skillful talented genius uproarious
joyful cheerful happy glorious majestic lush breathtaking marvel* sparkling sublime stirring vibrant playful
whimsical quirky loving pretty cute adorable humane astonishing smooth professional dependable welcome
sensational courage triumph* engag* involving""".split()
NEG_STRONG = """worst awful terrible horrible dreadful stupid boring bore tedious dull mess bad pointless lame unfunny
mediocre painful ugly trash garbage waste disappointing annoying tiresome dreary lifeless unwatchable abysmal
atrocious ridiculous junk flat forgettable disappoint* predictab* clich*""".split()
NEG_MILD = """poor weak silly trite derivative cheap tired slow empty shallow bland hollow plodding pretentious
unpleasant* offensive failure fails fail flawed sloppy clumsy awkward incoherent confusing murky limp stale
lazy dumb absurd tasteless vulgar grating obnoxious repetitive redundant bloated overwrought melodramatic
contrived forced inept amateurish unbearable insipid vapid worthless lacks lack fake generic formulaic
uninspired unoriginal uneven unconvincing unfocused drags mindless stilted wooden humorless hackneyed shoddy
crude grotesque gross nasty disgusting inane embarrassing laughable sadly unfortunately wasted mechanical
sluggish ponderous soulless bleak disjointed rehash* exhausting grim ludicrous preachy stiff smug sappy
saccharine cloying tepid thin unnecessary needless misguided muddled meandering shapeless trivial bogus
phony nothing never n't annoy* irritat* mediocr* pointles* botched collapse idiocy idiot* despicable devoid
hokey insult* nausea dry frustrat* sour unsuccessful stereotyp* hoary dismiss moody negativity punishment
questionable simplistic uneasy loose poorly sucks stink* tedium bored pathetic ineffective irrelevant fiasco
disaster* hate* hated dislike overrated overlong overdone overly excessive pedestrian ordinary
unimaginative banal manipulative convoluted messy laborious strained shrill unattractive unappealing
uninteresting not bomb dud drivel crap rotten lousy dismal depressing miserable pitiful wretched appalling
hideous vile cruel senseless nonsensical yawn sleep* snooze slog stagnant flaccid toothless lackluster
inconsequential imitation retread recycled unrealistic implausible artificial gimmick* tacky cheesy corny
childish juvenile crass sorry flop flops blunder* hackney* tiring weary wasteful maudlin neither nor rote
trouble hopeless cynical rage unacceptable devastating* tragic impersonal intolerable sinister feeble
twisted superficial* soporific charmless torture disturbing creepy inferior clunky obvious flimsy
unlikable unsympathetic gloomy unremarkable undistinguished unmemorable unsatisfying unexciting
unimpressive underwritten mawkish subpar substandard tawdry useless""".split()
NEG_PHRASES = """not good|not funny|not fun|not great|not very|not even|not much|not as|not so|no wit|no point|
no sense|n't work|n't care|n't add|n't make|n't have|n't want|n't quite|n't funny|n't good|n't much|n't been|
n't get|no joy|no one|no real|no new|too much|too many|too long|too little|too slow|too late|too dark|too busy|
too silly|too clever|so bad|so far|never quite|never really|nothing new|little more|far too|just too|
without any|without a|lacks a|lack of|fall flat|falls flat|falls short|no surprises|so stupid""".replace("\n", "").split("|")
POS_PHRASES = """well made|well done|well worth|well acted|well written|well crafted|well drawn|well shot|a joy|
a treat|a gem|a delight|a must|must see|a winner|a hoot|a blast|big hit|high marks|first rate|top notch|
one of the best|the best|very good|so good|so funny|so much fun|much fun|a lot of fun|rare treat|feel good|
good natured|good time|good fun|good humor|heart warming|laugh out|deeply moving|at its best|rings true|
worth seeing|worth a look|worth the|high spirits|big heart|real heart|smart and|funny and|fresh and|
sharp and|charming and|warm and|fun and|moving and""".replace("\n", "").split("|")

NEG_MILD += """seesaw* sappi* choppy dehumaniz* airless prepackaged broken milquetoast dupe moralis* tamer cheesi*
baffled quick buck ransack* cheapo degraded gratuitous wannabe underdone""".split()
POS_MILD += """genial buoyant diversion inviting decency unique timely adequate interesting interestingly charm
sophisticated intrigue colorful fluffy okay prefer potential success* charged passion* heroes perfect* pleasingly
unusual hearted insight* likeable lyrical romantic sexy tender clever* delicious fabulous fun* graceful* hug
kind kindness laugh* lovely* magic* masterful* power* praise* provocative reward* rich* smile* star* stun* sweet*
terrif* thank* treasure* unforgettable* wow* yes""".split()
NEG_MILD += """boring* bored* dull* lame* stupid* awful* bad* worse worst* ugly* poor* cheap* cliche* mess* weak*
ridiculous* laughable* nonsense* tedious* tiresome* trash* junk* flop* fail* dumb* hate* horrib* terribl*
dreary* depress* disast* disgust* offens* annoy* irritat* frustrat* pathet* mediocre* pointless* worthless*
predict* repetit* unfunny* waste* wasted* garbage* crap* sucks* stale* lifeless* lackluster* flat*""".split()
NEG_MILD += """dopey struggl* undramatic reek* impossible aground commercialis* strain virulent foul frantic
mildly unsuccessful* desired""".split()
NEG_MILD += """crippled dawdle* anemic lewd venality flicker* sterile lack* detached mumbo jumbo haphazard* dodge
farts excuse fell apart wore down apart jerking""".split()
POS_MILD += """honor landmark nifty dreamy durable adequate pizazz innovation""".split()
NEG_MILD += """queasy* horror* horrif* fetid lumbering scant tossed forgotten unromantic unentertaining pitifully
exploitive obligatory lament* bother rip""".split()
NEG_MILD += """drown* problematic evade inactive oblivious anachronistic frightful vanity misuse weirdness
off game stultifying uselessly shamelessly grubbing""".split()
NEG_PHRASES += ["no insight", "no new insight", "sorely lacking", "off his game", "no sense of"]
NEG_PHRASES += ["run of the", "no business", "rip off", "low rent", "hard to like", "too easy", "too crude"]
NEG_PHRASES += ["money back", "pay per view", "wait to see", "instead of", "run amok", "just another"]
NEG_PHRASES += ["all time low", "sit through", "to be desired", "unless you"]
STRONG, MILD, PHRASE = 1, 1, 1
LEXICON: dict[str, int] = {}
for _w in POS_MILD: LEXICON[_w] = MILD
for _w in NEG_MILD: LEXICON[_w] = -MILD
for _w in POS_STRONG: LEXICON[_w] = STRONG
for _w in NEG_STRONG: LEXICON[_w] = -STRONG
USE_PHRASES = False                          # ablation switch: hand-listed multi-word phrases
if USE_PHRASES:
    for _w in NEG_PHRASES: LEXICON[_w] = -PHRASE
    for _w in POS_PHRASES: LEXICON[_w] = PHRASE
# Generated negation phrases: "not X" / "n't X" for every short positive word X (flips it, net -1 or less).
for _w in list(LEXICON):
    if LEXICON[_w] > 0 and ' ' not in _w and not _w.endswith('*') and len(_w) <= 4:
        for _neg in ():   # ablation: generated negation phrases disabled ("not ", "n't ")
            LEXICON[_neg + _w] = -2
# A negative word that contains a positive word as its tail (un-funny, un-pleasant, dis-like...) must also cancel
# the embedded positive neuron, which fires inside it: weight -2.
for _w in list(LEXICON):
    if LEXICON[_w] < 0:
        for _pre in ():   # ablation: prefix cancel disabled ("un", "in", "non", "dis", "mis")
            if _w.startswith(_pre) and LEXICON.get(_w[len(_pre):].rstrip('*'), 0) > 0:
                LEXICON[_w] = -2
PATTERNS = list(LEXICON.items())


# ---------------------------------------------------------------------------
# Architecture
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
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 210,
        n_heads: int = 5,
        n_layers: int = 2,
        d_ff: int = len(PATTERNS),
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # only layer 1 needs a real MLP (the word detectors); layer 2 is attention-only (MLP width 1, zeroed)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff if i == 0 else 1) for i in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    D, H = model.d_model, model.n_heads
    dh = D // H
    stoi = task.stoi
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():                    # LayerNorms are plain normalisers (gain 1, bias 0)
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)

        # ---- residual stream layout (every written vector sums to 0 so LayerNorm only rescales)
        TOK, CMP, POS, PP, PM, BLK = 0, 32, 33, 45, 46, 47   # BLK + 32*(j-1): chars seen j steps back
        THETA = [2 * math.pi / 96, 2 * math.pi / 8, 2 * math.pi / 5]
        EPS = 1e-5

        def ln_scale(norm_sq):  # LayerNorm gain for a zero-mean vector with this squared norm
            return 1.0 / math.sqrt(norm_sq / D + EPS)

        # ---- embeddings: one-hot char, constant -1 "compensator", antipodal sin/cos position code
        for c in range(model.vocab_size):
            model.token_emb.weight[c, TOK + c] = 1.0
            model.token_emb.weight[c, CMP] = -1.0
        for s in range(model.max_seq_len):
            for f, th in enumerate(THETA):
                b = POS + 4 * f
                model.pos_emb.weight[s, b:b + 4] = torch.tensor(
                    [math.cos(th * s), math.sin(th * s), -math.cos(th * s), -math.sin(th * s)])
        a1 = ln_scale(2.0 + 2.0 * len(THETA))  # |x0|^2 = 1 (tok) + 1 (cmp) + 2 per frequency (pos)

        # ---- layer 1 attention: head h copies the character h+1 steps back into block h+1
        blk = model.blocks[0]
        G = 8.0
        for h in range(H):
            j, r = h + 1, h * dh
            for f, th in enumerate(THETA):
                c_, s_ = POS + 4 * f, POS + 4 * f + 1
                cj, sj = math.cos(th * j), math.sin(th * j)
                # query = pos code rotated back by j steps; key = pos code  =>  score ~ sum_f cos(th_f (s-(t-j)))
                blk.attn.W_q.weight[r + 2 * f, c_] = G * cj
                blk.attn.W_q.weight[r + 2 * f, s_] = G * sj
                blk.attn.W_q.weight[r + 2 * f + 1, s_] = G * cj
                blk.attn.W_q.weight[r + 2 * f + 1, c_] = -G * sj
                blk.attn.W_k.weight[r + 2 * f, c_] = 1.0
                blk.attn.W_k.weight[r + 2 * f + 1, s_] = 1.0
            for c in range(32):                      # value = char one-hot; out -> block j
                blk.attn.W_v.weight[r + c, TOK + c] = 1.0
                blk.attn.W_o.weight[BLK + 32 * (j - 1) + c, r + c] = 1.0
            blk.attn.W_v.weight[r + 32, CMP] = 1.0   # compensator keeps the sum at zero
            blk.attn.W_o.weight[CMP, r + 32] = 1.0

        # ---- layer 1 MLP: one neuron per lexicon entry = AND over the last few characters
        norm2 = 1 + 2 * len(THETA) + (1 + H * a1) ** 2 + H * a1 ** 2
        k2 = a1 * ln_scale(norm2)                    # LN2 gain applied to a matched character
        ends = [stoi[' '], stoi['_'], stoi['=']]
        for n, (pat, pol) in enumerate(PATTERNS):
            stem = pat.endswith('*')
            word = pat.rstrip('*')
            if (not stem and len(word) > H) or len(word) > H + 1:   # window only reaches H chars back
                word, stem = word[:H + 1], True
            L = len(word)
            conds = []                               # (offset, char)
            for i, ch in enumerate(word):
                conds.append((L - 1 - i if stem else L - i, stoi[ch]))
            ncond = len(conds) + (0 if stem else 1)
            for off, c in conds:
                if off == 0:
                    blk.mlp.fc1.weight[n, TOK + c] = a1
                else:
                    blk.mlp.fc1.weight[n, BLK + 32 * (off - 1) + c] = 1.0
            if not stem:                             # word must be followed by space / pad / '='
                for c in ends:
                    blk.mlp.fc1.weight[n, TOK + c] = a1
            blk.mlp.fc1.bias[n] = -(ncond - 0.5) * k2
            amp = 2.0 * pol / k2                     # neuron fires with value 0.5*k2
            blk.mlp.fc2.weight[PP, n] = amp
            blk.mlp.fc2.weight[PM, n] = -amp

        # ---- layer 2 attention: head 0 averages the polarity over the whole prefix
        b2 = model.blocks[1]
        RECENCY = 0.0                             # attention logit per unit of cos(theta_0 (t - s)); 0 = plain mean
        if RECENCY:
            g2 = RECENCY * math.sqrt(dh) / ln_scale(norm2) ** 2
            for q_in in (POS, POS + 1):              # query and key both read the slow (period 96) sin/cos pair
                b2.attn.W_q.weight[q_in - POS, q_in] = g2
                b2.attn.W_k.weight[q_in - POS, q_in] = 1.0
        b2.attn.W_v.weight[0, PP] = 1.0
        b2.attn.W_v.weight[1, PM] = 1.0
        b2.attn.W_o.weight[PP, 0] = 1.0
        b2.attn.W_o.weight[PM, 1] = 1.0

        # ---- readout: logit('1') ~ total polarity (+ tiny positive prior so ties -> positive)
        for tok, sgn, prior in (('1', 1.0, 0.002), ('0', -1.0, 0.0)):
            model.head.weight[stoi[tok], PP] = sgn
            model.head.weight[stoi[tok], PM] = -sgn
            # '=' only occurs at the last position: a constant boost that lifts '0'/'1' above every other token
            model.head.weight[stoi[tok], TOK + stoi['=']] = 1.0 + prior


model_shorthand_name = "W5_PlainWords_G8"
model_description = ("Pure single-word/stem vote counter (0.815) with 5 gather heads and attention gain doubled to 8 "
                     "(sharper position selection).")


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
