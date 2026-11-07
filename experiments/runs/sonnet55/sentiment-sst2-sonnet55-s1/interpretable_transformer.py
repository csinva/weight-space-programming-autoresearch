"""
Interpretable transformer for character-level sequence tasks (task: sentiment-sst2).

Hand-written circuit (no training, no LayerNorm), 3 layers:
  L1 attention: head k (k=1..N_SHIFT-1) attends to position p-k (sinusoidal position match) and copies
      that character's one-hot into slot k -> each position holds a window of its last N_SHIFT characters.
  L1 MLP: one ReLU neuron per lexicon stem (" good", " bad", ...). Fires iff the window ends with the stem.
      Writes +w / -w to SCORE.  Negator neurons (" not ", " n't", ...) write NEGF (flag) and CUE (-1).
  L2 attention: score(p, j) = cos(w(p-j)) + NEG_B*NEGF_j. Self wins unless a negator is within ~14 chars,
      so NEGNEAR = "negator shortly before me".  L2 MLP: 2 neurons flip SCORE's sign when NEGNEAR = 1.
  L3 attention: uniform average of SCORE + CUE over all positions -> AVG.
  Head: logit('1') = AVG + c1, logit('0') = -AVG + c0   (sign of the summed lexicon score).
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
    def __init__(self, d_model: int, n_heads: int, d_head: int):
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
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer without LayerNorm (so circuits stay exactly readable)."""

    def __init__(self, vocab_size, max_seq_len, d_model, layer_cfgs):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = len(layer_cfgs)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, h, dh, f) for (h, dh, f) in layer_cfgs])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Lexicon: stems are matched as " stem" (leading space, first 6 letters), ending at the current character.
# Words containing '-' are skipped (not in the vocabulary).
# ---------------------------------------------------------------------------

POS = """good great best funny fun beautiful love charm enjoy wonderful excellent brilliant delight
entertain engaging moving powerful smart clever fascinat touching hilarious perfect stunning superb
remarkab masterfu masterp rich warm fresh compelli impressi amazing terrific heart strong original
intellig witty gem solid well fine thrill pleasur inventiv sweet satisfy honest uplift lively crisp
poignant cool gorgeou inspir vivid nicely appeali captiva refresh importa human worth vibrant dazzl
genuine lovely sharp elegant fantast gripping beauty favorit tender engross sophist funniest joy
exhilar ingenio witty riveting spirited magnifi enthrall stirring accessi nuanced absorbi graceful
unique triumph charmin hip fully thoughtf wise heartfel smartly breath masterly deeply sincere
playful moves resonan haunting quirky effecti flawless""".split()

NEG = """bad worst dull boring mess fail tedious stupid waste poor flat lame weak annoying pointles
mediocre unfunny predicta cliche clich silly ugly awful terribl horribl disappo nothing lack trite
bland worse dumb pretenti plod clumsy forgetta empty tiresom irritat insult painful ridicul sloppy
muddled stale generic incohere shallow mindles slow lifeless repetit offensi cheap bore uninspi
pathetic tired overlon dreary unfortu neither fake contriv drag trash junk unpleas sadly lacks
worthle unconvin crude hollow stilted gimmick tasteles mundane overwro sluggish ineptly inept
trivial rehash derivat obvious mindless soulles unintent annoy fails nonsens rambling
disjoint misguid murky seldom unnecess amateur bogus dreadful atrocio laughab uneven
stiff mildly exhaust failure bomb wasted""".split()

MORE_POS = """romp laugh comic humor humour wit charis likab admir adora affect agile alive artful artist
astute attract auth bold bright brave buoyan captur celebra classic comfort confid courag creativ cute
dramat dynamic eager earnest emotion energ engag enchant enlight ethere exceptio exciting extraor
fierce fluid fond forceful frisky gentle gift glorio glow gold grace grand gratif happy
harmoni heroic hopeful hypnot imagina immers incisiv ingenu insight intens intrigu invigor kind
legend likeable luminou lush marvel mesmer meticul mild miracul natural noble oscar outstan
passion pleasan polish popular positive precise prais precious profoun proud pure radiant rare
reward riveted robust romant sassy savvy seduct sensit sensual serene sexy shrewd skill sleek
smooth snappy soar soulful spectac splendi stellar stylish sublime succe sumptuo sunny surpris
swift tasteful thoughtf thrilli tight timeless tough tremend trium truly trust twist unforget
upbeat valuabl vibran virtuos vitalit warmth wholesom winning wonder worthy zest zippy resourc
rousing exuber playfu delicat elabora epic evoc exquisit fabulou fantas flair""".split()

MORE_NEG = """abysmal aimless alienat angry annoyi appall arrogan artless awkward badly banal barely
bitter bizarre bleak blunt bombas boredom botch brutal bumbl callous careles chaotic chore clunky
coarse comatos complic condesc confus convolu corny crass creepy cringe cruel cynical
damp deadly deceit defici dense depress derail desper dim disast discord disgust dismal disorie
dispiri dreadf drab drivel dud dubious dumbed dystop effort embarr emotion enervat erratic excess
exploit failed fatuous feeble fetid filler flaccid flimsy flounder foolish forced formula frantic
frivol frustra futile gaudy ghastly gloomy grating grim gross grotesq gruesom halfhea hackney
hamfist harsh hateful heavyha hokey hopeles horrend humorle hype idiot ignoran implaus inane
inapt incompe indiffe inferio inflate insipid insuffe interm irrelev jarring jumble lackadai
languid laughs lazy leaden lethargi limp listles lousy ludicro lumber mawkish meander meaning
meh melodra messy misfire mistake monoton moronic muddle nasty nauseat negativ nonsen noisy
numbing obnoxio odd overbea overcoo overdon overkil overlon padded pale pandering
passive pedestr pedantic perfunc pitiful plain plodding ponder poorly preachy pretent
pseudo puerile pulp puny random ramble rancid repulsi resist ridicu rote rubbish ruin
sappy savage scant scarce schlock scream sedate sentimen shabby shame shoddy shrill sick
simplis sinister skimpy sleep sloppy smug sneaky snooze sophomo sour spiritl squand stagnan
stereoty stifled stilted stinks stink strained stretch stuffy stunted subpar suffer
sugary superfi tacky tame tasteles tawdry tepid thin thudd tiring toothles torpid tortur tragic
trashy trying tuneles turgid unambit unappea unbeara unconvi uncompel undevel undone
unfocus unfunny unimagin unimpre uninvol unlikab unoriginal unremark unwatch unwieldy uptight
vacuous vapid vague vulgar wan wasteful weary weirdl whiny witles woeful wooden woeful
wrongh yawn lacking too hardly never without barely mediocr""".split()

# weak function-word cues (weight WEAK_W)
WEAK_NEG = "even less least mere merely tries try attempt seems seem suppose instead nor sit rest".split()
WEAK_POS = "both deliver gives offers manages succeed easily ably remains always also enough better".split()

# unanchored word-internal endings
SUFFIXES = {"less": -1.0, "lessly": -1.0, "dull": -1.0, "tedium": -1.0,
            " lack": -1.0, " fails": -1.0, " mess": -1.0}

NEGATORS = [" not ", " n't", " no ", " never", " withou", " hardly", " nor ", " barely"]


N_SHIFT = 7          # window = last 7 characters (shifts 0..6)
WEAK_W = 0.5
V = 32               # vocab size
SW = V + 1           # slot width: one-hot char + BOS flag (dim V)
CS = N_SHIFT * SW    # cos(w p), sin(w p) positional dims
CONST = CS + 2
SCORE = CS + 3
AVG = CS + 4
NEGF = CS + 5        # layer 1: "this position ends a negator word"
NEGNEAR = CS + 6     # layer 2: "a negator ended within ~R chars before this position"
CUE = CS + 7         # negator's own (unflipped) negative score
D_MODEL = CS + 8
D_HEAD = SW
CUE_W = 1.0
FLIP = 2.0           # 2.0 = full sign flip
NEG_B = 0.227        # gate width: negator counts if cos(w*dist) + NEG_B > 1  (dist <~ 14 chars)
PRIOR = 0.006        # logit('1') - logit('0') bias; one lexicon hit of weight 1 contributes ~0.012


def lexicon():
    """stem -> score.  Positive words +1, negative -1, weak cues +-WEAK_W."""
    words = {}
    for w in POS:
        words[" " + w[:7]] = +1.0
    for w in NEG:
        words[" " + w[:7]] = -1.0
    for w in MORE_POS:
        words.setdefault(" " + w[:7], +1.0)
    for w in MORE_NEG:
        words.setdefault(" " + w[:7], -1.0)
    for w in WEAK_POS:
        words.setdefault(" " + w[:7], +WEAK_W)
    for w in WEAK_NEG:
        words.setdefault(" " + w[:7], -WEAK_W)
    words.update(SUFFIXES)
    ok = set("abcdefghijklmnopqrstuvwxyz '")
    out = {}
    for k, v in words.items():
        if set(k) <= ok:
            out[k[:N_SHIFT] if k[0] == " " else k[-N_SHIFT:]] = v
    return out


def write_ngram_neuron(m, i, stem, stoi):
    """Neuron i fires (=1) iff the last len(stem) characters equal stem (a leading space also matches BOS)."""
    L = len(stem)
    for off in range(L):
        ch = stem[L - 1 - off]
        m.fc1.weight[i, off * SW + stoi[ch]] = 1.0
        if off == L - 1 and ch == " ":
            m.fc1.weight[i, off * SW + V] = 1.0
    m.fc1.bias[i] = -(L - 1)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    stoi = task.stoi
    lex = lexicon()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # --- embeddings: token one-hot in slot 0; positions as (cos, sin) + constant 1
        for t in range(V):
            model.token_emb.weight[t, t] = 1.0
        w = 2 * math.pi / 128
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, CS] = math.cos(w * p)
            model.pos_emb.weight[p, CS + 1] = math.sin(w * p)
            model.pos_emb.weight[p, CONST] = 1.0
        model.pos_emb.weight[0, V] = 1.0        # BOS flag at position 0 (acts as a virtual leading space)

        # --- layer 1 attention: head k looks exactly k tokens back, copies char to slot k
        a = model.blocks[0].attn
        S = 3e4 * math.sqrt(D_HEAD)
        for k in range(1, N_SHIFT):
            r = (k - 1) * D_HEAD      # head index k-1 handles shift k
            c, s = math.cos(w * k), math.sin(w * k)
            # q = S * rotation(-k) applied to (cos wp, sin wp)  -> (cos w(p-k), sin w(p-k))
            a.W_q.weight[r + 0, CS + 0] = S * c
            a.W_q.weight[r + 0, CS + 1] = S * s
            a.W_q.weight[r + 1, CS + 0] = -S * s
            a.W_q.weight[r + 1, CS + 1] = S * c
            a.W_k.weight[r + 0, CS + 0] = 1.0
            a.W_k.weight[r + 1, CS + 1] = 1.0
            for t in range(SW):                 # value: copy char one-hot + BOS flag (slot 0)
                a.W_v.weight[r + t, t] = 1.0
                a.W_o.weight[k * SW + t, r + t] = 1.0

        # --- layer 1 MLP: one n-gram detector neuron per lexicon stem, then one per negator
        m = model.blocks[0].mlp
        for i, (stem, sc) in enumerate(lex.items()):
            write_ngram_neuron(m, i, stem, stoi)
            m.fc2.weight[SCORE, i] = sc
        for j, stem in enumerate(NEGATORS):
            i = len(lex) + j
            write_ngram_neuron(m, i, stem, stoi)
            m.fc2.weight[NEGF, i] = 1.0
            m.fc2.weight[CUE, i] = -CUE_W

        # --- layer 2 attention: "is there a negator shortly before me?" -> NEGNEAR
        # score(p, j) = S*(cos(w(p-j)) + NEG_B*NEGF_j): self wins unless a flagged key is within range.
        b = model.blocks[1].attn
        S2 = 1e3 * math.sqrt(3)
        b.W_q.weight[0, CS + 0] = S2
        b.W_q.weight[1, CS + 1] = S2
        b.W_q.weight[2, CONST] = S2 * NEG_B
        b.W_k.weight[0, CS + 0] = 1.0
        b.W_k.weight[1, CS + 1] = 1.0
        b.W_k.weight[2, NEGF] = 1.0
        b.W_v.weight[0, NEGF] = 1.0
        b.W_o.weight[NEGNEAR, 0] = 1.0
        # layer 2 MLP: flip the sign of SCORE when NEGNEAR=1   (y = s - 2*s*neg)
        m2 = model.blocks[1].mlp
        m2.fc1.weight[0, SCORE], m2.fc1.weight[0, NEGNEAR], m2.fc1.bias[0] = 1.0, 2.0, -2.0
        m2.fc1.weight[1, SCORE], m2.fc1.weight[1, NEGNEAR], m2.fc1.bias[1] = -1.0, 2.0, -2.0
        m2.fc2.weight[SCORE, 0] = -FLIP
        m2.fc2.weight[SCORE, 1] = +FLIP

        # --- layer 3 attention: uniform average of the (flipped) SCORE (+ negator CUE) over all positions -> AVG
        a3 = model.blocks[2].attn
        a3.W_v.weight[0, SCORE] = 1.0
        a3.W_v.weight[0, CUE] = 1.0
        a3.W_o.weight[AVG, 0] = 1.0

        # --- readout: '1' if the mean score is >= ~0 (slight positive prior)
        model.head.weight[stoi["1"], AVG] = 1.0
        model.head.weight[stoi["1"], CONST] = 0.05 + PRIOR / 2   # constants keep '0'/'1' above other tokens
        model.head.weight[stoi["0"], CONST] = 0.05 - PRIOR / 2
        model.head.weight[stoi["0"], AVG] = -1.0


model_shorthand_name = "NegationFlip_v50_cleaned"
model_description = ("Cleaned v49: dead word lists removed, shared n-gram neuron helper. 3 layers, no LayerNorm: "
                     "7-char window via shift heads, lexicon + negator neurons, gated negation flip, uniform average.")


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
    model = SimpleTransformer(
        vocab_size=task.vocab_size, max_seq_len=max_seq_len, d_model=D_MODEL,
        layer_cfgs=[(N_SHIFT - 1, D_HEAD, len(lexicon()) + len(NEGATORS)), (1, 3, 2), (1, 1, 0)],
    )
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
