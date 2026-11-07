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
feel+good top+notch first+rate tour+de+force crowd+pleas* stands+out
absorb* adept adroit affable animated apt arresting astute attentive balanced beguil* benevolen*
bliss* breez* brisk candid capable cerebral chic compact confidence conviction cozy craft*
creative credib* cunning curious dedicat* deliberate determined distinguished dramatic dreamlike
eager edgy electric* enigmatic entranc* erudite ethereal expert* expressive exotic fanciful fierce
finesse firm formidable frank frisky frothy gritty gutsy harmon* haunting* honed hysterical*
idiosyncratic immense* impish integrity irreverent jaunty jolly juicy kinetic lavish lean lifelike
lighthearted likeable lilting literate lithe loyal mellow merry mesmeric mischiev* moody muscular
mystical naturalistic neat nervy nostalgi* nourish* opulent ornate outrageous* peppy persuasive
pithy poise* prescient pristine prowess pungent rapt rapturous raucous reassur* redeem* redempt*
reflective rejuvenat* reliable reverent rigorous rollicking rowdy sage scrappy seasoned sharply
sly soothing sophisticat* sparse spellbind* spunky steady stately stoic suave supportive
sustain* tangy tantaliz* tenacious thorough tireless tough tricky tuneful unabashed unafraid
unbridled undeniab* unfailing unfussy unique unpredictable unsentimental unsparing upright
venerable verve vigilant vintage voluptuous welcome well+acted well+made well+written winning
winsome wistful youthful
accomplish* ace acclaim* acute admir* adorabl* adventurous affecting affection* agile alive
alluring amaz* ambitious amiable amus* appeal* appreciat* artful artistry assured astonish*
astounding attractive audaci* authentic* awesome beautiful* beauty beloved best better bittersweet
blessed bold bracing brave bravura breathtaking bright brilliant* buoyant captivat* carefully
caring celebrat* charisma* charm* cheer* classic classy clear clever* coherent colorful comfort*
comic commanding compassion* compelling competent confident* convincing cool courag* crackling
crisp cute dandy daring dazzl* decent deep deft* delectabl* delicate* delicious*
delight* dependable deserv* devastating* devoted dignity disarming distinctive* divine dynamic
earnest* easy ebullient effective* effervescent elegan* elegant eloquen* embrace emotional*
empath* enchant* endearing* energetic* energy engag* engross* enhanc* enjoy* enlighten* enrich*
enterprising entertain* enthrall* enthusias* epic essential* evocative* exceeding* excellen*
exceptional* excit* exhilarat* exquisite* extraordinar* exuberan* fabulous* faithful fantastic
fascinat* favorite fearless feisty fervent fest fine finely finest flair
flawless* fluid focus* fond forceful* fresh* friendly fun funnier funniest funny gem generous*
genial genius gentle* genuine* gifted giddy glad glorious* glory glow* good goodness gorgeous*
graceful* gracious* grand grateful great greater greatest grip* groovy gusto handsome* happi* happy
hearten* heartfelt heartwarming hearty heroic* highest hilarious* hip honest* honor* hook* hope*
hot humane humanity humor* hypnotic ideal idealis* imaginati* immaculate* immersive impeccabl*
importan* impress* incisive* incredibl* indelible ingenious* ingenuity innovati* insight* inspir*
intelligen* intense intimate intoxicating intrigu* intricate inventive involving irresistibl* jewel
joy* jubilant keen knowing laugh* legend* likable liked lively lovable love* lovely
loving lucid luminous luscious lush lyric* magic* magnificen* majestic marvel* master* mature
meaningful* memorabl* mesmeriz* meticulous* mighty mirth modest moved moving mysterious nifty nimble
noble nuance* observant offbeat optimis* original* outstanding passion* patient peaceful perceptive
perfect* phenomenal* picturesque playful* pleasant* pleasing pleasur* plucky poetic* poignan*
polish* popular potent powerful* precious precise pretty prize* profound* promis* proud provocative
punch* pure quirky radiant rare ravishing realistic recommend* refresh* rejoic* relevant remarkabl*
resilien* resonan* respect* restrain* revelat* rewarding rich* ripe riveting robust romantic
rousing sassy satisf* savvy scintillat* searing seductive* seamless* sensational* sensitiv*
sensual* sentimental serene sexy sharp* shimmer* shine shining shrewd sincer* skill* slick smart*
smooth* snappy solid soulful* soaring spark* special spectacular* spiffy spirit* splendid*
spontaneous sprightly stellar stimulat* stirring strength* striking strong* stun*
sturdy stylish* sublime* substan* subtle* subversive succe* suitabl* sumptuous* sunny super superb*
superior superlative supple surprising* suspense* sweet* swift sympath* talent* tasteful*
tasty tender* tense terrific* thoughtful* thrill* tight timeless timely tingl* touch*
tragic transcend* transform* treasure* treat tremendous* triumph* truly trust*
truth* unassuming uncanny uncompromising understated unexpected* unflinching unforgettabl* unique*
universal unpretentious unusual* uplift* upbeat urgent valuable versatile vibran* victor* vigor*
virtuos* visceral vision* vital* vivacious vivid* warm* watchable wholesome wicked* wise* wit
witty wonder* wondrous worth* wow zany zest*
""".split()

NEGATIVE = """
adolescent anemic artless ashamed aggravat* agony bereft bitter* blah blandly bludgeon* boneheaded
bogus bothersome brainless bratty brutish bumbl* callow careless* cacophon* caricature* clumsily
cloying comatose commonplace cop+out cringe* cutesy dawdl* deadening deflat* degrad* dimwitted
disagreeable disarray disdain* dishonest* dishearten* disinterest* dispirit* displeas* dissatisf*
ditzy dorky downer dross dud duds enfeebl* exasperat* eyesore fatuous fiasco flail* flaccid
foible fractured frivolous fussy futile ghoulish gooey grubby gruesome gullible headache hokum
hopeless* humdrum ignoble ignominious ill+conceived ill+advised imbecil* implod* impotent inaccura*
inattentive inconsequential indecipher* indigestible insensitive insubstantial interchangeable
irredeem* jaded jerky joyless juvenile klutz* lachrymose lamentable languish* lax lobotom* lowbrow
lumpy maddening* marred mean+spirited messy misanthrop* mock* moldy moot naff nauseous nitwit noxious
oafish overbearing overcome overindulgent overplay* oversimplif* overuse* paper+thin paranoid
pallid pedantic peevish pestering petty pitiless plagiar* plastic pompous ponderously pretense
prolong* prudish putrid quaint rancorous raunchy repressed resentful rickety rotting rudderless
sanctimonious sentimentality shallowness shrug* sleepwalk* slipshod smirk* snooze sophomoric
spasmodic spineless squalid stingy strident stuffy stultify* subliterate sugary sullen supercilious
tangled tardy tatty tawdriness tiresomeness tortuous trivial* twaddle unaccountabl* unamusing
unconcerned undisciplined unearned unenlightened unfathomabl* unfeeling ungainly unholy unimaginable
unkempt unmemorable unmoving unnerving unpolished unprofessional unsavory unskilled unsophisticated
unstable unsubtle untidy unwitting uproot* vexing wearying whiff wimpy witless wobbly worthlessness
half+baked plot+holes rip+off second+rate self+conscious self+indulgent warmed+over watered+down
low+grade low+rent b+movie one+note too+long too+much bloat* blatant blight* blunder*
chaotic chore clunk* collapse* creak* crumbl* decay* diminish* dreck
drags drain* drown* erode* fizzl* fluff grind* groan* haphazard* heavy+handed
hype irksome lifelessly loathe* lukewarm malaise meaningless*
mired muck* naive nightmare oblivious overload overwhelm* phony plague* pretense
ragged rehash* relentless* routine sag* seldom shrill sicken* sinister slog*
spoil* stifl* stoop* sully* tepid thankless tiresomely tripe tumbl* undermin*
unfocused unwise uninvit* unjust* unruly wither* woeful*
abomination abrasive absurd* abysmal* aimless* alienat* amateur* angry annoy* appalling
arduous arrogan* artificial atrocious* awful* awkward* bad badly baffl* bafflement banal*
barrage bastard belabor* bewilder* bizarre bland* bleak blunt bogged bombast* bore* boring botch*
bother* broken bungl* burdened cartoonish cheap* cheesy childish choppy
clich* clueless clumsy* coarse* complain* complicated conceit* condescend* confus* contempt*
contrived convoluted corny costly crass* crap* crude* cruel* crummy cumbersome cursory cynic* damn*
dated deadly dearth debacle deceit* defeat* deficien* dehumaniz* demean* depress* derivative
desperate* despair* dire dirty disappoint* disaster* disastrous* discomfort* disconnect*
discouraging disgrace* disgust* disingenuous disjointed dislike dismal* disorient* disposable
disrespect* distaste* distract* distress* disturb* dopey doze dragged dragging dreadful*
dreary drivel drone drowsy dubious* dull* dumb* dumped dysfunctional egregious* embarrass* empty
endless* enervating excess* excruciating exhaust* exploit* extended fail* failure faint fake*
falls falter* farce* fatal* fatigu* faulty feeble* filler filth* flabby flat* flaw* flimsy* flop
flounder* foolish* forced forgettable forgotten formula* foul fragmented frantic* frustrat* fumbl*
garbage garbled gaudy generic ghastly gimmick* glib gloomy glum graceless grating gratuitous*
greasy grim gross* grotesque* grueling hack* hampered hamfisted hapless
harsh* hate* hideous* hollow hoary hokey horrend* horribl* horrid* horrific* humorless hurt*
hypocrit* idiot* ignoran* ill illogical imitat* immature impenetrable impersonal implausib*
impossible improbable inadequate* inane* inarticulate incapable incoherent* incompetent*
incomprehensib* inconsistent* indifferent* indulgent* ineffective* inept* inert
inferior* inexcusable* infantile inferior insane insignificant insincere insipid* insufferabl*
insufficient insult* interminabl* intolerabl* irrelevan* irritat* jarring jejune jumbled junk* kitsch
lackluster lack* laborious* labored lame* languid laughable lazy* leaden lethargic lifeless
limp* listless* lost lousy* ludicrous* lugubrious lumbering lurid mannered manipulat* marginal
maudlin mawkish meager meandering mechanical mediocr* melodrama* meretricious mess* mildly
mindless* misbegotten miscast misconceived* miserabl* misery misfire* mishandl* misguided mishmash
misleading missed misses mistake* misuse* monotonous* monstrous* moronic* muddl* mundane murky mushy
nagging nasty* naught nauseat* needless* neglect* nonsens* nothing notorious* numb* obligatory
obnoxious* obscure obsolete obtuse obvious* odious* offal offend* offensive* oppressive* ordinary
outdated overacted overblown overcooked overdone overkill overlong overly overproduced overrated
overstuffed overwrought painful* pains pale paltry pandering pathetic* patchy pedestrian perfunctor*
pity pitiful* plodding pointless* ponderous* poor* preachy predictab* prefabricated
preposterous* pretentious* primitive problem* problematic processed prosaic* puerile pummel*
punish* puzzl* quagmire queasy rambl* rancid* recycled redundant regrettabl* regret*
repellent repetiti* repuls* repugnant* repulsive* retread* revolting* ridicul* ripoff
rote rotten rubbish rude* ruin* rushed sad sadly sappy scant schlock* schmaltz* scattered
screech* scream* senseless* shabby shallow shambles
shameless* shapeless shoddy* shortcoming* silly sill* simplistic sinking skimpy slapdash
sleazy sleep* slight slip* sloppy* sloppily slow* sludge* sluggish* slummy smarmy smug* snail*
soapy sodden soggy solemn* soporific sorely sorry soulless sour spiritless squander* stagnant
stale* stereotyp* stiff* stilted stink* stodgy strain* struggl* stumbl* stunt stupid* stupor
stultifying subpar substandard suck* suffer* suffocat* superficial* swamped tacky tainted
tame tasteless* tawdry tedious* tedium terribl* thin thoughtless* threadbare tiresome*
tired tiring tortur* toothless toxic trash* trifling trite* trouble* turgid ugly unappealing
unattractive unbearabl* unbelievabl* uncertain* uncomfortable unconvincing uncreative
underachiev* undercooked underdeveloped underwhelm* undistinguished uneasy unemotional unengaging
uneven unexciting unfocused unfortunate* unfulfilled unfunny unhappy unimaginativ* unimpressive
uninspir* unintelligib* unintentional* uninterest* uninvolving unlikable unlikeable unlikely
unnecessar* unoriginal unpalatabl* unpleasant* unravel* unrealistic unrelenting unremarkable
unsatisf* unseemly unsettling* unsure unsuccessful* untalented unwatchable unwieldy unworthy upset*
useless* vacant vacuous* vague* vain vanity vapid* vicious* violence violent* vile* vulgar* wan
wannabe waste* weak* wearisome* weary* weird whine* whiny wince*
wooden worn worse worst worthless* wreck* wretched* wrong* yawn* yuck*
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
FLIP_POS = 1.0                  # a negated positive counts as this much negative
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
model_shorthand_name = "big_w8_affix"
model_description = "Full big_vocab4 lexicon with every core winner: no W_o, 8-char truncated stems, negative affix rules, +/-1 weights, ungated negatives."


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
