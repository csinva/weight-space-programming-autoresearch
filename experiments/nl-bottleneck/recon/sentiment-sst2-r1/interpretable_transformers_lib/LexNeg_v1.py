"""
Interpretable transformer for character-level sequence tasks.

Hand-written lexicon-and-negation sentiment scorer for `sentiment-sst2`.

  Layer 1 attention : 12 heads = shift register (head k copies char at t-k, k=0..11)
  Layer 1 MLP       : one AND-gate neuron per lexicon entry / weak cue / negator
  Layer 2 attention : one head flags positions within 15 chars after a negator
  Layer 2 MLP       : flagged positive evidence is replaced by -1.0 per match
  Layer 3 attention : uniform attention averages per-position scores
  Head              : '1' if total >= 0 else '0'

No training: every parameter is written in closed form in write_weights().
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
# Lexicon (written by hand from general knowledge; not fitted to data)
#   plain word  -> whole word (boundary on both sides; boundary = ' ', '_', start)
#   "stem*"     -> prefix (boundary on the left only)
#   "*suffix"   -> suffix (boundary on the right only)
#   "a b"       -> short phrase (whole, internal space literal)
# ---------------------------------------------------------------------------

POSITIVE = """
good, great, best, better, fine, finest, nice, nicely, excellent, excellence, wonder*, wondrous,
amazing, amazingly, awesome, superb*, brilliant*, brilliance, fantastic*, terrific*, marvel*,
outstanding, remarkabl*, extraordinar*, exceptional*, magnificent*, splendid*, splendor, stunning*,
beautiful*, beauty, lovely, gorgeous*, elegan*, grace, graceful*, gracious, charm*, delight*, enjoy*,
fun, funny, funniest, hilarious*, hilarity, witty, wit, wits, clever*, smart*, intellig*, insight*,
thoughtful*, moving, touching*, touches, poignan*, heartfelt, heartwarm*, warm*, tender*, sweet*,
gentle*, engag*, engross*, absorbing, compelling*, gripping, riveting, thrill*, excit*, entertain*,
pleasant*, pleasur*, pleasing*, satisf*, rewarding, refresh*, fresh, freshness, original, originality,
inventive*, imagina*, creativ*, innovat*, inspired, inspiring, inspiration*, uplift*, powerful*,
power, strong, strongest, strength*, solid*, impress*, memorabl*, unforgettable, masterpiece*,
masterful*, mastery, masterwork, classic, gem, gems, treasure*, triumph*, success*, wins, winning,
winner*, accomplish*, achiev*, admir*, adorabl*, affecting, affection*, amus*, appeal*, astonish*,
astound*, authentic*, bold, boldly, brave*, breathtak*, bright, captivat*, compassion*, confident*,
convincing*, cool, courag*, crisp, dazzl*, deft, deftly, depth, distinctive, dynamic, earnest*,
effective*, effortless*, eloquen*, emotional*, energetic, energy, enthrall*, entranc*, epic,
essential, evocative, exhilarat*, exquisite*, fascinat*, favorite, favourite, fearless*, flawless*,
glorious*, glory, gift, gifted, grand, gutsy, handsome*, happy, happiness, haunting*, heart, hearts,
hearty, heroic, honest*, humane, humanity, humanist*, humor, humour, humorous*, humane, ingenious*,
ingenuity, intense, intimate, intrigu*, irresistibl*, joy, joyous, joyful*, kind, laugh, laughs,
laughter, laughing, lively, liveliness, love, loved, loves, loving*, lovable, luminous, lush,
lyrical*, magic, magical*, majestic, mesmeriz*, miracle*, miraculous*, noble, nuanced, offbeat,
passion*, perfect*, playful*, poetic*, poetry, polish*, potent, precise*, profound*, provocative*,
quirky, rare, rich, richly, richness, robust, romantic, rousing, savvy, sensitiv*, sensual, sharp*,
sincer*, skill*, smooth*, soulful*, sparkl*, spectacular*, spirited, spirit, stellar, stirring*,
striking*, stylish*, subtle, subtly, subtlety, surpris*, suspense*, sympathetic*, talent*, tasty,
timeless, touching, tremendous*, unique*, valuable, vibrant*, vital*, vivid*, warmth, wise, wisely,
wisdom, worthwhile, worthy, wow, zest*, zany, delicious*, delicate*, diverting, droll, ebullient*,
enchant*, endear*, enduring, enlighten*, enthusias*, expert*, fabulous*, genius, genuine*, glow*,
goodness, gratif*, hit, joyride, likable, likeable, masterly, meticulous*, mature, meaningful*,
nimble, nifty, positive*, praise*, pure, radiant*, resonan*, resonat*, respect*, revelation*,
rollicking, sexy, shine*, shining, sly, slyly, snappy, sophisticat*, soar*, spellbind*, sublime*,
succeed*, superior, sweeping, transcend*, truthful*, unpretentious*, vigor*, virtuos*, visionary,
watchable, welcome, whimsical*, wonderful*, affable, agreeable, alive, ambitious*, appreciat*,
artful*, artistry, assured, attractive*, beloved, breezy, buoyant, candid, celebrat*, cheer*,
cherish*, cohesive, commanding, compassionate, complex, confidence, cute, daring, decent, dedicat*,
detailed, devoted, dignity, dignified, dramatic*, dreamy, eager, ecstatic, edgy, elegance,
embrac*, empath*, enrich*, exuberan*, faithful*, feisty, festive, finely, finesse, first rate,
fond, fondly, fondness, friendly, fulfill*, giddy, glee*, grateful, groundbreak*, guilty pleasure,
handsomely, hearten*, heartbreak*, heaven*, highlight*, hip, honor*, honour*, hooked, hope, hopeful*,
hysterical*, iconic, illuminat*, imagination, impeccabl*, important, incisive*, incredibl*,
indelibl*, infectious*, informative, insightful, integrity, interest*, intoxicat*, invigorat*,
jubilant, keen, keenly, knockout, laudabl*, legendary, liberat*, lightheart*, lucid, lucky,
magnetic, meditative, mischiev*, natural, naturally, naturalis*, neat, nostalgi*, observant,
oscar*, passionate, peaceful, perceptive, persuasive, phenomenal*, pitch perfect, plucky, popular,
precision, pride, prodigious, promis*, proud, quality, rapturous, realistic*, refined, relevant,
reliable, relish*, resilien*, revel, revels, reverent*, riotous*, sassy, scintillat*, seamless*,
sensational, serene, shrewd*, skilful*, sleek, smile*, smiling, soothing, special, spontaneous,
sprightly, standout, steady, stimulating, stunner, sturdy, sumptuous*, superlative, supple, supreme*,
sure footed, tasteful*, tense, thoroughly, tidy, timely, tingl*, top notch, tour de force, treat,
tribute, triumphant, trust*, upbeat, valiant, versatil*, victor*, vitality, vulnerab*, warmly,
well made, well acted, well crafted, well done, well written, wholesome, winningly, wry, wryly,
youthful, zippy, accessible, acclaim*, acute, adept, adroit, adventur*, aglow, alluring, amiable,
amiably, ample, animated, artistic*, astute, audacious*, balanced, bedazzl*, benefit*, bless*,
blissful*, bonus, bravura, brisk, buzz, capable, careful*, caring, catchy, champion*, charismatic,
clarity, classy, colorful, colourful, comely, comfort*, comic genius, compact, competent, complexity,
composed, consummate, cracking, craft, crafted, craftsman*, credible, crowd pleas*, cunning,
curious*, dandy, dashing, dazzling, deeply, delectabl*, deliciously, dependabl*, deserv*, devilish*,
dexter*, diligent, discerning, divine, dreamlike, dumbfound*, durable, dynamism, easygoing,
eccentric, edifying, electric*, elevat*, eloquent, emotive, empower*, enamor*, encourag*, endless fun,
energiz*, enhanc*, enjoyabl*, ennobl*, enrapt*, entic*, entrancing, enviabl*, epic, euphori*,
evocativ*, exalt*, exceed*, excellently, exemplar*, exhilarating, expressive*, exuberance, fanciful, fascination, fast paced, feelgood, feel good, fervent*, fierce*, fiery, fine tuned,
flair, flamboyan*, flavor*, flavour*, fluent*, focused, forceful, formidabl*, fortunate*,
frank*, freewheel*, frolic*, gallant*, generous*, genial*, gleeful*, glamor*, glamour*, glitter*,
glistening, golden, goodhearted, goofy fun, gratifying, gravitas, grin*, gusto, hale, harmon*,
headlong, healing, hearty, heroine, heroism, highest, hilarity, honesty, hospitab*, humble,
humbling, idyllic, illustrious, immaculate*, immense*, immersive, impassion*, improve*, incomparabl*,
indispensabl*, ingratiat*, innocence, insightfully, inspir*, instructive, intelligent, intrepid,
inventiveness, involving, irreverent*, jaunty, jolly, jovial, kindly, kindness, kinetic, lavish*,
liveli*, lucidity, ludicrous fun, luxuriant, lyric, magnificence, majesty, marvelous,
masterfully, memorable, merit*, merry, mesmerising, mighty, mirth*, moral, motivat*, moving,
mythic, nicest, notable, noteworthy, novel, nurtur*, optimis*, opulen*, ornate, outdo*, outshine*,
outstand*, paradise, peerless, perky, personable, pertinent, picturesque, pioneer*, pithy, pizzazz,
playfulness, plentiful, pleasantly, plenty, poise*, polite, potency, powerhouse, prais*, prestig*,
pretty good, priceless, prime, pristine, prize*, proficien*, prolific, prominent, prosper*,
protect*, proven, prowess, punchy, quaint, quick witted, radiance, rapt, rapture, raptur*, ravish*,
razor sharp, reassur*, recommend*, redeem*, redemption*, rejoic*, rejuvenat*, relax*, remedy,
renown*, resolute, resourceful*, resplendent, restorative, retro fun, revitaliz*, revolution*,
right on, righteous*, rigorous, ripping, rocks, romp, rousing, saintly, salient, sane, satire,
scrumptious, seductive*, selfless*, sensibl*, serendip*, sharpest, sharply, shimmer*, shrewdly,
significant, simplicity, sizzl*, skillfully, slickly, smashing, smartest, snazzy, solace,
sonorous, soulful, spectacle, speedy, spellbinding, spiffy, splash*, sprawling, sterling, stirring,
straightforward, stupendous*, sturdy, stylized, suave, sufficient, sunny, super, superbly, supporting,
surefire, surreal, sustain*, sweetly, swift*, swoon*, sympath*, tantaliz*, tastefully, teem*,
tenacious*, thankful*, thorough, thoughtfully, thrilled, thriving, tickl*, tight, timeless, titillat*,
top, topnotch, toughness, tranquil*, transform*, treasured, trendy, trim, triumphantly, true,
truly, truth, unabashed*, unassuming, unblinking, uncanny, uncompromis*, understated, undeniabl*,
unflinching*, unforced, unusual, upliftingly, urbane, useful, valor, venerab*, verve, viable,
vibrancy, vigilant, virtue*, virtuous, visceral*, vision, vivacious*, vividly, voluptuous, warmhearted,
welcoming, whimsy, whiz, wholeheart*, willing, wink*, winsome, wistful*, witt*, won, workmanlike,
worth seeing, worth watching, worthiness, youthfully, zeal*, zesty
"""

NEGATIVE = """
bad, badly, worst, worse, awful*, terribl*, horribl*, horrid*, dreadful*, dire, poor, poorly, weak*,
lame, lamest, dull*, boring*, bore, bores, bored, tedious*, tedium, tiresome*, tired, stale, flat,
flatly, bland*, mediocr*, mess, messy, muddl*, mushy, sloppy, sloppi*, clumsy, clumsi*, awkward*,
inept*, incompeten*, amateur*, stupid*, dumb, dumber, dumbest, idiot*, silly, sillier, silliness,
foolish*, ridiculous*, laughabl*, ludicrous*, preposterous*, pretentious*, pompous*, self indulgent,
indulgent, overwrought, overlong, overblown, overdone, overcooked, overacted, overacting, overrated,
overstuffed, overkill, bloated, plodding, sluggish*, slow, slowly, slog, drag, drags, dragged,
dragging, interminabl*, endless, lifeless*, listless*, pointless*, worthless*, mindless*, senseless*,
joyless*, humorless*, humourless*, charmless*, witless*, gutless*, aimless*, soulless*, hollow,
empty, emptiness, shallow*, superficial*, thin, flimsy, contrived, contrivance*, formulaic, formula,
predictabl*, clich*, hackneyed, trite, banal*, derivative, generic, routine, uninspired, unimaginativ*,
unoriginal, unfunny, unwatchabl*, unbearabl*, unpleasant*, unconvincing*, uninteresting, unsatisf*,
uneven, unfocused, annoy*, irritat*, grating, obnoxious*, offensive*, ugly, ugliness, unappealing,
repellent, repulsive*, disgust*, distasteful*, gross, crude*, crass, vulgar*, cheap, cheaply, cheesy,
corny, schmaltz*, sappy, saccharine, maudlin, mawkish, melodramatic*, manipulat*, heavy handed,
ham fisted, wooden, stiff*, stilted, forced, strained, labored, laboured, convoluted, confused,
confusing, confusion, incoheren*, nonsens*, garbage, trash*, junk, rubbish, crap, crappy, sucks,
sucked, stinks, stinker, stink*, disaster*, failure*, flop, flops, botch*, misfire*, misguided,
misstep*, mistake*, wrong, problem*, flaw, flaws, flawed, waste, wasted, wasting, wasteful, lazy,
lazily, careless*, shoddy, tacky, tasteless*, insult*, insipid*, vapid*, inane*, inanity, idiotic*,
moronic, juvenile, sophomoric, childish, puerile, depressing*, dreary, drab, dismal*, gloomy, bleak,
sour, miserabl*, misery, painful*, painfully, agony, agoniz*, excruciat*, torture*, tortur*,
embarrass*, cringe*, wince, shame*, pathetic*, pitiful*, sorry, regret*, disappoint*, let down,
letdown, underwhelm*, mediocrity, exhausting, numbing, mind numbing, sleep*, snooze*, yawn*, dud,
clunky, clunker, choppy, disjointed, jumbled, haphazard*, rambling, rambl*, meander*, unfocused,
lengthy, padded, repetitive*, repetiti*, monotonous*, monoton*, one note, one dimension*, cardboard,
cartoonish, caricature*, stereotyp*, simplistic*, simple minded, dimwit*, brainless*, numb, numbly,
pedestrian, ordinary, mundane, humdrum, unremarkabl*, forgettabl*, negligibl*, meager, paltry, nasty,
mean spirited, cruel*, cynical*, hateful*, hate, hated, hates, loath*, despis*, ugh, yuck, awfully,
horribly, atrocious*, abysmal*, appall*, wretched*, lousy, rotten, putrid, abomination, travesty,
debacle, fiasco, chore, ordeal, tiring, wearisome, wearying, sloppily, ineptitude, misjudg*, miscast,
hammy, campy, exploitati*, exploitive, gratuitous*, pander*, smug*, preachy, sanctimon*,
condescend*, patroniz*, obvious, overused, recycled, retread, rehash*, rip off, ripoff, knockoff,
lackluster, lacklustre, limp, tepid, lukewarm, half hearted, halfhearted, uninvolving, uninspiring,
unengaging, unmemorable, unnecessar*, unlikabl*, unlikeabl*, unsympathetic, unfortunate*, sadly,
alas, frustrat*, boredom, dullness, mishmash, hodgepodge, muddle, morass, slapdash, sketchy,
threadbare, skimpy, undernourish*, underdevelop*, undercook*, unfinished, implausibl*, improbabl*,
far fetched, gimmick*, glib, facile, hokey, hokum, stupidly, dumbed, sterile, antiseptic, cold,
heartless*, artless*, shapeless*, plotless*, toothless*, clueless*, hopeless*, useless*, needless*,
tuneless*, uninvolved, unmoving, emotionless*, dumbs, irrelevan*, inconsequential, inferior*,
insufferabl*, intolerabl*, unendurabl*, unwieldy, unworthy, worn, worn out, ill conceived,
illogical, imitation, impotent, inadequate*, inauthentic, incomprehensibl*, inconsistent, ineffect*,
inert, inexcusabl*, insensitive*, insincere, irksome, jarring, joyless, kitsch*, labor, languid,
leaden, lethargic, lumbering, mangled, meaningless*, messiness, middling, misconceived, misogyn*,
mistaken, muddy, murky, naive, negative*, nightmare*, noisy, obnoxiously, odious, oppressive*,
overbearing, overplayed, overwritten, pallid, pandering, paper thin, perfunctory, pitiable,
plodding, ponderous*, poorer, posturing, pretension*, prosaic, protracted, puzzling, rancid,
redundant, repugnant, ridicul*, routinely, sadistic*, sanitized, scattershot, schlock*, screech*,
second rate, shallowly, shrill, sickening, sickly, sleazy, sloppiness, sludg*, sluggishness, smarmy,
snore*, sour, spoof? , squander*, stagnant, stale, stifl*, stodg*, strident, stunted, stupefy*,
subpar, substandard, tawdry, tediously, thankless, third rate, tin eared, tired out, tortuous,
toxic, trashy, trivial*, troubled, troubling, turgid, unacceptabl*, unappetiz*, unbelievabl*,
uncomfortabl*, undistinguished, unexciting, unfulfill*, unhappy, uninformative, unintelligibl*,
unintentional*, uninvited, unjustifi*, unnatural, unpleasantly, unpolished, unprofessional, unsavory,
unsettling, unsubtle, unsuccessful, untalented, unwelcome, unwise, uptight, vacuous*, vague*,
waste of, weakest, weakly, weary, weirdly, whiny, whiny*, wimpy, wince*, woeful*, worsen*, wreck*,
yawner, zero, bogus, bombastic, bungl*, cloying*, cold blooded, contempt*, crummy, cumbersome,
dated, dead, deadly dull, deadening, deadpan dull, debased, defeat*, deficien*, deflat*, degrad*,
dehumaniz*, deplorabl*, depress*, deriv*, desperate*, despair*, destroy*, detestabl*,
dim, dimly, dingy, dirty, discomfort*, disgrac*, disingenuous, dislik*, dismay*, distract*,
disturbing, dizzy*, dopey, dorky, doze*, drained, drearily, drivel, droning, drudg*, dumbing, dumpy,
egregious*, empty headed, endur*, erratic*, excess*, exhaust*, fake, falls apart, falls flat,
falter*, farce*, fatigu*, faulty, feeble*, filler, fizzl*, flabby, flail*, flatulen*, flounder*,
fluff, foul, frantic*, fumbl*, fussy, garish, ghastly, gloom*, goofy, grim, grimly, grind*, gruel*,
haggard, hapless, harsh*, hideous*, hokiest, horrendous*, humiliat*, hysteria, ignoran*, ill,
illness, imbecil*, immature*, impatien*, implod*, incessant*, indifferen*, indigest*, ineptly,
infantil*, innocuous, insane, insignifican*, insubstantial, irritable, junky, klutz*,
lamentabl*, lazier, lax, limping, losers, loser, lost, lousiest, low budget, low rent, lugubrious,
maddening, malaise, marred, mawkishly, melodrama*, merely, mess up, messed, mindlessly, miserably,
misfit, misshapen, misuse*, mock*, mope*, morose, muck*, mumbl*, mundanity, naff, nauseat*, neurotic,
nonexistent, numbing, obnoxious, obscen*, oddball, offend*, overpriced, overreach*, overstat*,
pain, pains, paltry, paranoi*, parody, pedantic, peril*, pest*, petty, phony, phoney, pitiless,
platitud*, pointlessly, pollut*, poser, prattl*, predictably, pretend*, primitive, prolong*,
puny, questionabl*, ragged, rancor*, rant*, rash, rattl*, rawness, regrettabl*, repel*, repuls*,
revolting, rickety, rote, rough, rubbishy, rude, ruin*, rusty, sad, scrappy, scream*, scuttl*,
shabby, shaky, shambl*, shoddily, sick, sicken*, sink*, skimp*, slack, slapstick, sleaz*, slight,
slimy, sloth*, slow moving, slugg*, smirk*, snide, sorely, sorrow*, spineless*, spiritless*,
splatter, spoil*, stale, starv*, stereotypical, stingy, stupidity, sucker*, sullen, suffer*, sulk*,
sully, tasteless, tawdriness, tear jerk*, temper tantrum, tense dull, terrib*, thoughtless*, threat*,
thud*, timid*, tiresomely, trash, trite, troubl*, ugliest, unamusing, uncaring, unclear, uncouth,
undermin*, undignified, unduly, uneasy, unequal, unfair*, unfaithful, unhinged, unimpress*,
uninspir*, unkind, unnerv*, unoriginality, unravel*, unruly, unseemly, unstable, unsteady, untrue,
unwatch*, unwieldy, unworthy, upset*, useless, vain, vapid, vex*, vile, villain? , violent*,
wallow*, wan, wane*, wanting, war? , washed out, watered down, wayward, weaken*, weird*, whine*,
wobbl*, woe*, wore, worry*, worthlessness, wounded, wrongly, yawning
"""

# weak cues (+/-0.4)
WEAK_POS = "film, performance, and"
WEAK_NEG = "movie, plot, just, would, un*, dis*, over*, *less"

# negators (-1.0 each; flag the next 15 characters)
NEGATORS = """
not, *n't, no, never, without, too, less, lacks, fails, rather than, nothing, none, nor, neither,
hardly, barely, scarcely, lack, lacking, lacked, fail, failed, failing, cannot, nobody, nowhere,
instead, absent, devoid, missing, minus, unable, seldom, rarely, far from, anything but, short of,
fewer, nary, loses
"""

WEIGHT_POS, WEIGHT_NEG, WEIGHT_WEAK, WEIGHT_NEGATOR, WEIGHT_FLIPPED = 1.0, -1.5, 0.4, -1.0, -1.0
WINDOW = 12          # shift-register width (offsets 0..11)
NEG_SPAN = 15        # characters flagged after a negator
SEQ_LEN = 82


def parse_entries(text: str) -> list[str]:
    out, seen = [], set()
    for raw in text.replace("\n", ",").split(","):
        e = " ".join(raw.strip().split())
        if not e or "?" in e:
            continue
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def entry_to_conditions(entry: str):
    """Return list of (offset, kind, char) conditions; offset 0 = current position.
    kind: 'c' literal char, 'b' boundary (space/'_'/sentence start)."""
    if entry.startswith("*"):                 # suffix: chars + right boundary
        body, left, right = entry[1:], False, True
    elif entry.endswith("*"):                 # stem: left boundary + chars
        body, left, right = entry[:-1], True, False
    else:                                     # whole word / phrase
        body, left, right = entry, True, True
    if left and right and len(body) > WINDOW - 2:   # too long -> 11-char stem
        body, right = body[: WINDOW - 1], False
    if len(body) + int(left) + int(right) > WINDOW:  # long stem -> cut to 11
        body = body[: WINDOW - 1] if left else body[-(WINDOW - 1):]
    seq = (["B"] if left else []) + list(body) + (["B"] if right else [])
    conds = []
    for off, ch in enumerate(reversed(seq)):
        conds.append((off, "b", None) if ch == "B" else (off, "c", ch))
    return conds


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Causal multi-head attention with free q/k and v head widths (no 1/sqrt(d) scaling)."""

    def __init__(self, d_model: int, n_heads: int, d_qk: int, d_v: int):
        super().__init__()
        self.n_heads, self.d_qk, self.d_v = n_heads, d_qk, d_v
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, self.d_qk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.d_qk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.d_v).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
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
        self.attn = Attention(d_model, n_heads, d_qk, d_v)
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None

    def forward(self, x):
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class Layout:
    """Residual-stream layout."""

    def __init__(self, vocab_size: int, seq_len: int):
        self.V = vocab_size
        self.CH = vocab_size + 1                    # char one-hot + boundary flag
        self.SRC = 0                                # current token's char block
        self.blk = [self.CH * (1 + k) for k in range(WINDOW)]   # copied char at t-k
        o = self.CH * (1 + WINDOW)
        self.POS = o; o += seq_len
        self.CONST = o; o += 1
        self.P = o; o += 1          # positive evidence value at this position
        self.C = o; o += 1          # number of positive matches at this position
        self.SCORE = o; o += 1      # negative evidence + negator penalties + flips
        self.NEGR = o; o += 1       # negator ends here
        self.FLAG = o; o += 1       # within NEG_SPAN chars after a negator
        self.TOTAL = o; o += 1      # mean score over prefix
        self.D = o


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(self, vocab_size: int, max_seq_len: int = SEQ_LEN, **_):
        super().__init__()
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len
        self.L = L = Layout(vocab_size, max_seq_len)
        self.pos_list = parse_entries(POSITIVE)
        self.neg_list = parse_entries(NEGATIVE)
        self.wpos_list = parse_entries(WEAK_POS)
        self.wneg_list = parse_entries(WEAK_NEG)
        self.negator_list = parse_entries(NEGATORS)
        n_ff1 = (len(self.pos_list) + len(self.neg_list) + len(self.wpos_list)
                 + len(self.wneg_list) + len(self.negator_list))
        self.token_emb = nn.Embedding(vocab_size, L.D)
        self.pos_emb = nn.Embedding(max_seq_len, L.D)
        self.blocks = nn.ModuleList([
            Block(L.D, WINDOW, max_seq_len, L.CH, n_ff1),      # shift register + lexicon
            Block(L.D, 1, max_seq_len + 1, 1, 1),              # negation flag + flip
            Block(L.D, 1, 1, 1, 0),                            # uniform sum
        ])
        self.head = nn.Linear(L.D, vocab_size, bias=False)

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
    L, V, T = model.L, model.vocab_size, model.max_seq_len
    stoi = task.stoi
    for p in model.parameters():
        p.zero_()

    # --- embeddings -------------------------------------------------------
    for c, i in stoi.items():
        model.token_emb.weight[i, L.SRC + i] = 1.0
        if c in (" ", "_"):
            model.token_emb.weight[i, L.SRC + V] = 1.0        # boundary flag
    for t in range(T):
        model.pos_emb.weight[t, L.POS + t] = 1.0
        model.pos_emb.weight[t, L.CONST] = 1.0
        k = t + 1                                             # offset k sits at position -1
        if k < WINDOW:
            model.pos_emb.weight[t, L.blk[k] + V] = 1.0       # sentence-start boundary

    # --- layer 1 attention: shift register --------------------------------
    a1 = model.blocks[0].attn
    lam = 60.0
    for k in range(WINDOW):
        qo, vo = k * T, k * L.CH
        for t in range(T):
            a1.W_q.weight[qo + t, L.POS + t] = 1.0
        # key for position j: score(t, j) = lam if j == t-k (or j==0 when t<k)
        for j in range(T):
            for t in range(T):
                if j == max(t - k, 0):
                    a1.W_k.weight[qo + t, L.POS + j] = lam
        for c in range(L.CH):
            a1.W_v.weight[vo + c, L.SRC + c] = 1.0
            a1.W_o.weight[L.blk[k] + c, vo + c] = 1.0

    # --- layer 1 MLP: AND gates -------------------------------------------
    m1 = model.blocks[0].mlp
    groups = [
        (model.pos_list, {L.P: WEIGHT_POS, L.C: 1.0}),
        (model.wpos_list, {L.P: WEIGHT_WEAK, L.C: 1.0}),
        (model.neg_list, {L.SCORE: WEIGHT_NEG}),
        (model.wneg_list, {L.SCORE: -WEIGHT_WEAK}),
        (model.negator_list, {L.SCORE: WEIGHT_NEGATOR, L.NEGR: 1.0}),
    ]
    n = 0
    for entries, outs in groups:
        for e in entries:
            conds = entry_to_conditions(e)
            for off, kind, ch in conds:
                dim = L.blk[off] + (V if kind == "b" else stoi[ch])
                m1.fc1.weight[n, dim] += 1.0
            m1.fc1.bias[n] = -(len(conds) - 1)
            for d, w in outs.items():
                m1.fc2.weight[d, n] = w
            n += 1
    assert n == m1.fc1.out_features

    # --- layer 2 attention: negation flag ----------------------------------
    a2 = model.blocks[1].attn
    big, pull = 100.0, 30.0
    for s in range(T):
        a2.W_q.weight[s, L.POS + s] = 1.0
    a2.W_q.weight[T, L.CONST] = 1.0
    for s in range(T):
        for j in range(T):
            if not (1 <= s - j <= NEG_SPAN):
                a2.W_k.weight[s, L.POS + j] = -big
    a2.W_k.weight[T, L.NEGR] = pull                 # prefer negator positions in window
    a2.W_v.weight[0, L.NEGR] = 1.0
    a2.W_o.weight[L.FLAG, 0] = 1.0

    # --- layer 2 MLP: flagged positives -> -1.0 each ----------------------
    m2 = model.blocks[1].mlp
    M = 100.0
    m2.fc1.weight[0, L.P] = 1.0
    m2.fc1.weight[0, L.C] = 1.0
    m2.fc1.weight[0, L.FLAG] = M
    m2.fc1.bias[0] = -M
    m2.fc2.weight[L.SCORE, 0] = -1.0      # subtract (P + C): cancels P, adds -1.0 per match

    # --- layer 3 attention: uniform average of per-position score ---------
    a3 = model.blocks[2].attn                 # q = k = 0 -> uniform over prefix
    a3.W_v.weight[0, L.SCORE] = 1.0
    a3.W_v.weight[0, L.P] = 1.0
    a3.W_o.weight[L.TOTAL, 0] = 1.0

    # --- unembedding: '1' iff total >= 0 ----------------------------------
    K = 1e4
    eps = 0.05 / T
    one, zero = stoi["1"], stoi["0"]
    for i in range(V):
        if i not in (one, zero):
            model.head.weight[i, L.CONST] = -K
    model.head.weight[one, L.TOTAL] = K
    model.head.weight[one, L.CONST] = K * eps


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "LexNeg_v1"
model_description = ("12-head shift register + AND-gate lexicon (+1.0 pos / -1.5 neg / +-0.4 weak), "
                     "negators -1.0 flag next 15 chars and flip positives to -1.0; uniform-sum sign readout.")


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
