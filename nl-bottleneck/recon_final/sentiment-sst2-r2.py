"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) lexicon-and-negation sentiment scorer for
`sentiment-sst2`:

  Layer 1  attention : 12 heads = shift register (head i copies the char i
                       steps back via sharp quadratic positional matching).
           MLP       : one ReLU AND-gate neuron per lexicon entry / negator,
                       reading the 12-char window.  Writes
                         SCORE += value  (+1.0 pos, -1.5 neg, +-0.4 weak, -1 negator)
                         PX    += value+1 (positive entries only; used for flipping)
                         NEGF  += 1       (negators)
  Layer 2  attention : 15 heads copy NEGF from offsets 1..15 -> NF (count of
                       negators in the preceding 15 chars).
           MLP       : if NF>=1, SCORE -= PX (positive entry worth v becomes -1).
  Layer 3  attention : uniform causal attention -> FINAL = mean(SCORE).
  Head               : logit('1') - logit('0') = K * (FINAL + eps)  (>=0 -> '1').
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

class Attn(nn.Module):
    """Causal multi-head attention with separate QK and V head dims, no scaling."""

    def __init__(self, d_model: int, n_heads: int, d_qk: int, d_v: int):
        super().__init__()
        self.H, self.dqk, self.dv = n_heads, d_qk, d_v
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.H
        q = self.W_q(x).view(B, T, H, self.dqk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.dqk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.dv).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, H * self.dv)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_qk, d_v, d_ff):
        super().__init__()
        self.attn = Attn(d_model, n_heads, d_qk, d_v)
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None

    def forward(self, x):
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Lexicon (written by hand, guided by the description; not fit to data)
# ---------------------------------------------------------------------------
# Syntax: "word" whole word, "stem*" word-initial stem, "*suf" word-final
# suffix; entries may contain spaces (phrases).

POSITIVE = """
good, great, best, better, fine, nice, nicely, love*, lovely, beautiful*, beauty, beauties, wonderful*,
excellent*, excellence, superb*, brilliant*, brilliance, terrific*, fantastic*, amazing*, amazed, astonish*,
stunning*, stunner, gorgeous*, charm*, delight*, enjoy*, fun, funny, funnier, funniest, hilarious*, hilarity,
witty, wit, wits, wittily, clever*, smart, smartly, intelligent*, intelligence, insight*, moving, movingly,
touching*, touched, poignant*, poignancy, heartfelt, heartwarming, heart warming, warm, warmly, warmth,
tender*, sweet, sweetly, sweetness, engag*, engross*, absorbing, compelling*, gripping*, riveting*,
thrill*, exciting*, excitement, entertain*, pleasur*, pleasant*, pleasing*, satisf*, rewarding*, refreshing*,
refreshed, fresh, freshness, original, originality, inventive*, imaginative*, imagination, creative*,
creativity, innovative*, unique*, remarkab*, memorab*, impressive*, impressed, triumph*, success*, win,
wins, winning, winner, winners, powerful*, power, powers, strong, strongly, strength*, solid, solidly, rich,
richly, richness, vivid*, lively, liveliness, vibrant*, energetic*, energy, spirited, spirit, inspir*, uplift*,
joy, joys, joyous*, joyful*, happy, happily, happiness, cheer*, laugh*, smile*, smiling, humor, humour,
humorous*, amusing*, amused, beautifully, perfect*, flawless*, admir*, respect*, honest*, sincer*,
genuine*, authentic*, elegan*, grace, graceful*, gracious, subtle*, subtlety, nuance*, deft*, skill*,
talent*, gifted, accomplish*, polished, crafted, well made, well acted, well done, well crafted,
well written, worth, worthy, worthwhile, recommend*, must see, classic, classics, gem, gems, treasure*,
marvel*, wonder, wonders, wondrous*, magnificen*, glorious*, glory, splendid*, superior, sublime*,
exquisite*, dazzl*, captivat*, fascinat*, intrigu*, provocative*, thought provoking, profound*, profundity,
wise, wisely, wisdom, thoughtful*, sharp, sharply, crisp, crisply, taut, suspense*, haunting*, lyrical*,
lyricism, poetic*, poetry, mesmeri*, hypnotic*, seductive*, sexy, cool, sassy, playful*, zany, quirky,
endear*, likab*, likeab*, lovab*, appealing*, attractive*, irresistib*, crowd pleaser, feel good, rousing,
stirring*, exhilarat*, breathtak*, spectacular*, epic, ambitious, ambition, bold, boldly, boldness, daring,
audaci*, brave, bravely, bravery, courage*, confident*, confidence, assured, assurance, achievement*,
sparkl*, shine, shines, shining, glow*, radiant*, luminous*, bright, genius, evocative*, atmospheric,
gentle*, kind, kindness, generous*, generosity, compassion*, humane, humanity, hope, hopeful*, optimis*,
tribute, celebrat*, unforgettable, timeless, enthrall*, enchant*, magic*, spellbind*, transcend*,
transfix*, illuminat*, revelat*, revelation, eloquen*, articulate, intimate*, intimacy, honesty, truthful*,
resonan*, resonate*, affecting, affectionate*, affection, empath*, sympath*, heart, hearts, soulful*,
uplifting, inspirational, rare, rarity, special, extraordinar*, exceptional*, outstanding*, phenomenal*,
incredib*, awesome*, superlative, stellar, top notch, first rate, first class, a plus, ingenious*,
ingenuity, resourceful*, intricate*, sophisticat*, savvy, shrewd*, astute*, keen, perceptive*, observant,
detailed, textured, layered, complex, complexity, ambitio*, fearless*, gutsy, edgy, provoca*, energiz*,
invigorat*, rollicking, rambunctious, raucous, boisterous, giddy, exuberan*, ebullien*, effervescen*,
buoyant*, breezy, bouncy, zippy, snappy, zesty, spunky, spry, nimble, agile, fluid, fluent*, seamless*,
effortless*, assured, confident, commanding, masterful*, masterpiece*, masterwork*, masterly, mastery,
virtuos*, tour de force, showcase*, standout*, stand out, standouts, gem of, scene stealing, scene stealer,
star making, star turn, terrific, wonderfully, marvelous*, marvellous*, fabulous*, splendor*, splendour*,
lavish*, sumptuous*, opulent*, ravishing*, lush, lushly, handsome*, pretty, prettily, beautif*,
picturesque, scenic, stunningly, dazzlingly, gloriously, sublimely, perfectly, brilliantly, cleverly,
intelligently, smartly, skillfully, skilfully, expertly, superbly, deftly, gracefully, elegantly,
tenderly, sensitively, sensitive*, delicate*, delicacy, finesse, poise, poised, dignity, dignified,
noble, nobility, decency, decent, integrity, heroic*, hero, heroes, heroine, champion*, victor*,
admirable*, laudab*, commendab*, praise*, praiseworthy, applau*, acclaim*, kudos, bravo, hail*,
cherish*, adore*, adorab*, cute, cuteness, charismatic*, charisma, magnetic*, magnetism, presence,
compelling, irresistible, addictive, unmissable, essential, important, relevant, timely, urgent,
necessary, valuable, invaluable, priceless, rich in, wealth of, worth seeing, worth watching,
worth the, worth your, enjoyable, entertaining, engaging, delightful, delightfully, charming,
charmingly, funny and, refreshingly, surprising*, surpris*, unexpected*, unpredictab*, twist*,
suspenseful, tense, tension, chilling*, creepy, scary, frightening*, spooky, eerie, eerily, unsettling,
intense*, intensity, gripping, harrowing*, wrenching, heartbreaking*, heartbreak*, devastating*,
tearjerk*, tear jerk*, tears, cathartic, catharsis, redempti*, redeem*, transform*, grown up, mature*,
maturity, adult, sensible, sane, wholesome, family friendly, good natured, good hearted,
big hearted, warm hearted, kind hearted, light hearted, lighthearted, high spirited, free spirited,
crowd pleasing, people pleasing, sure fire, surefire, slick*, stylish*, style, stylized, stylised,
visual*, visually, cinematic*, cinema, filmmaking, artistry, artful*, artistic*, art, craft, craftsman*,
craftsmanship, workmanship, rigorous*, disciplined, ambitious, accomplished, confident, assured,
sure handed, steady, consistent*, reliable, dependable, trustworthy, faithful*, loyal, respectful*,
intelligent, informative*, educational, enlighten*, illuminating, eye opening, eye opener,
mind blowing, jaw dropping, awe, awed, awesome, awestruck, wow*, gasp*, spectacle, grand*, grandeur,
majestic*, monumental*, landmark, milestone, breakthrough*, groundbreak*, ground break*, pioneer*,
visionar*, vision, visions, auteur, maestro, legend*, iconic, icon, immortal, enduring, endure*,
lasting, perennial, evergreen, cult, beloved, favorite*, favourite*, darling, sweetheart,
treat, treats, gift, gifts, blessing, bonus, delicious*, tasty, savory, scrumptious, yummy, juicy,
meaty, hearty, robust*, vigorous*, muscular, punchy, potent*, forceful*, dynamic*, dynamite,
explosive, electric*, kinetic*, propulsive, pulse pounding, heart pounding, adrenaline, rush,
fast paced, briskly, brisk, tight, tightly, economical, efficient*, concise, lean, sleek, streamlined,
clean, cleanly, pure, purely, simple pleasures, simplicity, unpretentious, modest*, humble, charm,
quaint, cozy, comforting*, comfort*, soothing*, calm*, serene*, peaceful*, tranquil*, gentle, lovingly,
loving, caring, nurturing, tender, earnest*, sincere, heartening, encourag*, reassur*, satisfying,
fulfilling*, enriching*, nourishing, wholesome, pleasingly, agreeabl*, amiab*, genial*, affab*,
congenial*, convivial, jovial*, jolly, merry, mirth*, gleeful*, glee, playfully, impish*, mischiev*,
sly, slyly, wry, wryly, droll, deadpan, ironic*, irony, satir*, sardonic, acerbic, biting, incisive*,
pointed, trenchant, scathing, cutting edge, cutting, savage*, sharp witted, quick witted, razor sharp,
laughs, laughter, chuckle*, giggle*, guffaw*, hoot, riot, uproarious*, side splitting, sidesplitting,
knee slapping, belly laugh*, howl*, howler, gag*, jokes, comedy, comedies, comedic, comic, comical*,
farcical, slapstick, screwball, romp, caper, frolic*, lark, blast, joyride, ride, adventure*, adventurous,
fantasy, dreamy, dreamlike, whimsical*, whimsy, fanciful, magical, mystical, mysterious, enigmatic*,
alluring, allure, mysterious, gripping, absorbing, involving, immersive, immerse*, transporting,
escapis*, sweeping, soaring, swooning, swoon*, romantic*, romance, passion*, sensual*, erotic*, steamy,
sizzl*, chemistry, sparks, smitten, infatuat*, love story, sweetheart, heartthrob, winsome*, fetching,
lovely, comely, stunning, ravishing, exotic, vivacious*, bubbly, sunny, upbeat, positive*, bright,
brightly, cheerful*, chipper, optimistic, idealis*, inspiring, aspir*, triumphant*, victorious, glorious,
rousing, stirring, anthem*, euphori*, ecstatic*, ecstasy, bliss*, rapture, rapturous*, heavenly,
divine*, angelic, wondrous, miraculous*, miracle*, phenomenon, sensation*, hit, smash, blockbuster,
knockout, knock out, knocks, sock*, bang up, top drawer, ace, aces, primo, prime, peak, pinnacle,
zenith, crown*, jewel*, highlight*, high point, high points, best of, finest, greatest, nicest,
smartest, sweetest, funniest, strongest, richest, freshest, brightest, boldest, bravest, wisest,
loveliest, prettiest, coolest, warmest, kindest, deepest, truest, purest, most original
""" + """,
gratifying, proud, pride, worthiest, ably, adept*, adroit*, consummate, impeccab*, meticulous*,
painstaking*, scrupulous*, careful*, attentive*, inspired, inventively, imaginatively, freshly,
originally, uniquely, singular*, distinctive*, distinct, idiosyncratic, personal, personality,
humanist*, humanis*, ennobl*, edify*, edifying, stimulating, stimulate*, challenging, thrillingly,
excitingly, beguil*, bewitch*, tantaliz*, titillat*, enrapt*, entranc*, rapt, riveted, hooked,
glued, mesmerized, transfixed, spellbound, moved, uplifted, inspired by, cheered, delighted, thrilled,
pleased, satisfied, impressed by, enjoyed, loved it, liked, like it, applause, standing ovation,
oscar, oscars, award*, honor*, honour*, merit*, deserv*, earn*, succeeds, succeed*, works, worked,
working, pays off, payoff, pay off, rewards, gains, benefit*, strength, strengths, asset*, virtue*,
plus, pluses, perk*, upside, highlights, wins over, winningly, captures, capture*, nails, nailed,
gets it right, hits, hit home, rings true, true to, truth, truthfully, real, realistic*, realism,
believab*, credib*, convincing*, persuasive*, compelling performance, plausib*, natural*, naturalis*,
unaffected, unforced, effortless, easy going, easygoing, relaxed, laid back, assured
"""

NEGATIVE = """
bad, badly, worst, worse, awful*, terribl*, horribl*, horrid*, dreadful*, dull*, boring*, bore, bores,
bored, boredom, tedious*, tedium, tiresome*, tired, tiring, monoton*, mediocr*, lame*, stupid*, dumb*,
silly, silliness, idiot*, inane*, inanity, pointless*, mess, messy, sloppy*, sloppily, clumsy, clumsi*,
awkward*, flat, flatly, bland*, stale, trite*, cliche*, predictab*, formula*, routine, generic,
derivative, unoriginal, contrived, contrivance*, forced, hackneyed, banal*, shallow*, superficial*,
empty, emptiness, hollow*, thin, thinly, weak, weakly, weakest, weakness*, poor, poorly, poorest,
failure*, disappoint*, letdown, let down, waste*, worthless*, useless*, junk, garbage, trash*,
crap*, rubbish, ugly, ugliness, unpleasant*, annoy*, irritat*, grating, obnoxious*, offensive*, offend*,
repuls*, repel*, disgust*, gross, vile, nasty, mean spirited, cheap, cheaply, cheapen*, shoddy*,
amateur*, incompetent*, incompetence, incoheren*, confus*, muddl*, murky, convoluted*, pretentious*,
pretension*, self indulgent, self important, smug*, preachy, sanctimonious*, heavy handed, overwrought,
overblown, overlong, overdone, overcooked, overstuff*, bloated, plodding, plod*, sluggish*, slow,
slowly, slowness, drag, drags, dragged, dragging, interminab*, endless*, long winded, sleep*, snooze*,
yawn*, lifeless*, listless*, soulless*, joyless*, humorless*, humourless*, mindless*, senseless*,
charmless*, unfunny, laughabl*, ridiculous*, absurd*, preposterous*, ludicrous*, implausib*, sappy,
schmaltz*, maudlin, mawkish*, cloying*, saccharine, syrupy, manipulat*, exploit*, cynical*, cynicism,
sour, disaster*, fiasco, flop, flops, misfire*, misguided, misstep*, mistake*, mistaken, wrong, wrongly,
problem*, flaw, flaws, flawed, unfortunate*, sadly, alas, regret*, sorry, pity, pitiful*, pitiab*,
shame, shameful*, shameless*, embarrass*, cringe*, wince*, painful*, pain, torture*, tortur*,
excruciat*, unbearab*, insufferab*, intolerab*, unwatchab*, avoid*, skip, dud, duds, turkey, stinker,
stink*, stinks, stunk, suck*, sucks, lousy, crummy, rotten, putrid, abysmal*, atrocious*, appalling*,
deplorab*, dismal*, dire, wretched*, miserab*, pathetic*, lamentab*, inept*, ineptitude, hapless,
hopeless*, clueless*, aimless*, mess of, muddle, jumble*, hodgepodge, patchwork, disjointed,
incoherent, haphazard*, scattershot, uneven*, unfocused, meandering, rambl*, wander*, aimlessly,
tired old, worn, worn out, threadbare, recycled, rehash*, retread*, warmed over, by the numbers,
paint by numbers, cookie cutter, run of the mill, middling, so so, forgettab*, unmemorab*, negligible,
inconsequential, trivial*, insignificant, slight, flimsy*, paper thin, wafer thin, one note,
one dimensional, cardboard, stock, stereotyp*, caricatur*, cartoonish*, wooden, stiff, stilted*,
lifeless, leaden, ponderous*, turgid, stodgy, sodden, soggy, limp, tepid, lukewarm, half baked,
half hearted, halfhearted, uninspired, uninspiring, uninteresting, uninvolving, unengaging,
unconvincing, unsatisfying, unsympathetic, unlikable, unlikeable, unappealing, unwatchable,
unnecessary, undistinguished, underwhelm*, underdeveloped, undercooked, unfulfilled, unrealized,
uneventful, unpleasant, unbelievable, uncomfortabl*, unfocused, unclear, unfinished, unpolished,
unimaginative, unremarkable, unexciting, unintentional*, unintelligible, unbearable, untalented,
witless, gutless, spineless, toothless, pointlessly, needless*, tasteless*, tactless, artless,
graceless, clueless, brainless, heartless, shapeless, plotless, characterless, airless, bloodless,
lackluster, lacklustre, lacking in, fizzl*, sputter*, stall*, stumbl*, falter*, flounder*, sink*,
sinks, sank, sunk, drown*, collaps*, crumbl*, implod*, derail*, undermin*, undone, ruin*, spoil*,
botch*, bungl*, mangl*, butcher*, squander*, fumbl*, wreck*, trainwreck, train wreck, car crash,
ham fisted, hamfisted, heavy hand*, clunky, clunk*, creaky, creak*, rickety, shaky, wobbly, labored,
laboured, laborious*, strained, straining, effortful, overly, overacted, overact*, overplay*, hammy,
ham, hams, mugging, scenery chewing, shrill*, strident, grating, abrasive*, jarring, cacophon*,
noisy, noise, loud, racket, headache*, migraine, nausea*, nauseat*, sickening*, sick, sickly, queasy,
revolting*, loathsome, loath*, hate*, hateful*, hatred, despis*, detest*, abhor*, contempt*,
contemptible, odious, repugnant, repellent, distasteful, unsavory, sleazy, sleaze, sordid, seedy,
smarmy, slimy, oily, crass*, vulgar*, crude*, coarse*, gratuitous*, exploitative, exploitation,
pandering, pander*, condescend*, patroniz*, insult*, demean*, degrad*, debas*, cheesy, cheese,
corny, corn, hokey, hokum, schlock*, kitsch*, tacky, tawdry, gaudy, garish, campy, bombast*,
pompous*, portentous*, grandios*, self serious, solemn*, dour, dreary*, drab, dingy, dismal,
gloomy, glum, bleak, depressing*, dispirit*, deflat*, demoraliz*, enervat*, exhaust*, wearisome,
wearying, weary, numbing, mind numbing, numbingly, stultify*, soporific*, sedative, narcotic,
comatose, catatonic, inert, static, stagnant, stuck, stalled, airless, lumbering, lumber*, ponderous,
plodding, tedium, longueurs, padded, padding, filler, overlong, too long, long, lengthy, protracted,
dragged out, drawn out, stretched, thin stretched, repetitive*, repetitious, redundant*, monotonous,
same old, rote, mechanical*, perfunctory*, lazy, lazily, laziness, sloppiness, careless*, hasty,
hastily, rushed, choppy, jumpy, erratic*, incoherently, nonsensical*, nonsense, gibberish, drivel,
twaddle, claptrap, hogwash, tripe, swill, dreck, schlock, muck, sludge, sludgy, slog, slogging, chore,
ordeal, endurance test, punishing, punishment, penance, purgatory, hell, hellish, nightmare*,
horrendous*, horrific*, hideous*, monstrous*, grotesque*, ghastly, gruesome, grisly, ugly, unsightly,
amateurish, amateurishly, low budget, low rent, second rate, third rate, b movie, straight to video,
direct to video, made for tv, tv movie, sitcom, sketch, skit, overrated, overpraised, overhyped, hype,
hyped, pretentiousness, artsy, arty, artsy fartsy, navel gazing, self absorbed, self satisfied,
self congratulatory, smugness, glib*, facile, simplistic*, superficially, shallowly, emptily,
vapid*, vacuous*, vacant, inane, insipid*, banality, fatuous, asinine, moronic, imbecil*, cretin*,
retarded, brain dead, dimwit*, dopey, goofy, doofus, nitwit, ninny, buffoon*, clown*, oafish, boorish,
juvenile, puerile, sophomoric, childish*, infantile, immature, adolescent, crude, crass, gross out,
grossout, toilet, potty, fart*, poop, vomit*, puke, barf, bodily, scatological, raunch*, lewd, smutty,
sleazy, trashy, exploitive, misogyn*, racist*, sexist*, homophob*, bigot*, hateful, mean, meanness,
cruel*, cruelty, callous*, cold, coldly, chilly, distant, detached, aloof, remote, clinical*, sterile,
antiseptic, bloodless, sanitized, tame, toothless, defanged, neutered, watered down, diluted, bland,
blandness, vanilla, milquetoast, wimpy, wishy washy, mushy, mush, gooey, goo, sugary, treacly, treacle,
sticky, soppy, weepy, tearjerker, melodram*, histrionic*, overheated, hysterical*, hysteria, shrill,
screechy, whiny, whining, whine*, complain*, petulant, sulky, mopey, mopes, mope*, sulk*, grim, grimly,
humorless, dismal, depressing, downer, bummer, letdown, comedown, anticlimax, anticlimactic*,
deflating, disappointingly, frustrat*, exasperat*, infuriat*, maddening*, aggravat*, vexing, galling,
irksome, bothersome, tiresome, troubl*, troubled, problematic, questionable, dubious*, suspect,
iffy, sketchy, shoddy, flimsy, slipshod, lousy, terrible, horrible, awful, dreadful, worse than,
worst of, nothing but, little more than, barely, merely, mere, only, simply, pass, passable, okay,
ok, mild*, mildly, modest, modestly, meager, meagre, scant, sparse, paltry, puny, measly, minor,
lesser, inferior, subpar, sub par, below par, below average, average, ordinary, mundane, humdrum,
pedestrian, prosaic, workmanlike, competent, serviceable, adequate, acceptable, tolerable,
watchable, inoffensive, harmless, innocuous, benign, mediocre, meh, blah, ho hum, whatever, lame,
nowhere, neither here, missed, misses, miss, missing the, off key, off putting, offputting,
off base, off target, off the mark, wide of, misfires, mishandl*, misjudg*, miscalculat*, miscast*,
misconceiv*, misdirect*, mismatch*, misplac*, misread*, misus*, mess up, messed, screw*, goof*,
blunder*, gaffe, faux pas, error*, failing at, failed attempt, wasted on, waste of, squandered,
thrown away, throwaway, disposable, forgotten, pallid, pale, paler, pales, pallor, wan, sallow,
anemic, anaemic, sickly, feeble*, frail, weakling, puny, impotent, ineffectual*, ineffective*,
inefficient, inadequa*, insufficient*, deficient*, deficien*, defect*, faulty, broken, busted,
damaged, crippl*, hobbl*, hamper*, hinder*, burden*, encumber*, saddled, weighed down, weigh*,
bogged, bog*, mired, sink, sunk, swamp*, drown, choke*, smother*, stifl*, suffocat*, strangl*,
constrict*, cramped, claustrophob*, confin*, stiflingly, oppressive*, overbearing, intrusive, heavy,
heavily, heavy going, tough going, hard going, rough going, trying, testing, taxing, demanding,
difficult, tough to, hard to, impossible to, unable, cannot, can not
""" + """,
distracting*, distract*, irrelevant, pointless, meaningless*, purposeless, motiveless, reasonless,
baffl*, bewilder*, perplex*, puzzl*, mystif*, confounding, incomprehensib*, impenetrab*, opaque,
obscur*, obtuse, dense, muddy, murk*, foggy, hazy, vague*, fuzzy, unclear, sketchy, incomplete,
fragment*, scrappy, ragged, frayed, tattered, shabby, threadbare, mangy, scruffy, grubby, grimy,
grungy, cheaply made, poorly made, badly made, badly written, badly acted, poorly written,
poorly acted, ill conceived, ill advised, ill fated, misbegotten, wrongheaded, wrong headed,
pig headed, dunderheaded, empty headed, dull witted, half witted, numbskull, schmuck, shmuck, jerk,
loser, losers, losing, lose, loses, lost, worthlessness, futil*, fruitless, vain, in vain, sterile,
barren, desolate, arid, dry, dryly, drily, parched, dusty, musty, moldy, mouldy, rancid, rank,
fetid, foul, stench, stale, mothball*, fossil*, antiquat*, dated, outdated, old fashioned, obsolete,
passe, creaky old, hoary, moth eaten, clapped out, past its prime, over the hill, has been, washed up
"""

# Words I listed above while brainstorming but that are too neutral / ambiguous
# to be lexicon entries (they are removed before compiling).
DROP = set("""kind,heart,art,real,plus,works,worked,working,ride,rush,hits,hit,personal,adult,complex,
presence,important,cinema,style,like it,liked,earn*,visual*,visually,rare,pass,only,simply,long,heavy,
heavily,lost,lose,loses,cold,mean,competent,dense,rank,ok,okay,trying,testing,difficult,sketch,skit,
sitcom,toilet,bodily,corn,cheese,stock,loud,noise,sick,miss,average,ordinary,minor,static,remote,
distant,dry,weigh*,bog*,sink,sinks,tough to,hard to,merely,mere,vain,whatever,demanding,taxing,
tv movie,made for tv,cult,clean,pure,purely,lean,adventure*,fantasy,gag*,ham,hams,pale,suspect,
neither here,nowhere,barely,cannot,can not,unable,missing the""".replace("\\n", "").replace("\n", "").split(","))

WEAK_POS = "film, performance*, and"
WEAK_NEG = "movie, plot, just, would, un*, dis*, over*, *less"

NEGATORS = """
not, n't, no, never, without, too, less, lacks, lack, lacking, lacked, fails, fail, failed,
failing, rather than, nothing, none, nor, neither, cannot, hardly, barely, scarcely, rarely,
seldom, nobody, nowhere, instead of, devoid, absent, missing, unable, far from, short of,
minus, nary, isn't, wasn't, doesn't
"""

BOUND = (" ", "_")
WIN = 12          # chars visible: offsets 0..11
MAXSTEM = 11


def parse_list(s: str) -> list[str]:
    out, seen = [], set()
    for e in s.replace("\n", " ").split(","):
        e = " ".join(e.split())
        if e and e not in seen:
            seen.add(e)
            out.append(e)
    return out


def compile_entry(e: str):
    """Return (conds, left_boundary_offset or None).
    conds: list of (offset, tuple_of_chars) that must all hold at the firing position."""
    if e.startswith("*"):                       # suffix: ...xyz followed by boundary
        s = e[1:][-MAXSTEM:]
        conds = [(0, BOUND)] + [(j + 1, (s[-1 - j],)) for j in range(len(s))]
        return conds, None
    if e.endswith("*"):                         # stem: boundary + abc...
        s = e[:-1][:MAXSTEM]
        return [(j, (s[-1 - j],)) for j in range(len(s))], len(s)
    if len(e) + 2 <= WIN:                       # whole word / phrase
        conds = [(0, BOUND)] + [(j + 1, (e[-1 - j],)) for j in range(len(e))]
        return conds, len(e) + 1
    s = e[:MAXSTEM]                             # too long: 11-char stem
    return [(j, (s[-1 - j],)) for j in range(len(s))], len(s)


class SimpleTransformer(nn.Module):
    """3-layer lexicon/negation transformer (no LayerNorm)."""

    N_SHIFT = 12
    N_NEG_HEADS = 15

    def __init__(self, vocab_size: int, max_seq_len: int = 32, **_):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        V = vocab_size
        # residual layout
        self.TOK = 0
        self.WIN0 = V                                   # 12 blocks of V
        self.P_T = V + WIN * V                          # t
        self.P_T2 = self.P_T + 1                        # t^2
        self.ONE = self.P_T + 2                         # constant 1
        self.IND0 = self.P_T + 3                        # [t==k], k=0..10
        self.SCORE = self.IND0 + MAXSTEM
        self.PX = self.SCORE + 1
        self.NEGF = self.SCORE + 2
        self.NF = self.SCORE + 3
        self.FINAL = self.SCORE + 4
        d = self.FINAL + 1
        self.d_model = d

        self.pos_list = [e for e in parse_list(POSITIVE) if e not in DROP]
        neg_ex = set(self.pos_list) | set(parse_list(NEGATORS)) | DROP
        self.neg_list = [e for e in parse_list(NEGATIVE) if e not in neg_ex]
        self.wpos_list = parse_list(WEAK_POS)
        self.wneg_list = parse_list(WEAK_NEG)
        self.negator_list = parse_list(NEGATORS)
        n_ff1 = (len(self.pos_list) + len(self.neg_list) + len(self.wpos_list)
                 + len(self.wneg_list) + len(self.negator_list))

        self.token_emb = nn.Embedding(V, d)
        self.pos_emb = nn.Embedding(max_seq_len, d)
        self.blocks = nn.ModuleList([
            Block(d, self.N_SHIFT, 2, V, n_ff1),        # shift register + lexicon
            Block(d, self.N_NEG_HEADS, 2, 1, 2),        # negation scope + flip
            Block(d, 1, 1, 1, 0),                       # uniform sum
        ])
        self.head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    m = model
    V = m.vocab_size
    stoi = task.stoi
    A = 30.0                 # positional sharpness
    M = 100.0                # gating constant
    K = 1000.0               # output logit scale
    EPS = 0.05               # ties (sum == 0) -> '1'

    with torch.no_grad():
        for p in m.parameters():
            p.zero_()

        # embeddings
        for c, i in stoi.items():
            m.token_emb.weight[i, m.TOK + i] = 1.0
        for t in range(m.max_seq_len):
            m.pos_emb.weight[t, m.P_T] = float(t)
            m.pos_emb.weight[t, m.P_T2] = float(t * t)
            m.pos_emb.weight[t, m.ONE] = 1.0
            if t < MAXSTEM:
                m.pos_emb.weight[t, m.IND0 + t] = 1.0

        # ---- layer 1 attention: shift register -------------------------
        at = m.blocks[0].attn
        for i in range(m.N_SHIFT):
            # score(s) = A*(2(t-i)s - s^2), maximal at s = t-i (or 0 if t<i)
            at.W_q.weight[2 * i, m.P_T] = 2 * A
            at.W_q.weight[2 * i, m.ONE] = -2 * A * i
            at.W_q.weight[2 * i + 1, m.ONE] = -A
            at.W_k.weight[2 * i, m.P_T] = 1.0
            at.W_k.weight[2 * i + 1, m.P_T2] = 1.0
            for c in range(V):
                at.W_v.weight[i * V + c, m.TOK + c] = 1.0
                at.W_o.weight[m.WIN0 + i * V + c, i * V + c] = 1.0

        # ---- layer 1 MLP: lexicon AND gates -----------------------------
        mlp = m.blocks[0].mlp
        entries = ([(e, 1.0, True) for e in m.pos_list]
                   + [(e, -1.5, False) for e in m.neg_list]
                   + [(e, 0.4, True) for e in m.wpos_list]
                   + [(e, -0.4, False) for e in m.wneg_list]
                   + [(e, None, False) for e in m.negator_list])
        for n, (e, val, is_pos) in enumerate(entries):
            conds, lb = compile_entry(e)
            for off, chars in conds:
                for ch in chars:
                    mlp.fc1.weight[n, m.WIN0 + off * V + stoi[ch]] += 1.0
            nc = len(conds)
            if lb is not None:
                for ch in BOUND:
                    mlp.fc1.weight[n, m.WIN0 + lb * V + stoi[ch]] += 1.0
                mlp.fc1.weight[n, m.IND0 + lb - 1] += 1.0      # sentence start
                nc += 1
            mlp.fc1.bias[n] = -(nc - 1)
            if val is None:                                     # negator
                mlp.fc2.weight[m.SCORE, n] = -1.0
                mlp.fc2.weight[m.NEGF, n] = 1.0
            else:
                mlp.fc2.weight[m.SCORE, n] = val
                if is_pos:
                    mlp.fc2.weight[m.PX, n] = val + 1.0

        # ---- layer 2 attention: copy NEGF from offsets 1..15 -> NF -------
        at = m.blocks[1].attn
        for h in range(m.N_NEG_HEADS):
            k = h + 1
            at.W_q.weight[2 * h, m.P_T] = 2 * A
            at.W_q.weight[2 * h, m.ONE] = -2 * A * k
            at.W_q.weight[2 * h + 1, m.ONE] = -A
            at.W_k.weight[2 * h, m.P_T] = 1.0
            at.W_k.weight[2 * h + 1, m.P_T2] = 1.0
            at.W_v.weight[h, m.NEGF] = 1.0
            at.W_o.weight[m.NF, h] = 1.0

        # ---- layer 2 MLP: flagged positives -> -1 each --------------------
        # delta = -(ReLU(PX + M(NF-1)) - ReLU(M(NF-1)))  = -PX if NF>=1 else 0
        mlp = m.blocks[1].mlp
        mlp.fc1.weight[0, m.PX] = 1.0
        mlp.fc1.weight[0, m.NF] = M
        mlp.fc1.bias[0] = -M
        mlp.fc1.weight[1, m.NF] = M
        mlp.fc1.bias[1] = -M
        mlp.fc2.weight[m.SCORE, 0] = -1.0
        mlp.fc2.weight[m.SCORE, 1] = 1.0

        # ---- layer 3 attention: uniform mean of SCORE ---------------------
        at = m.blocks[2].attn            # W_q, W_k stay zero -> uniform
        at.W_v.weight[0, m.SCORE] = 1.0
        at.W_o.weight[m.FINAL, 0] = 1.0

        # ---- unembedding ---------------------------------------------------
        T_last = task.prompt_len         # number of positions averaged at '='
        m.head.weight[stoi["1"], m.FINAL] = K
        m.head.weight[stoi["1"], m.ONE] = K * EPS / T_last
        m.head.weight[stoi["0"], m.ONE] = 0.0
        for c, i in stoi.items():
            if c not in "01":
                m.head.weight[i, m.ONE] = -1e4


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "LexNeg_v1"
model_description = ("Shift-register (12 heads) + AND-gate lexicon MLP (+1 pos, -1.5 neg, +-0.4 weak), "
                     "negators -1 and flip positives to -1 within next 15 chars, uniform-sum sign readout.")


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
