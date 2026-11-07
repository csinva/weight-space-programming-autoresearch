"""Interpretable transformer embedder for fMRI language encoding.

Strategy
--------
Each word in the 10-gram is tokenized into a small set of "feature tokens"
(function-word class, semantic category, perceptual modality, concreteness,
animacy, valence/arousal, frequency bucket, morphology, person reference).
The token embedding row of each feature token is a one-hot at its dedicated
feature dimension, replicated across attention head slices.

A single hand-wired attention layer pools those one-hot tokens over the
n-gram at several recency time-scales (each head uses position-keyed scoring).
The final-token hidden state is therefore a multi-scale recency-weighted
"bag of interpretable lexical features" for the 10-gram, and ridge maps it
to voxels.

Everything goes through the real `SimpleTransformer.forward` pass. No
training, no gradient descent, no pretrained weights, no external libraries
for embedding computation.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results, plot_corr_over_iterations,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ---------------------------------------------------------------------------
# Char vocab kept for optional orthographic content; the real "vocabulary"
# is the feature-token set built below.
# ---------------------------------------------------------------------------
_VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\"
_BASE_CHARS = ['<pad>', '<unk>'] + list(_VOCAB_CHARS)
N_CHAR = len(_BASE_CHARS)

PAD_ID = 0
UNK_ID = 1
POS_DIM = 0       # residual dim 0 holds j (position index inside the ngram timeline)
BIAS_DIM = 1      # residual dim 1 holds a constant 1 (the bias channel)
CAT_OFFSET = 2    # feature dims start here within each head slice

_stoi = {c: i for i, c in enumerate(_BASE_CHARS)}

# Multi-scale recency lambdas for the per-head position-keyed attention scores.
# lambda<0 -> primacy (look at the front of the n-gram), 0 -> uniform mean,
# >0 -> recency (the larger, the more concentrated on the last word).
LAMBDAS = (-1.0, 0.0, 4.0, 16.0)  # v96: weaker primacy (was -2)

# Words actually consumed from the end of the n-gram (10-gram).
N_APPEND_WORDS = 12

# Each word emits 'reps' copies of its feature tokens. Final-word emphasis
# increases its weight in the uniform-mean head (lambda=0).
RECENCY_REPS = (2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)  # v218 emphasize last word
# Content words emit features CONTENT_BONUS extra times on top of RECENCY_REPS
# (so the lambda=0 "global mean" head is biased toward content over function).
CONTENT_BONUS = 1  # v211 retry at new baseline

USE_CHAR_CONTENT = False  # reverted in v11 (random char hurt heavily)
CHAR_CONTENT_STD = 1.0


# ---------------------------------------------------------------------------
# Hand-coded lexicons (compact, human-readable). These are derived from
# well-known cognitive/neuro categories; nothing here was trained or
# computed from corpus statistics.
# ---------------------------------------------------------------------------
_SEM_CATEGORIES = {
    "MOTION": "go goes went going gone come comes came coming run ran running runs walk walks walked walking move moves moved moving fly flies flew flown drive drove driven ride rides rode jump jumps jumped falls fall fell fallen throw throws threw catch caught catches turn turned turns turning rush rushes chase chased chases climb climbs climbed crawl crawls slide slides rolls roll spin spins march marches step stepped steps leave left leaves arrive arrives arrived enter enters entered exit exits return returns returned follow followed approaches approach approached escape escaped flee fleeing swim swam dive dived sit sits sitting sat stand stood standing stands lay laying lying lies swing swings swung".split(),
    "SPACE": "up down left right above below under over inside outside near far here there front back top bottom between among around through across along beside behind beyond edge edges corner corners middle center centre side sides north south east west forward forwards backward backwards backwards upward upwards downward downwards out off away apart together onto toward towards against next ahead amid amidst within without alongside beneath underneath atop opposite adjacent surrounding".split(),
    "TIME": "time times now then today tomorrow yesterday soon later before after early late always never often sometimes year years month months week weeks day days hour hours minute minutes second seconds moment moments morning mornings night nights evening evenings afternoon afternoons noon midnight past future present while during until since again ago already yet still when whenever whence forever instantly immediately recently currently briefly".split(),
    "QUANTITY": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety zero many few several all some none most least more less much little half double twice huge tiny count number numbers dates lot lots dozen hundred thousand million plenty enough single whole total each every first second third last hundreds thousands millions billion billions count countless multiple multiples couple bunch piles handful many fewer fewer additional extra".split(),
    "BODY": "head face eye eyes ear ears nose mouth lip lips tooth teeth hand hands arm arms leg legs foot feet finger fingers hair skin heart blood bone bones back chest shoulder shoulders knee knees throat stomach brain neck chin cheek wrist elbow thumb nail body skull beard tears tear scar scars".split(),
    "PERSON": "man men woman women boy boys girl girls child children people person guy guys lady kid kids baby friend friends mother father mom dad sister brother son daughter wife husband family neighbor stranger crowd human folk gentleman".split(),
    "SOCIAL": "together alone meet met meeting marry married wedding party group team gang community share shared help helped helping agree argue argued fight fought war peace trust betray join visit invite welcome greet".split(),
    "EMOTION_POS": "happy happily happiness joy joyful joys glad gladly love loved loves loving like liked likes enjoy enjoyed enjoying enjoys excited exciting excitement wonderful wonderfully great greatness amazing amazingly beautiful beautifully pleasure pleasures pleasant smile smiled smiles smiling laugh laughs laughed laughing proud proudly pride hope hoped hoping hopes hopeful delight delighted delights delightful cheerful cheerfully cheered pleased grateful gratitude relief relieved calm calmly thrill thrilled thrilling thrills euphoric ecstatic content contented contentment satisfaction satisfied jubilant elated overjoyed".split(),
    "EMOTION_NEG": "sad sadness angry anger afraid afraidly fear fears feared scared scarier frightened frightens frightening worried worry worries worrying cry cried cries crying pain pains painful hurt hurts terrible awful awfully horrible hate hated hates hating disgust disgusts disgusted grief griefs sorrow lonely loneliness nervous nervously anxious anxiously upset miserable misery depressed depression guilt guilty shame jealous jealousy fears scares scared fearful sadly sadder sorrow sorrows grieve grieving grieved cries crying sorrowful angered jealousy resentful bitter bitterly miserably wept weep weeping mournful frustrated frustration shame shameful regret regretted regretting fury furious annoyed annoying anguish anguished dreaded dread despair desperate".split(),
    "COMMUNICATION": "say said says saying tell told telling tells speak spoke spoken speaking talk talked talking talks ask asked asking asks answer answers answered answering call called calling calls shout shouts shouted shouting yell yelled whisper whispers whispered word words voice voices question questions story stories explain explained explains read reads write wrote writing writes letter letters book books reply replies replied discuss discussed mention mentioned mentions describe described describes name names news conversation conversations promise promised promises thank thanks thanked thanking announce announced state stated".split(),
    "MENTAL": "think thought thinks thinking know knew known knows believe believed believes believing remember remembered remembers remembering forget forgot forgotten forgets forgetting understand understood understands understanding realize realized realizes realizing wonder wondered wonders wondering imagine imagined imagines imagining guess guessed guesses guessing idea ideas mind learn learned learns learning dream dreamed dreams dreaming decide decided decides deciding suppose supposed supposing consider considered considering considers expect expected expects expecting assume assumed assumes doubt doubted doubts notice noticed notices noticing want wanted wants wanting wish wished wishes need needed needs needing hope hoped hopes hoping mean meant means meaning figure figured figures figuring plan planned plans planning try tried tries trying care cared cares caring teach taught teaches teaching experience experienced experiences truth knows believes thinks remembers forgets understands realizes wonders imagines guesses thinking knowing believing remembering forgetting understanding decision decisions choice choices opinion opinions reflect reflected reflecting reflects meditate meditates meditated considering pondering ponder pondered recall recalled recalls recalling perceive perceived perceives".split(),
    "PERCEPTION": "see saw seen seeing look looked looking looks watch watched watching watches hear heard hears hearing listen listened listens listening smell smelled smells smelling taste tasted tastes tasting touch touched touches touching feel felt feels feeling notice noticed notices noticing stare stared stares staring glance glanced glances observe observed observes observing gaze gazed gazes gazing peer peered peering glimpse glimpsed sniff sniffed peeking peeked peek".split(),
    "FOOD": "eat ate eaten eating food drink drank drinking water bread meat fruit apple orange meal meals breakfast lunch dinner cook cooked cooking hungry thirsty sweet bitter sour salt sugar coffee tea wine beer milk egg cheese cake soup rice eats drinks cooks ate ate eaten eaten cooked cookies cookie pizza burger pasta noodles salad sandwich sandwiches dessert desserts chocolate candy candies snack snacks butter sauce sauces juice juices yogurt cereal pancakes waffles bacon ham turkey chicken vegetable vegetables potato potatoes tomato tomatoes onion garlic carrot peach peaches berry berries strawberry banana grape grapes lemon mushroom pizza".split(),
    "PLACE": "house home homes room rooms door doors window windows wall walls floor street streets road roads city cities town towns country school church store shop office building park garden field forest mountain river ocean sea lake beach sky world land farm village ground cabin central downtown".split(),
    "OBJECT": "thing things stuff box book books table chair bed car cars key keys money paper bag bottle cup phone clock machine tool tools wheel stone wood metal glass cloth knife pen door chain rope ball gun camera computer screen coin coins triangle".split(),
    "NATURE": "tree trees fire air earth wind rain snow storm sun moon star stars cloud clouds animal dog cat bird birds fish horse flower flowers grass leaf leaves rock rocks soil dirt mud ice wave hill valley river rivers stream forests forest woods bush bushes branch branches roots leafy pine oak maple sand sands sandy sandbar shore shores beach beaches pond ponds creek meadow meadows lake lakes mountain mountains mountainous waterfall waterfalls".split(),
    "HEALTH": "doctor doctors nurse nurses surgeon surgeons surgery hospital hospitals emergency patient patients sick ill illness illnesses disease diseases pain pains ache aches hurt hurts hurting injured injury injuries wound wounded blood bloody heal healed healing cure cured pill pills medicine medicines drug drugs treatment treatments cancer fever cough coughing epilepsy seizure seizures ambulance clinic operation operations recovery dying health insurance bandage stitches diagnosis symptoms therapy therapist".split(),
    "QUALITY": "good bad new old young right wrong true false real fake strange normal important hard easy soft strong weak rich poor clean dirty empty full heavy light bright dark sharp dull fresh nice fine perfect big small large little long short tall wide narrow huge tiny crazy weird wild quiet loud sorry able main different same fancy plain rough smooth thick thin deep flat round fast slow quick ready tough best worst slowly quickly excellent terrific awful brilliant ordinary unusual common typical pleasant unpleasant powerful gentle obvious clear unclear vague".split(),
    "WORK_MONEY": "work worked working job jobs money pay paid buy bought buying sell sold selling business company boss market price cost dollar dollars trade build built building factory worker wage profit bank store customer".split(),
    "COLOR": "red blue green yellow white black gray grey brown orange purple pink color colors colour golden silver dark bright pale crimson scarlet rosy beige tan ivory navy maroon teal violet hue shade tint colored".split(),
    "KINSHIP": "mother father mom dad parent parents son daughter sister brother wife husband child children baby uncle aunt cousin grandmother grandfather grandma grandpa family nephew niece sons daughters sisters brothers wives husbands children babies kids cousins aunts uncles relative relatives stepfather stepmother stepson stepdaughter inlaw inlaws spouse spouses".split(),
    "ANIMAL": "dog dogs cat cats bird birds fish fishes horse horses cow cows pig pigs sheep chicken chickens duck ducks lion lions tiger tigers bear bears wolf wolves fox foxes deer rabbit rabbits mouse rat rats snake snakes frog frogs insect insects bug bugs bee bees ant ants spider spiders animal animals creature creatures puppy puppies kitten kittens kitty mouse mice rats squirrel squirrels chipmunk chipmunks goat goats elephant elephants giraffe giraffes whale whales dolphin dolphins shark sharks shrimp crab crabs eagle eagles hawk hawks owl owls parrot parrots pigeon pigeons hen hens rooster bull bulls cattle calf calves lamb lambs piglet pony ponies".split(),
    "WEATHER": "rain rained raining rains snow snowed snowing snows wind windy storm storms stormy sunny sunshine cloudy clouds cloud cold hot warm cool freezing freeze frozen fog foggy ice icy frost frosty heat hail hailstorm lightning thunder thunderstorm drizzle downpour blizzard hurricane tornado winter summer spring autumn fall season seasons weather temperature climate humid humidity dry damp".split(),
    "POSSESSION": "have has had having own owns owned get gets getting got gotten give gave given take took taken takes keep kept hold held lose lost find found bring brought carry carried receive offer put set place placed leave left use used using wait waited waiting stay stayed staying stand stood sit sat sent send check checked wear wore worn".split(),
    "CHANGE": "become became becoming becomes change changed changes changing grow grew grown growing grows turn turned turns turning increase increased increases increasing decrease decreased decreases rise rose risen rises rising fall fell falls falling fallen break broke broken breaks breaking make made makes making build built builds building create created creates creating destroy destroyed destroys destroying form formed forms forming develop developed develops developing begin began begun begins beginning start started starts starting stop stopped stops stopping end ended ends ending finish finished finishes finishing open opened opens opening close closed closes closing happen happened happens happening cut cuts cutting hit hits drop dropped drops dropping remove removed removes removing appear appeared appears appearing spend spent spends spending bend bent burst bursting transform transformed shrink shrank expand expanded".split(),
    "INTENSITY": "very really so too quite rather extremely incredibly absolutely totally completely almost nearly barely hardly just only even much way super pretty kind sorta kinda highly utterly purely simply fully entirely truly genuinely particularly especially significantly somewhat slightly mildly".split(),
    "CLOTHING": "shoe shoes shirt shirts pants dress dresses coat coats jacket hat hats sock socks glove gloves scarf belt tie suit boot boots sweater skirt jeans clothes clothing button pocket sleeve collar zipper cap garment garments uniform shorts blouse blouses sweatshirt hoodie hoodies trench shawl shawls slipper slippers sandal sandals sneaker sneakers gown gowns robe vest vests trousers tights stockings ribbon ribbons necklace bracelet earring earrings ring rings".split(),
    "VEHICLE": "car cars truck trucks bus buses train trains plane planes boat boats bike bikes motorcycle taxi cab subway seat seatbelt wheel engine brake brakes gas drive driving road traffic helicopter pilot flight airport jet".split(),
    "TECH": "phone phones computer screen tv television radio camera internet email text message call button machine wire battery light switch electric power".split(),
    "LIFE_DEATH": "life live lived lives living alive born birth grow grew growing grows age aged ages aging young old die died dies dying death dead deaths kill killed kills killing survive survived survives surviving breathe breathed breathing breath heartbeat exist existed existing existence murder murdered drown drowned suicide dead corpse buried funeral grave casket coffin tombstone".split(),
    "SCHOOL": "school university college campus class classroom student students teacher professor study studied learn lesson grade exam test homework library team club semester".split(),
    "RELIGION": "god gods church churches pray prayed praying prayer prayers faith religion religious holy heaven hell soul souls spirit spirits bible jesus christ christian christians christianity sin sins angel angels devil demon devils priest priests worship worshipped sacred divine blessed pastor bishop monk nun temple temples synagogue mosque rabbi muslim islamic jewish catholic protestant ritual rituals sacrifice baptism".split(),
    "GAME_PLAY": "play played plays playing game games sport sports football basketball baseball soccer tennis golf hockey team teams score scored win wins won winning lose loses lost losing ball bat field coach coaches player players fun funny awesome joke jokes toy toys puzzle puzzles cards card deck race raced racing trophy medal champion competition tournament match round goal".split(),
    "SELF_MOTION": "go goes went going gone come comes came coming run ran running walk walked walking move moved moving fly flew flies flying swim swam swims swimming climb climbed climbs climbing jump jumped jumps jumping fall fell falls falling rise rose risen rises rising arrive arrived arrives enter entered enters leave left leaves leaving return returned returns wander wandered wanders crawl crawled crawls roll rolled rolls slide slid sliding step stepped stepping march marched approach approached approaches".split(),
    "CAUSED_MOTION": "throw throws threw thrown throwing push pushes pushed pushing pull pulls pulled pulling carry carries carried carrying lift lifts lifted lifting drag drags dragged dragging drop drops dropped dropping kick kicks kicked kicking toss tosses tossed shove shoves shoved grab grabs grabbed grabbing hand hands handed bring brings brought roll rolls rolled rolling slide slides slid sliding fling flings flung hurl hurls hurled".split(),
    "SPEECH_ACT": "say said says saying tell told telling tells ask asked asking asks answer answered answers answering speak spoke speaks speaking spoken talk talked talks talking shout shouted shouts shouting yell yelled yells yelling whisper whispered whispers whispering call called calls calling reply replied replies replying explain explained explains explaining mention mentioned mentions mentioning suggest suggested suggests suggesting confess confessed admit admitted insist insisted argued argue argues arguing".split(),
    "DISCOURSE": "because so then therefore thus hence since although though however but yet still meanwhile afterward afterwards consequently whereas otherwise nonetheless besides moreover anyway when while after before until once whenever unless instead finally eventually suddenly".split(),
    "NUMERIC": "one two three four five six seven eight nine ten hundred thousand million first second third twice double half".split(),
    "MOTOR": "grab push pull pulled lift throw kick grip hold holding held carry hit punch grasp reach shave wipe squeeze press pick poke tap pat tug".split(),
    "ABSTRACT_REL": "cause caused because reason result effect purpose means kind sort type way ways form part parts whole sense point fact case matter problem question chance luck fate course rest bit influence experience risk risks truth".split(),
    "SUBSTANCE": "cigarette cigarettes smoke smoking smoked tobacco pack drug drugs alcohol beer wine pill pills medicine weed pot ash lighter match matches nicotine".split(),
    "MONEY_NUM": "dollar dollars cent cents penny dime buck bucks cost price cheap expensive free pay paid owe debt cash credit bill bills change worth value".split(),
    "PEOPLE_ROLE": "boyfriend girlfriend friend boss worker assistant director directors manager nurse teacher student officer guard leader member partner clerk agent owner customer guest host neighbor".split(),
    # v62: NAME and PLACE_PROPER cats dropped — proper-noun lexicons rarely
    # generalize to actual narrative names/places not in the list.
}

# Valence (sentiment).
_VAL_POS = set("good great love like happy joy nice kind beautiful wonderful best better win won success hope safe friend gift smile laugh warm bright fun pleasant gentle clean fresh free peace calm glad enjoy enjoyed proud excited amazing perfect lucky grateful cheerful delight pleased comfort sweet cool awesome favorite special brave strong healthy smart funny laughing celebrate party loved liked likes loves loved enjoying friendly trusted soft warmer kindness happiness joyful peaceful relaxed gorgeous heaven blessed marvelous talented charming hopeful caring tender precious treasure dream dreams victory wins easygoing".split())  # v130: expanded
_VAL_NEG = set("bad worse worst hate fear pain hurt sad angry death dead kill killed lost lose fail wrong sick ill dark cold cruel ugly dirty broken danger trouble war fight blood enemy evil sorry afraid scared worried worry cry terrible awful horrible nervous anxious lonely guilt shame mad upset stress hard tough struggle difficult problem problems wound injury cancer disease tears suffering frightened poor weak tired exhausted crying alone".split())
# v54: strong-magnitude valence subsets (extreme sentiment words).
_VAL_POS_INT = set("love amazing perfect wonderful best awesome beautiful brilliant excited thrilled grateful proud delighted celebrate".split())
_VAL_NEG_INT = set("hate hated terrible awful horrible worst evil cruel kill killed murder murdered furious enraged disgusted horror terror frightened devastated".split())

# Animacy.
_ANIMATE = set((
    "man men woman women boy girl child children people person friend mother father "
    "son daughter sister brother dog cat bird fish horse cow lion tiger bear wolf "
    "animal baby human teacher doctor king queen soldier worker player crowd folk"
).split())

# Arousal.
_HIGH_AROUSAL = set("scream shout run fight fire explode crash rush panic terror excited thrilled furious rage storm danger attack chase escape shock kill blood gun death dead crash smash burst slam violent fierce wild desperate frantic terrified horror screaming yelling".split())

# Perceptual modality.
_MODALITY = {
    "VISION": "see saw seen seeing look looked looking looks watch watched watching bright dark color colors red blue green yellow white black light lights shadow shadows glow shine shining appear appeared vision sight glance glanced stare stared gaze visible image picture view scene".split(),
    "SOUND": "hear heard hearing listen listened loud quiet sound sounds noise noises music song songs voice voices ring rang bell bang banging crash whisper whispered scream screamed shout yell echo silence silent loud tune hears listens noisily musical singing sing sang sung hum hummed humming buzz buzzing thunder thundered click clicked clicking knock knocked knocking tap tapped tapping squeak squeaking creak creaking".split(),
    "TOUCH": "touch touched feel felt soft hard rough smooth warm cold hot cool wet dry sharp press pressed grip held holding squeeze rub texture sticky slippery".split(),
    "TASTE": "taste tasted tastes tasting sweet bitter sour salty spicy delicious flavor flavour flavors flavours eat ate yummy bland savory tasty tangy gulp swallow swallowed crunchy crunch crunchy mouth chew chewed chewing".split(),
    "SMELL": "smell smelled scent odor odour fragrance stink stinky perfume aroma nose sniff".split(),
    "MOTOR": "grab grabbed push pushed pull pulled lift lifted throw threw kick kicked run ran walk walked jump jumped grip held hold carry carried hit punch grasp reach reached swing wave squeeze grasps grabbing pushes pushing pulls pulling lifts throws threw catches caught punches punching slams slammed slap slapped slapping wrestle wrestled bend bent bending stretching".split(),
}
_CAT2MOD = {"COLOR": "VISION", "FOOD": "TASTE", "BODY": "TOUCH", "MOTION": "MOTOR",
            "ANIMAL": "VISION", "NATURE": "VISION", "WEATHER": "VISION",
            "SPEECH_ACT": "SOUND", "COMMUNICATION": "SOUND"}

# Concreteness.
_CONCRETE = set("house tree dog cat car book table chair hand eye water fire stone door window food bird fish rock wall floor street road wood metal glass bottle cup phone money".split())
_ABSTRACT = set("idea thought love fear hope time truth freedom justice mind dream memory reason power belief fact chance luck soul spirit meaning".split())
_CONCRETE_CATS = {"BODY", "FOOD", "PLACE", "OBJECT", "NATURE", "ANIMAL", "COLOR",
                  "CLOTHING", "VEHICLE", "TECH", "PERSON",
                  "MOTOR", "WEATHER", "SUBSTANCE"}  # v122: drop GAME_PLAY/HEALTH (neutral noise)
_ABSTRACT_CATS = {"MENTAL", "EMOTION_POS", "EMOTION_NEG", "TIME", "QUANTITY", "CHANGE",
                  "SOCIAL", "RELIGION",
                  "ABSTRACT_REL", "MONEY_NUM",
                  "LIFE_DEATH", "POSSESSION",
                  "SCHOOL", "COMMUNICATION"}  # v118: also add communication

# Function-word classes.
_PRONOUN = set((
    "i you he she it we they me him her us them my your his its our their this that "
    "these those who what which whom whose myself himself herself yourself themselves "
    "ourselves something anything everything nothing someone anyone everyone "
    "somebody anybody everybody somewhere anywhere everywhere other another each "
    "ya yall mine yours hers ours theirs itself one some any none either neither"
).split())
_PREP = set("in on at to from of for with by about into over under after before between through during without within against among around above below behind across near beside inside outside off out upon underneath beneath beyond alongside opposite toward towards past throughout".split())
_SPATIAL_PREP = set("in on at into onto over under through between among around within "
                    "above below behind across near beside inside outside off out "
                    "upon underneath beneath beyond amid amidst alongside opposite "
                    "toward towards against past throughout".split())
_CONJ = set("and or but so because although though while if when as than nor yet whether unless once whereas wherever whenever since plus".split())
_ARTICLE = set("a an the".split())
_AUX = set((
    "is are was were be been being am do does did have has had will would can "
    "could should shall may might must "
    "gonna gotta wanna gimme lemme dunno hafta"
).split())
_NEG = set((
    "not no never none nothing nobody nowhere neither nor "
    "dont didnt doesnt cant cannot wont wouldnt couldnt shouldnt isnt arent "
    "wasnt werent havent hasnt hadnt aint mustnt mightnt neednt "
    "without lack lacks lacking lacked"
).split())
_SELF_REF = set("i me my mine myself we us our ours ourselves im id ive weve wed".split())
_OTHER_REF = set("he him his she her hers they them their theirs hes shes theyre theyve theyd theyll".split())
_PRON_CONTRACT = set((
    "im id ive youre youve youd youll hes shes its theyre theyve theyd theyll "
    "weve wed well theres thats whats wheres heres hows whos hed shed"
).split())
_INTERJ = set((
    "oh uh um yeah ok okay yes yep nope hmm huh ah eh wow hey alright "
    "hello hi bye well gosh wel umm uhh mmm mhm "
    "ouch oops oof yikes shh psst whoa wait alas geez damn gee"
).split())
_WH = set("where when why how what which who whom whose whatever whenever wherever however whichever whoever".split())
_DISC = set((
    "maybe sure actually exactly pretty kinda sorta really probably definitely "
    "basically literally honestly obviously apparently certainly perhaps possibly "
    "anyway somehow though although besides instead therefore suddenly "
    "essentially specifically usually generally frankly seriously truly indeed"
).split())

# Word frequency (rough rank list of most-common English words). Words not
# in the list are RARE -- frequency is a strong driver of language responses.
_FREQ_LIST = (
    "the be to of and a in that have i it for not on with he as you do at this but his "
    "by from they we say her she or an will my one all would there their what so up out "
    "if about who get which go me when make can like time no just him know take people "
    "into year your good some could them see other than then now look only come its over "
    "think also back after use two how our work first well way even new want because any "
    "these give day most us man find here thing tell very still should through where much "
    "before too same right around another himself old little place such again off went "
    "while away something both house world own being head down many never under last "
    "those great life always those once side might room"
).split()
_WORD2FREQRANK = {w: i for i, w in enumerate(_FREQ_LIST)}


def freq_bucket(w: str) -> str:
    r = _WORD2FREQRANK.get(w)
    if r is None:
        return "FREQ_RARE"
    # v73: collapse FREQ_HIGH into FREQ_MID too — keep only RARE vs not-rare.
    return "FREQ_MID"


def heuristic_pos(w: str) -> str:
    if len(w) <= 2:
        return "SHORT"
    if w.endswith("ing"):
        return "VBG"
    if w.endswith("tion") or w.endswith("sion"):
        return "N_TION"
    if w.endswith("ness"):
        return "N_NESS"
    if w.endswith("ment"):
        return "N_MENT"
    if w.endswith("ity"):
        return "N_ITY"
    if w.endswith("ly"):
        return "ADV_LY"
    if w.endswith("ful"):
        return "ADJ_FUL"
    if w.endswith("ous"):
        return "ADJ_OUS"
    if w.endswith("ive"):
        return "ADJ_IVE"
    if w.endswith("est"):
        return "SUPER_EST"
    if w.endswith("er"):
        return "COMPAR_ER"
    if w.endswith("ed"):
        return "VBD"
    if w.endswith("s"):
        return "PLURAL_S"
    return "OTHER"


def len_bucket(w: str) -> str:
    n = len(w)
    if n <= 2:
        return "L1_2"
    if n <= 4:
        return "L3_4"
    if n <= 6:
        return "L5_6"
    if n <= 8:
        return "L7_8"
    return "L9p"


def morph_prefix(w: str):
    for p in ("un", "re", "dis", "in", "over", "mis", "pre"):
        if w.startswith(p) and len(w) > len(p) + 2:
            return p.upper()
    return None


def func_type(w: str):
    if w in _PRONOUN or w in _PRON_CONTRACT:
        return "PRON"
    if w in _PREP:
        return "PREP"
    if w in _CONJ:
        return "CONJ"
    if w in _ARTICLE:
        return "ART"
    if w in _AUX:
        return "AUX"
    if w in _NEG:
        return "NEG"
    if w in _WH:
        return "WH"
    if w in _DISC:
        return "DISC"
    if w in _INTERJ:
        return "INTERJ"
    return None


# Build word -> categories / modalities lookup.
_CAT_NAMES = list(_SEM_CATEGORIES.keys())
_WORD2CATS: Dict[str, set] = {}
for _ci, _cn in enumerate(_CAT_NAMES):
    for _w in _SEM_CATEGORIES[_cn]:
        _WORD2CATS.setdefault(_w, set()).add(_ci)

_MOD_NAMES = list(_MODALITY.keys())
_WORD2MOD: Dict[str, set] = {}
for _mi, _mn in enumerate(_MOD_NAMES):
    for _w in _MODALITY[_mn]:
        _WORD2MOD.setdefault(_w, set()).add(_mi)


def word_features(w: str) -> List[str]:
    """Return the list of feature-name strings active for a single word."""
    raw = w
    w = w.replace("'", "")  # contracted forms (dont, im, its) are apostrophe-free
    feats: List[str] = [freq_bucket(w)]
    ft = func_type(w)
    if ft:
        feats.append("FUNC_" + ft)  # restored from v16 collapse
        if w in _SPATIAL_PREP:
            feats.append("SPATIAL_PREP")
    else:
        feats.append("CONTENT")
        if raw.endswith("'s") and len(w) > 1 and w not in _WORD2CATS:
            w = w[:-1]  # possessive -> base noun for lookups (mother's -> mother)
    cats = sorted(_WORD2CATS.get(w, []))
    for c in cats:
        feats.append("SEM_" + _CAT_NAMES[c])
    catset = {_CAT_NAMES[c] for c in cats}
    mods = sorted(_WORD2MOD.get(w, []))
    for m in mods:
        feats.append("MOD_" + _MOD_NAMES[m])
    for cat, mod in _CAT2MOD.items():
        if cat in catset and "MOD_" + mod not in feats:
            feats.append("MOD_" + mod)
    if catset & _CONCRETE_CATS:
        feats.append("CONC_HIGH")
    if catset & _ABSTRACT_CATS:
        feats.append("CONC_LOW")
    # v90: ANIMATE dropped — words overlap heavily with KINSHIP/PEOPLE_ROLE/
    # SEM_PERSON_GENERIC/_ANIMAL cats.
    if w in _SELF_REF:
        feats.append("SELF_REF")
    if w in _OTHER_REF:
        feats.append("OTHER_REF")
    # v82: AROUSAL_HIGH dropped — overlaps heavily with VAL_NEG/VAL_NEG_INT.
    if w in _VAL_POS:
        feats.append("VAL_POS")
    if w in _VAL_NEG:
        feats.append("VAL_NEG")
    # v83: VAL_POS_INT/VAL_NEG_INT dropped — they're subsets of VAL_POS/NEG
    # so they primarily add multicollinearity without orthogonal signal.
    return feats


# Master feature vocabulary (all feature names that word_features can emit).
def _build_feature_names() -> List[str]:
    names: List[str] = []
    for t in ["PRON", "PREP", "CONJ", "ART", "AUX", "NEG", "WH", "DISC", "INTERJ"]:
        names.append("FUNC_" + t)
    names.append("CONTENT")
    for c in _CAT_NAMES:
        names.append("SEM_" + c)
    for m in _MOD_NAMES:
        names.append("MOD_" + m)
    names += ["CONC_HIGH", "CONC_LOW",
              "FREQ_MID", "FREQ_RARE",
              "VAL_POS", "VAL_NEG",
              "NEG_SCOPE", "SPATIAL_PREP",
              "SELF_REF", "OTHER_REF"]
    return names


FEATURE_NAMES = _build_feature_names()
NFEAT_BASE = len(FEATURE_NAMES)

# Lightweight bigram collocation patterns. Some 2-word constructions carry
# distinct meaning beyond their parts (e.g., "going to" = future tense,
# "i think" = epistemic stance). Each match in the recent window emits a
# dedicated bigram-pattern feature, appended to the feature vocab.
BIGRAM_PATTERNS: List[Tuple[str, str, str]] = []  # disabled (hurt in v7)
_BIGRAM_NAMES = sorted({p[2] for p in BIGRAM_PATTERNS})
FEATURE_NAMES = FEATURE_NAMES + _BIGRAM_NAMES
NFEAT = len(FEATURE_NAMES)
_FEAT2IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

# v56: hand-coded MLP composition features. Each tuple (feat_a, feat_b) is
# detected as a soft-AND co-occurrence in the uniform-mean pooled context and
# written to a new unused residual slot, so the ridge can read off feature
# interactions the linear feature bag cannot capture.
MLP_COMPOSITIONS: List[Tuple[str, str]] = [
    ("NEG_SCOPE", "VAL_POS"),               # negated positive
    ("NEG_SCOPE", "VAL_NEG"),               # negated negative
    ("SEM_BODY", "MOD_MOTOR"),              # embodied action
    ("SELF_REF", "SEM_EMOTION_NEG"),        # self-directed distress
    ("SELF_REF", "SEM_EMOTION_POS"),        # self-directed joy
    ("MOD_SOUND", "SEM_COMMUNICATION"),     # speech perception
    ("VAL_NEG_INT", "ANIMATE"),             # someone hurt/killed
    ("AROUSAL_HIGH", "MOD_MOTOR"),          # violent action
    ("SEM_KINSHIP", "SEM_EMOTION_POS"),     # warm family
    ("SEM_KINSHIP", "SEM_EMOTION_NEG"),     # family conflict
    ("MOD_VISION", "SEM_COLOR"),            # color perception
    ("SEM_PLACE", "SEM_SELF_MOTION"),       # going somewhere
    ("FUNC_NEG", "MOD_MOTOR"),              # not doing
    ("FUNC_AUX", "SEM_MENTAL"),             # modal thinking (could/should think)
    ("SEM_TIME", "SEM_CHANGE"),             # temporal change
]


def _bigram_match(w1: str, w2: str) -> List[str]:
    w1 = w1.replace("'", ""); w2 = w2.replace("'", "")
    out: List[str] = []
    for a, b, name in BIGRAM_PATTERNS:
        if (a == "*" or a == w1) and (b == "*" or b == w2):
            out.append(name)
    return out

FEAT_TOKEN_BASE = N_CHAR

# ---------------------------------------------------------------------------
# Per-word identity tokens for the most common CONTENT words.
# Each word in this list gets its own dedicated feature dim (a one-hot), so
# the encoder distinguishes e.g. "house" vs "tree" beyond just SEM_PLACE/SEM_NATURE.
# Critically these are NOT random hashes (which overfit in v2) -- they are
# stable one-hot rows that ridge can weight individually. We curate the
# vocabulary by taking the first ~5 most-canonical entries from each semantic
# category so the resulting list is compact (a few hundred dims), interpretable,
# and biased toward high-coverage common content words.
# ---------------------------------------------------------------------------
PER_CAT_TOP = 5  # take first N words of each SEM_CATEGORIES list

# Tiny hand-curated word-id vocab (v15): just the 30 most common content
# words across narrative stories. Far fewer dims than v4's 214 so much
# less overfit risk while still giving ridge per-word identity.
_TINY_WORD_ID = (
    "time day night people way thing house door hand eye head face man woman "
    "boy girl kid friend car water god mother father room car place"
).split()

_MID_WORD_ID = (
    # 100 most common content words in narrative storytelling, hand-picked
    # to span persons, body parts, places, objects, mental verbs, and time.
    "time day night year week morning evening way thing world life house door "
    "room car water god mother father friend kid boy girl man woman people "
    "head hand eye face foot heart mind voice body skin hair "
    "love hate fear hope dream pain joy hurt smile laugh cry kiss "
    "see hear feel know think want need say speak ask answer remember "
    "give take walk run sit stand wait sleep eat drink work play hold "
    "place street city town home school church bed kitchen table chair "
    "money food light dark dog cat bird tree water"
).split()


def _content_word_id_vocab() -> List[str]:
    seen = set()
    out: List[str] = []
    for cat in _SEM_CATEGORIES.values():
        added_from_cat = 0
        for w in cat:
            if added_from_cat >= PER_CAT_TOP:
                break
            if w in _PRONOUN or w in _PREP or w in _CONJ or w in _ARTICLE \
                    or w in _AUX or w in _NEG or w in _WH or w in _DISC or w in _INTERJ:
                continue
            if w in seen:
                continue
            seen.add(w); out.append(w); added_from_cat += 1
    # Add the first ~4 words of each perceptual modality.
    for cat in _MODALITY.values():
        added = 0
        for w in cat:
            if added >= 4:
                break
            if w not in seen:
                seen.add(w); out.append(w); added += 1
    return out


WORD_ID_VOCAB = list(dict.fromkeys(_MID_WORD_ID))  # v21: ~90-word mid vocab
WORD_ID_TOKEN_BASE = FEAT_TOKEN_BASE + NFEAT
_WORD2IDTOK = {w: WORD_ID_TOKEN_BASE + i for i, w in enumerate(WORD_ID_VOCAB)}
N_WORD_ID = len(WORD_ID_VOCAB)

# Optional hashed-word identity dimension (overfits heavily; off by default).
USE_WORD_ID = False
WORD_ID_STD = 0.25
WORD_HASH_SIZE = 16384
WORD_HASH_BASE = WORD_ID_TOKEN_BASE + N_WORD_ID
VOCAB_SIZE = WORD_HASH_BASE + WORD_HASH_SIZE

# Enable content-word identity emission (off again - v15 overfit).
USE_CONTENT_WORD_ID = False


def _word_hash(word: str) -> int:
    h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
    return WORD_HASH_BASE + (h % WORD_HASH_SIZE)


# ---------------------------------------------------------------------------
# Architecture (LN -> identity; the forward pass IS the design.)
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

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        scores = scores + attn_bias
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
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_bias)
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """Causal token-level transformer. `forward` returns hidden states (no LM head);
    the embedder reads the final-token hidden state as the encoding feature."""

    def __init__(self, vocab_size: int, max_seq_len: int = 512, d_model: int = 1024,
                 n_heads: int = 4, n_layers: int = 1, d_ff: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.Identity()

    def forward(self, ids: torch.Tensor, pos_ids: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(pos_ids)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        bias = torch.zeros(B, 1, T, T, device=ids.device)
        bias = bias.masked_fill(causal[None, None], float("-inf"))
        bias = bias.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        for block in self.blocks:
            h = block(h, bias)
        return self.final_ln(h)


class InterpretableEmbedder:
    """Tokenizes each n-gram string into feature tokens, runs SimpleTransformer,
    and returns the final-token hidden state."""

    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.max_seq_len = model.max_seq_len

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        text = text.lower()
        words = text.split()
        ids: List[int] = []
        pos: List[int] = []
        if USE_CHAR_CONTENT:
            for i, c in enumerate(text):
                ids.append(_stoi.get(c, UNK_ID))
                pos.append(i)
        # locate each word's end position on the char timeline
        spans: List[int] = []
        idx = 0
        for w in words:
            s = text.find(w, idx)
            if s < 0:
                s = idx
            e = s + len(w)
            spans.append(e - 1 if e > s else s)
            idx = e
        recent = list(zip(words, spans))[-N_APPEND_WORDS:]
        nrec = len(recent)
        recent_words = [w for w, _ in recent]
        for k, (w, endpos) in enumerate(recent):
            dist = nrec - 1 - k  # 0 == last word
            reps = RECENCY_REPS[min(dist, len(RECENCY_REPS) - 1)]
            extra: List[str] = []
            negated = any(pw.replace("'", "") in _NEG for pw in recent_words[max(0, k - 2):k])
            wf = word_features(w)
            if negated:
                extra.append("NEG_SCOPE")
                _flip = {"VAL_POS": "VAL_NEG", "VAL_NEG": "VAL_POS",
                         "SEM_EMOTION_POS": "SEM_EMOTION_NEG",
                         "SEM_EMOTION_NEG": "SEM_EMOTION_POS"}
                wf = [_flip.get(f, f) for f in wf]
            # Bigram patterns: check pair (prev_word, this_word).
            if k > 0:
                for bg in _bigram_match(recent_words[k - 1], w):
                    extra.append(bg)
            # Content words get a bonus rep count.
            if "CONTENT" in wf:
                reps = reps + CONTENT_BONUS
            feat_ids = [FEAT_TOKEN_BASE + _FEAT2IDX[f] for f in (wf + extra)]
            if USE_CONTENT_WORD_ID and "CONTENT" in wf:
                lookup = w.replace("'", "")
                if lookup in _WORD2IDTOK:
                    feat_ids.append(_WORD2IDTOK[lookup])
                elif lookup.endswith("s") and lookup[:-1] in _WORD2IDTOK:
                    feat_ids.append(_WORD2IDTOK[lookup[:-1]])
            if USE_WORD_ID:
                feat_ids.append(_word_hash(w))
            for _ in range(reps):
                for fid in feat_ids:
                    ids.append(fid)
                    pos.append(endpos)
        if not ids:
            return [PAD_ID], [0]
        if len(ids) > self.max_seq_len:
            ids = ids[-self.max_seq_len:]
            pos = pos[-self.max_seq_len:]
        pos = [min(pp, self.max_seq_len - 1) for pp in pos]
        return ids, pos

    @torch.no_grad()
    def __call__(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            enc = [self.encode(t) for t in texts[i: i + batch_size]]
            lens = [len(e[0]) for e in enc]
            T = max(lens)
            ids = torch.full((len(enc), T), PAD_ID, dtype=torch.long)
            pos_ids = torch.zeros((len(enc), T), dtype=torch.long)
            pad_mask = torch.zeros((len(enc), T), dtype=torch.bool)
            for j, (e, pp) in enumerate(enc):
                ids[j, :len(e)] = torch.tensor(e, dtype=torch.long)
                pos_ids[j, :len(pp)] = torch.tensor(pp, dtype=torch.long)
                pad_mask[j, :len(e)] = True
            ids = ids.to(self.device)
            pos_ids = pos_ids.to(self.device)
            pad_mask = pad_mask.to(self.device)
            hidden = self.model(ids, pos_ids, pad_mask)
            last = torch.tensor([l - 1 for l in lens], device=self.device)
            emb = hidden[torch.arange(len(enc), device=self.device), last]
            embs.append(emb.float().cpu().numpy())
        return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# Hand-written weights (NO TRAINING)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer) -> None:
    D = model.d_model
    H = model.n_heads
    dh = D // H
    assert H == len(LAMBDAS), "n_heads must match number of decay lambdas"
    n_dense_feats = NFEAT + (N_WORD_ID if USE_CONTENT_WORD_ID else 0)
    assert CAT_OFFSET + n_dense_feats <= dh, \
        f"feats+word_id ({n_dense_feats}) must fit in head slice ({dh})"

    with torch.no_grad():
        model.token_emb.weight.zero_()
        # Each feature token's embedding is a one-hot at its feature dim,
        # REPLICATED across head slices.
        for f in range(NFEAT):
            tok = FEAT_TOKEN_BASE + f
            for hh in range(H):
                model.token_emb.weight[tok, hh * dh + CAT_OFFSET + f] = 1.0

        if USE_CONTENT_WORD_ID:
            # word-id one-hots placed immediately after the feature dims.
            wid_offset = CAT_OFFSET + NFEAT
            for i in range(N_WORD_ID):
                tok = WORD_ID_TOKEN_BASE + i
                for hh in range(H):
                    model.token_emb.weight[tok, hh * dh + wid_offset + i] = 1.0

        if USE_CHAR_CONTENT:
            g = torch.Generator().manual_seed(0)
            content_lo = CAT_OFFSET + NFEAT + (N_WORD_ID if USE_CONTENT_WORD_ID else 0)
            for hh in range(H):
                lo = hh * dh + content_lo
                hi = (hh + 1) * dh
                w = torch.empty(N_CHAR, hi - lo)
                w.normal_(mean=0.0, std=CHAR_CONTENT_STD / math.sqrt(hi - lo), generator=g)
                model.token_emb.weight[:N_CHAR, lo:hi] = w
            model.token_emb.weight[PAD_ID].zero_()

        if USE_WORD_ID:
            g2 = torch.Generator().manual_seed(123)
            content_lo = CAT_OFFSET + NFEAT + (N_WORD_ID if USE_CONTENT_WORD_ID else 0)
            for hh in range(H):
                lo = hh * dh + content_lo
                hi = (hh + 1) * dh
                w = torch.empty(WORD_HASH_SIZE, hi - lo)
                w.normal_(mean=0.0, std=WORD_ID_STD / math.sqrt(hi - lo), generator=g2)
                model.token_emb.weight[WORD_HASH_BASE:WORD_HASH_BASE + WORD_HASH_SIZE,
                                       lo:hi] = w

        # pos_emb: place index j into POS_DIM and constant 1 into BIAS_DIM.
        model.pos_emb.weight.zero_()
        js = torch.arange(model.max_seq_len, dtype=torch.float32)
        model.pos_emb.weight[:, POS_DIM] = js
        model.pos_emb.weight[:, BIAS_DIM] = 1.0

        # Attention block: per-head, q reads BIAS_DIM and k reads POS_DIM,
        # so q.k = lambda_h * j (independent of token content). Softmax then
        # produces multi-scale recency weights (smaller lambda -> flatter).
        blk = model.blocks[0]
        attn = blk.attn
        attn.W_q.weight.zero_()
        attn.W_k.weight.zero_()
        for hh, lam in enumerate(LAMBDAS):
            base = hh * dh
            attn.W_q.weight[base + 0, BIAS_DIM] = 1.0
            attn.W_k.weight[base + 0, POS_DIM] = lam * math.sqrt(dh)
        eye = torch.eye(D)
        eye[POS_DIM, POS_DIM] = 0.0
        eye[BIAS_DIM, BIAS_DIM] = 0.0
        attn.W_v.weight.copy_(eye)
        attn.W_o.weight.copy_(torch.eye(D))

        # v58: drop MLP comps (neutral or slightly negative in v56/v57). MLP
        # zeroed again to act as identity-add.
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()
        # v21: revert v20 LN, keep final_ln as Identity (the original).
        model.final_ln = nn.Identity()


# ---------------------------------------------------------------------------
# Identity + description
# ---------------------------------------------------------------------------

model_shorthand_name = "FeatBag_v237_ExpandEmoPosV3"
model_description = (
    "From v232, expand SEM_EMOTION_POS with elated/jubilant/content/satisfied."
)


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def build_embedder(device: str = 'cuda', d_model: int = 2048, n_heads: int = None,
                   n_layers: int = 1, d_ff: int = 1024, max_seq_len: int = 512) -> InterpretableEmbedder:
    model = SimpleTransformer(
        vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len,
        d_model=d_model, n_heads=len(LAMBDAS), n_layers=n_layers, d_ff=d_ff)
    write_weights(model)
    model.eval()
    return InterpretableEmbedder(model, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    t0 = time.time()
    cfg = EncodingConfig(subject=args.subject, num_train=args.num_train, num_test=args.num_test)
    embedder = build_embedder(device=args.device)
    r = run_encoding(embedder, cfg)
    n_params = sum(p.numel() for p in embedder.model.parameters())

    upsert_overall_results(
        [make_result_row(r, model_shorthand_name, n_params, model_description)], RESULTS_DIR)
    plot_corr_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"subject:        {cfg.subject}")
    print(f"test_corr:      {r['test_corr']:.4f}  (train_corr={r['corrs_train_mean']:.4f}, "
          f"median={r['corrs_test_median']:.4f}, frac>0.2={r['corrs_test_frac>0.2']:.4f}, "
          f"top5%={r['corrs_test_mean_top5_percentile']:.4f})")
    print(f"roi corrs:      " + ", ".join(f"{k}={v:.3f}" for k, v in r['roi_corrs'].items()))
    print(f"encoding_secs:  {r['encoding_seconds']:.1f}s")
    print(f"total_seconds:  {time.time() - t0:.1f}s")
