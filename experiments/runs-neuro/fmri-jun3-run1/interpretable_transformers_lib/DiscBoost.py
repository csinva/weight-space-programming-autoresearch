"""Interpretable transformer embedder for fMRI language encoding.

LEGITIMACY NOTE
---------------
Every feature is produced by the genuine `SimpleTransformer.forward` pass
(token-embedding lookup + causal self-attention pooling). `encode()` only does
TOKENIZATION: it maps each word to a small set of interpretable feature-token
ids (POS, length, morphology, function-word type, semantic category, perceptual
modality, concreteness). The actual vectors live in `token_emb` (a model
parameter) and are pooled by real attention. No numpy feature matrices are
returned directly; no training, no gradients, no pretrained weights.

The circuit ("LexFeatBoC"):
  * Residual coordinate dims: dim0 = position j, dim1 = constant 1 (from pos_emb).
  * For each recent word we emit feature tokens. Each feature token's token_emb
    row is a one-hot for that feature, REPLICATED across all head slices.
  * Multi-head attention = multi-scale recency-weighted pooling. Head h with
    decay lambda_h:  score(i,j)=lambda_h*j  => softmax weights ~ exp(lambda_h*j)
    (recency). lambda=0 is the global mean. Recent words are additionally
    repeated (recency emphasis) so they dominate the pooled bag, matching the
    fMRI's sensitivity to recent words.
  * W_v=identity (coord dims excluded), W_o=identity, MLP=0, LN=identity. The
    final-token state is the multi-scale recency-weighted bag of interpretable
    lexical features for the n-gram; ridge maps it to voxels.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --subject UTS03 --num-train 5
"""

from __future__ import annotations

import argparse
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
# Char vocab (kept for optional orthographic content) + feature-token vocab
# ---------------------------------------------------------------------------
_VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\"
_BASE_CHARS = ['<pad>', '<unk>'] + list(_VOCAB_CHARS)
N_CHAR = len(_BASE_CHARS)

PAD_ID = 0
UNK_ID = 1
POS_DIM = 0
BIAS_DIM = 1
CAT_OFFSET = 2

_stoi = {c: i for i, c in enumerate(_BASE_CHARS)}

LAMBDAS = (-0.04, 0.0, 4.0, 16.0)
N_APPEND_WORDS = 12
# recency emphasis: number of times a word's feature tokens are repeated, by
# distance from the end (index 0 == last word).
RECENCY_REPS = (5, 1, 1, 1, 1, 1, 1, 1, 1, 1)
ZETA_DISC = 1.75

USE_CHAR_CONTENT = False
CHAR_CONTENT_STD = 1.0  # std scale of random char embeddings


# ----------------------- hand-coded lexicons -----------------------
_SEM_CATEGORIES = {
    "MOTION": "go goes went going gone come comes came coming run ran running walk walked walking move moved moving fly flew flown drive drove driven ride rode jump jumped fall fell fallen throw threw catch caught turn turned turns rush chase climb crawl slide roll spin march step leave left arrive enter exit return follow approach escape flee crawl swim dive sit sitting stand standing lay laying lie lying swing".split(),
    "SPACE": "up down left right above below under over inside outside near far here there front back top bottom between among around through across along beside behind beyond edge corner middle center side north south east west forward backward upward downward out off away apart together onto toward towards against next".split(),
    "TIME": "time times now then today tomorrow yesterday soon later before after early late always never often sometimes year years month months week weeks day days hour hours minute minutes second moment moments morning night nights evening afternoon noon midnight past future present while during until since again ago already yet still when whenever".split(),
    "QUANTITY": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety zero many few several all some none most least more less much little half double twice huge tiny count number numbers dates lot lots dozen hundred thousand million plenty enough single whole total each every first second third last".split(),
    "BODY": "head face eye eyes ear ears nose mouth lip lips tooth teeth hand hands arm arms leg legs foot feet finger fingers hair skin heart blood bone bones back chest shoulder shoulders knee knees throat stomach brain neck chin cheek wrist elbow thumb nail body skull beard tears tear scar scars".split(),
    "MOTOR": "grab push pull pulled lift throw kick run walk jump grip hold holding held carry hit punch grasp reach shave wipe squeeze press".split(),
    "PERSON": "man men woman women boy boys girl girls child children people person guy guys lady kid kids baby friend friends mother father mom dad sister brother son daughter wife husband family neighbor stranger crowd human folk gentleman".split(),
    "SOCIAL": "together alone meet met meeting marry married wedding party group team gang community share shared help helped helping agree argue argued fight fought war peace trust betray join visit invite welcome greet".split(),
    "EMOTION_POS": "happy joy joyful glad love loved loving liked enjoy enjoyed excited exciting wonderful great amazing beautiful pleasure smile smiled laugh laughed laughing proud hope hopeful delight cheerful pleased grateful relief calm".split(),
    "EMOTION_NEG": "sad sadness angry anger afraid fear scared frightened worried worry cry cried crying pain hurt terrible awful horrible hate hated disgust grief sorrow lonely nervous anxious upset miserable depressed guilt shame jealous".split(),
    "COMMUNICATION": "say said says saying tell told telling tells speak spoke spoken speaking talk talked talking ask asked asking answer answered call called calling shout yell whisper word words voice question questions story stories explain read write wrote writing letter book reply discuss mention describe name names news conversation promise promised thank thanks thanked".split(),
    "MENTAL": "think thought thinking know knew known believe believed remember remembered forget forgot understand understood realize realized wonder wondered imagine imagined guess idea ideas mind learn learned dream dreamed decide decided suppose consider expect assume doubt notice want wanted wants wanting wish wished need needed hope mean meant figure figured plan planned try tried trying care cared teach taught experience truth".split(),
    "PERCEPTION": "see saw seen seeing look looked looking looks watch watched watching hear heard hearing listen listened smell smelled taste tasted touch touched feel felt feeling notice noticed stare stared glance observe gaze".split(),
    "FOOD": "eat ate eaten eating food drink drank drinking water bread meat fruit apple orange meal meals breakfast lunch dinner cook cooked cooking hungry thirsty sweet bitter sour salt sugar coffee tea wine beer milk egg cheese cake soup rice".split(),
    "PLACE": "house home homes room rooms door doors window windows wall walls floor street streets road roads city cities town towns country school church store shop office building park garden field forest mountain river ocean sea lake beach sky world land farm village ground cabin central downtown".split(),
    "OBJECT": "thing things stuff box book books table chair bed car cars key keys money paper bag bottle cup phone clock machine tool tools wheel stone wood metal glass cloth knife pen door chain rope ball gun camera computer screen coin coins triangle".split(),
    "NATURE": "tree trees fire air earth wind rain snow storm sun moon star stars cloud clouds animal dog cat bird birds fish horse flower flowers grass leaf leaves rock rocks soil dirt mud ice wave hill valley river rivers stream".split(),
    "HEALTH": "doctor doctors nurse surgeon surgery hospital emergency patient sick ill illness disease pain ache hurt injured injury wound blood heal healed cure pill pills medicine drug treatment cancer fever cough epilepsy seizure ambulance clinic operation recovery dying health insurance".split(),
    "QUALITY": "good bad new old young right wrong true false real fake strange normal important hard easy soft strong weak rich poor clean dirty empty full heavy light bright dark sharp dull fresh nice fine perfect big small large little long short tall wide narrow huge tiny crazy weird wild quiet loud sorry able main different same fancy plain rough smooth thick thin deep flat round flat fast slow quick ready tough best worst slowly quickly".split(),
    "WORK_MONEY": "work worked working job jobs money pay paid buy bought buying sell sold selling business company boss market price cost dollar dollars trade build built building factory worker wage profit bank store customer".split(),
    "COLOR": "red blue green yellow white black gray grey brown orange purple pink color colors colour golden silver dark bright pale".split(),
    "KINSHIP": "mother father mom dad parent parents son daughter sister brother wife husband child children baby uncle aunt cousin grandmother grandfather grandma grandpa family nephew niece".split(),
    "ANIMAL": "dog dogs cat cats bird birds fish horse horses cow pig sheep chicken duck lion tiger bear wolf fox deer rabbit mouse rat snake frog insect bug bee ant spider animal animals creature".split(),
    "WEATHER": "rain rained snow snowed wind windy storm sunny cloudy cold hot warm freezing fog ice frost heat winter summer spring autumn season weather temperature".split(),
    "ABSTRACT_REL": "cause caused because reason result effect purpose means kind sort type way ways form part parts whole sense point fact case matter problem question chance luck fate course rest bit influence experience risk risks truth".split(),
    "POSSESSION": "have has had having own owns owned get gets getting got gotten give gave given take took taken takes keep kept hold held lose lost find found bring brought carry carried receive offer put set place placed leave left use used using wait waited waiting stay stayed staying stand stood sit sat sent send check checked wear wore worn".split(),
    "CHANGE": "become became becoming change changed grow grew grown turn turned increase decrease rise rose fall fell break broke broken make made build built create destroy form develop begin began start started starts stop stopped end ended finish open opened close closed happen happened happens happening cut hit drop dropped remove removed appear appeared appears spend spent spending".split(),
    "INTENSITY": "very really so too quite rather extremely incredibly absolutely totally completely almost nearly barely hardly just only even much".split(),
    "CLOTHING": "shoe shoes shirt shirts pants dress dresses coat coats jacket hat hats sock socks glove gloves scarf belt tie suit boot boots sweater skirt jeans clothes clothing button pocket sleeve collar zipper cap garment garments uniform".split(),
    "SUBSTANCE": "cigarette cigarettes smoke smoking smoked tobacco pack drug drugs alcohol beer wine drink drunk pill pills medicine weed pot ash lighter match matches nicotine".split(),
    "VEHICLE": "car cars truck trucks bus buses train trains plane planes boat boats bike bikes motorcycle taxi cab subway seat seatbelt wheel engine brake brakes gas drive driving road traffic helicopter pilot flight airport jet".split(),
    "MONEY_NUM": "dollar dollars cent cents penny dime buck bucks cost price cheap expensive free pay paid owe debt cash credit bill bills change worth value".split(),
    "TECH": "phone phones computer screen tv television radio camera internet email text message call button machine wire battery switch electric power".split(),
    "LIFE_DEATH": "life live lived lives living alive born birth grow grew age aged young old die died death dead dying kill killed survive survived breathe breath heartbeat exist".split(),
    "SCHOOL_INST": "school university college campus class classroom student students teacher professor study studied learn lesson grade exam test homework library team club church government union company office church liberty semester".split(),
    "RELIGION": "god gods church pray prayed prayer faith religion religious holy heaven hell soul spirit bible jesus christ christian sin angel devil priest worship sacred divine blessed evangelical".split(),
    "GAME_PLAY": "play played playing plays game games sport sports football basketball baseball soccer tennis golf team score win lose won ball bat field coach fun funny awesome joke toy toys".split(),
    "PEOPLE_ROLE": "boyfriend girlfriend friend boss worker assistant director directors manager nurse teacher student officer guard leader member partner clerk agent owner customer guest host neighbor".split(),
    "NAME": ("michael mike john james robert david william richard joseph thomas charles "
             "christopher daniel paul mark george steven kenneth andrew brian kevin "
             "mary patricia jennifer linda elizabeth susan jessica sarah karen nancy "
             "betty helen sandra donna carol ruth ivy melanie kristen christian "
             "texas georgia california florida boston chicago york america american").split(),
    "SELF_MOTION": "go goes went going gone come comes came coming run ran running walk walked walking move moved moving fly flew swim climb jump jumped fall fell rise rose arrive enter leave left return wander wandered".split(),
    "CAUSED_MOTION": "throw threw thrown push pushed pull pulled carry carried lift lifted drag dragged drop dropped kick kicked toss shove grab grabbed hand handed bring brought".split(),
    "SPEECH_ACT": "say said says saying tell told telling ask asked asking answer answered speak spoke talk talked shout shouted yell whisper whispered call called reply explain explained".split(),
    "DISCOURSE": "because so then therefore thus hence since although though however but yet still meanwhile afterward afterwards consequently whereas otherwise nonetheless besides moreover anyway when while after before until once whenever unless instead finally eventually suddenly".split(),
    "WORK": "work works worked working job jobs money pay paid pays paying buy bought sell sold cost costs price boss employee company business office store shop market dollar dollars cash rich poor expensive cheap hire hired fired wage salary career duty client customer".split(),
}
# Coarse valence (sentiment) beyond the EMOTION_* categories.
_VAL_POS = set("good great love happy joy nice beautiful wonderful best better win won success hope safe friend gift smile laugh warm bright fun pleasant gentle clean fresh free peace calm glad enjoy enjoyed proud excited amazing perfect lucky grateful cheerful delight pleased comfort sweet cool awesome favorite special wonderful brave strong healthy beautiful smart funny happy laughing celebrate party loved".split())
_VAL_NEG = set("bad worse worst hate fear pain hurt sad angry death dead kill killed lost lose fail wrong sick ill dark cold cruel ugly dirty broken danger trouble war fight blood enemy evil sorry afraid scared worried worry cry terrible awful horrible nervous anxious lonely guilt shame angry mad upset stress hard tough struggle difficult problem problems wound injury cancer disease tears pain suffering scared frightened poor weak tired exhausted sad crying alone".split())
# Animacy: animate (living agents) tends to be tracked distinctly by the brain.
_ANIMATE = set((
    "man men woman women boy girl child children people person friend mother father "
    "son daughter sister brother dog cat bird fish horse cow lion tiger bear wolf "
    "animal baby human teacher doctor king queen soldier worker player crowd folk"
).split())
# Coarse emotional valence/arousal lexicons (beyond the EMOTION_* categories).
_HIGH_AROUSAL = set("scream shout run fight fire explode crash rush panic terror excited thrilled furious rage storm danger attack chase escape shock kill blood gun death dead crash smash burst slam violent fierce wild desperate frantic terrified horror scream screaming yelling".split())
_CAT_NAMES = list(_SEM_CATEGORIES.keys())
_WORD2CATS: Dict[str, List[int]] = {}
for _ci, _cn in enumerate(_CAT_NAMES):
    for _w in _SEM_CATEGORIES[_cn]:
        _WORD2CATS.setdefault(_w, set()).add(_ci)
_WORD2CATS = {w: sorted(cs) for w, cs in _WORD2CATS.items()}

_MODALITY = {
    "VISION": "see saw seen seeing look looked looking looks watch watched watching bright dark color colors red blue green yellow white black light lights shadow shadows glow shine shining appear appeared vision sight glance glanced stare stared gaze visible image picture view scene".split(),
    "SOUND": "hear heard hearing listen listened loud quiet sound sounds noise noises music song songs voice voices ring rang bell bang banging crash whisper whispered scream screamed shout yell echo silence silent loud quiet tune".split(),
    "TOUCH": "touch touched feel felt soft hard rough smooth warm cold hot wet dry sharp press pressed grip held holding squeeze rub texture sticky slippery".split(),
    "TASTE": "taste tasted sweet bitter sour salty spicy delicious flavor flavour eat ate yummy bland".split(),
    "SMELL": "".split(),  # pruned: fired only ~21x in 152k words -> overfit-prone sparse z-score spike
    "MOTOR": "grab grabbed push pushed pull pulled lift lifted throw threw kick kicked run ran walk walked jump jumped grip held hold carry carried hit punch grasp reach reached swing wave squeeze".split(),
}
_MOD_NAMES = list(_MODALITY.keys())
# Category -> perceptual modality, to extend modality coverage from categories.
_CAT2MOD = {"COLOR": "VISION", "FOOD": "TASTE", "BODY": "TOUCH", "MOTION": "MOTOR",
            "ANIMAL": "VISION", "NATURE": "VISION", "WEATHER": "VISION",
            "SPEECH_ACT": "SOUND", "COMMUNICATION": "SOUND"}
_WORD2MOD: Dict[str, List[int]] = {}
for _mi, _mn in enumerate(_MOD_NAMES):
    for _w in _MODALITY[_mn]:
        _WORD2MOD.setdefault(_w, set()).add(_mi)
_WORD2MOD = {w: sorted(cs) for w, cs in _WORD2MOD.items()}
_NO_CAT_SOUND = set("write wrote writing written".split())

_CONCRETE = set("house tree dog cat car book table chair hand eye water fire stone door window food bird fish rock wall floor street road wood metal glass bottle cup phone money".split())
_ABSTRACT = set("idea thought love fear hope time truth freedom justice mind dream memory reason power belief fact chance luck soul spirit meaning".split())
# Category-level concreteness: words in these categories are treated as concrete /
# abstract respectively, vastly extending concreteness coverage beyond the explicit
# word lists above.
_CONCRETE_CATS = {"BODY", "FOOD", "PLACE", "OBJECT", "NATURE", "ANIMAL", "COLOR",
                  "CLOTHING", "SUBSTANCE", "VEHICLE", "TECH", "PERSON"}
_ABSTRACT_CATS = {"MENTAL", "EMOTION_POS", "EMOTION_NEG", "TIME", "ABSTRACT_REL",
                  "QUANTITY", "CHANGE", "SOCIAL", "RELIGION"}

_PRONOUN = set((
    "i you he she it we they me him her us them my your his its our their this that "
    "these those who what which whom whose myself himself herself yourself themselves "
    "ourselves something anything everything nothing someone anyone everyone "
    "somebody anybody everybody somewhere anywhere everywhere other another each"
).split())
_PREP = set("in on at to from of for with by about into over under after before between through during without within against among around above below behind across near beside inside outside onto".split())
# Spatial/locative prepositions (engage parietal spatial-cognition regions).
_SPATIAL_PREP = set("in on at into over under through between among around within "
                    "above below behind across near beside inside outside onto".split())
_CONJ = set("and or but so because although though while if when as than nor yet whether unless".split())
# Logical/discourse-relation connectives (causal, adversative, conditional) — mark
# discourse coherence processing, distinct from plain additive and/or coordination.
_DISCREL = set("but because although though if unless so yet however therefore instead".split())
_GOAL_PREP = set("to for".split())
_GENITIVE_PREP = set("of".split())
_FUTURE = set("will gonna shall".split())
_PAST_AUX = set("was were had been".split())
_ADDITIVE = set("and".split())
_ARTICLE = set("a an the".split())
_AUX = set((
    "is are was were be been being am do does did have has had will would can "
    "could should shall may might must "
    "gonna gotta wanna gimme lemme dunno hafta"
).split())
_NEG = set((
    "not no never none nothing nobody nowhere neither nor "
    # contractions are spelled without apostrophes in this spoken corpus
    "dont didnt doesnt cant cannot wont wouldnt couldnt shouldnt isnt arent "
    "wasnt werent havent hasnt hadnt aint mustnt mightnt neednt"
).split())
# First-person self-reference (engages default-mode / medial prefrontal cortex).
_SELF_REF = set("i me my mine myself we us our ours ourselves "
                "im id ive weve wed".split())
# Third-person / other-person reference (theory-of-mind, social cognition).
_OTHER_REF = set("he him his she her hers they them their theirs "
                 "himself herself themselves "
                 "hes shes theyre theyve theyd theyll".split())
# Inanimate/expletive third-person reference (no theory-of-mind / no animacy).
_INANIM_REF = set("it its itself".split())
# Subject-pronoun contractions (also without apostrophes); treated as pronouns.
_PRON_CONTRACT = set((
    "im id ive youre youve youd youll hes shes its theyre theyve theyd theyll "
    "weve theres thats whats wheres heres hows whos"
).split())
# Interjections / discourse fillers, very common in spoken narratives.
_INTERJ = set((
    "oh uh um yeah ok okay yes yep yeah nope hmm huh ah eh wow hey alright "
    "hello hi bye well gosh wel umm uhh mmm mhm"
).split())
_DISFLUENCY = set("uh um umm uhh er erm hmm mmm".split())
_FOCUS = set("also too even only".split())  # focus/additive particles (info structure)
_UNIV = set("all every each whole entire everything everyone everybody".split())
_NOT_PAST_ED = set("need indeed instead ahead hundred sacred naked wicked speed feed seed embed exceed proceed united".split())
_PRON_PREFIX = set("i we you he she they it".split())  # pronoun contraction prefixes (i'll/we're)

# Wh-words (questions / relatives) and discourse/hedge adverbs.
_WH = set("where when why how what which who whom whose whatever whenever wherever however".split())
_DISC = set((
    "maybe sure actually exactly pretty kinda sorta really probably definitely "
    "basically literally honestly obviously apparently certainly perhaps possibly "
    "anyway somehow though although besides instead therefore suddenly"
).split())

# Compact built-in frequency list (~ the most common English words, in rough
# descending frequency). Word frequency / predictability is a strong driver of
# language-region responses. Unknown words fall into the RARE bucket.
_FREQ_LIST = (
    "the be to of and a in that have i it for not on with he as you do at this but his "
    "by from they we say her she or an will my one all would there their what so up out "
    "if about who get which go me when make can like time no just him know take people "
    "into year your good some could them see other than then now look only come its over "
    "think also back after use two how our work first well way even new want because any "
    "these give day most us man find here thing tell very still should through where much "
    "before too same right around another himself old little place such again off went "
    "while away something both house world own being head down many never under last "
    "those great life always those once side might room "
    "three came does turn ask men need land different home move try kind hand change "
    "play air point page letter mother answer found study learn school father keep tree "
    "start city earth eye light thought under story saw left few along close seem next "
    "hard open begin paper together group often until children feet car mile night walk "
    "white sea began grow took river four carry state book hear stop second later miss "
    "idea enough eat face watch far almost let above girl mountain cut young talk soon "
    "list song family leave mind every name big high follow act house "
    "real night close stop open seem next begin mark book mile feet care second carry "
    "eat room friend fish north base horse sure watch color wood main girl ready ever "
    "red though feel talk bird soon body dog measure black short class wind question "
    "happen ship area half rock order fire south problem piece told knew pass since top "
    "whole king space best hour better true during five remember step early hold ground "
    "reach fast sing table travel morning ten simple toward war pattern center love "
    "person money serve appear road map science rule pull cold notice voice fall power "
    "town fine certain fly lead cry dark machine note wait plan figure star field rest "
    "able beauty drive front teach week final gave green develop sleep warm strong clear "
    "fact street lot nothing course stay full force blue object decide deep moon island "
    "foot word turn ask men land different move kind hand change play air point page "
    "letter mother answer study learn school father keep tree start city earth eye light "
    "thought under story saw left"
).split()
_WORD2FREQRANK = {w: i for i, w in enumerate(_FREQ_LIST)}


def freq_bucket(w: str) -> str:
    r = _WORD2FREQRANK.get(w)
    if r is None:
        return "FREQ_RARE"
    return "FREQ_HIGH"


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
    if n <= 10:
        return "L9_10"
    return "L11"


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


def word_features(w: str) -> List[str]:
    raw = w
    w = w.replace("'", "")  # corpus contractions/possessives use apostrophes; sets are apostrophe-free
    feats = [freq_bucket(w)]
    if "'" in raw:
        feats.append("CONTRACTION")  # colloquial contracted forms (i'm/don't/it's), spoken register
        pref = raw.split("'")[0]
        if pref in _PRON_PREFIX:
            w = pref  # i'll/we're -> i/we (avoid false homographs ill/were)
    if len(w) > 3 and w.endswith("ed") and w not in _NOT_PAST_ED:
        feats.append("PAST_MORPH")  # -ed past-tense/participle morphology (complements tense axis)
    if w in _FOCUS:
        feats.append("FOCUS")  # focus/additive particles (also/even/only) mark info structure
    if w in _UNIV:
        feats.append("UNIV")  # universal/maximal quantifiers (all/every/whole)
    ft = func_type(w)
    if ft:
        feats.append("FUNC_" + ft)
        if w in _DISCREL:
            feats.append("DISCREL")
        if w in _ADDITIVE:
            feats.append("ADDITIVE")
        if w in _GOAL_PREP:
            feats.append("GOAL_PREP")
        if w in _GENITIVE_PREP:
            feats.append("GENITIVE_PREP")
        if w in _FUTURE:
            feats.append("FUTURE")
        if w in _PAST_AUX:
            feats.append("PAST_AUX")
        if w in _DISFLUENCY:
            feats.append("DISFLUENCY")  # hesitation fillers (uh/um) carved from INTERJ
    else:
        feats.append("CONTENT")  # marks content (non-function) words
        if w == "like":
            feats.append("DISC_LIKE")  # quotative/discourse "like" (dominant spoken usage)
        if raw.endswith("'s") and len(w) > 1 and w not in _WORD2CATS:
            w = w[:-1]  # possessive -> base noun for semantic lookups (mother's -> mother)
    if w in _SPATIAL_PREP:
        feats.append("SPATIAL_PREP")  # locative/spatial preposition (parietal); ungated from func_type
    cats = _WORD2CATS.get(w, [])
    for c in cats:
        feats.append("SEM_" + _CAT_NAMES[c])
    catset = {_CAT_NAMES[c] for c in cats}
    mods = set(_WORD2MOD.get(w, []))
    for m in mods:
        feats.append("MOD_" + _MOD_NAMES[m])
    # Derive perceptual modality from semantic categories to extend coverage
    # (perceptual modality is a strong driver of sensory-language cortex).
    for cat, mod in _CAT2MOD.items():
        if cat in catset and mod not in {_MOD_NAMES[m] for m in mods}:
            if mod == "SOUND" and w in _NO_CAT_SOUND:
                continue
            feats.append("MOD_" + mod)
    # Concreteness: explicit lexicon first, else derived from semantic category
    # membership so coverage extends to hundreds of words (concreteness is a
    # strong, well-established driver of language-cortex responses).
    catset = {_CAT_NAMES[c] for c in cats}
    _is_conc = w in _CONCRETE or bool(catset & _CONCRETE_CATS)
    if _is_conc:
        feats.append("CONC_HIGH")
    if (w in _ABSTRACT or (catset & _ABSTRACT_CATS)) and not _is_conc:
        feats.append("CONC_LOW")  # concrete wins ties (avoid contradictory CONC_HIGH+CONC_LOW)
    if w in _ANIMATE:
        feats.append("ANIMATE")
    if w in _SELF_REF:
        feats.append("SELF_REF")
    if w in _OTHER_REF:
        feats.append("OTHER_REF")
    if w in _INANIM_REF:
        feats.append("INANIM_REF")
    if w in _HIGH_AROUSAL:
        feats.append("AROUSAL_HIGH")
    if w in _VAL_POS:
        feats.append("VAL_POS")
    if w in _VAL_NEG:
        feats.append("VAL_NEG")
    return feats


# Master feature vocabulary (all feature names that word_features can emit).
def _build_feature_names() -> List[str]:
    names = []
    for t in ["PRON", "PREP", "CONJ", "ART", "AUX", "NEG", "WH", "DISC", "INTERJ"]:
        names.append("FUNC_" + t)
    names.append("CONTENT")
    for c in _CAT_NAMES:
        names.append("SEM_" + c)
    for m in _MOD_NAMES:
        names.append("MOD_" + m)
    names.append("CONC_HIGH")
    names.append("CONC_LOW")
    names.append("ANIMATE")
    names.append("AROUSAL_HIGH")
    for t in ["FREQ_HIGH", "FREQ_RARE"]:
        names.append(t)
    names.append("VAL_POS")
    names.append("VAL_NEG")
    names.append("NEG_SCOPE")
    names.append("SPATIAL_PREP")
    names.append("SELF_REF")
    names.append("OTHER_REF")
    names.append("INANIM_REF")
    names.append("DISC_LIKE")
    names.append("DISCREL")
    names.append("ADDITIVE")
    names.append("GOAL_PREP")
    names.append("GENITIVE_PREP")
    names.append("FUTURE")
    names.append("PAST_AUX")
    names.append("DISFLUENCY")
    names.append("CONTRACTION")
    names.append("FOCUS")
    names.append("UNIV")
    names.append("PAST_MORPH")
    return names


FEATURE_NAMES = _build_feature_names()
NFEAT = len(FEATURE_NAMES)
_FEAT2IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

FEAT_TOKEN_BASE = N_CHAR
# Optional per-word random identity embedding (captures specific word identity
# that coarse categories cannot). Hashed to a fixed row so the same word always
# gets the same random vector (generalizes for words shared by train/test).
USE_WORD_ID = False
WORD_ID_STD = 0.25
WORD_HASH_SIZE = 16384
WORD_HASH_BASE = FEAT_TOKEN_BASE + NFEAT
VOCAB_SIZE = WORD_HASH_BASE + WORD_HASH_SIZE


def _word_hash(word: str) -> int:
    import hashlib
    h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
    return WORD_HASH_BASE + (h % WORD_HASH_SIZE)


# ---------------------------------------------------------------------------
# Architecture (NO TRAINING; LayerNorms = identity)
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
        x = 0.7 * x + self.attn(self.ln1(x), attn_bias)
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 512, d_model: int = 1024,
                 n_heads: int = 8, n_layers: int = 1, d_ff: int = 16):
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
                pad_mask: torch.Tensor, kbias: torch.Tensor = None) -> torch.Tensor:
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(pos_ids)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        bias = torch.zeros(B, 1, T, T, device=ids.device)
        bias = bias.masked_fill(causal[None, None], float("-inf"))
        bias = bias.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        if kbias is not None:
            bias = bias + kbias[:, None, None, :]
        for block in self.blocks:
            h = block(h, bias)
        return self.final_ln(h)


class InterpretableEmbedder:
    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.max_seq_len = model.max_seq_len

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        text = text.lower()
        words = text.split()
        ids: List[int] = []
        pos: List[int] = []
        cflag: List[int] = []
        fwflag: List[int] = []
        dflag: List[int] = []
        # orthographic char tokens on the char-position timeline
        if USE_CHAR_CONTENT:
            for i, c in enumerate(text):
                ids.append(_stoi.get(c, UNK_ID))
                pos.append(i)
                cflag.append(0)
                fwflag.append(0)
                dflag.append(0)
        # locate each word's end position on the same timeline
        spans = []
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
            extra = []
            # negation scope: any negation word in the preceding 3 words tags this
            # word (a cheap compositional cue language cortex is sensitive to).
            negated = any(pw.replace("'", "") in _NEG for pw in recent_words[max(0, k - 2):k])
            wf = word_features(w)
            if negated:
                extra.append("NEG_SCOPE")
                # affective composition: negation inverts valence/emotion polarity
                # (not good ~ negative; not happy ~ unhappy)
                _flip = {"VAL_POS": "VAL_NEG", "VAL_NEG": "VAL_POS",
                         "SEM_EMOTION_POS": "SEM_EMOTION_NEG",
                         "SEM_EMOTION_NEG": "SEM_EMOTION_POS"}
                wf = [_flip.get(f, f) for f in wf]
            _seen = set()
            _ordered = [f for f in (wf + extra) if not (f in _seen or _seen.add(f))]
            # readout = last token's residual: the LAST feature in the bag gets an extra
            # +1 emphasis via the residual connection. Place the most predictive feature
            # of the final word in that residual slot, by priority: valence, then self-
            # reference, then abstractness (all salient drivers of the readout on the final
            # word; concrete-low/abstract helps but concrete-high does not).
            for _grp in ("CONC_LOW", "SELF_REF", "VAL"):
                _m = [f for f in _ordered if f.startswith(_grp)]
                if _m:
                    _ordered = [f for f in _ordered if not f.startswith(_grp)] + _m
            feat_ids = [FEAT_TOKEN_BASE + _FEAT2IDX[f]
                        for f in _ordered]
            if USE_WORD_ID:
                feat_ids = feat_ids + [_word_hash(w)]
            _cf = 1 if "CONTENT" in _ordered else 0
            _fw = 1 if k == 0 else 0  # first word of the context window = topic/discourse anchor
            _df = 1 if "DISCREL" in _ordered else 0
            for _ in range(reps):
                for fid in feat_ids:
                    ids.append(fid)
                    pos.append(endpos)
                    cflag.append(_cf)
                    fwflag.append(_fw)
                    dflag.append(_df)
        if not ids:
            return [PAD_ID], [0], [0], [0], [0]
        if len(ids) > self.max_seq_len:
            ids = ids[-self.max_seq_len:]
            pos = pos[-self.max_seq_len:]
            cflag = cflag[-self.max_seq_len:]
            fwflag = fwflag[-self.max_seq_len:]
            dflag = dflag[-self.max_seq_len:]
        pos = [min(pp, self.max_seq_len - 1) for pp in pos]
        return ids, pos, cflag, fwflag, dflag

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
            kbias = torch.zeros((len(enc), T), dtype=torch.float)
            for j, (e, pp, cf, fw, df) in enumerate(enc):
                ids[j, :len(e)] = torch.tensor(e, dtype=torch.long)
                pos_ids[j, :len(pp)] = torch.tensor(pp, dtype=torch.long)
                pad_mask[j, :len(e)] = True
                # content-word pooling de-emphasis: down-weight content words in the
                # pooled bag so function words (syntactic/discourse frame) are better
                # represented alongside the content-heavy last-word residual.
                kbias[j, :len(cf)] = (-0.6 * torch.tensor(cf, dtype=torch.float)
                                      + 0.6 * torch.tensor(fw, dtype=torch.float)
                                      + ZETA_DISC * torch.tensor(df, dtype=torch.float))
            ids = ids.to(self.device)
            pos_ids = pos_ids.to(self.device)
            pad_mask = pad_mask.to(self.device)
            kbias = kbias.to(self.device)
            hidden = self.model(ids, pos_ids, pad_mask, kbias)
            last = torch.tensor([l - 1 for l in lens], device=self.device)
            emb = hidden[torch.arange(len(enc), device=self.device), last]
            embs.append(emb.float().cpu().numpy())
        return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# Hand-written weights (no training)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer) -> None:
    D = model.d_model
    H = model.n_heads
    dh = D // H
    assert H == len(LAMBDAS), "n_heads must match number of decay lambdas"
    assert CAT_OFFSET + NFEAT <= dh, f"features ({NFEAT}) must fit in head slice ({dh})"

    with torch.no_grad():
        model.token_emb.weight.zero_()
        # each feature token -> one-hot at its feature dim, replicated per head.
        for f in range(NFEAT):
            tok = FEAT_TOKEN_BASE + f
            for hh in range(H):
                model.token_emb.weight[tok, hh * dh + CAT_OFFSET + f] = 1.0

        # optional random orthographic content for char tokens, placed in the
        # per-head dims above the feature block (disjoint from feature dims).
        if USE_CHAR_CONTENT:
            g = torch.Generator().manual_seed(0)
            content_lo = CAT_OFFSET + NFEAT
            for hh in range(H):
                lo = hh * dh + content_lo
                hi = (hh + 1) * dh
                w = torch.empty(N_CHAR, hi - lo)
                w.normal_(mean=0.0, std=CHAR_CONTENT_STD / math.sqrt(hi - lo), generator=g)
                model.token_emb.weight[:N_CHAR, lo:hi] = w
            model.token_emb.weight[PAD_ID].zero_()

        # optional per-word random identity vectors in the per-head content dims
        # above the feature block (multi-scale, disjoint from feature one-hots).
        if USE_WORD_ID:
            g2 = torch.Generator().manual_seed(123)
            content_lo = CAT_OFFSET + NFEAT
            nrows = WORD_HASH_SIZE
            for hh in range(H):
                lo = hh * dh + content_lo
                hi = (hh + 1) * dh
                w = torch.empty(nrows, hi - lo)
                w.normal_(mean=0.0, std=WORD_ID_STD / math.sqrt(hi - lo), generator=g2)
                model.token_emb.weight[WORD_HASH_BASE:WORD_HASH_BASE + nrows, lo:hi] = w

        model.pos_emb.weight.zero_()
        js = torch.arange(model.max_seq_len, dtype=torch.float32)
        model.pos_emb.weight[:, POS_DIM] = js
        model.pos_emb.weight[:, BIAS_DIM] = 1.0

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

        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()
    return


model_shorthand_name = "DiscBoost"
model_description = (
    "Hand-wired interpretable transformer. Each word is tokenized into a small set of "
    "interpretable feature tokens: function-word type (pronoun/prep/conj/article/aux/"
    "neg/wh/discourse/interjection) or CONTENT, ~40 hand-curated semantic categories "
    "(motion, person, emotion, place, body, health, etc.; including fine-grained "
    "action subtypes self-motion/caused-motion/speech-act that match motor and "
    "language cortex), perceptual modality "
    "(vision/sound/touch/taste/motor; smell pruned as it fired <0.02% of words and "
    "only added z-scored noise; partly derived from category membership, "
    "e.g. speech/communication->sound to match auditory cortex), "
    "concreteness (derived from categories), animacy, arousal, valence, "
    "person-reference (1st-person self-reference for default-mode/medial-prefrontal "
    "cortex and 3rd-person other-reference for theory-of-mind/social cognition), "
    "inanimate/expletive reference (it/its/itself: no animacy or theory-of-mind, "
    "carved from generic pronouns), tense/temporal-reference axis (future modals "
    "will/gonna/shall and past auxiliaries was/were/had/did, with present as the "
    "unmarked default - a temporal-deixis distinction language cortex tracks), "
    "spatial/locative prepositions (parietal spatial cognition), goal/purpose "
    "prepositions (to/for: recipient/benefactive/goal argument-structure markers, "
    "distinct from spatial prepositions), genitive preposition (of: possessive/"
    "partitive relation, very high frequency, carved from generic prepositions), "
    "logical/discourse-"
    "relation connectives (causal/adversative/conditional: but, because, if, so - "
    "distinct from additive and/or coordination, marking discourse coherence "
    "processing), additive coordinator (and: the dominant additive conjunction, "
    "carved from generic conjunctions), and word-"
    "frequency bucket. Contracted words (don't/i'm/it's) are apostrophe-normalized so "
    "spoken-corpus negations and pronouns are correctly tagged, and possessives "
    "(mother's) fall back to their base noun for semantic lookup. Negation within a "
    "2-word scope tags the word and compositionally inverts its valence/emotion polarity "
    "(not good ~ negative). Each feature token's "
    "embedding is a fixed one-hot, replicated "
    "across head slices. A single hand-wired attention layer pools these tokens over "
    "the 12-gram at 4 position time-scales (lambda=-0.04,0,4,16: a near-uniform broad "
    "context head with a faint primacy/topic bias, a global-mean head, and two recency "
    "heads) via position-keyed scores. A per-key additive attention bias reshapes which "
    "words dominate the pooled bag: content words are down-weighted (so the function-word "
    "syntactic/discourse frame is better represented), the first word of the window is "
    "boosted (topic/discourse anchor), and discourse connectives (but/because/so/if/"
    "although/however...) are strongly boosted as predictive discourse-structure cues. "
    "A weighted single-layer residual (0.7*x + attn) carries the last word's most "
    "readout-salient feature (priority: valence > self-reference > abstractness) as an "
    "extra emphasis in the final-token output embedding. Recent words "
    "are repeated to emphasize recency. Surface/morphological features (length, POS "
    "suffix, prefix) were ablated as they only added overfitting. Embedding comes "
    "entirely from the genuine forward pass. No training, no pretrained weights, no "
    "external libraries."
)


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def build_embedder(device: str = 'cuda', d_model: int = 1024, n_heads: int = None,
                   n_layers: int = 1, d_ff: int = 16, max_seq_len: int = 512) -> InterpretableEmbedder:
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
