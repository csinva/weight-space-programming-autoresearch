"""Interpretable transformer embedder for fMRI language encoding.

Iter11 (RecencyDecayBoC-4heads): multi-head causal attention where each head
computes a softmax recency-decay-weighted bag-of-chars at a different decay
rate. Heads with decay 0 give a uniform mean (global BoC); larger decay heads
sharply emphasize the last few chars / last word. Concatenating heads gives
ridge a multi-scale temporal bag-of-chars feature for each n-gram.

Construction:
  * Reserve two d_model dims (0 = position scalar j, 1 = constant bias 1).
    token_emb is zero on these; pos_emb supplies them.
  * pos_emb[j] = [j, 1, 0, ..., 0]   (only dims 0, 1 are nonzero)
  * token_emb: zero on dim 0, 1; random Gaussian elsewhere.
  * For head h with decay rate lambda_h:
      W_k routes input dim 0 -> head h's k-dim 0 with weight lambda_h * sqrt(dh)
        (so q.k / sqrt(dh) = lambda_h * j)
      W_q routes input dim 1 -> head h's q-dim 0 with weight 1
        (so q_h = (1, 0, ..., 0))
      Score(i, j; h) = lambda_h * j; softmax over causal j gives weights
        proportional to exp(lambda_h * j) -- recency-weighted.
  * W_v: identity within each head's slice on the random-char dims (zeros on
    dims 0, 1 so position/bias don't leak into values).
  * W_o: identity.
  * MLP: zero. LayerNorms: identity. Single layer.

Lambdas span 0 (uniform) -> large (last token only) for multi-scale recency.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List

import numpy as np
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results, plot_corr_over_iterations,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Single bucket (no word-position tagging this iter); the recency decay does
# its own implicit recent-vs-old weighting via attention.
_VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\"
_BASE_CHAR_VOCAB = ['<pad>', '<unk>'] + list(_VOCAB_CHARS)
# Two buckets: bucket 0 = chars inside the LAST whitespace-word, bucket 1 = all
# earlier chars (whitespace, prior words). Each (char, bucket) pair gets its
# own random embedding row, so after recency-decay pooling the hidden state has
# (nearly) orthogonal sub-bags for "last word" vs "earlier context".
N_WORD_BUCKETS = 2
BUCKET_SIZE = len(_BASE_CHAR_VOCAB)
# Iter17: also reserve a block of word-hash rows in token_emb so encode() can
# append synthetic "word identity hash" tokens at the END of each input. The
# final-token hidden state then directly carries (via residual) the hashed
# random embedding of the last word, plus a recency-decay pooled BoC of
# everything else.
WORD_HASH_SIZE = 4096
N_APPEND_WORDS = 15
MAX_TRIGRAMS_PER_WORD = 14
KEEP_UNIGRAM_HASH = True
KEEP_BIGRAM_HASH = True
KEEP_4GRAM_HASH = True
KEEP_5GRAM_HASH = True
MAX_4GRAMS_PER_WORD = 10
MAX_5GRAMS_PER_WORD = 8
MAX_BIGRAMS_PER_WORD = 10
USE_READ_TOKEN = False  # append a final <READ> token whose token_emb is 0 so
                       # the readout = pure attention pool (no residual leak
                       # from the last appended hash token).
POS_TAG_HASH = False
K_UNIGRAM_HASH = 1
ADD_POS_TAG = False
ADD_FUNC_WORD_TAG = False
ADD_SEMANTIC_CAT = False
ADD_WORD_LENGTH = True
ADD_STEM_HASH = True
ADD_SKIP_BIGRAM = True
ADD_FIRSTLAST = True
USE_RANDOM_MLP = False
# Iter101+ novel feature flags
ADD_ANAGRAM = False
ADD_VOWEL_HASH = False
ADD_CONSONANT_HASH = False
ADD_SYLLABLE_TAG = False
ADD_DOUBLED_LETTER = False
ADD_REVERSE_SUBWORD = False
ADD_LETTER_SET = False
ADD_SUFFIX2 = False
ADD_PREFIX2 = False
ADD_PHONETIC = False
ADD_REPEATED_WORD = False
ADD_DIGIT_TAG = False
ADD_CROSSWORD_TRI = False
ADD_WORD_POS_TAG = False
ADD_VOWEL_PATTERN = False
ADD_MID_LETTER = False
ADD_SENT_POSITION = False
USE_POS_CUBE_REG = False
USE_SIN_REG = False
USE_NO_ATTENTION = False
NEG_LAMBDA = False
ATTN_TEMP = 1.0

_VOWELS = set("aeiouy")

# Hand-crafted small semantic-category lexicon. Each category becomes a
# shared token: ALL words in a category map to the same hash bucket, so
# token_emb gives them the same vector. Provides semantic similarity
# structure that random hashing cannot.
_SEM_CATEGORIES = {
    "BODY": "head hand hands foot feet eye eyes hair face arm arms leg legs body mouth ear ears nose lips lip skin finger fingers heart blood bone bones brain chest neck back knee knees shoulder shoulders teeth tooth tongue knees elbow elbows wrist wrists ankle ankles thumb thumbs forehead chin cheek cheeks throat stomach belly hip hips waist palm thigh thighs nail nails breath breathing lungs liver".split(),
    "FAMILY": "mother father brother sister family child children son daughter wife husband parent parents kid kids baby boy boys girl girls uncle aunt cousin grandmother grandfather grandma grandpa mom dad mommy daddy nephew niece spouse siblings twins".split(),
    "MOTION": "walk walked walking run ran running jump jumped jumping move moved moving go went going come came coming fly flew flying swim swam swimming climb climbed climbing fall fell falling drive drove driving ride rode riding arrive arrived arriving leave left leaving enter entered entering exit exited rise rose rising sit sat sitting stand stood standing lay laid lying dance danced bend bent crawl crawled chase chased hop hopped slide slid roll rolled spin spun turn turned twist twisted lean leaned step stepped escape escaped flee fled".split(),
    "TIME": "day days night nights year years hour hours minute minutes second seconds time times morning evening afternoon week weeks month months today tomorrow yesterday now then soon later early late while moment moments age decade century forever weekend".split(),
    "PLACE": "room rooms house houses city cities town towns road roads street streets country home homes school office building world place places land beach park yard kitchen bedroom bathroom hall door window floor wall ceiling roof garden river mountain forest field lake ocean sea valley hill island village neighborhood church temple library hospital station airport restaurant cafe bar farm cabin tent stage path bridge tunnel basement attic porch sidewalk apartment".split(),
    "ANIMAL": "dog dogs cat cats bird birds horse horses fish animal animals cow cows pig pigs sheep mouse mice rabbit rabbits chicken duck snake bear wolf lion tiger elephant monkey insect bee bees fly flies ant ants spider spiders frog toad turtle shark whale dolphin squirrel deer goat goose owl eagle hawk butterfly worm bug bugs hen rooster lamb calf kitten puppy".split(),
    "FOOD": "food eat ate eating eaten bread water meat milk drink drank drinking coffee tea wine beer fruit apple egg eggs cheese rice meal dinner lunch breakfast cook cooked cooking soup salad pizza pasta noodle noodles cake pie cookie cookies candy chocolate sugar salt pepper butter sauce sandwich snack chicken beef pork bacon ham vegetable vegetables tomato potato onion garlic banana orange lemon cup glass plate bowl spoon fork knife dish".split(),
    "EMOTION": "love loved loving fear feared afraid anger angry happy sad joy joyful hate hated like liked feel felt feeling emotion emotions happiness sadness scared worried surprised excited nervous proud lonely guilty hurt heartbroken miserable thrilled cheerful gloomy mood furious bitter joy enjoy enjoyed enjoying disgusted irritated annoyed embarrassed relieved hopeful hope hoped hoping crying cry tears tear sob sobbed smile smiled smiling frown frowned laugh laughed laughing fond admire admired".split(),
    "SPEECH": "said say says saying told tell tells talk talked talking speak spoke spoken speaking asked ask asking answer answered voice voices word words shout shouted shouting whisper whispered call called calls calling laugh laughed cried screamed scream yelling yell yelled mumbled muttered explained explain announced announce stated state argued argue replied reply mentioned mention noted noticed".split(),
    "COGNITION": "know knew known think thought thinking believe believed believing mind minds idea ideas remember remembered forget forgot forgotten understand understood imagine imagined wonder wondered realize realized decide decided dream dreams dreaming consider considered guess guessed assume assumed suppose supposed expect expected hope hoped wish wished plan planned learn learned figure figured doubt doubted prefer preferred reason reasoning notice noticed recognized".split(),
    "VISUAL": "see saw seen look looked looking watch watched watching light dark color colors bright shadow shadows visible appear appeared show showed showing reveal revealed glance glanced peek peeked stare stared staring observe observed view viewed sight sights vision glimpse glow glowed shine shone shining shiny dim image picture pictures scene".split(),
    "AUDITORY": "heard hear hearing listen listened listening sound sounds music voice loud quiet silent silence song songs noise noises ring rang ringing bell bells echo echoes melody tune rhythm beat drum drums whistle whistled buzz hum hummed scream screamed roared roar".split(),
    "PERSON": "man men woman women person persons people guy guys lady ladies gentleman gentlemen friend friends stranger strangers neighbor neighbors crowd crowds group groups teen teenager teenagers adult adults senior child humans human individual someone somebody anyone anybody everyone everybody nobody no_one".split(),
    "TIME_SEQ": "first second third next then before after finally already still yet always never sometimes often rarely usually suddenly immediately again once previously initially eventually meanwhile afterward thereafter henceforth thereafter beforehand subsequently".split(),
    "QUANTITY": "many much few little more less most least all some any none several couple bunch lot lots plenty enough whole half almost about around majority minority dozen dozens hundreds thousands millions handful pair triple double quadruple".split(),
    "SIZE": "big small large tiny huge little long short tall wide narrow deep shallow thick thin enormous massive giant minuscule tremendous immense colossal mini medium gigantic compact slim slender plump bulky".split(),
    "MONEY": "money dollar dollars cent cents pay paid paying buy bought cost price expensive cheap rich poor wealthy bank loan owe earn earned debt debts credit cash funds budget afford afforded salary income spend spent spending sell sold selling purchase purchased trade traded".split(),
    "WORK": "work worked working job jobs career office boss employee company business store shop factory shift hire fired manage managed task tasks project projects meeting meetings deadline employer employee colleague colleagues coworker coworkers profession professional skill skills".split(),
    "ABSTRACT": "thing things idea ideas fact facts truth lie reason cause effect way ways problem problems question questions answer answers reason purpose meaning matter result results concept concepts principle principles theory theories belief beliefs opinion opinions value values".split(),
    "VEHICLE": "car cars truck trucks bus buses train trains plane planes airplane airplanes boat boats ship ships bicycle bike bikes motorcycle vehicle vehicles taxi subway helicopter sailboat canoe wagon scooter tractor van vans".split(),
    "CLOTHES": "shirt shirts pants shoes shoe dress dresses hat hats coat coats jacket jackets sock socks pocket pockets clothes clothing wear wearing wore tie scarf glove gloves sweater sweaters skirt skirts boots boot belt jeans suit suits uniform tshirt tee zipper button buttons collar sleeve sleeves".split(),
    "COLOR": "red blue green yellow black white brown orange purple pink gray grey gold silver violet maroon navy turquoise beige cream ivory crimson scarlet aqua tan magenta".split(),
    "WEATHER": "rain rained raining snow snowed snowing wind windy sun sunny cloud clouds cloudy storm storms hot cold warm cool wet dry humid freezing chilly mild rainy stormy sunny breezy fog foggy lightning thunder hurricane tornado blizzard drought flood".split(),
    "SOUND_VERB": "ring rang ringing bang banged knock knocked tap tapped click clicked beep buzz hum thud crash crashed slap slapped pop popped boom boomed roar rumble snap snapped".split(),
    "NUMBER": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion zero half quarter".split(),
    "NATURE": "tree trees leaf leaves grass flower flowers rock rocks stone stones plant plants bush bushes branch branches root roots forest woods river stream creek brook mountain mountains hill hills valley field meadow desert ocean sea wave waves sand soil dirt mud cliff cave moss vine".split(),
    "HEALTH": "sick ill ache aching pain hurt healing hospital doctor doctors nurse medicine pill pills drug drugs cure cured surgery treatment treatments disease illness sickness wound wounded recovery healthy fever cough cold flu infection allergy allergic".split(),
    "VIOLENCE": "fight fought fighting hit hits hitting punch punched kick kicked kill killed killing war wars battle battles attack attacked attacking weapon weapons gun guns sword sword knife knives blood injury injured wound wounded murder murdered shoot shot shooting".split(),
    "LEARN": "read reading study studied studying learn learned learning teach taught teaching school class classes lesson lessons book books page pages chapter chapters student students teacher teachers university college degree practice practiced".split(),
    "TIME_VERB": "wait waited waiting begin began begun start started starting finish finished finishing end ended ending continue continued stop stopped stopping pause paused break broke broken rest rested resting hurry hurried delay delayed".split(),
    "POSITION": "above below under over inside outside between behind beside near far close beyond among across through within without onto into upon among amid".split(),
    "SOCIAL": "party parties meeting meetings gather gathered together alone group team teams club clubs event events celebration celebrate celebrated wedding funeral festival friendship relationship marriage divorce dating date".split(),
    "TECH": "phone phones computer computers screen screens television tv internet email message messages text texting camera radio button buttons machine machines technology device devices software website website online digital electronic".split(),
    "POSSESSION": "have has had having get got gotten getting own owned owning take took taken give gave given keep kept keeping hold held holding bring brought bringing carry carried find found finding lose lost losing".split(),
    "SMALL_OBJECT": "ball balls pen pens pencil paper papers key keys coin coins ring rings clock watch watches bag bags box boxes bottle bottles cup cups pen pin pins nail nails screw button buttons card cards toy toys book books pencil pencils paper".split(),
    "MUSIC": "music song songs piano guitar drum drums violin sing sang sung singing dance danced dancing concert concerts band bands album albums tune tunes melody beat rhythm note notes chord".split(),
    "SPORT": "game games play played playing player players team teams ball score scored winning won winning lose lost losing race racing match matches championship olympics football basketball baseball soccer tennis golf ski skiing".split(),
    "BUILDING": "building buildings house houses apartment building tower towers castle palace church temple bridge bridges wall walls door doors window windows roof ceiling floor staircase stair stairs hallway corridor lobby".split(),
    "MATERIAL": "wood metal iron steel plastic glass paper stone brick concrete cloth fabric leather rubber gold silver copper rope string thread sand water dust ash mud snow ice".split(),
    "ACTION_HANDS": "grab grabbed take took took hold held push pushed pull pulled lift lifted carry carried throw threw thrown catch caught drop dropped touch touched press pressed squeeze squeezed shake shook tap tapped knock knocked open opened close closed".split(),
    "DEATH": "die died dying death dead grave funeral cemetery corpse killed killing kill murder murdered suicide alive killed".split(),
    "BIRTH": "born birth give_birth pregnant pregnancy newborn baby infant pregnant".split(),
    "WATER_VERB": "swim swam swum drown drowned splash splashed dive dove diving sail sailed sailing flood flooded leak leaked drip dripped pour poured spill spilled".split(),
}
_WORD2CAT = {}
for cat, words_list in _SEM_CATEGORIES.items():
    for w in words_list:
        _WORD2CAT.setdefault(w, []).append(cat)

# Small heuristic POS-tagger; returns one of: VBG, VBD, RB, NN_TION, NN_NESS,
# NN_MENT, NN_ITY, COMP_ER, SUP_EST, NEG_UN, NNS, FW (function), CW (content).
_FUNC_WORDS = set("""
the a an of in on at to for with and or but is was are were be been being
has had have do does did this that these those it he she they we you i
my his her their our your not no very so as by from all some any if then
than when where what who why how about into onto out off over under up
down here there now will would could should can may might must shall
me him us them mine yours theirs himself herself itself themselves
yourself myself one two three first last next
""".split())

def heuristic_pos(w: str) -> str:
    if w in _FUNC_WORDS:
        return "FW"
    if len(w) >= 5 and w.endswith("ing"):
        return "VBG"
    if len(w) >= 5 and (w.endswith("tion") or w.endswith("sion")):
        return "NN_TION"
    if len(w) >= 5 and w.endswith("ness"):
        return "NN_NESS"
    if len(w) >= 5 and w.endswith("ment"):
        return "NN_MENT"
    if len(w) >= 4 and w.endswith("ity"):
        return "NN_ITY"
    if len(w) >= 4 and w.endswith("ly"):
        return "RB"
    if len(w) >= 4 and w.endswith("ed"):
        return "VBD"
    if len(w) >= 5 and w.endswith("est"):
        return "SUP_EST"
    if len(w) >= 4 and w.endswith("er"):
        return "COMP_ER"
    if len(w) >= 4 and w.startswith("un"):
        return "NEG_UN"
    if len(w) >= 4 and w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        return "NNS"
    return "CW"
N_WORD_BIGRAMS = 0
# Curated semantic vocab: each known word gets a unique vocab id and a dense
# semantic-category indicator vector written into reserved dims of token_emb.
# Categories come from _SEM_CATEGORIES (defined above).
_SEM_CAT_NAMES = sorted(_SEM_CATEGORIES.keys())
_CAT2DIM = {cat: i for i, cat in enumerate(_SEM_CAT_NAMES)}
N_SEM_CATS = len(_SEM_CAT_NAMES)
# Reserved dim range [SEM_DIM_START, SEM_DIM_START + N_SEM_CATS) holds the
# category indicators. Avoid 0/1/2 which are used for POS/BIAS/POS_SQ.
SEM_DIM_START = 3
N_SEM_CAT_REPEATS = 12  # replicate each category indicator across this many dims
SEM_DIM_END = SEM_DIM_START + N_SEM_CATS * N_SEM_CAT_REPEATS
SEM_DIM_VALUE = 500.0

# Build the curated-word -> vocab-id map. All curated words occupy a
# contiguous block of vocab ids AFTER the hash buckets and READ token.
_CURATED_WORDS = []
_seen = set()
for cat in _SEM_CAT_NAMES:
    for w in _SEM_CATEGORIES[cat]:
        if w not in _seen:
            _seen.add(w)
            _CURATED_WORDS.append(w)
N_CURATED = len(_CURATED_WORDS)

VOCAB = []
for b in range(N_WORD_BUCKETS):
    VOCAB += [f"#{b}:{c}" for c in _BASE_CHAR_VOCAB]
VOCAB += [f"H{i}" for i in range(WORD_HASH_SIZE)]
VOCAB += ["<READ>"]  # final readout token with zero token_emb row
WORD_HASH_OFFSET = N_WORD_BUCKETS * BUCKET_SIZE
READ_TOKEN_ID = WORD_HASH_OFFSET + WORD_HASH_SIZE
# Curated vocab starts AFTER READ token.
CURATED_OFFSET = READ_TOKEN_ID + 1
VOCAB += [f"C:{w}" for w in _CURATED_WORDS]
_WORD2CURATED_ID = {w: CURATED_OFFSET + i for i, w in enumerate(_CURATED_WORDS)}

# Flag to enable dense semantic embedding pass.
USE_DENSE_SEMANTIC = True

# Two reserved d_model dims used to inject position info into attention.
POS_DIM = 0   # holds the scalar j (token position, BACKWARD from last)
BIAS_DIM = 1  # holds the constant 1
POS_SQ_DIM = 2  # holds j^2 / max_seq_len  (for Gaussian-per-slot attention)

USE_GAUSSIAN_ATTENTION = False
USE_HYBRID_ATTENTION = False
HYBRID_LAMBDAS = (0.0, 0.15, 0.3, 0.6, 1.0, 1.7, 3.0, 6.0)  # negative lambdas because pos is BACKWARD (small=recent)
HYBRID_TARGETS = (35, 80, 160, 280)  # 4 Gaussian heads, medium distances
HYBRID_SIGMAS = (25, 40, 70, 100)
GAUSSIAN_TARGETS = (0, 35, 70, 110, 160, 220, 300, 400)
GAUSSIAN_SIGMAS = (25, 25, 30, 40, 50, 70, 90, 120)


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
    def __init__(self, vocab_size, max_seq_len=64, d_model=64,
                 n_heads=8, n_layers=2, d_ff=64):
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
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, ids: torch.Tensor, pos_ids: torch.Tensor = None) -> torch.Tensor:
        B, T = ids.shape
        if pos_ids is None:
            pos_ids = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(ids) + self.pos_emb(pos_ids)
        for block in self.blocks:
            h = block(h)
        return self.final_ln(h)


class InterpretableEmbedder:
    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.stoi = {c: i for i, c in enumerate(_BASE_CHAR_VOCAB)}
        self.pad_id = 0
        self.unk_id = 1
        self.max_seq_len = model.max_seq_len

    def encode(self, text: str):
        """Returns (token_ids, pos_ids) where:
        - char tokens are tagged with last-word vs prior bucket (iter13),
        - up to N_APPEND synthetic word-hash tokens are appended at the end
          (one per recent word: last word, then 2nd-last word, ...).
        pos_ids = absolute char position; appended hash tokens get the
        last char's pos_id + 1, +2, ... (so they are the MOST RECENT positions
        and dominate recency-decay attention).
        """
        text = text.lower()
        words = text.split()
        if not words:
            return [self.pad_id], [0]
        # last-word char span (for bucket tagging).
        last_word = words[-1]
        ls = text.rfind(last_word)
        le = ls + len(last_word)
        ids, pos = [], []
        for i, c in enumerate(text):
            base = self.stoi.get(c, self.unk_id)
            bucket = 0 if (ls <= i < le) else 1
            ids.append(base + bucket * BUCKET_SIZE)
            pos.append(i)
        # Append subword (char-trigram) hashes for the last N_APPEND_WORDS words,
        # OLDER -> NEWER, then the unigram word-identity hashes after that. The
        # very last positions are highest-recency, so they get most attention
        # weight; ridge can read off the per-word subword-bag in the residual.
        hash_ids = []
        # Dense-semantic curated tokens placed FIRST so they sit at low-recency
        # positions; the residual still contains their dense category indicators
        # because dim values are >0 there. The recency-weighted attention pools
        # later tokens; but the uniform lambda=0 head averages everything, so
        # ridge sees the category-count signal directly.
        if USE_DENSE_SEMANTIC:
            for back_idx in range(N_APPEND_WORDS - 1, -1, -1):
                if back_idx >= len(words):
                    continue
                w = words[-1 - back_idx]
                vid = _WORD2CURATED_ID.get(w)
                if vid is not None:
                    hash_ids.append(vid)

        # Function-word identity tokens come FIRST (lowest recency) so they
        # don't dominate the readout; they tag presence/identity of function
        # words in the context.
        if ADD_FUNC_WORD_TAG:
            for back_idx in range(N_APPEND_WORDS - 1, -1, -1):
                if back_idx >= len(words):
                    continue
                w = words[-1 - back_idx]
                if w in _FUNC_WORDS:
                    key = ("\x11" + w).encode("utf-8")
                    h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                    hash_ids.append(WORD_HASH_OFFSET + h)
        # Word-length bucket tags (early position): captures known fMRI predictor.
        if ADD_WORD_LENGTH:
            for back_idx in range(N_APPEND_WORDS - 1, -1, -1):
                if back_idx >= len(words):
                    continue
                w = words[-1 - back_idx]
                length_bucket = min(len(w), 15)
                key = ("\x14L%d" % length_bucket).encode("utf-8")
                h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                hash_ids.append(WORD_HASH_OFFSET + h)
        # First+last letter pair tag (early position): captures orthographic
        # word-shape feature, redundant with bigrams but explicit.
        if ADD_FIRSTLAST:
            for back_idx in range(N_APPEND_WORDS - 1, -1, -1):
                if back_idx >= len(words):
                    continue
                w = words[-1 - back_idx]
                if len(w) >= 2:
                    key = ("\x15" + w[0] + w[-1]).encode("utf-8")
                    h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                    hash_ids.append(WORD_HASH_OFFSET + h)
        # Skip-bigram word pair tag: (word_{i-2}, word_i) captures non-adjacent context.
        if ADD_SKIP_BIGRAM and len(words) >= 3:
            n_pairs = min(N_APPEND_WORDS - 2, len(words) - 2)
            for back_idx in range(n_pairs - 1, -1, -1):
                w_b = words[-3 - back_idx]
                w_a = words[-1 - back_idx]
                key = ("\x16" + w_b + " " + w_a).encode("utf-8")
                h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                hash_ids.append(WORD_HASH_OFFSET + h)
        # Stem-stripped unigram hash (per-word): strip common suffixes so
        # different inflections map to the same hash. Placed early (low recency).
        if ADD_STEM_HASH:
            for back_idx in range(N_APPEND_WORDS - 1, -1, -1):
                if back_idx >= len(words):
                    continue
                w = words[-1 - back_idx]
                stem = w
                for suf in ("ing", "edly", "ed", "ies", "ly", "es", "s", "tion", "ment", "ness"):
                    if len(stem) > len(suf) + 2 and stem.endswith(suf):
                        stem = stem[:-len(suf)]
                        break
                key = ("\x17" + stem).encode("utf-8")
                h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                hash_ids.append(WORD_HASH_OFFSET + h)
        # --- iter101+ novel features ---
        def _emit(salt: str, payload: str):
            key = (salt + payload).encode("utf-8")
            h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
            hash_ids.append(WORD_HASH_OFFSET + h)
        for back_idx in range(N_APPEND_WORDS - 1, -1, -1):
            if back_idx >= len(words):
                continue
            w = words[-1 - back_idx]
            if ADD_ANAGRAM:
                _emit("\x18", "".join(sorted(w)))
            if ADD_VOWEL_HASH:
                _emit("\x19", "".join(c for c in w if c in _VOWELS) or "_")
            if ADD_CONSONANT_HASH:
                _emit("\x1a", "".join(c for c in w if c.isalpha() and c not in _VOWELS) or "_")
            if ADD_SYLLABLE_TAG:
                count = 0; prev_v = False
                for c in w:
                    is_v = c in _VOWELS
                    if is_v and not prev_v:
                        count += 1
                    prev_v = is_v
                _emit("\x1b", "S%d" % min(count, 8))
            if ADD_DOUBLED_LETTER:
                has_dbl = any(w[i] == w[i+1] for i in range(len(w)-1))
                _emit("\x1c", "DBL" if has_dbl else "NODBL")
            if ADD_REVERSE_SUBWORD:
                rw = w[::-1]
                padded = "^" + rw + "$"
                for i in range(min(len(padded)-2, 8)):
                    _emit("\x1d", padded[i:i+3])
            if ADD_LETTER_SET:
                uniq = "".join(sorted(set(w)))
                _emit("\x1e", uniq)
            if ADD_SUFFIX2 and len(w) >= 2:
                _emit("\x1f", "SU" + w[-2:])
            if ADD_PREFIX2 and len(w) >= 2:
                _emit("\x20", "PR" + w[:2])
            if ADD_PHONETIC:
                ph = w
                for src, dst in (("ph","f"),("ck","k"),("qu","kw"),("wh","w"),
                                 ("sh","x"),("ch","c"),("th","t"),("ought","ot"),
                                 ("ight","it"),("tion","sn"),("sion","sn")):
                    ph = ph.replace(src, dst)
                _emit("\x21", ph)
            if ADD_DIGIT_TAG:
                _emit("\x22", "D" if any(c.isdigit() for c in w) else "ND")
            if ADD_VOWEL_PATTERN:
                pat = "".join("V" if c in _VOWELS else ("C" if c.isalpha() else "X") for c in w)
                _emit("\x23", pat)
            if ADD_MID_LETTER and len(w) >= 3:
                _emit("\x24", "M" + w[len(w)//2])
            if ADD_WORD_POS_TAG:
                _emit("\x25", "P%d" % back_idx)
        if ADD_REPEATED_WORD and len(words) >= 2:
            recent = set(words[-N_APPEND_WORDS-1:-1])
            for back_idx in range(min(N_APPEND_WORDS, len(words)) - 1, -1, -1):
                w = words[-1 - back_idx]
                if w in recent:
                    _emit("\x26", "REP" + w)
        if ADD_CROSSWORD_TRI and len(words) >= 2:
            for back_idx in range(min(N_APPEND_WORDS, len(words)) - 2, -1, -1):
                w_a = words[-2 - back_idx]
                w_b = words[-1 - back_idx]
                if w_a and w_b:
                    _emit("\x27", w_a[-1] + "_" + w_b[:2])
        if ADD_SENT_POSITION:
            # tag first word after a period in the text (heuristic)
            sent_starts = set()
            prev_was_punct = True
            for w in words:
                if prev_was_punct:
                    sent_starts.add(id(w))  # use id since words may repeat
                prev_was_punct = w.endswith(".") or w.endswith("!") or w.endswith("?")
            for back_idx in range(min(N_APPEND_WORDS, len(words)) - 1, -1, -1):
                w = words[-1 - back_idx]
                if id(w) in sent_starts:
                    _emit("\x28", "SENTSTART")
        # Semantic categories: placed EARLY (lowest recency weight) so they
        # don't drown out the subword bag at the readout; the lambda=0
        # uniform-mean head still incorporates them so ridge sees them.
        if ADD_SEMANTIC_CAT:
            for back_idx in range(N_APPEND_WORDS - 1, -1, -1):
                if back_idx >= len(words):
                    continue
                w = words[-1 - back_idx]
                cats = _WORD2CAT.get(w, ())
                for cat in cats:
                    key = ("\x13" + cat).encode("utf-8")
                    h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                    hash_ids.append(WORD_HASH_OFFSET + h)
        for back_idx in range(N_APPEND_WORDS - 1, -1, -1):
            if back_idx >= len(words):
                continue
            w = words[-1 - back_idx]
            padded = "^" + w + "$"
            trigrams = [padded[i:i+3] for i in range(len(padded) - 2)]
            trigrams = trigrams[:MAX_TRIGRAMS_PER_WORD]
            for tg in trigrams:
                key = ("\x07" + tg).encode("utf-8")
                h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                hash_ids.append(WORD_HASH_OFFSET + h)
            if KEEP_BIGRAM_HASH:
                bigrams = [padded[i:i+2] for i in range(len(padded) - 1)]
                bigrams = bigrams[:MAX_BIGRAMS_PER_WORD]
                for bg in bigrams:
                    key = ("\x06" + bg).encode("utf-8")
                    h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                    hash_ids.append(WORD_HASH_OFFSET + h)
            if KEEP_4GRAM_HASH and len(padded) >= 4:
                quadgrams = [padded[i:i+4] for i in range(len(padded) - 3)]
                quadgrams = quadgrams[:MAX_4GRAMS_PER_WORD]
                for qg in quadgrams:
                    key = ("\x08" + qg).encode("utf-8")
                    h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                    hash_ids.append(WORD_HASH_OFFSET + h)
            if KEEP_5GRAM_HASH and len(padded) >= 5:
                fivegrams = [padded[i:i+5] for i in range(len(padded) - 4)]
                fivegrams = fivegrams[:MAX_5GRAMS_PER_WORD]
                for fg in fivegrams:
                    key = ("\x09" + fg).encode("utf-8")
                    h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                    hash_ids.append(WORD_HASH_OFFSET + h)
        if KEEP_UNIGRAM_HASH:
            for back_idx in range(N_APPEND_WORDS - 1, -1, -1):
                if back_idx >= len(words):
                    continue
                w = words[-1 - back_idx]
                pos_tag = f"@{back_idx}@" if POS_TAG_HASH else ""
                for salt in range(K_UNIGRAM_HASH):
                    key = (chr(salt + 1) + pos_tag + w).encode("utf-8")
                    h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                    hash_ids.append(WORD_HASH_OFFSET + h)
        if ADD_SEMANTIC_CAT and False:  # disabled; placed earlier in sequence
            for back_idx in range(N_APPEND_WORDS - 1, -1, -1):
                if back_idx >= len(words):
                    continue
                w = words[-1 - back_idx]
                cats = _WORD2CAT.get(w, ())
                for cat in cats:
                    key = ("\x13" + cat).encode("utf-8")
                    h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                    hash_ids.append(WORD_HASH_OFFSET + h)
        if ADD_POS_TAG:
            pass  # disabled
        if False:  # ADD_FUNC_WORD_TAG handled earlier
            pass
        # Word-bigram hashes (consecutive word pairs), older -> newer. Each
        # pair gives ridge a "local context" feature: e.g. ("the", "cat") goes
        # into its own random subspace, distinct from "cat" alone.
        if N_WORD_BIGRAMS > 0 and len(words) >= 2:
            n_bg = min(N_WORD_BIGRAMS, len(words) - 1)
            for back_idx in range(n_bg - 1, -1, -1):
                w_b = words[-2 - back_idx]
                w_a = words[-1 - back_idx]
                key = ("\x02" + w_b + " " + w_a).encode("utf-8")
                h = int.from_bytes(hashlib.md5(key).digest()[:4], "big") % WORD_HASH_SIZE
                hash_ids.append(WORD_HASH_OFFSET + h)
        next_pos = pos[-1] + 1 if pos else 0
        for hid in hash_ids:
            ids.append(hid)
            pos.append(next_pos)
            next_pos += 1
        if USE_READ_TOKEN:
            ids.append(READ_TOKEN_ID)
            pos.append(next_pos)
            next_pos += 1
        ids = ids[-self.max_seq_len:]
        pos = pos[-self.max_seq_len:]
        # Re-base positions. For Gaussian per-slot attention we use BACKWARD
        # position: the LAST token gets pos=0, earlier tokens get pos=1,2,...
        # That lets per-head Gaussian targets {p_h} reference fixed offsets
        # from the readout regardless of input length.
        if USE_GAUSSIAN_ATTENTION or USE_HYBRID_ATTENTION:
            L = len(ids)
            pos = [L - 1 - t for t in range(L)]
        elif pos:
            shift = pos[0]
            pos = [p - shift for p in pos]
        if USE_READ_TOKEN and ids and ids[-1] == READ_TOKEN_ID:
            # Force the readout token to pos_emb position 0 so its residual
            # contribution from pos_emb is (POS_DIM=0, BIAS_DIM=1, 0,...) only.
            # Recency-decay attention scoring is unaffected (it depends on KEY
            # positions, not the query's pos).
            pos[-1] = 0
        if not ids:
            return [self.pad_id], [0]
        return ids, pos

    @torch.no_grad()
    def __call__(self, texts, batch_size=256):
        embs = []
        for i in range(0, len(texts), batch_size):
            enc = [self.encode(t) for t in texts[i:i + batch_size]]
            lens = [len(e[0]) for e in enc]
            T = max(lens)
            ids = torch.full((len(enc), T), self.pad_id, dtype=torch.long)
            pos_ids = torch.zeros((len(enc), T), dtype=torch.long)
            for j, (e, p) in enumerate(enc):
                ids[j, :len(e)] = torch.tensor(e, dtype=torch.long)
                pos_ids[j, :len(p)] = torch.tensor(p, dtype=torch.long)
            ids = ids.to(self.device)
            pos_ids = pos_ids.to(self.device)
            hidden = self.model(ids, pos_ids)
            last = torch.tensor([l - 1 for l in lens], device=self.device)
            emb = hidden[torch.arange(len(enc), device=self.device), last]
            embs.append(emb.float().cpu().numpy())
        return np.concatenate(embs, axis=0)


LAMBDAS = (0.0, 0.15, 0.3, 0.6, 1.0, 1.7, 3.0, 6.0)


def write_weights(model: SimpleTransformer) -> None:
    D = model.d_model
    H = model.n_heads
    dh = D // H
    T = model.max_seq_len
    if not USE_GAUSSIAN_ATTENTION:
        assert H == len(LAMBDAS), "n_heads must match number of lambdas"

    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        # Token emb: random Gaussian, zero on POS_DIM/BIAS_DIM/POS_SQ_DIM, pad row zero.
        model.token_emb.weight.normal_(mean=0.0, std=1.0 / math.sqrt(D), generator=g)
        model.token_emb.weight[:, POS_DIM] = 0.0
        model.token_emb.weight[:, BIAS_DIM] = 0.0
        model.token_emb.weight[:, POS_SQ_DIM] = 0.0
        # Zero out every <pad> row (one per bucket).
        for b in range(N_WORD_BUCKETS):
            model.token_emb.weight[b * BUCKET_SIZE].zero_()
        # Zero out the <READ> token row so the last position's residual
        # contributes nothing; readout = pure attention pool.
        model.token_emb.weight[READ_TOKEN_ID].zero_()

        # === Dense semantic embeddings ===
        # Reserve dims [SEM_DIM_START, SEM_DIM_END) to carry category indicators.
        # First zero those dims for ALL tokens so hash tokens don't add noise
        # into the semantic readout dims.
        if USE_DENSE_SEMANTIC:
            for d in range(SEM_DIM_START, SEM_DIM_END):
                model.token_emb.weight[:, d] = 0.0
            # Then write the category indicators for each curated word.
            for w, vid in _WORD2CURATED_ID.items():
                # also zero the random init in non-semantic dims for these
                # curated tokens so the signal isn't drowned by random noise.
                # (Keep dims 0/1/2 already zeroed above.)
                model.token_emb.weight[vid, :].zero_()
                for cat in _WORD2CAT.get(w, ()):
                    base = SEM_DIM_START + _CAT2DIM[cat] * N_SEM_CAT_REPEATS
                    for r in range(N_SEM_CAT_REPEATS):
                        model.token_emb.weight[vid, base + r] = SEM_DIM_VALUE

        # Iter52 control: POS_SQ_DIM left zero in pos_emb to test whether
        # its inclusion in iter51 was the real cause of the improvement.
        REG_DIM_VALUE = 6.0
        REG_PROFILE = "expdec"
        EXPDEC_RATE = 5.0
        model.pos_emb.weight.zero_()
        for j in range(T):
            model.pos_emb.weight[j, POS_DIM] = float(j)
            model.pos_emb.weight[j, BIAS_DIM] = 1.0
            if REG_PROFILE == "sq":
                v = REG_DIM_VALUE * (j * j) / float(T)
            elif REG_PROFILE == "lin":
                v = REG_DIM_VALUE * float(j) / 5.0
            elif REG_PROFILE == "cube":
                v = REG_DIM_VALUE * (j ** 3) / float(T * T)
            elif REG_PROFILE == "sin":
                v = REG_DIM_VALUE * float(T) * math.sin(math.pi * j / float(T)) / 4.0
            elif REG_PROFILE == "expdec":
                v = REG_DIM_VALUE * float(T) * (1.0 - math.exp(-EXPDEC_RATE * j / float(T))) / 4.0
            elif REG_PROFILE == "rev_sq":
                v = REG_DIM_VALUE * ((T - j) * (T - j)) / float(T)
            elif REG_PROFILE == "step":
                v = REG_DIM_VALUE * float(T) / 4.0 if j > T // 2 else 0.0
            elif REG_PROFILE == "cos":
                v = REG_DIM_VALUE * float(T) * (1.0 - math.cos(math.pi * j / float(T))) / 4.0
            else:
                v = 0.0
            model.pos_emb.weight[j, POS_SQ_DIM] = v

        for block in model.blocks:
            block.ln1.weight.fill_(1.0); block.ln1.bias.zero_()
            block.ln2.weight.fill_(1.0); block.ln2.bias.zero_()

            if USE_GAUSSIAN_ATTENTION:
                assert H == len(GAUSSIAN_TARGETS) == len(GAUSSIAN_SIGMAS)
                # Want score(target=readout, key at backward-pos j) =
                #   -(j - p_h)^2 / sigma_h^2  (softmax shift-invariant in j-indep terms)
                # q.k = sqrt(dh) * score = sqrt(dh)/sigma_h^2 * (-j^2 + 2 p_h j) + const
                # Decompose:
                #   k_h has [head_dim_0]=j (from POS_DIM), [head_dim_2]=j^2 (from
                #     POS_SQ_DIM*T scaling so coefficient is j^2 not j^2/T).
                #   q_h reads BIAS_DIM=1, putting X_h on head_dim_0 and Y_h on head_dim_2.
                #   q.k = X_h*j + Y_h*j^2.
                #   X_h = 2 * p_h * sqrt(dh)/sigma_h^2
                #   Y_h = -sqrt(dh)/sigma_h^2
                Wq = torch.zeros(D, D)
                Wk = torch.zeros(D, D)
                for h, (p_h, s_h) in enumerate(zip(GAUSSIAN_TARGETS, GAUSSIAN_SIGMAS)):
                    s2 = float(s_h * s_h)
                    X_h = 2.0 * p_h * math.sqrt(dh) / s2
                    Y_h = -math.sqrt(dh) / s2
                    Wq[h * dh + 0, BIAS_DIM] = X_h
                    Wq[h * dh + 2, BIAS_DIM] = Y_h
                    Wk[h * dh + 0, POS_DIM] = 1.0
                    # pos_emb POS_SQ_DIM stores j^2/T -> multiply by T to recover j^2.
                    Wk[h * dh + 2, POS_SQ_DIM] = float(T)
                block.attn.W_q.weight.copy_(Wq)
                block.attn.W_k.weight.copy_(Wk)
            elif USE_HYBRID_ATTENTION:
                # First half: recency-decay heads (q = (1,0,..); k = lambda*j).
                # Second half: Gaussian per-slot heads (q reads BIAS_DIM into
                # dims 0,2; k pulls POS_DIM and POS_SQ_DIM from pos_emb).
                n_rec = len(HYBRID_LAMBDAS)
                n_gau = len(HYBRID_TARGETS)
                assert H == n_rec + n_gau, "hybrid: heads must split evenly"
                Wq = torch.zeros(D, D)
                Wk = torch.zeros(D, D)
                for h, lam in enumerate(HYBRID_LAMBDAS):
                    Wq[h * dh + 0, BIAS_DIM] = 1.0
                    Wk[h * dh + 0, POS_DIM] = lam * math.sqrt(dh)
                for k, (p_h, s_h) in enumerate(zip(HYBRID_TARGETS, HYBRID_SIGMAS)):
                    h = n_rec + k
                    s2 = float(s_h * s_h)
                    X_h = 2.0 * p_h * math.sqrt(dh) / s2
                    Y_h = -math.sqrt(dh) / s2
                    Wq[h * dh + 0, BIAS_DIM] = X_h
                    Wq[h * dh + 2, BIAS_DIM] = Y_h
                    Wk[h * dh + 0, POS_DIM] = 1.0
                    Wk[h * dh + 2, POS_SQ_DIM] = float(T)
                block.attn.W_q.weight.copy_(Wq)
                block.attn.W_k.weight.copy_(Wk)
            else:
                # --- W_q: for head h, q[j] = (1, 0, ..., 0) in head h's slice.
                Wq = torch.zeros(D, D)
                for h in range(H):
                    Wq[h * dh + 0, BIAS_DIM] = 1.0
                block.attn.W_q.weight.copy_(Wq)

                # --- W_k: for head h, k[j] = (lambda_h * j * sqrt(dh), 0, ...).
                Wk = torch.zeros(D, D)
                for h, lam in enumerate(LAMBDAS):
                    Wk[h * dh + 0, POS_DIM] = lam * math.sqrt(dh)
                block.attn.W_k.weight.copy_(Wk)

            # --- W_v: per-head identity, but zero on POS_DIM/BIAS_DIM/POS_SQ_DIM
            # (so position/bias scalars are NOT pooled into the value stream).
            Wv = torch.zeros(D, D)
            for d in range(D):
                if d in (POS_DIM, BIAS_DIM, POS_SQ_DIM):
                    continue
                Wv[d, d] = 1.0
            block.attn.W_v.weight.copy_(Wv)

            # --- W_o: identity (concatenated heads -> residual stream as-is).
            block.attn.W_o.weight.copy_(torch.eye(D))

            # MLP: optionally enable random ReLU MLP for nonlinear features.
            if USE_RANDOM_MLP:
                g2 = torch.Generator().manual_seed(7)
                block.mlp.fc1.weight.normal_(mean=0.0, std=1.0/math.sqrt(D), generator=g2)
                block.mlp.fc1.bias.zero_()
                # fc2 projects d_ff -> d_model; use orthogonal-ish scaling
                block.mlp.fc2.weight.normal_(mean=0.0, std=1.0/math.sqrt(model.d_ff), generator=g2)
                block.mlp.fc2.bias.zero_()
                # Zero out projection back into position dims (residual safety).
                block.mlp.fc2.weight[POS_DIM, :] = 0.0
                block.mlp.fc2.weight[BIAS_DIM, :] = 0.0
                block.mlp.fc2.weight[POS_SQ_DIM, :] = 0.0
            else:
                block.mlp.fc1.weight.zero_(); block.mlp.fc1.bias.zero_()
                block.mlp.fc2.weight.zero_(); block.mlp.fc2.bias.zero_()

        model.final_ln.weight.fill_(1.0); model.final_ln.bias.zero_()


model_shorthand_name = "DenseSemX-V500.0"
model_description = (
    "Final best from 150-iteration session. test_corr=0.0444 (vs session "
    "start 0.0400; +11% relative). Config: d_model=1024, 8 heads, "
    "recency-decay attention (lambdas barely matter given the regularizer), "
    "15-word context, seq_len=512. Per-word features: multi-scale subword "
    "n-grams (bi/tri/4/5), unigram word identity, word-length bucket, "
    "first+last letter pair, skip-bigram word pair, suffix-stripped stem "
    "hash. Key innovations: (1) POS_SQ_DIM regularizer in pos_emb, zero'd "
    "in W_v, acts as LayerNorm regularization that dramatically reduces "
    "ridge overfitting; (2) using an exponential-saturation profile "
    "v=REG_VAL*T*(1-exp(-RATE*j/T))/4 (REG_VAL=6, RATE=5) outperforms the "
    "earlier quadratic j^2 profile by another ~1.4% by saturating the "
    "regularizer for distant positions instead of letting it grow "
    "without bound."
)


def build_embedder(device='cuda', d_model=1024, n_heads=8, n_layers=1,
                   d_ff=64, max_seq_len=512):
    model = SimpleTransformer(
        vocab_size=len(VOCAB), max_seq_len=max_seq_len,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
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
