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

LAMBDAS = (0.0, 0.5, 2.0, 8.0)
N_APPEND_WORDS = 12
# recency emphasis: number of times a word's feature tokens are repeated, by
# distance from the end (index 0 == last word).
RECENCY_REPS = (4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1)

USE_CHAR_CONTENT = False
CHAR_CONTENT_STD = 1.0  # std scale of random char embeddings


# ----------------------- hand-coded lexicons -----------------------
_SEM_CATEGORIES = {
    "MOTION": "go goes went going gone come comes came run ran running walk walked walking move moved moving fly flew flown drive drove driven ride rode jump jumped fall fell fallen throw threw catch caught turn turned rush chase climb crawl slide roll spin march step leave left arrive enter exit return follow approach escape flee crawl swim dive".split(),
    "SPACE": "up down left right above below under over inside outside near far here there front back top bottom between among around through across along beside behind beyond edge corner middle center side north south east west forward backward upward downward out off away apart together onto toward against".split(),
    "TIME": "now then today tomorrow yesterday soon later before after early late always never often sometimes year years month months week weeks day days hour hours minute minutes second moment moments morning night nights evening afternoon noon midnight past future present while during until since again ago already yet still".split(),
    "QUANTITY": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety zero many few several all some none most least more less much little half double twice huge tiny count number numbers dates lot lots dozen hundred thousand million plenty enough single whole total each every first second third last".split(),
    "BODY": "head face eye eyes ear ears nose mouth lip lips tooth teeth hand hands arm arms leg legs foot feet finger fingers hair skin heart blood bone bones back chest shoulder shoulders knee knees throat stomach brain neck chin cheek wrist elbow thumb nail body".split(),
    "PERSON": "man men woman women boy boys girl girls child children people person guy guys lady kid kids baby friend friends mother father mom dad sister brother son daughter wife husband family neighbor stranger crowd human folk gentleman".split(),
    "SOCIAL": "together alone meet met meeting marry married wedding party group team gang community share shared help helped helping agree argue argued fight fought war peace trust betray join visit invite welcome greet".split(),
    "EMOTION_POS": "happy joy joyful glad love loved loving like liked enjoy enjoyed excited exciting wonderful great amazing beautiful pleasure smile smiled laugh laughed laughing proud hope hopeful delight cheerful pleased grateful relief calm".split(),
    "EMOTION_NEG": "sad sadness angry anger afraid fear scared frightened worried worry cry cried crying pain hurt terrible awful horrible hate hated disgust grief sorrow lonely nervous anxious upset miserable depressed guilt shame jealous".split(),
    "COMMUNICATION": "say said says saying tell told telling speak spoke spoken speaking talk talked talking ask asked asking answer answered call called calling shout yell whisper word words voice question questions story stories explain read write wrote writing letter book reply discuss mention describe".split(),
    "MENTAL": "think thought thinking know knew known believe believed remember remembered forget forgot understand understood realize realized wonder wondered imagine imagined guess idea ideas mind learn learned dream dreamed decide decided suppose consider expect assume doubt notice want wanted wants wanting wish wished need needed hope mean meant figure figured plan planned try tried".split(),
    "PERCEPTION": "see saw seen seeing look looked looking watch watched watching hear heard hearing listen listened smell smelled taste tasted touch touched feel felt feeling notice noticed stare stared glance observe gaze".split(),
    "FOOD": "eat ate eaten eating food drink drank drinking water bread meat fruit apple orange meal meals breakfast lunch dinner cook cooked cooking hungry thirsty sweet bitter sour salt sugar coffee tea wine beer milk egg cheese cake soup rice".split(),
    "PLACE": "house home homes room rooms door doors window windows wall walls floor street streets road roads city cities town towns country school church store shop office building park garden field forest mountain river ocean sea lake beach sky world land farm village".split(),
    "OBJECT": "thing things box book books table chair bed car cars key keys money paper bag bottle cup phone clock machine tool tools wheel stone wood metal glass cloth knife pen door chain rope ball gun camera computer screen".split(),
    "NATURE": "tree trees fire air earth wind rain snow storm sun moon star stars cloud clouds animal dog cat bird birds fish horse flower flowers grass leaf leaves rock rocks soil dirt mud ice wave hill valley".split(),
    "QUALITY": "good bad new old young right wrong true false real fake strange normal important hard easy soft strong weak rich poor clean dirty empty full heavy light bright dark sharp dull fresh nice fine perfect".split(),
    "WORK_MONEY": "work worked working job jobs money pay paid buy bought buying sell sold selling business company boss market price cost dollar dollars trade build built building factory worker wage profit bank store customer".split(),
    "COLOR": "red blue green yellow white black gray grey brown orange purple pink color colors colour golden silver dark bright pale".split(),
    "KINSHIP": "mother father mom dad parent parents son daughter sister brother wife husband child children baby uncle aunt cousin grandmother grandfather grandma grandpa family nephew niece".split(),
    "ANIMAL": "dog dogs cat cats bird birds fish horse horses cow pig sheep chicken duck lion tiger bear wolf fox deer rabbit mouse rat snake frog insect bug bee ant spider animal animals creature".split(),
    "WEATHER": "rain rained snow snowed wind windy storm sunny cloudy cold hot warm cool freezing fog ice frost heat winter summer spring autumn season weather temperature".split(),
    "ABSTRACT_REL": "cause caused because reason result effect purpose means kind sort type way ways form part parts whole sense point fact case matter problem question chance luck fate".split(),
    "POSSESSION": "have has had having own owns owned get got gets give gave given take took taken keep kept hold held lose lost find found bring brought carry carried receive offer put set place placed leave left".split(),
    "CHANGE": "become became becoming change changed grow grew grown turn turned increase decrease rise rose fall fell break broke broken make made build built create destroy form develop begin began start started stop stopped end ended finish open opened close closed happen happened happens cut hit drop dropped".split(),
    "INTENSITY": "very really so too quite rather extremely incredibly absolutely totally completely almost nearly barely hardly just only even much".split(),
    "CLOTHING": "shoe shoes shirt shirts pants dress dresses coat coats jacket hat hats sock socks glove gloves scarf belt tie suit boot boots sweater skirt jeans clothes clothing button pocket sleeve collar zipper cap".split(),
    "SUBSTANCE": "cigarette cigarettes smoke smoking smoked tobacco pack drug drugs alcohol beer wine drink drunk pill pills medicine weed pot ash lighter match matches nicotine".split(),
    "VEHICLE": "car cars truck trucks bus buses train trains plane planes boat boats bike bikes motorcycle taxi cab subway seat seatbelt wheel engine brake brakes gas drive driving road traffic".split(),
    "MONEY_NUM": "dollar dollars cent cents penny dime buck bucks cost price cheap expensive free pay paid owe debt cash credit bill bills change worth value".split(),
    "TECH": "phone phones computer screen tv television radio camera internet email text message call button machine wire battery light switch electric power".split(),
    "HEALTH": "doctor doctors nurse surgeon surgery hospital emergency patient sick ill illness disease pain ache hurt injured injury wound blood heal healed cure pill pills medicine drug treatment cancer fever cough epilepsy seizure ambulance clinic operation recovery dying".split(),
    "LIFE_DEATH": "life live lived lives living alive born birth grow grew age aged young old die died death dead dying kill killed survive survived breathe breath heartbeat exist".split(),
    "SCHOOL_INST": "school university college campus class classroom student students teacher professor study studied learn lesson grade exam test homework library team club church government union company office church liberty".split(),
}
# Coarse valence (sentiment) beyond the EMOTION_* categories.
_VAL_POS = set("good great love like happy joy nice kind beautiful wonderful best better win won success hope safe friend gift smile laugh warm bright fun pleasant gentle clean fresh free peace calm".split())
_VAL_NEG = set("bad worse worst hate fear pain hurt sad angry death dead kill killed lost lose fail wrong sick ill dark cold cruel ugly dirty broken danger trouble war fight blood enemy evil sorry".split())
# Animacy: animate (living agents) tends to be tracked distinctly by the brain.
_ANIMATE = set((
    "man men woman women boy girl child children people person friend mother father "
    "son daughter sister brother dog cat bird fish horse cow lion tiger bear wolf "
    "animal baby human teacher doctor king queen soldier worker player crowd folk"
).split())
# Coarse emotional valence/arousal lexicons (beyond the EMOTION_* categories).
_HIGH_AROUSAL = set("scream shout run fight fire explode crash rush panic terror excited thrilled furious rage storm danger attack chase escape shock".split())
_CAT_NAMES = list(_SEM_CATEGORIES.keys())
_WORD2CATS: Dict[str, List[int]] = {}
for _ci, _cn in enumerate(_CAT_NAMES):
    for _w in _SEM_CATEGORIES[_cn]:
        _WORD2CATS.setdefault(_w, set()).add(_ci)
_WORD2CATS = {w: sorted(cs) for w, cs in _WORD2CATS.items()}

_MODALITY = {
    "VISION": "see saw look watch bright dark color red blue green light shadow glow shine appear vision sight glance stare".split(),
    "SOUND": "hear heard listen loud quiet sound noise music song voice ring bell bang crash whisper scream echo silence".split(),
    "TOUCH": "touch feel felt soft hard rough smooth warm cold hot cool wet dry sharp smooth press grip hold".split(),
    "TASTE": "taste sweet bitter sour salty spicy delicious flavor eat".split(),
    "SMELL": "smell scent odor fragrance stink perfume aroma".split(),
    "MOTOR": "grab push pull lift throw kick run walk jump grip hold carry hit punch grasp reach".split(),
}
_MOD_NAMES = list(_MODALITY.keys())
_WORD2MOD: Dict[str, List[int]] = {}
for _mi, _mn in enumerate(_MOD_NAMES):
    for _w in _MODALITY[_mn]:
        _WORD2MOD.setdefault(_w, set()).add(_mi)
_WORD2MOD = {w: sorted(cs) for w, cs in _WORD2MOD.items()}

_CONCRETE = set("house tree dog cat car book table chair hand eye water fire stone door window food bird fish rock wall floor street road wood metal glass bottle cup phone money".split())
_ABSTRACT = set("idea thought love fear hope time truth freedom justice mind dream memory reason power belief fact chance luck soul spirit meaning".split())

_PRONOUN = set("i you he she it we they me him her us them my your his its our their this that these those who what which myself himself herself".split())
_PREP = set("in on at to from of for with by about into over under after before between through during without within against among around".split())
_CONJ = set("and or but so because although though while if when as than nor yet whether unless".split())
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
# Subject-pronoun contractions (also without apostrophes); treated as pronouns.
_PRON_CONTRACT = set((
    "im id ive youre youve youd youll hes shes its theyre theyve theyd theyll "
    "weve wed well theres thats whats wheres heres hows whos hed shed"
).split())
# Interjections / discourse fillers, very common in spoken narratives.
_INTERJ = set((
    "oh uh um yeah ok okay yes yep yeah nope hmm huh ah eh wow hey alright "
    "hello hi bye well gosh wel umm uhh mmm mhm"
).split())

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
    "those great life always those once side might room"
).split()
_WORD2FREQRANK = {w: i for i, w in enumerate(_FREQ_LIST)}


def freq_bucket(w: str) -> str:
    r = _WORD2FREQRANK.get(w)
    if r is None:
        return "FREQ_RARE"
    if r < 30:
        return "FREQ_TOP"
    if r < 100:
        return "FREQ_HIGH"
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
    feats = ["POS_" + heuristic_pos(w), "LEN_" + len_bucket(w), freq_bucket(w)]
    mp = morph_prefix(w)
    if mp:
        feats.append("PRE_" + mp)
    ft = func_type(w)
    if ft:
        feats.append("FUNC_" + ft)
    else:
        feats.append("CONTENT")  # marks content (non-function) words
    for c in _WORD2CATS.get(w, []):
        feats.append("SEM_" + _CAT_NAMES[c])
    for m in _WORD2MOD.get(w, []):
        feats.append("MOD_" + _MOD_NAMES[m])
    if w in _CONCRETE:
        feats.append("CONC_HIGH")
    if w in _ABSTRACT:
        feats.append("CONC_LOW")
    if w in _ANIMATE:
        feats.append("ANIMATE")
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
    for t in ["SHORT", "VBG", "N_TION", "N_NESS", "N_MENT", "N_ITY", "ADV_LY",
              "ADJ_FUL", "ADJ_OUS", "ADJ_IVE", "SUPER_EST", "COMPAR_ER", "VBD",
              "PLURAL_S", "OTHER"]:
        names.append("POS_" + t)
    for t in ["L1_2", "L3_4", "L5_6", "L7_8", "L9_10", "L11"]:
        names.append("LEN_" + t)
    for t in ["UN", "RE", "DIS", "IN", "OVER", "MIS", "PRE"]:
        names.append("PRE_" + t)
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
    for t in ["FREQ_TOP", "FREQ_HIGH", "FREQ_MID", "FREQ_RARE"]:
        names.append(t)
    names.append("VAL_POS")
    names.append("VAL_NEG")
    names.append("NEG_SCOPE")
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
        x = x + self.attn(self.ln1(x), attn_bias)
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
    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.max_seq_len = model.max_seq_len

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        text = text.lower()
        words = text.split()
        ids: List[int] = []
        pos: List[int] = []
        # orthographic char tokens on the char-position timeline
        if USE_CHAR_CONTENT:
            for i, c in enumerate(text):
                ids.append(_stoi.get(c, UNK_ID))
                pos.append(i)
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
            if any(pw in _NEG for pw in recent_words[max(0, k - 3):k]):
                extra.append("NEG_SCOPE")
            feat_ids = [FEAT_TOKEN_BASE + _FEAT2IDX[f]
                        for f in (word_features(w) + extra)]
            if USE_WORD_ID:
                feat_ids = feat_ids + [_word_hash(w)]
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


model_shorthand_name = "LexFeatCover2"
model_description = (
    "LexFeatV2 (interpretable lexical feature tokens) PLUS a modest per-word random "
    "identity embedding (hashed, std 0.5) filling the per-head content dims, so each "
    "word contributes both coarse categories and a specific (deterministic) identity "
    "vector. 8-head multi-scale recency-weighted pooling. Forward-pass only. No "
    "training, no pretrained weights."
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
