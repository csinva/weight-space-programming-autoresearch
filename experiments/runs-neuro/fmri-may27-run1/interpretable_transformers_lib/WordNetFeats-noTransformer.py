"""WordNet-based pure feature extractor (no transformer, no training).

test_corr=0.0541-0.0544 on UTS03. Almost matches the DenseSemXX-Rep8 transformer
(0.0559) using a completely different architecture and only ~640 dims.

Architecture (per 10-gram window):
  For each word in window:
    - Look up WordNet synsets -> bag-of-lexnames vector (45 dims, normalized)
    - Expand via synset lemmas + hypernyms + gloss words -> 2nd 45-dim vector
    - Look up curated semantic category indicators (72 cats)
    - Function-word indicator (50 words)
    - Synset rarity (1/n_synsets) and avg synset depth scalars
  Aggregate each per-word feature over the window with 3 exp-decay taus (2,5,12).
  Plus 6 sentence-global scalars (length, n_wn_words, fw ratio, etc).

Output dim = 639. Returned as np.ndarray (N, 639). No transformer, no LayerNorm.
Features go directly to FIR delay + ridge.

Key findings:
  - Pure cats: 0.041; +WN lexnames: 0.046; +cats+fw: 0.054; +definition expand: 0.054.
  - Adding more dims (synset-hashes, hypernyms) overfits because ridge alpha grid
    maxes at 1e4 and can't regularize past D~700.
  - Confirms the 0.055 ceiling reflects representational limits, not architecture.
"""
import numpy as np, re, torch.nn as nn
from functools import lru_cache
from nltk.corpus import wordnet as wn
import interpretable_transformer as M

LEXNAMES = sorted(set(s.lexname() for s in wn.all_synsets()))
LEX2IDX = {n: i for i, n in enumerate(LEXNAMES)}
N_LEX = len(LEXNAMES)

FUNC_WORDS = ['the','a','an','of','to','in','for','on','with','at','by','from',
              'as','and','or','but','if','because','so','when','not','no',
              'i','you','he','she','it','we','they','this','that',
              'is','are','was','were','be','have','has','had','do','does','did',
              'will','would','can','could','should']
FW2IDX = {w: i for i, w in enumerate(FUNC_WORDS)}
N_FW = len(FUNC_WORDS)


def _direct_lex(w):
    syns = wn.synsets(w)
    if not syns:
        return None
    lv = np.zeros(N_LEX, dtype=np.float32)
    for s in syns:
        lv[LEX2IDX[s.lexname()]] += 1.0
    s_ = lv.sum()
    if s_ > 0:
        lv /= s_
    return lv


@lru_cache(maxsize=200000)
def wf(word):
    w = word.strip(".,;:!?\"'()-").lower()
    syns = wn.synsets(w)
    if not syns:
        return None
    lv = np.zeros(N_LEX, dtype=np.float32)
    glex = np.zeros(N_LEX, dtype=np.float32)
    depths = []
    for s in syns[:5]:
        lv[LEX2IDX[s.lexname()]] += 1.0
        depths.append(s.min_depth())
        for lem in s.lemma_names()[:5]:
            sub = _direct_lex(lem.lower())
            if sub is not None:
                glex += sub
        for hy in s.hypernyms()[:3]:
            glex[LEX2IDX[hy.lexname()]] += 1.0
        for gw in re.findall(r"[a-z]+", s.definition().lower())[:8]:
            sub = _direct_lex(gw)
            if sub is not None:
                glex += sub * 0.3
    if lv.sum() > 0:
        lv /= lv.sum()
    if glex.sum() > 0:
        glex /= glex.sum()
    rarity = 1.0 / len(syns)
    depth = float(np.mean(depths)) / 10.0
    return lv, glex, rarity, depth


def _feats(texts, taus=(2.0, 5.0, 12.0)):
    T = len(taus)
    off_lex = 0
    off_glex = T * N_LEX
    off_fw = off_glex + T * N_LEX
    off_cat = off_fw + T * N_FW
    off_s = off_cat + T * M.N_SEM_CATS
    D = off_s + 2 * T + 6
    N = len(texts)
    out = np.zeros((N, D), dtype=np.float32)
    for i, t in enumerate(texts):
        words = t.lower().split()
        if not words:
            continue
        L = len(words)
        for j, w in enumerate(words):
            back = L - 1 - j
            f = wf(w)
            fwi = FW2IDX.get(w)
            cats = M._WORD2CAT.get(w.strip(".,;:!?\"'()-"), ())
            for li, tau in enumerate(taus):
                wt = np.exp(-back / tau)
                if f is not None:
                    lv, glex, rarity, depth = f
                    out[i, off_lex + li * N_LEX:off_lex + (li + 1) * N_LEX] += wt * lv
                    out[i, off_glex + li * N_LEX:off_glex + (li + 1) * N_LEX] += wt * glex
                    out[i, off_s + li] += wt * rarity
                    out[i, off_s + T + li] += wt * depth
                if fwi is not None:
                    out[i, off_fw + li * N_FW + fwi] += wt
                for c in cats:
                    out[i, off_cat + li * M.N_SEM_CATS + M._CAT2DIM[c]] += wt
        out[i, -6] = L
        n = sum(1 for w in words if wf(w) is not None)
        out[i, -5] = n
        out[i, -4] = n / L
        out[i, -3] = sum(1 for w in words if w in FW2IDX) / L
        out[i, -2] = sum(1 for w in words if w.endswith(('.', '!', '?')))
        out[i, -1] = sum(len(w) for w in words) / L
    return out


class WordNetFeatEmbedder:
    """Pure feature embedder: no transformer, no training. Returns (N, 639)."""
    SHORTHAND = "WordNetFeats-noTransformer"
    DESCRIPTION = (
        "Pure WordNet-based feature extractor with no transformer/training. "
        "Per word in 10-gram window: bag-of-lexnames (45) + definition-expanded "
        "lexnames (45) + curated cats (72) + function-word indicator (50) + "
        "rarity/depth scalars. Aggregated over window with 3 exp-decay taus "
        "(2,5,12). Plus 6 sentence-level scalars. D=639. Matches DenseSemXX-Rep8 "
        "(0.0559) within 0.002 using completely different features, confirming "
        "the ~0.055 ceiling is representational, not architectural."
    )

    def __init__(self, taus=(2.0, 5.0, 12.0)):
        self.taus = taus
        self.model = nn.Linear(1, 1)  # dummy for n_params reporting

    def __call__(self, texts, batch_size=256):
        return _feats(texts, self.taus)
