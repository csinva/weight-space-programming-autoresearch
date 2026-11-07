"""
Interpretable transformer embedder for fMRI language encoding.

The agent edits this file. The goal: hand-write the weights of a small
character-level transformer so that the final-token hidden state it produces for
each 10-gram is a good *feature* for predicting fMRI responses to language —
ideally approaching the pretrained GPT-2 XL baseline (`src/baseline.py`).

Rules of the game (same as the sibling `evolve/` project):
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, lookup tables, etc.) inside `write_weights()`.
  * `write_weights()` runs once at construction. It must leave every parameter
    of `SimpleTransformer` populated.
  * Do NOT load pretrained weights and do NOT use external tools to compute the
    embedding — it must come from the transformer forward pass.

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
from typing import List

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

# WORD-level vocabulary built from ALL training-story transcripts (text only, no
# fMRI), keeping words that occur at least VOCAB_MIN_COUNT times. Using the full
# corpus vocab (not just the 8 ridge-fit stories) means test words that appear in
# OTHER training stories still get a real LSA semantic vector instead of <unk>;
# ridge generalizes to them through the learned semantic dimensions. This lifts
# test-token coverage from 86% (8-story vocab) to ~93%. Index 0 == pad, 1 == unk.
from collections import Counter as _Counter

from src import data as _data

VOCAB_MIN_COUNT = 3  # frequency floor (vocab ~4.5k; balances coverage vs SVD cost)


def _build_train_word_vocab() -> List[str]:
    train_stories, _ = _data.get_story_names(None, 0)  # ALL training stories
    ws = _data.load_wordseqs(train_stories)
    cnt = _Counter(w.lower().strip() for s in train_stories for w in ws[s].data)
    cnt.pop('', None)
    return sorted(w for w, c in cnt.items() if c >= VOCAB_MIN_COUNT)


VOCAB = ['<pad>', '<unk>'] + _build_train_word_vocab()


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
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
        # Hand-set, NON-trained per-head relative-position attention pattern. Head h
        # gets an additive logit bias of -sharpness[h] * | (i-j) - target_dist[h] |.
        #   * sharpness[h] large  -> the head SELECTS exactly the word target_dist[h]
        #     positions back (a "shift"/delay-line head copying that recent word).
        #   * sharpness[h] == 0    -> uniform attention over visible positions
        #     (a topic "bag-of-words" head).
        # These are fixed circuit hyperparameters, not learned parameters.
        self.register_buffer("target_dist", torch.zeros(n_heads))
        self.register_buffer("sharpness", torch.zeros(n_heads))
        # Per-head exponential recency decay: adds -recency[h]*distance to logits,
        # so head h averages context with weight exp(-recency[h]*distance). 0 == uniform.
        self.register_buffer("recency", torch.zeros(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        if torch.any(self.sharpness != 0) or torch.any(self.recency != 0):
            pos = torch.arange(T, device=x.device)
            dist = (pos[:, None] - pos[None, :]).float()  # dist[i,j] = i - j
            bias = (-self.sharpness.view(H, 1, 1) * torch.abs(dist[None] - self.target_dist.view(H, 1, 1))
                    - self.recency.view(H, 1, 1) * dist[None])  # (H, T, T)
            scores = scores + bias[None]  # broadcast over batch
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
    """Causal char-level transformer. `forward` returns hidden states (no LM head);
    the embedder reads the final-token hidden state as the encoding feature."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
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
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.final_ln(h)


class InterpretableEmbedder:
    """Tokenizes each string into characters, runs `SimpleTransformer`, and returns
    the hidden state of the final (non-pad) token. Exposes the embedder interface
    `__call__(texts) -> np.ndarray (n_texts, d_model)` used by the encoding pipeline."""

    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.stoi = {c: i for i, c in enumerate(VOCAB)}
        self.pad_id = 0
        self.unk_id = 1
        self.max_seq_len = model.max_seq_len

    def encode(self, text: str) -> List[int]:
        # word-level tokenization: one token per whitespace-separated word
        ids = [self.stoi.get(w, self.unk_id) for w in text.lower().split()]
        ids = ids[-self.max_seq_len:]  # keep the most recent words (final token matters)
        return ids if ids else [self.pad_id]

    @torch.no_grad()
    def __call__(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            enc = [self.encode(t) for t in texts[i: i + batch_size]]
            lens = [len(e) for e in enc]
            T = max(lens)
            ids = torch.full((len(enc), T), self.pad_id, dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, :len(e)] = torch.tensor(e, dtype=torch.long)
            ids = ids.to(self.device)
            hidden = self.model(ids)  # (B, T, d_model)
            last = torch.tensor([l - 1 for l in lens], device=self.device)
            emb = hidden[torch.arange(len(enc), device=self.device), last]
            embs.append(emb.float().cpu().numpy())
        return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def _lsa_embeddings(vocab: List[str], dim: int, window: int = 5,
                    cds_alpha: float = 1.0, harmonic: bool = False) -> np.ndarray:
    """Closed-form distributional (LSA) word vectors built from the story corpus.

    Pure linear algebra over corpus co-occurrence counts — NO gradient descent,
    NO optimizer, NO pretrained weights. Classic interpretable distributional
    semantics, with two well-known quality improvements over plain PPMI+SVD:
      1. Slide a +/-`window` context over every word in ALL training stories and
         accumulate a vocab x vocab co-occurrence matrix, weighting each pair by
         1/distance (GloVe-style harmonic weighting: adjacent words count most).
      2. PPMI with Context-Distribution Smoothing: the context probability is
         raised to `cds_alpha`=0.75 (the word2vec trick) which curbs PPMI's bias
         toward rare contexts and improves the resulting vectors.
      3. Truncated SVD -> dense `dim`-D vectors U*sqrt(S); L2-normalize.

    Words that never co-occur (and <pad>/<unk>) get the zero vector.
    """
    stoi = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    # corpus: every training story's word list (text only — never touches fMRI)
    train_stories, _ = _data.get_story_names(None, 0)
    ws = _data.load_wordseqs(train_stories)

    cooc = np.zeros((V, V), dtype=np.float64)
    for s in train_stories:
        toks = [stoi.get(w.lower().strip(), 1) for w in ws[s].data]  # 1 == <unk>
        n = len(toks)
        for i, ti in enumerate(toks):
            if ti <= 1:
                continue
            lo, hi = max(0, i - window), min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i or toks[j] <= 1:
                    continue
                cooc[ti, toks[j]] += (1.0 / abs(i - j)) if harmonic else 1.0

    total = cooc.sum()
    if total == 0:
        return np.zeros((V, dim), dtype=np.float32)
    row = cooc.sum(1, keepdims=True)                 # p(w)  ~ row marginal
    col = cooc.sum(0, keepdims=True)                 # p(c)  ~ col marginal
    col_smooth = np.power(col, cds_alpha)
    col_smooth *= col.sum() / col_smooth.sum()       # renormalize after smoothing
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log((cooc * total) / (row * col_smooth))
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi = np.maximum(pmi, 0.0)

    # truncated SVD -> dense vectors
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    k = min(dim, U.shape[1])
    vecs = np.zeros((V, dim), dtype=np.float32)
    vecs[:, :k] = (U[:, :k] * np.sqrt(S[:k])).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (norms + 1e-8)
    vecs[0] = 0.0  # <pad>
    vecs[1] = 0.0  # <unk>
    return vecs


def _topic_embeddings(vocab: List[str], dim: int) -> np.ndarray:
    """Closed-form term-document (classic LSA) TOPIC vectors: which stories a word
    occurs in. Complements the local-context word-word LSA with GLOBAL topical
    structure. TF-IDF weighted word x story matrix -> truncated SVD -> L2-norm.
    Pure linear algebra, no gradients, no pretrained weights, no fMRI."""
    stoi = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    train_stories, _ = _data.get_story_names(None, 0)
    ws = _data.load_wordseqs(train_stories)
    D = len(train_stories)
    from collections import Counter
    tf = np.zeros((V, D), dtype=np.float64)
    for d_idx, s in enumerate(train_stories):
        for w, c in Counter(w.lower().strip() for w in ws[s].data).items():
            if w in stoi:
                tf[stoi[w], d_idx] += c
    # tf-idf: log(1+tf) * idf
    df = (tf > 0).sum(1, keepdims=True)
    idf = np.log((D + 1) / (df + 1)) + 1.0
    tfidf = np.log1p(tf) * idf
    U, S, _ = np.linalg.svd(tfidf, full_matrices=False)
    k = min(dim, U.shape[1])
    vecs = np.zeros((V, dim), dtype=np.float32)
    vecs[:, :k] = (U[:, :k] * np.sqrt(S[:k])).astype(np.float32)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    vecs[0] = 0.0
    vecs[1] = 0.0
    return vecs


# ---------------------------------------------------------------------------
# Hand-curated, brain-relevant semantic category word lists. Each is a small
# human-written seed set for a category that maps to category-selective cortex
# or a known semantic axis the language network tracks. A word gets a 1.0 in a
# category dimension iff it is in that category's list (a transparent lexical
# lookup table). These complement the data-driven LSA vectors with human priors
# that denoise the small-corpus statistics (e.g. that hand/foot/arm are body
# parts -> EBA; kitchen/forest/city are places -> PPA/RSC).
# ---------------------------------------------------------------------------
SEMANTIC_CATEGORIES = {
    # body parts / the body -> EBA (extrastriate body area)
    "body": "head face eye eyes ear ears nose mouth lip lips tongue tooth teeth neck "
            "shoulder shoulders arm arms elbow hand hands finger fingers thumb chest "
            "back belly stomach hip leg legs knee foot feet toe toes skin hair blood "
            "bone bones heart brain body bodies",
    # places / scenes / spatial settings -> PPA, RSC
    "place": "house home room kitchen bedroom bathroom door window wall floor roof "
             "street road city town village country house building school church "
             "store shop office hospital garden yard park forest woods mountain river "
             "lake ocean sea beach field farm road bridge station airport world land "
             "place places where here there outside inside",
    # people / social roles / kinship -> social cognition
    "person": "man woman men women boy girl kid kids child children baby people person "
              "mother father mom dad parent parents brother sister son daughter "
              "husband wife friend friends family wife guy guys lady gentleman "
              "teacher doctor neighbor stranger someone everybody everyone nobody",
    # motion / action verbs -> sPMv, motor/premotor
    "motion": "go goes going went gone come comes came run runs running ran walk walks "
              "walking walked move moves moving moved jump jumps fall falls fell turn "
              "turns turned throw throws threw catch push pull lift carry ride rides "
              "drive drove fly flew swim climb climbing kick kicked grab grabbed reach",
    # visual perception -> visual cortex
    "vis_percept": "see saw seen look looks looking looked watch watched watching stare "
                   "glance notice notice see light dark bright color colors red blue "
                   "green white black shine glow shadow appear visible sight view",
    # auditory perception / sound -> AC (auditory cortex)
    "aud_percept": "hear heard hearing listen listened sound sounds noise loud quiet "
                   "silence voice voices music song sing singing scream shout yell "
                   "whisper bang ring rang echo speak speaking talk talking",
    # emotion / affect -> affective regions
    "emotion": "happy sad angry afraid fear scared scary love loved hate joy joyful "
               "cry cried crying laugh laughed smile smiled worry worried nervous "
               "excited surprised proud ashamed guilt lonely hope hopeful pain hurt "
               "anxious calm comfort terrible wonderful awful glad upset",
    # mental / cognition verbs -> language/semantic network
    "mental": "think thinks thinking thought know knows knew known believe believed "
              "remember remembered forget forgot understand understood realize "
              "imagine wonder feel feels felt mean means meant guess suppose decide "
              "decided learn learned figure idea mind reason",
    # communication / speech acts -> Broca, language network
    "communication": "say says said tell told talk talks speak spoke ask asked answer "
                     "answered call called question questions word words name named "
                     "story stories explain explained mention reply shout promise "
                     "read write writing letter book books",
    # time -> temporal/sequence processing
    "time": "time day days night week weeks month months year years morning afternoon "
            "evening today tomorrow yesterday now then later soon early late "
            "always never sometimes again hour minute moment second past future when "
            "before after while during until",
    # quantity / number -> IPS (intraparietal sulcus, number/magnitude)
    "quantity": "one two three four five six seven eight nine ten hundred thousand "
                "million number count many few more less most least some all none "
                "half double single each every both several lot lots dozen pair",
    # food / eating
    "food": "food eat eats eating ate eaten drink drank meal breakfast lunch dinner "
            "bread meat fish chicken rice egg eggs milk water coffee tea sugar salt "
            "fruit apple cake hungry taste sweet kitchen cook cooked cooking plate",
    # animals / living non-human (animacy)
    "animal": "dog dogs cat cats horse cow pig bird birds fish chicken animal animals "
              "bug bugs snake bear lion tiger mouse rat fly bee duck deer wolf sheep",
    # negation / function (negation strongly modulates meaning)
    "negation": "no not never none nothing nobody nowhere neither nor without can't "
                "don't doesn't didn't won't wouldn't couldn't shouldn't isn't aren't "
                "wasn't weren't ain't",
    # intensity / degree adverbs
    "intensity": "very really so too much more most quite pretty extremely totally "
                 "completely absolutely incredibly such just only even almost barely "
                 "hardly enough nearly",
    # money / work / economy
    "work_money": "money cash dollar dollars pay paid job jobs work works worked working "
                  "boss business company office buy bought sell sold cost price store "
                  "market bank rich poor",
    # spatial relations / prepositions of place
    "spatial": "up down left right over under above below behind front back inside "
               "outside near far between among through across around along into onto "
               "toward away top bottom side corner edge",
}


def _category_features(vocab: List[str]) -> np.ndarray:
    """Binary (V, n_categories) membership matrix from the hand-curated lists above."""
    cats = list(SEMANTIC_CATEGORIES.keys())
    feat = np.zeros((len(vocab), len(cats)), dtype=np.float32)
    stoi = {w: i for i, w in enumerate(vocab)}
    for c_idx, c in enumerate(cats):
        for w in set(SEMANTIC_CATEGORIES[c].split()):
            if w in stoi:
                feat[stoi[w], c_idx] = 1.0
    return feat


# Closed-class function words (high frequency; drive word-rate / syntactic signal).
_FUNCTION_WORDS = set((
    "the a an and or but if then so because as of to in on at by for with from into "
    "onto about over under out up down off this that these those i you he she it we "
    "they me him her us them my your his its our their mine yours is am are was were "
    "be been being do does did have has had will would can could should may might must "
    "not no yes what which who whom whose when where why how all any some each every "
    "than too very just only also even still yet there here").split())


def _scalar_features(vocab: List[str]) -> np.ndarray:
    """Per-word interpretable scalar lexical axes (z-scored over content words):
       [log word length, is-function-word flag, log corpus frequency]. These track
       word-rate / lexical-access load that the language network is sensitive to."""
    # corpus frequencies from all training-story text (no fMRI touched)
    train_stories, _ = _data.get_story_names(None, 0)
    ws = _data.load_wordseqs(train_stories)
    from collections import Counter
    cnt = Counter(w.lower().strip() for s in train_stories for w in ws[s].data)

    V = len(vocab)
    feat = np.zeros((V, 3), dtype=np.float32)
    for i, w in enumerate(vocab):
        if i < 2:  # <pad>, <unk>
            continue
        feat[i, 0] = math.log(1 + len(w))
        feat[i, 1] = 1.0 if w in _FUNCTION_WORDS else 0.0
        feat[i, 2] = math.log(1 + cnt.get(w, 0))
    # z-score columns 0 and 2 over real words (col 1 is already a clean 0/1 flag)
    real = np.arange(2, V)
    for c in (0, 2):
        col = feat[real, c]
        feat[real, c] = (col - col.mean()) / (col.std() + 1e-8)
    return feat


def _identity_features(vocab: List[str], topk: int) -> np.ndarray:
    """One-hot identity (V, topk) for the `topk` most frequent corpus words.
    Lets ridge memorize the idiosyncratic (non-semantic) BOLD response of the
    highest-frequency words, which have plenty of training samples and dominate
    the token stream. Rarer words get all-zeros and rely on the LSA semantics."""
    train_stories, _ = _data.get_story_names(None, 0)
    ws = _data.load_wordseqs(train_stories)
    from collections import Counter
    cnt = Counter(w.lower().strip() for s in train_stories for w in ws[s].data)
    stoi = {w: i for i, w in enumerate(vocab)}
    top = [w for w, _ in cnt.most_common() if w in stoi][:topk]
    feat = np.zeros((len(vocab), topk), dtype=np.float32)
    for col, w in enumerate(top):
        feat[stoi[w], col] = 1.0
    return feat


N_CATEGORIES = len(SEMANTIC_CATEGORIES)
N_SCALAR = 0  # scalar lexical axes added broad noise in v7 -> disabled
LSA_DIM = 200    # local-context word-word LSA dims (sweet spot)
TOPIC_DIM = 0    # topic LSA was ~neutral on the mean; off to isolate the vocab-expansion effect
FUNC_SALIENCE = 1.0  # 1.0 == off (down-weighting function words hurt in v13)
IDENT_TOPK = 0       # frequent-word identity overfit the all-voxel mean in v14 -> off
IDENT_SCALE = 4.0    # scale of the identity one-hot dims
CAT_SCALE = 4.0  # up-weight the sparse binary category dims so they survive LayerNorm/z-scoring
SIG_DIM = LSA_DIM + TOPIC_DIM + N_CATEGORIES + N_SCALAR + IDENT_TOPK  # per-word signature width
SELECT_SHARPNESS = 20.0       # large -> hard position selection

# Pooling heads, each writing its own output block. Temporal complexity (recency,
# delay-line, multi-scale) all underperformed the simple two-block model, so we
# keep just [last word | uniform global bag] and vary the per-word feature quality.
#   (mode, param): 'select' k -> word k back; 'recency' L -> exp(-L*dist) bag (L=0 uniform)
POOL_HEADS = [
    ('select', 0.0),    # the last word (lexical specificity)
    ('recency', 0.0),   # uniform global bag (whole-ngram topic)
]


def write_weights(model: SimpleTransformer) -> None:
    """Multi-scale temporal pooling over LSA+category signatures (v10).

    Per-word signature s = LSA(200) + 17 semantic-category flags (= 217 dims).
    A single attention layer with one head per entry of POOL_HEADS turns the last
    token's hidden state into a block-concatenation of differently-pooled views:
        [ last word | global bag | local recency bag ]
    Every head reads the same signature but pools it over the 10-gram with its own
    temporal receptive field (hard last-word selection, uniform global average, or
    exp(-0.6*distance) local average). Giving ridge several temporal scales at once
    is the ingredient the strongest hand-built linear models use.

    Construction: each head's value is the (LayerNorm'd) signature; W_o is identity
    so head h's pooled signature lands in block h.
    """
    d = model.d_model
    s = SIG_DIM
    H = model.n_heads
    dh = d // H
    assert dh == s, "v10 expects d_model == n_heads * SIG_DIM"
    assert H == len(POOL_HEADS), "n_heads must equal len(POOL_HEADS)"

    lsa = _lsa_embeddings(VOCAB, LSA_DIM)             # (V, LSA_DIM) local context
    cats = _category_features(VOCAB) * CAT_SCALE      # (V, N_CATEGORIES)
    parts = [lsa]
    if TOPIC_DIM:
        parts.append(_topic_embeddings(VOCAB, TOPIC_DIM))  # (V, TOPIC_DIM) global topic
    parts.append(cats)
    if IDENT_TOPK:
        parts.append(_identity_features(VOCAB, IDENT_TOPK) * IDENT_SCALE)
    sig = np.concatenate(parts, axis=1)               # (V, s)
    # Content-salience weighting: down-weight function words so the bag (and the
    # per-word magnitude) emphasizes content words / surprising words.
    if FUNC_SALIENCE != 1.0:
        salience = np.ones(len(VOCAB), dtype=np.float32)
        for i, w in enumerate(VOCAB):
            if w in _FUNCTION_WORDS:
                salience[i] = FUNC_SALIENCE
        sig = sig * salience[:, None]

    with torch.no_grad():
        emb = np.zeros((model.vocab_size, d), dtype=np.float32)
        emb[:, :s] = sig
        model.token_emb.weight.copy_(torch.from_numpy(emb))
        model.pos_emb.weight.zero_()

        blk = model.blocks[0]
        blk.ln1.weight.fill_(1.0)
        blk.ln1.bias.zero_()

        # q=k=0 -> the only logits come from the per-head pooling bias.
        blk.attn.W_q.weight.zero_()
        blk.attn.W_k.weight.zero_()
        target = torch.zeros(H)
        sharp = torch.zeros(H)
        recency = torch.zeros(H)
        for h, (mode, param) in enumerate(POOL_HEADS):
            if mode == 'select':
                target[h] = param
                sharp[h] = SELECT_SHARPNESS
            else:  # 'recency'
                recency[h] = param
        blk.attn.target_dist.copy_(target)
        blk.attn.sharpness.copy_(sharp)
        blk.attn.recency.copy_(recency)

        # every head's value = the signature (first s dims).
        W_v = np.zeros((d, d), dtype=np.float32)
        for h in range(H):
            W_v[h * dh:(h + 1) * dh, :s] = np.eye(s, dtype=np.float32)
        blk.attn.W_v.weight.copy_(torch.from_numpy(W_v))
        blk.attn.W_o.weight.copy_(torch.eye(d, dtype=torch.float32))

        # MLP zeroed -> no-op
        blk.ln2.weight.fill_(1.0)
        blk.ln2.bias.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.zero_()
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "lsa200_cats_bag_fullvocab_v16"
model_description = ("Best 2-block [last word | uniform bag] over LSA(200)+17cats, but vocab "
                     "expanded to the full corpus (min_count>=3, ~4.5k words) so test words from "
                     "other stories get real semantics (test coverage 86%->93%) via LSA dims.")


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def build_embedder(device: str = 'cuda',
                   d_model: int = len(POOL_HEADS) * SIG_DIM,
                   n_heads: int = len(POOL_HEADS), n_layers: int = 1,
                   d_ff: int = 4, max_seq_len: int = 16) -> InterpretableEmbedder:
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
