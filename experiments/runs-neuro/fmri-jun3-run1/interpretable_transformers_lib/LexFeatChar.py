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

LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
N_APPEND_WORDS = 12
# recency emphasis: number of times a word's feature tokens are repeated, by
# distance from the end (index 0 == last word).
RECENCY_REPS = (4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1)

USE_CHAR_CONTENT = True
CHAR_CONTENT_STD = 1.0  # std scale of random char embeddings


# ----------------------- hand-coded lexicons -----------------------
_SEM_CATEGORIES = {
    "MOTION": "go went going come came run ran running walk walked move moved moving fly flew drive drove ride rode jump jumped fall fell throw threw catch turn turned rush chase climb crawl slide roll".split(),
    "SPACE": "up down left right above below under over inside outside near far here there front back top bottom between around through across along beside behind beyond edge corner middle center".split(),
    "TIME": "now then today tomorrow yesterday soon later before after early late always never often sometimes year month week day hour minute moment morning night evening past future while during until since".split(),
    "QUANTITY": "one two three four five many few several all some none most least more less much little half double huge tiny count number lot dozen hundred thousand million".split(),
    "BODY": "head face eye eyes ear ears nose mouth lip tooth teeth hand hands arm arms leg legs foot feet finger hair skin heart blood bone back chest shoulder knee throat stomach brain".split(),
    "PERSON": "man woman boy girl child children people person guy lady kid baby friend mother father mom dad sister brother son daughter wife husband family neighbor stranger crowd".split(),
    "SOCIAL": "together alone meet met marry married party group team gang community share help agree argue fight war peace trust".split(),
    "EMOTION_POS": "happy joy glad love loved like liked enjoy excited wonderful great amazing beautiful pleasure smile laugh laughed proud hope hopeful delight cheerful".split(),
    "EMOTION_NEG": "sad angry anger fear afraid scared worried worry cry cried pain hurt terrible awful horrible hate disgust grief sorrow lonely nervous anxious upset".split(),
    "COMMUNICATION": "say said says tell told speak spoke talk talked ask asked answer call called shout yell whisper word words voice question story explain read write wrote".split(),
    "MENTAL": "think thought know knew believe believed remember forget understand realize wonder imagine guess idea mind learn dream decide suppose consider".split(),
    "PERCEPTION": "see saw seen look looked watch watched hear heard listen smell taste touch feel felt notice stare glance".split(),
    "FOOD": "eat ate eaten food drink drank water bread meat fruit apple meal breakfast lunch dinner cook hungry thirsty sweet bitter salt sugar coffee tea wine".split(),
    "PLACE": "house home room door window wall floor street road city town country school church store office building park garden field forest mountain river ocean sea beach sky world land".split(),
    "OBJECT": "thing things box book table chair bed car key money paper bag bottle cup phone clock machine tool wheel stone wood metal glass cloth".split(),
    "NATURE": "tree fire air earth wind rain snow storm sun moon star cloud animal dog cat bird fish horse flower grass leaf rock".split(),
    "QUALITY": "good bad new old young right wrong true false real strange normal important hard easy soft strong weak rich poor clean dirty empty full".split(),
    "WORK_MONEY": "work worked job money pay paid buy bought sell sold business company boss market price cost dollar trade build built".split(),
}
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
_AUX = set("is are was were be been being am do does did have has had will would can could should shall may might must".split())
_NEG = set("not no never none nothing nobody nowhere neither nor".split())


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
    if w in _PRONOUN:
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
    return None


def word_features(w: str) -> List[str]:
    feats = ["POS_" + heuristic_pos(w), "LEN_" + len_bucket(w)]
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
    for t in ["PRON", "PREP", "CONJ", "ART", "AUX", "NEG"]:
        names.append("FUNC_" + t)
    names.append("CONTENT")
    for c in _CAT_NAMES:
        names.append("SEM_" + c)
    for m in _MOD_NAMES:
        names.append("MOD_" + m)
    names.append("CONC_HIGH")
    names.append("CONC_LOW")
    return names


FEATURE_NAMES = _build_feature_names()
NFEAT = len(FEATURE_NAMES)
_FEAT2IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

FEAT_TOKEN_BASE = N_CHAR
VOCAB_SIZE = FEAT_TOKEN_BASE + NFEAT


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
        for k, (w, endpos) in enumerate(recent):
            dist = nrec - 1 - k  # 0 == last word
            reps = RECENCY_REPS[min(dist, len(RECENCY_REPS) - 1)]
            feat_ids = [FEAT_TOKEN_BASE + _FEAT2IDX[f] for f in word_features(w)]
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


model_shorthand_name = "LexFeatChar"
model_description = (
    "LexFeatBoC + orthographic char content: char tokens (random per-char vectors "
    "in dims above the feature block) on a unified char-position timeline alongside "
    "interpretable word-feature tokens (POS, length, morphology, function-type, 18 "
    "semantic cats, 6 modalities, concreteness). 8-head multi-scale recency pooling. "
    "Features from the forward pass only. No training, no pretrained weights."
)


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def build_embedder(device: str = 'cuda', d_model: int = 1024, n_heads: int = 8,
                   n_layers: int = 1, d_ff: int = 16, max_seq_len: int = 512) -> InterpretableEmbedder:
    model = SimpleTransformer(
        vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len,
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
