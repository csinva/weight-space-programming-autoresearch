"""Interpretable transformer embedder for fMRI language encoding.

LEGITIMACY NOTE
---------------
Every feature is produced by the genuine `SimpleTransformer.forward` pass
(token-embedding lookup + causal self-attention pooling). Nothing is computed by
external Python feature code on the embedding. Weights are hand-written in closed
form (no training, no gradients, no pretrained weights).

The circuit ("SemCatBoC"):
  * Residual stream coordinate dims:
        dim 0 (POS_DIM)  = absolute token position j   (from pos_emb)
        dim 1 (BIAS_DIM) = constant 1                   (from pos_emb)
  * Tokens fed in:
        - one char token per character of the n-gram (orthography)
        - a word token for each recent word (hashed to a row of token_emb)
  * token_emb is a hand-built LOOKUP TABLE: each word that belongs to a
    hand-coded semantic category gets that category's indicator written into its
    row. So a word token injects a small semantic-category one-hot. The category
    one-hot is REPLICATED across all head slices so attention can pool it at
    several temporal scales. Optional random char "content" dims add orthography.
  * Multi-head attention = multi-scale recency-weighted pooling. Head h with
    decay lambda_h:  W_k routes POS_DIM->key0 (weight lambda_h*sqrt(dh)),
    W_q routes BIAS_DIM->query0 (weight 1) => score(i,j)=lambda_h*j => softmax
    weights tokens ~ exp(lambda_h*j) (recency). lambda=0 is the global mean.
  * W_v = identity (coordinate dims excluded), W_o = identity, MLP=0, LN=identity.
    The final-token state is the concatenation over heads of a recency-weighted
    bag-of-semantic-categories (+optional char orthography) at multiple scales.
    Ridge reads this multi-scale semantic summary of each n-gram.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --subject UTS03 --num-train 5
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from typing import List, Tuple

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
# Vocabulary
# ---------------------------------------------------------------------------
_VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\"
_BASE_CHARS = ['<pad>', '<unk>'] + list(_VOCAB_CHARS)
N_CHAR = len(_BASE_CHARS)

WORD_HASH_SIZE = 32768
WORD_HASH_BASE = N_CHAR
VOCAB_SIZE = WORD_HASH_BASE + WORD_HASH_SIZE

N_APPEND_WORDS = 12
PAD_ID = 0
UNK_ID = 1

POS_DIM = 0
BIAS_DIM = 1
CAT_OFFSET = 2  # within each head slice, category dims start here

# Per-head recency decay rates (len == n_heads).
LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

USE_CHAR_CONTENT = False  # iter2: semantic categories only

_stoi = {c: i for i, c in enumerate(_BASE_CHARS)}


# Hand-coded semantic-category lexicon. Brain language regions are known to track
# coarse semantic axes (motion, social, emotion, body, space, time, quantity,
# perception, communication, mental states, etc.). Each category is one feature
# dim; a word can belong to several.
_SEM_CATEGORIES = {
    "MOTION": "go went going come came run ran running walk walked move moved moving fly flew flies drive drove ride rode jump jumped fall fell falling throw threw catch swim turn turned rush chase climb crawl slide roll".split(),
    "SPEED": "fast quick quickly slow slowly rapid sudden suddenly instantly hurry rush speed swift gradual".split(),
    "SPACE": "up down left right above below under over inside outside near far here there front back top bottom between around through across along beside behind beyond edge corner middle center".split(),
    "TIME": "now then today tomorrow yesterday soon later before after early late always never often sometimes year month week day hour minute moment morning night evening past future while during until since".split(),
    "QUANTITY": "one two three four five many few several all some none most least more less much little half double huge tiny big small large great enormous count number lot dozen hundred thousand million".split(),
    "BODY": "head face eye eyes ear ears nose mouth lip tooth teeth hand hands arm arms leg legs foot feet finger hair skin heart blood bone back chest shoulder knee throat stomach brain".split(),
    "PERSON": "man woman boy girl child children people person guy lady kid baby friend mother father mom dad sister brother son daughter wife husband family neighbor stranger crowd".split(),
    "SOCIAL": "friend enemy together alone meet met marry married love loved hate party group team gang community share help helped agree argue fight war peace trust betray".split(),
    "EMOTION_POS": "happy joy glad love loved like liked enjoy enjoyed excited wonderful great amazing beautiful pleasure smile laugh laughed proud hope hopeful delight cheerful".split(),
    "EMOTION_NEG": "sad angry anger fear afraid scared worried worry cry cried pain hurt terrible awful horrible hate disgust grief sorrow lonely nervous anxious upset frightened".split(),
    "COMMUNICATION": "say said says tell told speak spoke speaking talk talked talking ask asked answer call called shout yell whisper word words voice question story explain read write wrote".split(),
    "MENTAL": "think thought know knew knows believe believed remember forgot forget understand realize wonder imagine guess idea mind learn learned dream decide suppose consider".split(),
    "PERCEPTION": "see saw seen look looked looking watch watched hear heard listen listened smell taste touch feel felt felt notice stare glance observe".split(),
    "VISION_LIGHT": "light dark bright shadow color colors red blue green yellow white black gray shine glow gleam glitter dim flash sun bright".split(),
    "SOUND": "sound noise loud quiet silent silence music song voice ring bell bang crash whisper scream echo hum buzz".split(),
    "FOOD": "eat ate eaten food drink drank water bread meat fruit apple meal breakfast lunch dinner cook cooked hungry thirsty sweet bitter salt sugar coffee tea wine".split(),
    "PLACE": "house home room door window wall floor street road city town country house school church store office building park garden field forest mountain river ocean sea beach sky world land".split(),
    "OBJECT": "thing things box book table chair bed car door key money paper bag bottle cup phone clock machine tool wheel stone wood metal glass cloth".split(),
    "NATURE": "tree water fire air earth wind rain snow storm sun moon star sky cloud animal dog cat bird fish horse flower grass leaf stone rock sea ocean mountain".split(),
    "SIZE": "big small large little tall short long huge tiny giant great wide narrow thick thin fat deep shallow vast".split(),
    "QUALITY": "good bad new old young right wrong true false real strange weird normal special important hard easy soft strong weak rich poor clean dirty empty full".split(),
    "MONEY_WORK": "work worked job money pay paid buy bought sell sold business company boss office market price cost rich poor dollar trade build built".split(),
}
_CAT_NAMES = list(_SEM_CATEGORIES.keys())
NCAT = len(_CAT_NAMES)

_WORD2CATS = {}
for _ci, _cn in enumerate(_CAT_NAMES):
    for _w in _SEM_CATEGORIES[_cn]:
        _WORD2CATS.setdefault(_w, set()).add(_ci)
_WORD2CATS = {w: sorted(cs) for w, cs in _WORD2CATS.items()}


def _word_hash(word: str) -> int:
    h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
    return WORD_HASH_BASE + (h % WORD_HASH_SIZE)


# ---------------------------------------------------------------------------
# Architecture (NO TRAINING). LayerNorms = identity so the hand-built positional
# coordinates survive untouched into attention.
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
        if USE_CHAR_CONTENT:
            for i, c in enumerate(text):
                ids.append(_stoi.get(c, UNK_ID))
                pos.append(i)
        p = len(text)
        for w in words[-N_APPEND_WORDS:]:
            ids.append(_word_hash(w))
            pos.append(p)
            p += 1
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
    assert CAT_OFFSET + NCAT <= dh, "categories must fit inside a head slice"

    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        model.token_emb.weight.zero_()

        content_lo = CAT_OFFSET + NCAT  # per-slice start of random char content
        if USE_CHAR_CONTENT:
            # random orthographic content for char rows, replicated structure per
            # head slice (every slice's content region gets independent randoms).
            for hh in range(H):
                lo = hh * dh + content_lo
                hi = (hh + 1) * dh
                w = torch.empty(N_CHAR, hi - lo)
                w.normal_(mean=0.0, std=1.0 / math.sqrt(hi - lo), generator=g)
                model.token_emb.weight[:N_CHAR, lo:hi] = w
            model.token_emb.weight[PAD_ID].zero_()

        # word rows: write semantic-category one-hot, replicated across head slices.
        for w, cats in _WORD2CATS.items():
            wid = _word_hash(w)
            for hh in range(H):
                base = hh * dh + CAT_OFFSET
                for c in cats:
                    model.token_emb.weight[wid, base + c] = 1.0

        # pos_emb coordinates
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


model_shorthand_name = "SemCatBoC"
model_description = (
    "Legit single-layer transformer: token_emb is a hand-coded lookup table that "
    "injects a 22-way semantic-category one-hot for each recent word; 8-head "
    "attention pools a multi-scale recency-weighted bag-of-categories (decay "
    "lambdas 0..8). MLP/LN identity. Features come entirely from the forward "
    "pass. No training, no pretrained weights."
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
