"""Interpretable transformer embedder for fMRI language encoding.

LEGITIMACY NOTE
---------------
Every feature here is produced by the genuine `SimpleTransformer.forward` pass
(token-embedding lookup + causal self-attention pooling). Nothing is computed by
external Python feature code. The weights are hand-written in closed form (no
training, no gradients, no pretrained weights).

The circuit (iter1 = "RecencyBoC"):
  * The residual stream reserves two coordinate dims:
        dim 0 (POS_DIM)  = absolute token position j   (supplied by pos_emb)
        dim 1 (BIAS_DIM) = constant 1                   (supplied by pos_emb)
    All other dims carry random per-token "content" embeddings from token_emb.
  * Tokens fed to the model:
        - one char token per character of the n-gram (orthography / letters)
        - a random "word-identity hash" token appended for each recent word
          (a random bag-of-words embedding; word identity is the strongest cue)
  * Multi-head attention = multi-scale recency-weighted bag-of-tokens pooling.
    Head h has decay rate lambda_h:
        W_k routes POS_DIM  -> head h key-dim 0 with weight lambda_h * sqrt(dh)
        W_q routes BIAS_DIM -> head h query-dim 0 with weight 1
        => score(i, j) = q . k / sqrt(dh) = lambda_h * j
        => softmax over causal positions weights tokens ~ exp(lambda_h * j),
           i.e. recency-weighted. lambda=0 is a uniform mean (global bag);
           large lambda selects the last token(s) / last word.
    W_v copies each token's content into its head's value slice (POS/BIAS dims
    excluded so coordinates never leak into the pooled content). W_o = identity.
  * MLP = 0, all LayerNorms = identity (single layer). So the final-token hidden
    state is exactly the concatenation over heads of the recency-weighted
    bag-of-(chars+words), each head at a different temporal scale. Ridge then
    reads a multi-scale orthographic + word-identity summary of each n-gram.

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

# A block of random "word identity" rows. encode() hashes each recent word into
# one of these rows, so the model sees a random bag-of-words embedding pooled by
# attention. (Lookup table of random word vectors -- explicitly allowed.)
WORD_HASH_SIZE = 8192
WORD_HASH_BASE = N_CHAR
VOCAB_SIZE = WORD_HASH_BASE + WORD_HASH_SIZE

N_APPEND_WORDS = 12        # how many recent words get a word-hash token
PAD_ID = 0
UNK_ID = 1

# Reserved coordinate dims in the residual stream.
POS_DIM = 0
BIAS_DIM = 1

# Per-head recency decay rates (must have len == n_heads). 0 == global uniform
# mean; large == last-token selection. Spans many temporal scales.
LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

_stoi = {c: i for i, c in enumerate(_BASE_CHARS)}


def _word_hash(word: str) -> int:
    h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
    return WORD_HASH_BASE + (h % WORD_HASH_SIZE)


# ---------------------------------------------------------------------------
# Architecture (edited freely -- NO TRAINING). LayerNorms replaced by identity
# so the hand-built positional coordinates survive untouched into attention.
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
        scores = scores + attn_bias  # additive causal/pad mask (B,1,T,T)
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
        # additive attention bias: causal + pad positions -> -inf
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        bias = torch.zeros(B, 1, T, T, device=ids.device)
        bias = bias.masked_fill(causal[None, None], float("-inf"))
        bias = bias.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        for block in self.blocks:
            h = block(h, bias)
        return self.final_ln(h)


class InterpretableEmbedder:
    """Tokenizes each n-gram into char tokens + appended word-hash tokens, runs
    `SimpleTransformer`, and returns the final (most-recent) token hidden state."""

    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.max_seq_len = model.max_seq_len

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        text = text.lower()
        words = text.split()
        ids: List[int] = []
        pos: List[int] = []
        # char tokens (orthography of the whole n-gram)
        for i, c in enumerate(text):
            ids.append(_stoi.get(c, UNK_ID))
            pos.append(i)
        p = len(text)
        # word-identity hash tokens for the most recent words, oldest -> newest
        recent = words[-N_APPEND_WORDS:]
        for w in recent:
            ids.append(_word_hash(w))
            pos.append(p)
            p += 1
        if not ids:
            return [PAD_ID], [0]
        # keep the most recent positions if we overflow
        if len(ids) > self.max_seq_len:
            ids = ids[-self.max_seq_len:]
            pos = pos[-self.max_seq_len:]
        # clamp positions into the pos-embedding table range
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
            hidden = self.model(ids, pos_ids, pad_mask)  # (B, T, D)
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

    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        # token_emb: random Gaussian content; zero on coordinate dims; pad zero.
        model.token_emb.weight.normal_(mean=0.0, std=1.0 / math.sqrt(dh), generator=g)
        model.token_emb.weight[:, POS_DIM] = 0.0
        model.token_emb.weight[:, BIAS_DIM] = 0.0
        model.token_emb.weight[PAD_ID].zero_()

        # pos_emb: dim POS_DIM = j, dim BIAS_DIM = 1, else 0.
        model.pos_emb.weight.zero_()
        js = torch.arange(model.max_seq_len, dtype=torch.float32)
        model.pos_emb.weight[:, POS_DIM] = js
        model.pos_emb.weight[:, BIAS_DIM] = 1.0

        blk = model.blocks[0]
        attn = blk.attn
        attn.W_q.weight.zero_()
        attn.W_k.weight.zero_()
        attn.W_v.weight.zero_()
        attn.W_o.weight.zero_()
        for hh, lam in enumerate(LAMBDAS):
            base = hh * dh
            # query for head hh: dim0 reads BIAS_DIM (=1)
            attn.W_q.weight[base + 0, BIAS_DIM] = 1.0
            # key for head hh: dim0 reads POS_DIM (=j) scaled so score = lam * j
            attn.W_k.weight[base + 0, POS_DIM] = lam * math.sqrt(dh)
        # value/output: identity on content dims, exclude coordinate dims.
        eye = torch.eye(D)
        eye[POS_DIM, POS_DIM] = 0.0
        eye[BIAS_DIM, BIAS_DIM] = 0.0
        attn.W_v.weight.copy_(eye)
        attn.W_o.weight.copy_(torch.eye(D))

        # MLP = 0 (no-op this iteration).
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()
    return


model_shorthand_name = "RecencyBoC"
model_description = (
    "Legit single-layer char transformer: token_emb = random per-char + random "
    "per-word-hash vectors; 8-head attention pools a multi-scale recency-weighted "
    "bag-of-(chars+words) (decay lambdas 0..8). MLP/LN identity. Features come "
    "entirely from the forward pass. No training, no pretrained weights."
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
