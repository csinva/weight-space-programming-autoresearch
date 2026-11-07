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
VOCAB = []
for b in range(N_WORD_BUCKETS):
    VOCAB += [f"#{b}:{c}" for c in _BASE_CHAR_VOCAB]

# Two reserved d_model dims used to inject position info into attention.
POS_DIM = 0   # holds the scalar j (token position)
BIAS_DIM = 1  # holds the constant 1


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
                 n_heads=4, n_layers=2, d_ff=256):
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
    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.stoi = {c: i for i, c in enumerate(_BASE_CHAR_VOCAB)}
        self.pad_id = 0
        self.unk_id = 1
        self.max_seq_len = model.max_seq_len

    def encode(self, text: str) -> List[int]:
        text = text.lower()
        words = text.split()
        if not words:
            return [self.pad_id]
        # Locate each word's char span.
        spans = []
        cursor = 0
        for w in words:
            start = text.find(w, cursor)
            spans.append((start, start + len(w)))
            cursor = start + len(w)
        # Default bucket = OLDEST (N_WORD_BUCKETS-1) for whitespace / older.
        bucket_of = [N_WORD_BUCKETS - 1] * len(text)
        for back_idx in range(min(N_WORD_BUCKETS - 1, len(spans))):
            s, e = spans[-1 - back_idx]
            for i in range(s, e):
                bucket_of[i] = back_idx  # 0 = last word, 1 = 2nd-last, ...
        ids = []
        for i, c in enumerate(text):
            base = self.stoi.get(c, self.unk_id)
            ids.append(base + bucket_of[i] * BUCKET_SIZE)
        ids = ids[-self.max_seq_len:]
        return ids if ids else [self.pad_id]

    @torch.no_grad()
    def __call__(self, texts, batch_size=256):
        embs = []
        for i in range(0, len(texts), batch_size):
            enc = [self.encode(t) for t in texts[i:i + batch_size]]
            lens = [len(e) for e in enc]
            T = max(lens)
            ids = torch.full((len(enc), T), self.pad_id, dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, :len(e)] = torch.tensor(e, dtype=torch.long)
            ids = ids.to(self.device)
            hidden = self.model(ids)
            last = torch.tensor([l - 1 for l in lens], device=self.device)
            emb = hidden[torch.arange(len(enc), device=self.device), last]
            embs.append(emb.float().cpu().numpy())
        return np.concatenate(embs, axis=0)


# Decay rates per head: 0 = uniform, larger = sharper recency emphasis.
LAMBDAS = (0.0, 0.08, 0.15, 0.25, 0.4, 0.6, 0.85, 1.2, 1.7, 2.3, 3.0, 4.0, 5.5, 7.0, 9.0, 12.0)


def write_weights(model: SimpleTransformer) -> None:
    D = model.d_model
    H = model.n_heads
    dh = D // H
    T = model.max_seq_len
    assert H == len(LAMBDAS), "n_heads must match number of lambdas"

    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        # Token emb: random Gaussian, zero on POS_DIM and BIAS_DIM, pad row zero.
        model.token_emb.weight.normal_(mean=0.0, std=1.0 / math.sqrt(D), generator=g)
        model.token_emb.weight[:, POS_DIM] = 0.0
        model.token_emb.weight[:, BIAS_DIM] = 0.0
        # Zero out every <pad> row (one per bucket).
        for b in range(N_WORD_BUCKETS):
            model.token_emb.weight[b * BUCKET_SIZE].zero_()

        # Pos emb: dim POS_DIM = j (raw position), dim BIAS_DIM = 1, rest 0.
        model.pos_emb.weight.zero_()
        for j in range(T):
            model.pos_emb.weight[j, POS_DIM] = float(j)
            model.pos_emb.weight[j, BIAS_DIM] = 1.0

        for block in model.blocks:
            block.ln1.weight.fill_(1.0); block.ln1.bias.zero_()
            block.ln2.weight.fill_(1.0); block.ln2.bias.zero_()

            # --- W_q: for head h, q[j] = (1, 0, ..., 0) in head h's slice.
            # Need input dim BIAS_DIM -> output dim (h*dh + 0) with weight 1.
            Wq = torch.zeros(D, D)
            for h in range(H):
                Wq[h * dh + 0, BIAS_DIM] = 1.0
            block.attn.W_q.weight.copy_(Wq)

            # --- W_k: for head h, k[j] = (lambda_h * j * sqrt(dh), 0, ...).
            # input dim POS_DIM -> output dim (h*dh + 0) with weight lambda_h*sqrt(dh).
            # Combined with 1/sqrt(dh) scaling -> score = lambda_h * j.
            Wk = torch.zeros(D, D)
            for h, lam in enumerate(LAMBDAS):
                Wk[h * dh + 0, POS_DIM] = lam * math.sqrt(dh)
            block.attn.W_k.weight.copy_(Wk)

            # --- W_v: per-head identity, but zero on POS_DIM and BIAS_DIM
            # (so position/bias scalars are NOT pooled into the value stream).
            # i.e., V[:, head h slice] = x[:, head h slice] except dims POS_DIM/BIAS_DIM -> 0.
            Wv = torch.zeros(D, D)
            for d in range(D):
                if d == POS_DIM or d == BIAS_DIM:
                    continue
                Wv[d, d] = 1.0
            block.attn.W_v.weight.copy_(Wv)

            # --- W_o: identity (concatenated heads -> residual stream as-is).
            block.attn.W_o.weight.copy_(torch.eye(D))

            # MLP off.
            block.mlp.fc1.weight.zero_(); block.mlp.fc1.bias.zero_()
            block.mlp.fc2.weight.zero_(); block.mlp.fc2.bias.zero_()

        model.final_ln.weight.fill_(1.0); model.final_ln.bias.zero_()


model_shorthand_name = "RecencyDecayBoC-16h+LastWordBucket"
model_description = (
    "iter13 widened to 16 recency-decay heads (lambdas densely spaced 0..12) "
    "and d_model=1024. 2 word-position buckets (last word vs prior). 1 layer, "
    "MLP off. Doubles ridge feature dim for finer multi-scale BoC."
)


def build_embedder(device='cuda', d_model=1024, n_heads=16, n_layers=1,
                   d_ff=64, max_seq_len=64):
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
