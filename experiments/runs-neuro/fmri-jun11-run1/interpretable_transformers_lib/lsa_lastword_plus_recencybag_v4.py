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

# WORD-level vocabulary: every distinct (lowercased) word that occurs in the
# 8 training stories (2076 words). Built from the training transcripts only —
# exactly how the original char vocab was built — so test-story words that
# never occur in training map to <unk> (ridge could not have learned a weight
# for them anyway). Index 0 == pad, 1 == unk.
from src import data as _data


def _build_train_word_vocab(num_train: int = 8) -> List[str]:
    train_stories, _ = _data.get_story_names(num_train, 0)
    ws = _data.load_wordseqs(train_stories)
    words = {w.lower().strip() for s in train_stories for w in ws[s].data}
    words.discard('')
    return sorted(words)


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
        # Hand-set, NON-trained per-head recency decay. When recency_lambdas[h] > 0
        # a fixed additive bias of -lambda*(i-j) is added to head h's attention
        # logits, so (with q=k=0) head h averages the context with exponentially
        # decaying weight exp(-lambda*distance) toward the most recent word. This
        # is a fixed circuit hyperparameter, not a learned parameter.
        self.register_buffer("recency_lambdas", torch.zeros(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        if torch.any(self.recency_lambdas != 0):
            # dist[i, j] = i - j (>=0 for visible causal positions)
            pos = torch.arange(T, device=x.device)
            dist = (pos[:, None] - pos[None, :]).float()  # (T, T)
            bias = -self.recency_lambdas.view(H, 1, 1) * dist[None]  # (H, T, T)
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

SIG_DIM = 200  # half of d_model; distributional (LSA) semantic dimensionality


def _lsa_embeddings(vocab: List[str], dim: int, window: int = 5) -> np.ndarray:
    """Closed-form distributional (LSA) word vectors built from the story corpus.

    Pure linear algebra over corpus co-occurrence counts — NO gradient descent,
    NO optimizer, NO pretrained weights. Pipeline (classic, interpretable
    distributional semantics):
      1. Slide a +/-`window` word context over every (lowercased) word in ALL the
         training stories and accumulate a vocab x vocab co-occurrence matrix.
      2. Convert counts to Positive Pointwise Mutual Information (PPMI), the
         standard association measure (down-weights frequent function words).
      3. Truncated SVD of the PPMI matrix -> dense `dim`-D vectors U*sqrt(S).
         Each row is a word's meaning as "the contexts it keeps company with".
      4. L2-normalize so every word has a comparable-magnitude signature.

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
                if j == i:
                    continue
                tj = toks[j]
                if tj <= 1:
                    continue
                cooc[ti, tj] += 1.0

    # PPMI
    total = cooc.sum()
    if total == 0:
        return np.zeros((V, dim), dtype=np.float32)
    row = cooc.sum(1, keepdims=True)
    col = cooc.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log((cooc * total) / (row * col))
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


RECENCY_LAMBDA = 0.3  # exponential decay rate of context attention with word distance


def write_weights(model: SimpleTransformer) -> None:
    """LSA-semantics last-word + RECENCY-weighted bag-of-words model (v4).

    Same two-block circuit as v3, but the context average is now RECENCY-weighted
    instead of uniform: the single head's attention logits get a fixed additive
    bias of -RECENCY_LAMBDA * (distance) (set on `attn.recency_lambdas`), so a
    word `k` positions back contributes weight proportional to exp(-lambda*k).
    Motivation: the fMRI BOLD signal mostly reflects the most recent words, so
    recent context should dominate the pooled meaning.
      * lower half  = the LAST word's LSA semantic vector (residual stream)
      * upper half  = recency-weighted mean of context words' LSA vectors
    """
    d = model.d_model
    s = SIG_DIM
    assert d == 2 * s, "v4 expects d_model == 2*SIG_DIM"
    sig = _lsa_embeddings(VOCAB, s)

    with torch.no_grad():
        # token embedding: signature in lower half, zeros in upper half
        emb = np.zeros((model.vocab_size, d), dtype=np.float32)
        emb[:, :s] = sig
        model.token_emb.weight.copy_(torch.from_numpy(emb))
        model.pos_emb.weight.zero_()

        blk = model.blocks[0]
        # ln1: standard (unit gain, zero bias). Normalizes each token's vector;
        # since every token has the same [sig|0] structure the per-token scale is
        # ~constant, so this acts as a fixed rescale that z-scoring later undoes.
        blk.ln1.weight.fill_(1.0)
        blk.ln1.bias.zero_()

        # attention: recency-weighted averaging that moves the (normalized)
        # lower-half signature into the upper half. q=k=0 so the only logit signal
        # is the fixed recency bias -> weights decay as exp(-lambda*distance).
        blk.attn.W_q.weight.zero_()
        blk.attn.W_k.weight.zero_()
        blk.attn.recency_lambdas.fill_(RECENCY_LAMBDA)
        W_v = np.zeros((d, d), dtype=np.float32)
        W_v[:s, :s] = np.eye(s, dtype=np.float32)   # value = lower-half signature
        blk.attn.W_v.weight.copy_(torch.from_numpy(W_v))
        W_o = np.zeros((d, d), dtype=np.float32)
        W_o[s:, :s] = np.eye(s, dtype=np.float32)    # write averaged sig into UPPER half
        blk.attn.W_o.weight.copy_(torch.from_numpy(W_o))

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
model_shorthand_name = "lsa_lastword_plus_recencybag_v4"
model_description = ("v3 + recency: lower half = last word's LSA vector, upper half = "
                     "recency-weighted mean of context LSA vectors (attention bias "
                     "exp(-0.3*distance)). Recent words dominate the pooled context.")


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def build_embedder(device: str = 'cuda',
                   d_model: int = 400, n_heads: int = 1, n_layers: int = 1,
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
