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

# Character vocabulary the embedder tokenizes over (index 0 == pad, 1 == unk).
# Kept for compatibility / fallback; iter5 onward switches to a WORD-level
# tokenization where each input 10-gram is split on whitespace and every word
# is hashed into one of WORD_VOCAB_SIZE deterministic random embedding rows.
_VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\"
_BASE_CHAR_VOCAB = ['<pad>', '<unk>'] + list(_VOCAB_CHARS)

# Iter9: tag each char with which word-from-end it belongs to. Bucket 0 = last
# word, bucket 1 = 2nd-to-last word, bucket 2 = 3rd-to-last word, bucket 3 =
# everything older (or for chars that aren't in any word). Each bucket uses
# its own random embedding row per char, so the uniform-mean attention
# decomposes the bag-of-chars into per-position-from-end sub-bags in disjoint
# random subspaces of the hidden state.
N_WORD_BUCKETS = 3
BUCKET_SIZE = len(_BASE_CHAR_VOCAB)
LAST_WORD_OFFSET = BUCKET_SIZE  # kept for backward-compat with iter8 snapshot
VOCAB = []
for b in range(N_WORD_BUCKETS):
    VOCAB += [f"#{b}:{c}" for c in _BASE_CHAR_VOCAB]
# Total vocab = N_WORD_BUCKETS * BUCKET_SIZE

WORD_VOCAB_SIZE = 8192  # unused after iter5


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
    the hidden state of the final (non-pad) token."""

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
        # Find each word's [start, end) span in `text` (whitespace-delimited).
        spans = []
        cursor = 0
        for w in words:
            start = text.find(w, cursor)
            spans.append((start, start + len(w)))
            cursor = start + len(w)
        # Map char-index -> bucket (which word-from-end it belongs to, or the
        # "older" bucket if outside the last few words / inside whitespace).
        bucket_of = [N_WORD_BUCKETS - 1] * len(text)
        for back_idx in range(min(N_WORD_BUCKETS - 1, len(spans))):
            b = back_idx  # 0 = last word, 1 = 2nd-last, ...
            s, e = spans[-1 - back_idx]
            for i in range(s, e):
                bucket_of[i] = b
        ids = []
        for i, c in enumerate(text):
            base = self.stoi.get(c, self.unk_id)
            ids.append(base + bucket_of[i] * BUCKET_SIZE)
        ids = ids[-self.max_seq_len:]
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

def write_weights(model: SimpleTransformer) -> None:
    """Populate `model`'s parameters in-place. No training.

    Iter9 (WordPosTaggedBoC-4buckets): same BoC circuit as iter1/8, but each
    char is tagged with which word-from-end it belongs to (4 buckets: last
    word, 2nd-last, 3rd-last, older / whitespace). Token_emb gives every
    (char, bucket) pair its own random Gaussian row, so the uniform-mean
    attention decomposes the bag-of-chars into four word-position-specific
    sub-bags in (nearly) orthogonal random subspaces of the hidden state.
    Ridge thus sees per-word-position char histograms -- a far stronger
    word-identity signal than a single global BoC.

    Construction (single layer, identical to iter1 modulo larger vocab):
      * token_emb : random Gaussian (std=1/sqrt(d_model)); pad rows zeroed.
      * pos_emb   : zero.
      * attn      : Q=K=0 (uniform causal mean), V=O=I.
      * MLP       : zero.
      * Every LayerNorm = identity.
    """
    D = model.d_model
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        model.token_emb.weight.normal_(mean=0.0, std=1.0 / math.sqrt(D), generator=g)
        # Zero out every <pad> row (one per bucket).
        for b in range(N_WORD_BUCKETS):
            model.token_emb.weight[b * BUCKET_SIZE].zero_()
        model.pos_emb.weight.zero_()

        for block in model.blocks:
            block.ln1.weight.fill_(1.0); block.ln1.bias.zero_()
            block.ln2.weight.fill_(1.0); block.ln2.bias.zero_()
            block.attn.W_q.weight.zero_()
            block.attn.W_k.weight.zero_()
            block.attn.W_v.weight.copy_(torch.eye(D))
            block.attn.W_o.weight.copy_(torch.eye(D))
            block.mlp.fc1.weight.zero_(); block.mlp.fc1.bias.zero_()
            block.mlp.fc2.weight.zero_(); block.mlp.fc2.bias.zero_()

        model.final_ln.weight.fill_(1.0); model.final_ln.bias.zero_()


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "WordPosTaggedBoC-3buckets"
model_description = (
    "Same word-position-tagged BoC as iter9 but with N_WORD_BUCKETS=3 "
    "(last word / 2nd-last word / older + whitespace). iter8 (2 buckets) was "
    "best so far at 0.0309; iter9 (4 buckets) diluted the signal to 0.0297; "
    "try 3 buckets to see if there's a sweet spot."
)


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def build_embedder(device: str = 'cuda',
                   d_model: int = 256, n_heads: int = 4, n_layers: int = 1,
                   d_ff: int = 64, max_seq_len: int = 64) -> InterpretableEmbedder:
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
