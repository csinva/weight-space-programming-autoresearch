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
VOCAB = ['<pad>', '<unk>'] + list(_VOCAB_CHARS)

# Word-level vocabulary used by iter5+. Each word string is mapped to a row of
# token_emb via Python's built-in hash, modulo WORD_VOCAB_SIZE (with bucket 0
# reserved as <pad>). This is hashed feature hashing: collisions occur but
# scale gracefully and require no training to assign embeddings.
WORD_VOCAB_SIZE = 8192


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
    """Tokenizes each string into WORDS (whitespace-split), hashes every word
    into one of `WORD_VOCAB_SIZE` deterministic buckets, runs `SimpleTransformer`,
    and returns the hidden state of the final (non-pad) token."""

    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.pad_id = 0
        self.unk_id = 1
        self.max_seq_len = model.max_seq_len
        self.vocab_size = model.vocab_size

    @staticmethod
    def _hash_word(w: str, vocab_size: int) -> int:
        # Deterministic stable hash via md5 (Python's builtin hash is salted).
        import hashlib
        h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
        # bucket 0 reserved for <pad>, 1 for <unk>; map words to 2..vocab_size-1
        return 2 + (h % (vocab_size - 2))

    def encode(self, text: str) -> List[int]:
        words = text.lower().split()
        if not words:
            return [self.pad_id]
        words = words[-self.max_seq_len:]  # keep most recent words (10-gram)
        return [self._hash_word(w, self.vocab_size) for w in words]

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

    Iter5 (HashedBoW): WORD-level feature hashing. Tokenizer (above) hashes
    every word in the 10-gram into one of vocab_size=8192 buckets; token_emb
    is a random Gaussian table that assigns each bucket a fixed near-orthogonal
    vector. A single-layer transformer averages over the (<=10) word vectors:

        h_T = token_emb[w_T] + LayerNorm( mean_{t<=T} token_emb[w_t] )

    The residual carries the LAST word's random embedding; attention adds the
    bag-of-words signature of the 10-gram. With d_model=256 >> 10, both
    components are near-orthogonal, so ridge can recover (i) which word is at
    the final position and (ii) which words occurred recently -- the same
    representational structure GPT-2 XL exploits, just with random instead of
    learned word vectors.

    Construction matches iter1's BoCharMean circuit, only the *tokenization*
    moved from chars to hashed words:
      * token_emb : random Gaussian (std=1/sqrt(d_model)); <pad> row zeroed.
      * pos_emb   : zero (no positional info needed for BoW).
      * attn      : Q=K=0 (uniform causal mean), V=O=I.
      * MLP       : zero.
      * Every LayerNorm = identity.
    """
    D = model.d_model
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        model.token_emb.weight.normal_(mean=0.0, std=1.0 / math.sqrt(D), generator=g)
        model.token_emb.weight[0].zero_()  # <pad>
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
model_shorthand_name = "HashedBoW-d64"
model_description = (
    "Same word-hashing BoW circuit as HashedBoW-d256, narrowed to d_model=64 "
    "(64*4-delay=256 ridge features) to combat the heavy overfitting we saw "
    "with d=256 (train 0.47 vs test 0.027). 1 layer, uniform causal mean."
)


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def build_embedder(device: str = 'cuda',
                   d_model: int = 64, n_heads: int = 4, n_layers: int = 1,
                   d_ff: int = 64, max_seq_len: int = 10) -> InterpretableEmbedder:
    model = SimpleTransformer(
        vocab_size=WORD_VOCAB_SIZE, max_seq_len=max_seq_len,
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
