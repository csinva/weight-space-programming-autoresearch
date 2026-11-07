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

SIG_DIM = 1024  # half of d_model; per-word random signature dimensionality


def _word_signatures(vocab_size: int, sig_dim: int) -> np.ndarray:
    """Fixed deterministic unit-norm random signature per word (rows near-orthogonal)."""
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((vocab_size, sig_dim)).astype(np.float32)
    sig /= (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-8)
    sig[0] = 0.0  # <pad>
    return sig


def write_weights(model: SimpleTransformer) -> None:
    """Last-word + bag-of-words model (v2).

    The hidden state is split into two interpretable halves of `SIG_DIM` dims:
      * lower half  = the LAST word's identity signature (from the residual stream)
      * upper half  = the BAG-OF-WORDS: the uniform average of every word's
                      signature across the 10-gram (built by one attention head).

    Circuit (1 layer, 1 head, MLP zeroed):
      - token_emb[w] = [signature_w (SIG_DIM) | zeros (SIG_DIM)]; the residual
        stream therefore carries the last word's raw signature in the lower half.
      - attention query/key are zero so the softmax is UNIFORM over all visible
        (causal) positions -> the last position averages every word in the ngram.
      - W_v copies the lower-half signature; W_o writes that average into the
        UPPER half. So after the residual add: [last word | mean over context].
    Keeping the two signals in disjoint dimension blocks means they never
    interfere, and the per-dimension z-scoring downstream fixes their scales.
    Ridge then reads both "which word just occurred" and "what words are in the
    surrounding context" — a bag-of-words-in-context encoder.
    """
    d = model.d_model
    s = SIG_DIM
    assert d == 2 * s, "v2 expects d_model == 2*SIG_DIM"
    sig = _word_signatures(model.vocab_size, s)

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

        # attention: uniform averaging that moves the (normalized) lower-half
        # signature into the upper half.
        blk.attn.W_q.weight.zero_()   # q == 0  -> all scores equal -> uniform softmax
        blk.attn.W_k.weight.zero_()
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
model_shorthand_name = "lastword_plus_bagofwords_v2"
model_description = ("1-layer/1-head circuit: lower half of the hidden state = last word's "
                     "random signature, upper half = uniform bag-of-words average over the "
                     "10-gram (q=k=0 -> uniform attention). Adds context to the v1 lookup.")


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def build_embedder(device: str = 'cuda',
                   d_model: int = 2048, n_heads: int = 1, n_layers: int = 1,
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
