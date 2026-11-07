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
# These are exactly the characters that occur in the (lowercased) Huth story
# transcripts: letters, space, and apostrophe dominate; digits and the
# punctuation `-.!()[]{}\\` show up only in hyphenated/alphanumeric tokens
# (e.g. "7-up", "t-shirt", "co2") and non-speech annotation markers (e.g.
# "{cg}", "{ns}", "{ls}"). Anything unseen maps to <unk>.
_VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\"
VOCAB = ['<pad>', '<unk>'] + list(_VOCAB_CHARS)


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
        ids = [self.stoi.get(c, self.unk_id) for c in text.lower()]
        ids = ids[-self.max_seq_len:]  # keep the most recent chars (final token matters)
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
    """Populate `model`'s parameters in-place. No training allowed."""
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.zeros_(p)
    
    with torch.no_grad():
        d_model = model.d_model
        max_seq_len = model.max_seq_len
        vocab_size = model.vocab_size

        # The best model so far was WordBoundaryFeatures (0.0364) which used explicit is_space and letter identity,
        # plus attention to the recent 5 tokens, and strong random MLPs.
        # Let's expand on that: "Words" are really what language regions track.
        # Let's build a feature vector that represents the CURRENT word, and the PREVIOUS word.
        # This requires identifying spaces.
        
        # Token embedding: 
        # Dim 0: is_space
        # Dim 1: is_letter
        # Dim 2-27: letter identity
        # Dim 28-55: random projection of letter identity (for richer representation)
        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = 1.0
            elif c.isalpha():
                token_emb[i, 1] = 1.0
                token_emb[i, 2 + (ord(c) - ord('a'))] = 1.0
        # Add random projection
        rand_proj = torch.randn(vocab_size, 28)
        token_emb[:, 28:56] = rand_proj
        model.token_emb.weight.data.copy_(token_emb)

        # Positional embedding:
        # Dim 56: absolute position
        # Dim 57: exponential decay
        # Dim 58: constant 1
        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 56] = p / max_seq_len
            pos_emb[p, 57] = math.exp(p / max_seq_len)
            pos_emb[p, 58] = 1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        # Layer 1 Attention: Accumulate letters in the CURRENT word
        # We want to stop accumulation at the last space, but causal attention can't do that easily without a recurrent state.
        # Approximation: exponentially decay attention, but penalize looking past spaces.
        # Actually, if we just use a small window (e.g. 5 chars), it usually captures the current word.
        l1_attn = model.blocks[0].attn
        H = l1_attn.n_heads
        dh = l1_attn.d_head
        
        for h in range(H):
            # Q is constant, K is position -> decaying attention
            l1_attn.W_q.weight[h * dh, 58] = 5.0
            l1_attn.W_k.weight[h * dh, 56] = 1.0
            
            # Pass all features
            for i in range(dh):
                l1_attn.W_v.weight[h * dh + i, h * dh + i] = 1.0
                l1_attn.W_o.weight[h * dh + i, h * dh + i] = 1.0
                
        # MLP1: High variance mixing of current char + recent context
        mlp1 = model.blocks[0].mlp
        nn.init.normal_(mlp1.fc1.weight, std=1.0)
        nn.init.normal_(mlp1.fc1.bias, std=0.5)
        nn.init.normal_(mlp1.fc2.weight, std=1.0)
        nn.init.normal_(mlp1.fc2.bias, std=0.5)
        
        # Layer 2 Attention: Look at the last space!
        # Head 0: looks at the most recent space, fetching its features (which represent the previous word)
        l2_attn = model.blocks[1].attn
        
        # Q wants space, K is space
        l2_attn.W_q.weight[0, 58] = 100.0  # Look at recent
        l2_attn.W_k.weight[0, 56] = 1.0    # Expose pos
        l2_attn.W_k.weight[0, 0] = 50.0    # BONUS FOR SPACE
        l2_attn.W_k.weight[0, 1] = -50.0   # PENALTY FOR LETTER
        
        # Head 1-3: just random or decaying
        for h in range(1, H):
            l2_attn.W_q.weight[h * dh, 58] = 2.0
            l2_attn.W_k.weight[h * dh, 56] = 1.0
            
        # Pass features
        for i in range(d_model):
            l2_attn.W_v.weight[i, i] = 1.0
            l2_attn.W_o.weight[i, i] = 1.0
            
        # MLP2: Mix the current word representation with the previous word representation
        mlp2 = model.blocks[1].mlp
        nn.init.normal_(mlp2.fc1.weight, std=1.0)
        nn.init.normal_(mlp2.fc1.bias, std=0.5)
        nn.init.normal_(mlp2.fc2.weight, std=1.0)
        nn.init.normal_(mlp2.fc2.bias, std=0.5)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.ones_(block.ln2.weight)
                
        nn.init.ones_(model.final_ln.weight)

# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "WordAndPrevWordContext"
model_description = "L1 forms current word features. L2 attention explicitly seeks the most recent space to fetch the previous word's features. High variance MLPs."


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def build_embedder(device: str = 'cuda',
                   d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                   d_ff: int = 256, max_seq_len: int = 64) -> InterpretableEmbedder:
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
