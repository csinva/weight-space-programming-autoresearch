"""
Interpretable transformer embedder for fMRI language encoding.
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
from top_words import TOP_WORDS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

_VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\"
VOCAB = ['<pad>', '<unk>'] + list(_VOCAB_CHARS)

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
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int,
                 n_heads: int, n_layers: int, d_ff: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return x

class InterpretableEmbedder(nn.Module):
    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        super().__init__()
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, input_strings: List[str]) -> torch.Tensor:
        B = len(input_strings)
        T = self.model.max_seq_len
        input_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
        for i, s in enumerate(input_strings):
            for j, char in enumerate(s[-T:]):
                if char in VOCAB:
                    input_ids[i, j] = VOCAB.index(char)
                else:
                    input_ids[i, j] = VOCAB.index('<unk>')
        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :].cpu()

def write_weights(model: SimpleTransformer) -> None:
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.zeros_(p)
    
    with torch.no_grad():
        d_model = model.d_model
        max_seq_len = model.max_seq_len
        vocab_size = model.vocab_size
        d_ff = model.blocks[0].mlp.fc1.out_features
        n_heads = model.blocks[0].attn.n_heads
        d_head = model.blocks[0].attn.d_head

        num_nets = 15
        dim_per_net = 64
        ff_per_net = 256
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0 
                
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        ling_start = num_nets * dim_per_net
        
        char_to_dim = {'i': 0, 'n': 1, 'g': 2, 'e': 3, 'd': 4, 's': 5, 'l': 6, 'y': 7, 
                       'r': 8, 'o': 9, 't': 10, 'a': 11, 'c': 12, 'm': 13}
        
        for i, c in enumerate(VOCAB):
            if c in char_to_dim:
                token_emb[i, ling_start + char_to_dim[c]] = S * 1.0
                
            if c.isalpha():
                token_emb[i, ling_start + 14] = S * 1.0
            elif c == ' ':
                token_emb[i, ling_start + 15] = S * 1.0
                
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0
            
            pos_emb[p, ling_start+19] = S * (p / max_seq_len)
            
        for net in range(num_nets):
            start = net * dim_per_net
            pos_emb[:, start+28:start+30] = pos_emb[:, 28:30]
            
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        for net in range(num_nets):
            d_start = net * dim_per_net
            f_start = net * ff_per_net
            h_start = (net % n_heads) * d_head
            
            # --- LAYER 1: Extremely sharp local extraction ---
            l1_decay = 10.0 + float(net) * (70.0 / 14.0) 
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay
            
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
                
            std_dev = 0.5
            pass
            pass
            
            # --- LAYER 2: Staggered Integration ---
            l2_decay = 0.05 + float(net) * (11.95 / 14.0)
            
            l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, d_start + 28] = l2_decay
            
            # Staggered logic: 3-way split
            net_b = (net + 6) % 15
            net_c = (net + 12) % 15
            d_start_b = net_b * dim_per_net
            d_start_c = net_c * dim_per_net
            
            for i in range(22):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 1.0
                
            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            pass
            pass

        # LINGUISTIC DETECTORS (Head 9)
        h_start = 9 * d_head 
        d_start = ling_start
        
        # Layer 1
        l1_attn.W_q.weight[h_start + 0, d_start + 19] = 5.0
        l1_attn.W_k.weight[h_start + 0, d_start + 19] = 20.0
        
        for i in range(16): 
            l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
            l1_attn.W_o.weight[d_start + 20 + i, h_start + i] = S * 1.0
            
        f_start = 15 * 256
        morph_scale = S * 12.0
        
        # 'ing'
        mlp1.fc1.weight[f_start + 0, d_start + 20 + 0] = 1.0
        mlp1.fc1.weight[f_start + 0, d_start + 20 + 1] = 1.0
        mlp1.fc1.weight[f_start + 0, d_start + 20 + 2] = 1.0
        mlp1.fc1.bias[f_start + 0] = -2.0 
        mlp1.fc2.weight[d_start + 40, f_start + 0] = morph_scale
        
        # 'ed'
        mlp1.fc1.weight[f_start + 1, d_start + 20 + 3] = 1.0
        mlp1.fc1.weight[f_start + 1, d_start + 20 + 4] = 1.0
        mlp1.fc1.bias[f_start + 1] = -1.5 
        mlp1.fc2.weight[d_start + 41, f_start + 1] = morph_scale
        
        # 's' 
        mlp1.fc1.weight[f_start + 2, d_start + 20 + 5] = 1.0
        mlp1.fc1.bias[f_start + 2] = -0.5
        mlp1.fc2.weight[d_start + 42, f_start + 2] = morph_scale
        
        # 'ly'
        mlp1.fc1.weight[f_start + 3, d_start + 20 + 6] = 1.0
        mlp1.fc1.weight[f_start + 3, d_start + 20 + 7] = 1.0
        mlp1.fc1.bias[f_start + 3] = -1.5 
        mlp1.fc2.weight[d_start + 43, f_start + 3] = morph_scale
        
        # 'er'
        mlp1.fc1.weight[f_start + 4, d_start + 20 + 3] = 1.0
        mlp1.fc1.weight[f_start + 4, d_start + 20 + 8] = 1.0
        mlp1.fc1.bias[f_start + 4] = -1.5 
        mlp1.fc2.weight[d_start + 44, f_start + 4] = morph_scale
        
        # 'ion'
        mlp1.fc1.weight[f_start + 5, d_start + 20 + 0] = 1.0
        mlp1.fc1.weight[f_start + 5, d_start + 20 + 9] = 1.0
        mlp1.fc1.weight[f_start + 5, d_start + 20 + 1] = 1.0
        mlp1.fc1.bias[f_start + 5] = -2.0 
        mlp1.fc2.weight[d_start + 45, f_start + 5] = morph_scale

        # 'est'
        mlp1.fc1.weight[f_start + 6, d_start + 20 + 3] = 1.0
        mlp1.fc1.weight[f_start + 6, d_start + 20 + 5] = 1.0
        mlp1.fc1.weight[f_start + 6, d_start + 20 + 10] = 1.0
        mlp1.fc1.bias[f_start + 6] = -2.0 
        mlp1.fc2.weight[d_start + 46, f_start + 6] = morph_scale
        
        # 'al'
        mlp1.fc1.weight[f_start + 7, d_start + 20 + 11] = 1.0
        mlp1.fc1.weight[f_start + 7, d_start + 20 + 6] = 1.0
        mlp1.fc1.bias[f_start + 7] = -1.5 
        mlp1.fc2.weight[d_start + 47, f_start + 7] = morph_scale

        # 'ic'
        mlp1.fc1.weight[f_start + 8, d_start + 20 + 0] = 1.0
        mlp1.fc1.weight[f_start + 8, d_start + 20 + 12] = 1.0
        mlp1.fc1.bias[f_start + 8] = -1.5 
        mlp1.fc2.weight[d_start + 48, f_start + 8] = morph_scale

        # 'ment'
        mlp1.fc1.weight[f_start + 9, d_start + 20 + 13] = 1.0
        mlp1.fc1.weight[f_start + 9, d_start + 20 + 3] = 1.0
        mlp1.fc1.weight[f_start + 9, d_start + 20 + 1] = 1.0
        mlp1.fc1.weight[f_start + 9, d_start + 20 + 10] = 1.0
        mlp1.fc1.bias[f_start + 9] = -3.0 
        mlp1.fc2.weight[d_start + 49, f_start + 9] = morph_scale

        # Word Length approx
        mlp1.fc1.weight[f_start + 10, d_start + 20 + 14] = 1.0
        mlp1.fc1.weight[f_start + 10, d_start + 20 + 15] = -3.0
        mlp1.fc2.weight[d_start + 50, f_start + 10] = morph_scale

        # Layer 2
        l2_attn.W_q.weight[h_start + 0, d_start + 19] = 5.0
        l2_attn.W_k.weight[h_start + 0, d_start + 19] = 1.0
        
        for i in range(11): 
            l2_attn.W_v.weight[h_start + i, d_start + 40 + i] = 1.0
            l2_attn.W_o.weight[d_start + 40 + i, h_start + i] = S * 1.0

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "Interpretable_No_Noise_Ablation"
model_description = "Ablation test: removing all random normal noise from the MLP blocks. (0.0401) with 15 staggered delay networks + 11 explicitly wired morphological concept detectors. This is the finalized structural architecture."

def build_embedder(device: str = 'cuda',
                   d_model: int = 1020, n_heads: int = 10, n_layers: int = 2,
                   d_ff: int = 4000, max_seq_len: int = 64) -> InterpretableEmbedder:
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
    parser.add_argument("--num-test", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    embedder = build_embedder(device=args.device)
    
    config = EncodingConfig()
    config.subject = args.subject
    config.num_train = args.num_train
    config.num_test = args.num_test

    t0 = time.time()
    try:
        results = run_encoding(embedder, config)
        test_corr = results["test_corr"]
        print(f"Mean test correlation: {test_corr:.4f}")
        
        n_params = sum(p.numel() for p in embedder.model.parameters())
        row = make_result_row(results, model_shorthand_name, n_params, model_description, "success")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during evaluation: {e}")
        row = {
            "subject": args.subject,
            "test_corr": 0.0, "train_corr": 0.0, "frac_test_voxels_above_0.2": 0.0,
            "encoding_seconds": time.time() - t0,
            "status": "error", "model_shorthand_name": model_shorthand_name,
            "n_params": sum(p.numel() for p in embedder.model.parameters()),
            "description": model_description,
            "corrs_test_frac>0.1": 0.0, "corrs_test_frac>0.05": 0.0, "corrs_test_frac>0.0": 0.0,
            "corrs_test_median": 0.0, "corrs_test_p75": 0.0, "corrs_test_p90": 0.0, "corrs_test_p95": 0.0, "corrs_test_p99": 0.0
        }

    upsert_overall_results([row], RESULTS_DIR)
