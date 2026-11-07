import math
import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
try:
    cmu = nltk.corpus.cmudict.dict()
except:
    nltk.download('cmudict')
    cmu = nltk.corpus.cmudict.dict()

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results
)
from src import data

# Build CMU phoneme vocab
all_phones = set()
for word, prons in cmu.items():
    for pron in prons:
        all_phones.update(pron)
all_phones = sorted(list(all_phones))
VOCAB = ['<space>'] + all_phones + ['<unk>', '<pad>']

def get_phonemes(text: str) -> List[str]:
    """Convert space-separated words into a sequence of phonemes with <space> between words."""
    words = text.strip().split()
    seq = []
    for i, w in enumerate(words):
        w = "".join(c for c in w.lower() if c.isalpha())
        if w in cmu:
            seq.extend(cmu[w][0])
        else:
            # fallback: just put an unk for the word
            seq.append('<unk>')
            
        if i < len(words) - 1:
            seq.append('<space>')
    return seq

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=mask)
        x = x + attn_out
        
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

def write_weights(model: SimpleTransformer) -> None:
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.zeros_(p)
    
    with torch.no_grad():
        d_model = model.d_model
        max_seq_len = model.max_seq_len
        vocab_size = model.vocab_size
        d_ff = model.blocks[0].mlp[0].out_features
        n_heads = model.blocks[0].attn.num_heads
        d_head = d_model // n_heads

        num_nets = 14  # Decrease to 14 to fit within 1040 dims for 2 layer staggering
        dim_per_net = 68
        ff_per_net = 256
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        # Random orthogonal projection for each phoneme into the first 28 dimensions
        rng = np.random.RandomState(42)
        random_vectors = rng.randn(vocab_size, 28)
        random_vectors /= np.linalg.norm(random_vectors, axis=1, keepdims=True)
        random_vectors *= S
        
        token_emb[:, 0:28] = torch.from_numpy(random_vectors).float()
                
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        token_emb[:, 1000:1020] = token_emb[:, 0:20] # truncate slightly for the remainder dims
            
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1018] = C
            pos_emb[p, 1019] = -C
        model.pos_emb.weight.data.copy_(pos_emb)

        scales = np.linspace(15, 80, num_nets)
        
        layer = model.blocks[0]
        layer.ln1.weight.fill_(1.0)
        layer.ln2.weight.fill_(1.0)

        in_proj = torch.zeros(d_model, 3 * d_model)
        out_proj = torch.zeros(d_model, d_model)
        for net in range(num_nets):
            start = net * dim_per_net
            for i in range(28):
                in_proj[start+i, d_model + start+i] = C
                in_proj[1018, 2*d_model + start+i] = C
                
                scale = scales[net]
                out_proj[start+i, start+i+28] = 1.0 * scale
        
        layer.attn.in_proj_weight.data.copy_(in_proj.T)
        layer.attn.out_proj.weight.data.copy_(out_proj.T)

        ff1 = torch.zeros(d_model, d_ff)
        ff2 = torch.zeros(d_ff, d_model)
        for net in range(num_nets):
            start = net * dim_per_net
            f_start = net * ff_per_net
            
            for i in range(28):
                ff1[start+i+28, f_start+i] = C
                ff2[f_start+i, start+i+28] = 1.0
                
            ff1[1019, f_start+255] = C
            for i in range(28):
                ff2[f_start+255, start+i] = 1.0
                ff2[f_start+255, start+i+28] = -1.0
                
        layer.mlp[0].weight.data.copy_(ff1.T)
        layer.mlp[2].weight.data.copy_(ff2.T)

        scales_L2 = np.linspace(0.01, 14, num_nets)
        layer2 = model.blocks[1]
        layer2.ln1.weight.fill_(1.0)
        layer2.ln2.weight.fill_(1.0)
        
        in_proj2 = torch.zeros(d_model, 3 * d_model)
        out_proj2 = torch.zeros(d_model, d_model)
        for net in range(num_nets):
            start = net * dim_per_net
            for i in range(28):
                in_proj2[start+i+28, d_model + start+i] = C
                in_proj2[1018, 2*d_model + start+i] = C
                
                scale2 = scales_L2[net]
                out_proj2[start+i, start+i+28+28] = 1.0 * scale2
                
        layer2.attn.in_proj_weight.data.copy_(in_proj2.T)
        layer2.attn.out_proj.weight.data.copy_(out_proj2.T)

        ff1_2 = torch.zeros(d_model, d_ff)
        ff2_2 = torch.zeros(d_ff, d_model)
        for net in range(num_nets):
            start = net * dim_per_net
            f_start = net * ff_per_net
            
            for i in range(28):
                ff1_2[start+i+28+28, f_start+i] = C
                ff2_2[f_start+i, start+i+28+28] = 1.0
                
            ff1_2[1019, f_start+255] = C
            for i in range(28):
                ff2_2[f_start+255, start+i+28] = 1.0
                ff2_2[f_start+255, start+i+28+28] = -1.0
                
        layer2.mlp[0].weight.data.copy_(ff1_2.T)
        layer2.mlp[2].weight.data.copy_(ff2_2.T)

        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.fill_(1.18)

class InterpretableEmbedder(nn.Module):
    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        super().__init__()
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, input_strings: List[str]) -> torch.Tensor:
        # We need to process smaller chunks to avoid OOM
        # 1. First prepare input ids
        B = len(input_strings)
        T = self.model.max_seq_len
        input_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
        for i, s in enumerate(input_strings):
            phones = get_phonemes(s)
            for j, p in enumerate(phones[-T:]):
                if p in VOCAB:
                    input_ids[i, j] = VOCAB.index(p)
                else:
                    input_ids[i, j] = VOCAB.index('<unk>')
                    
        # 2. Process in mini-batches to save GPU RAM
        batch_size = 64
        outputs = []
        for i in range(0, B, batch_size):
            batch_ids = input_ids[i:i+batch_size]
            hidden_states = self.model(batch_ids)
            outputs.append(hidden_states[:, -1, :].cpu())
            
        return torch.cat(outputs, dim=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_shorthand_name = "Phonetic_Reservoir_CMUDict"
    model_description = "Uses the exact same SVD_Fixed_Char_Reservoir continuous-time exponential decay envelope that hit 0.0421, but mapped to Phonemes (CMUDict) instead of characters, providing deeper biologically grounded acoustic features."

    print(f"\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    # Use exact same architectural parameters as 0.0421 L2 reservoir
    model = SimpleTransformer(
        vocab_size=len(VOCAB),
        max_seq_len=256,
        d_model=1040,
        n_heads=13,
        n_layers=2,
        d_ff=3840
    )
    write_weights(model)
    
    embedder = InterpretableEmbedder(model, device=args.device)

    config = EncodingConfig()
    config.subject = args.subject
    config.num_train = args.num_train
    config.num_test = args.num_test

    print(f"\nExtracting features and running encoding for {model_shorthand_name}...")
    start_time = time.time()
    
    results_dict = run_encoding(embedder, config)
    mean_corr = results_dict["test_corr"]
    
    elapsed = time.time() - start_time
    print(f"\nModel: {model_shorthand_name}")
    print(f"Mean Correlation: {mean_corr:.4f}")
    print(f"Time taken: {elapsed:.1f}s")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    row = make_result_row(
        r=results_dict,
        model_shorthand_name=model_shorthand_name,
        n_params=sum(p.numel() for p in model.parameters()),
        description=model_description
    )
    upsert_overall_results([row], results_dir)
    print("Results appended.")
