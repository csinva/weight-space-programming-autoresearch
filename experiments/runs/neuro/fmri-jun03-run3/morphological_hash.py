import math
import argparse
import os
import sys
import time
import hashlib
from typing import List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results
)

class MorphologicalTrigramReservoir(nn.Module):
    def __init__(self, d_model=1500, num_layers=3, max_words=128, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_words = max_words
        self.device = device
        
        self.word_cache = {}
        
        self.alphas = []
        # Progressively slower timescales
        self.alphas.append([0.0, 0.3, 0.5, 0.7])  # Layer 1: fast
        self.alphas.append([0.5, 0.7, 0.8, 0.9])  # Layer 2: medium
        self.alphas.append([0.8, 0.9, 0.95, 0.99]) # Layer 3: slow
        
        self.W_in = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)
        ])
        
        for w in self.W_in:
            nn.init.normal_(w.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
            w.weight.requires_grad = False
            
    def get_word_vector(self, word: str) -> torch.Tensor:
        word = word.strip().lower()
        if word in self.word_cache:
            return self.word_cache[word]
            
        word_padded = f"^{word}$"
        trigrams = [word_padded[i:i+3] for i in range(len(word_padded)-2)]
        
        vec = torch.zeros(self.d_model, device=self.device)
        for tg in trigrams:
            seed = int(hashlib.md5(tg.encode('utf-8')).hexdigest(), 16) % (2**32)
            rng = torch.Generator(device=self.device)
            rng.manual_seed(seed)
            vec += torch.randn(self.d_model, generator=rng, device=self.device) / math.sqrt(self.d_model)
            
        if len(trigrams) > 0:
            vec /= math.sqrt(len(trigrams))
            
        self.word_cache[word] = vec
        return vec

    def forward(self, input_strings: List[str]) -> torch.Tensor:
        # Assuming input_strings are space-separated words
        
        batch_size = len(input_strings)
        assert batch_size == 1, "Only batch size 1 supported for now"
        
        words = input_strings[0].split()
        
        seq_len = min(len(words), self.max_words)
        
        out_features = []
        
        h_states = []
        for l in range(self.num_layers):
            h_states.append([torch.zeros(self.d_model, device=self.device) for _ in self.alphas[l]])
            
        for t in range(seq_len):
            w_vec = self.get_word_vector(words[t])
            
            curr_in = w_vec
            for l in range(self.num_layers):
                v = torch.tanh(self.W_in[l](curr_in))
                
                for a_idx, alpha in enumerate(self.alphas[l]):
                    h_states[l][a_idx] = alpha * h_states[l][a_idx] + (1 - alpha) * v
                    
                curr_in = torch.stack(h_states[l]).mean(dim=0)
                
            state_concat = []
            for l in range(self.num_layers):
                for h in h_states[l]:
                    state_concat.append(h)
            
            out_features.append(torch.cat(state_concat))
            
        # Pad if needed
        while len(out_features) < self.max_words:
            out_features.append(torch.zeros_like(out_features[0]))
            
        return torch.stack(out_features).unsqueeze(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device

    model = MorphologicalTrigramReservoir(
        d_model=1500,
        num_layers=3,
        max_words=128,  # Truncate sequence logic in run_encoding handles this normally?
        device=device
    ).to(device)

    # Note: Our custom encoding wrapper might need a specific text-to-batch formulation
    # Wait, the `eval.py` expects a Huggingface-like model/tokenizer or a direct feature extractor.
    
    # Actually, we should just run word_level_rc.py and check how it works
