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
from src import data

class MorphologicalReservoir(nn.Module):
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
            
        self.to(self.device)
            
    def get_word_vector(self, word: str) -> torch.Tensor:
        word = word.strip().lower()
        if word in self.word_cache:
            return self.word_cache[word]
            
        word_padded = f"^{word}$"
        trigrams = [word_padded[i:i+3] for i in range(max(1, len(word_padded)-2))]
        
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
        batch_size = len(input_strings)
        
        # Determine maximum sequence length in this batch (for words)
        words_batch = [s.split() for s in input_strings]
        seq_len = min(self.max_words, max(len(w) for w in words_batch))
        if seq_len == 0:
            return torch.zeros(batch_size, self.d_model * sum(len(a) for a in self.alphas)).numpy()
            
        out_batch = []
        
        for b in range(batch_size):
            words = words_batch[b][:seq_len]
            
            h_states = []
            for l in range(self.num_layers):
                h_states.append([torch.zeros(self.d_model, device=self.device) for _ in self.alphas[l]])
                
            for t in range(len(words)):
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
            
            out_batch.append(torch.cat(state_concat))
            
        return torch.stack(out_batch).cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_shorthand_name = "Morphological_Trigram_Reservoir"
    model_description = "A 3-layer continuous-time reservoir computer. Words are mapped to embeddings via structural hashing of their character trigrams (capturing orthographic/morphological similarity without semantic weights). Layers have progressively slower exponential smoothing."

    print(f"\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    embedder = MorphologicalReservoir(d_model=1500, num_layers=3, max_words=128, device=args.device)
    
    config = EncodingConfig()
    config.subject = args.subject
    config.num_train = args.num_train
    config.num_test = args.num_test

    print(f"\nExtracting features and running encoding for {model_shorthand_name}...")
    start_time = time.time()
    
    results_dict = run_encoding(embedder, config)
    mean_corr = results_dict["mean_corr"]
    
    elapsed = time.time() - start_time
    print(f"\nModel: {model_shorthand_name}")
    print(f"Mean Correlation: {mean_corr:.4f}")
    print(f"Time taken: {elapsed:.1f}s")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    row = make_result_row(
        model_name=model_shorthand_name,
        corr=mean_corr,
        run_name="word_level_rc",
        description=model_description,
        subject=args.subject,
        num_train=args.num_train,
        num_test=args.num_test
    )
    upsert_overall_results(row, results_dir)
    print("Results appended.")
