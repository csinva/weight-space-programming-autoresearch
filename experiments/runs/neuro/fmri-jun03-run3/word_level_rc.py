"""
Untrained Word-Level Reservoir Computer
"""

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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class WordLevelReservoir(nn.Module):
    def __init__(self, d_model=1500, num_layers=3, max_words=128, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_words = max_words
        self.device = device
        
        self.word_cache = {}
        
        self.alphas = []
        a1 = torch.linspace(0.0, 0.7, d_model)
        a2 = torch.linspace(0.5, 0.9, d_model)
        a3 = torch.linspace(0.8, 0.99, d_model)
        
        self.alphas.append(a1.to(device))
        self.alphas.append(a2.to(device))
        self.alphas.append(a3.to(device))
        
        torch.manual_seed(42)
        self.W12 = (torch.randn(d_model, d_model, device=device) / math.sqrt(d_model)) * 1.5
        self.W23 = (torch.randn(d_model, d_model, device=device) / math.sqrt(d_model)) * 1.5
        
        self.b12 = torch.randn(d_model, device=device) * 0.1
        self.b23 = torch.randn(d_model, device=device) * 0.1

    def get_word_vector(self, word: str) -> torch.Tensor:
        if word not in self.word_cache:
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % (2**32)
            torch.manual_seed(h)
            vec = torch.randn(self.d_model, device=self.device) / math.sqrt(self.d_model)
            self.word_cache[word] = vec
        return self.word_cache[word]

    @torch.no_grad()
    def forward(self, input_strings: List[str]) -> torch.Tensor:
        B = len(input_strings)
        outputs = torch.zeros(B, self.d_model * 3, device=self.device)
        
        for i, s in enumerate(input_strings):
            words = s.split()[-self.max_words:]
            
            h1 = torch.zeros(self.d_model, device=self.device)
            h2 = torch.zeros(self.d_model, device=self.device)
            h3 = torch.zeros(self.d_model, device=self.device)
            
            for w in words:
                v = self.get_word_vector(w)
                
                h1 = self.alphas[0] * h1 + (1 - self.alphas[0]) * v
                
                in2 = torch.tanh(h1 @ self.W12 + self.b12)
                h2 = self.alphas[1] * h2 + (1 - self.alphas[1]) * in2
                
                in3 = torch.tanh(h2 @ self.W23 + self.b23)
                h3 = self.alphas[2] * h3 + (1 - self.alphas[2]) * in3
                
            outputs[i, 0:self.d_model] = h1
            outputs[i, self.d_model:2*self.d_model] = h2
            outputs[i, 2*self.d_model:] = h3
            
        return outputs.cpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_shorthand_name = "Untrained_Word_Level_Reservoir"
    model_description = "A 3-layer purely random continuous-time reservoir computer operating on WORDS instead of characters. Each word receives a deterministic random 1500-dim hash. Layers have progressively slower exponential smoothing (L1: 0-0.7, L2: 0.5-0.9, L3: 0.8-0.99)."

    print(f"\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    embedder = WordLevelReservoir(d_model=1500, num_layers=3, max_words=128, device=args.device)
    
    config = EncodingConfig()
    config.subject = args.subject
    config.num_train = args.num_train
    config.num_test = args.num_test

    t0 = time.time()
    try:
        results = run_encoding(embedder, config)
        test_corr = results["test_corr"]
        print(f"Mean test correlation: {test_corr:.4f}")
        
        n_params = 0 # It's untrained!
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
            "n_params": 0,
            "description": model_description,
            "corrs_test_frac>0.1": 0.0, "corrs_test_frac>0.05": 0.0, "corrs_test_frac>0.0": 0.0,
            "corrs_test_median": 0.0, "corrs_test_p75": 0.0, "corrs_test_p90": 0.0, "corrs_test_p95": 0.0, "corrs_test_p99": 0.0
        }

    upsert_overall_results([row], RESULTS_DIR)
