import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class WordPositionEmbedder(nn.Module):
    def __init__(self, dim_per_word=512):
        super().__init__()
        self.dim_per_word = dim_per_word
        self.total_dim = dim_per_word * 10 # 10 words
        self.char_map = {}
        np.random.seed(42)

    def forward(self, texts: list[str]) -> np.ndarray:
        B = len(texts)
        out = np.zeros((B, self.total_dim), dtype=np.float32)
        
        for i, t in enumerate(texts):
            words = t.split()
            # Pad or truncate to exactly 10 words
            if len(words) < 10:
                words = ['<pad>'] * (10 - len(words)) + words
            else:
                words = words[-10:]
                
            vec = np.zeros(self.total_dim, dtype=np.float32)
            
            # Concatenate the word representations (explicitly maintaining order)
            for w_idx, word in enumerate(words):
                if word not in self.char_map:
                    self.char_map[word] = np.random.randn(self.dim_per_word).astype(np.float32)
                
                start = w_idx * self.dim_per_word
                end = start + self.dim_per_word
                vec[start:end] = self.char_map[word]
                
            out[i] = vec
            
        return out

embedder = WordPositionEmbedder(dim_per_word=512) # 5120 total dims

config = EncodingConfig(
    subject="UTS03",
    num_train=8,
    num_test=2,
    ngram_size=10,
    ndelays=4,
    nboots=5,
    chunklen=40,
    nchunks=20,
    trim_edges=True
)

model_name = "Word_Position_Concatenation"
print(f"\n--- Testing model: {model_name} ---", flush=True)
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 5120, "Words assigned random embeddings and concatenated to preserve exact position syntax.")
upsert_overall_results([row], RESULTS_DIR)
