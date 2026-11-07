import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class FastTextEmbedder(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = dim
        self.char_map = {}
        np.random.seed(42)

    def forward(self, texts: list[str]) -> np.ndarray:
        B = len(texts)
        out = np.zeros((B, self.dim), dtype=np.float32)
        
        for i, t in enumerate(texts):
            # Split the 10-gram into words
            words = t.split()
            # We want to represent the *current* word (the last one)
            # and perhaps some decaying history.
            
            # Simple word-level hash mapping to orthogonal vectors
            vec = np.zeros(self.dim, dtype=np.float32)
            
            for w_idx, word in enumerate(reversed(words)):
                if word not in self.char_map:
                    self.char_map[word] = np.random.randn(self.dim).astype(np.float32)
                
                # Exponential decay for older words
                decay = 0.5 ** w_idx
                vec += self.char_map[word] * decay
                
            out[i] = vec
            
        return out

embedder = FastTextEmbedder(dim=4096)

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

model_name = "Word_Level_Exponential_Decay"
print(f"\n--- Testing model: {model_name} ---", flush=True)
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 4096, "Word-level random orthogonal embeddings with exponential decay across the 10-gram.")
upsert_overall_results([row], RESULTS_DIR)
