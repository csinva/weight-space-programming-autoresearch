import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from final_model_0421 import build_embedder as build_0421_embedder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class HybridEmbedder(nn.Module):
    def __init__(self, transformer_embedder, bow_dim=4096):
        super().__init__()
        self.transformer_embedder = transformer_embedder
        self.bow_dim = bow_dim
        self.char_map = {}
        np.random.seed(42)

    def forward(self, texts: list[str]) -> np.ndarray:
        # Get transformer features
        with torch.no_grad():
            transformer_feats = self.transformer_embedder(texts).numpy()
            
        B = len(texts)
        bow_feats = np.zeros((B, self.bow_dim), dtype=np.float32)
        
        for i, t in enumerate(texts):
            words = t.split()
            vec = np.zeros(self.bow_dim, dtype=np.float32)
            
            for word in words[-10:]: # look at up to last 10 words
                if word not in self.char_map:
                    self.char_map[word] = np.random.randn(self.bow_dim).astype(np.float32)
                vec += self.char_map[word]
                
            bow_feats[i] = vec
            
        # Concatenate transformer and BoW features
        return np.concatenate([transformer_feats, bow_feats], axis=1)

print("Building 0.0421 Transformer...", flush=True)
transformer_embedder = build_0421_embedder(device=device)

embedder = HybridEmbedder(transformer_embedder, bow_dim=4096)

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

model_name = "Hybrid_0421_and_BoW"
print(f"\n--- Testing model: {model_name} ---", flush=True)
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 4096 + 1020, "Concatenation of 0.0421 Character Transformer and 4096-dim Bag of Words")
upsert_overall_results([row], RESULTS_DIR)
