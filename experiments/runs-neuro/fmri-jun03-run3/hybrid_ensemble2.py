import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from final_model_0421 import build_embedder as build_0421_embedder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LayeredHybridEmbedder(nn.Module):
    def __init__(self, transformer_embedder, bow_dim=1024):
        super().__init__()
        self.transformer_embedder = transformer_embedder
        
        self.char_map_word = {}
        self.char_map_bigram = {}
        self.dim = bow_dim
        np.random.seed(42)

    def forward(self, texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            transformer_feats = self.transformer_embedder(texts).numpy()
            
        B = len(texts)
        bow_feats = np.zeros((B, self.dim), dtype=np.float32)
        bigram_feats = np.zeros((B, self.dim), dtype=np.float32)
        
        for i, t in enumerate(texts):
            words = t.split()[-10:]
            
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in words:
                if word not in self.char_map_word:
                    self.char_map_word[word] = np.random.randn(self.dim).astype(np.float32)
                vec += self.char_map_word[word]
            bow_feats[i] = vec
            
            b_vec = np.zeros(self.dim, dtype=np.float32)
            for w1, w2 in zip(words[:-1], words[1:]):
                bigram = w1 + "_" + w2
                if bigram not in self.char_map_bigram:
                    self.char_map_bigram[bigram] = np.random.randn(self.dim).astype(np.float32)
                b_vec += self.char_map_bigram[bigram]
            bigram_feats[i] = b_vec
            
        return np.concatenate([transformer_feats, bow_feats, bigram_feats], axis=1)

print("Building 0.0421 Transformer...", flush=True)
transformer_embedder = build_0421_embedder(device=device)
embedder = LayeredHybridEmbedder(transformer_embedder, bow_dim=1024)

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

model_name = "Hybrid_0421_Word_Bigram"
print(f"\n--- Testing model: {model_name} ---", flush=True)
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 1024 + 1024 + 1020, "Hybrid of 0.0421 Char Transformer + Unigram BoW + Bigram BoW")
upsert_overall_results([row], RESULTS_DIR)
