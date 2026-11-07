import torch
import torch.nn as nn
import numpy as np
import os
import sys
import string

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

VOCAB = list(string.ascii_lowercase + " .,;'-")
VOCAB_SIZE = len(VOCAB)

class CharTrigramBagEmbedder(nn.Module):
    def __init__(self, mode="trigram"):
        super().__init__()
        self.mode = mode
        
        self.char_to_idx = {ch: i for i, ch in enumerate(VOCAB)}
        self.dim = VOCAB_SIZE * VOCAB_SIZE * VOCAB_SIZE

    def forward(self, texts: list[str]) -> np.ndarray:
        B = len(texts)
        semantic_feats = np.zeros((B, self.dim), dtype=np.float32)
        
        for i, t in enumerate(texts):
            # clean text
            t = "".join([c for c in t.lower() if c in self.char_to_idx])
            
            for j in range(len(t) - 2):
                c1 = self.char_to_idx[t[j]]
                c2 = self.char_to_idx[t[j+1]]
                c3 = self.char_to_idx[t[j+2]]
                idx = c1 * VOCAB_SIZE * VOCAB_SIZE + c2 * VOCAB_SIZE + c3
                semantic_feats[i, idx] += 1.0
                
        # Normalize
        norms = np.linalg.norm(semantic_feats, axis=1, keepdims=True)
        semantic_feats = semantic_feats / (norms + 1e-8)
                
        return semantic_feats

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

model_name = "Pure_CharTrigram_Bag"
print(f"\n--- Testing model: {model_name} ---", flush=True)
embedder = CharTrigramBagEmbedder()
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, VOCAB_SIZE ** 3, "Bag of Character Trigrams (Normalized)")
upsert_overall_results([row], RESULTS_DIR)
