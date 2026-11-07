import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class BagOfWordsEmbedder(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        self.dim = dim
        self.char_map = {}
        np.random.seed(42)

    def forward(self, texts: list[str]) -> np.ndarray:
        B = len(texts)
        out = np.zeros((B, self.dim), dtype=np.float32)
        
        for i, t in enumerate(texts):
            words = t.split()
            vec = np.zeros(self.dim, dtype=np.float32)
            
            # Pure bag of words (no decay, just sum of random vectors for words in the 10-gram context window)
            for word in words:
                if word not in self.char_map:
                    self.char_map[word] = np.random.randn(self.dim).astype(np.float32)
                vec += self.char_map[word]
                
            out[i] = vec
            
        return out

embedder = BagOfWordsEmbedder(dim=4096)

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

model_name = "Bag_Of_Words_10gram"
print(f"\n--- Testing model: {model_name} ---", flush=True)
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 4096, "Pure Bag of Words random embeddings over the 10-gram window.")
upsert_overall_results([row], RESULTS_DIR)
