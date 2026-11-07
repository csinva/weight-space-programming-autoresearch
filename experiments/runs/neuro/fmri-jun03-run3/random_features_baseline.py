import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class RandomEmbedder(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = dim
        self.char_map = {}
        np.random.seed(42)

    def forward(self, texts: list[str]) -> np.ndarray:
        # Just pure fixed random features per n-gram
        B = len(texts)
        out = np.zeros((B, self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if t not in self.char_map:
                self.char_map[t] = np.random.randn(self.dim).astype(np.float32)
            out[i] = self.char_map[t]
        return out

embedder = RandomEmbedder(dim=4096)

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

model_name = "Pure_Random_Orthogonal_Fingerprints"
print(f"\n--- Testing model: {model_name} ---", flush=True)
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 4096, "Pure random normal feature vector per unique 10-gram. Tests baseline dimensionality.")
upsert_overall_results([row], RESULTS_DIR)
