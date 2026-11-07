import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from final_model_0421 import build_embedder as build_0421_embedder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import spacy
nlp = spacy.load("en_core_web_lg")

class WeightedGloVeEmbedder(nn.Module):
    def __init__(self, transformer_embedder, glove_repeats=3):
        super().__init__()
        self.transformer_embedder = transformer_embedder
        self.glove_repeats = glove_repeats

    def forward(self, texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            transformer_feats = self.transformer_embedder(texts).numpy()
            
        B = len(texts)
        # Average GloVe only
        glove_feats = np.zeros((B, 300), dtype=np.float32)
        
        for i, t in enumerate(texts):
            doc = nlp(t)
            if doc.has_vector:
                glove_feats[i] = doc.vector
                
        # Repeat the GloVe features to reduce their L2 penalty relative to the Transformer
        repeated_glove = np.tile(glove_feats, (1, self.glove_repeats))
        
        return np.concatenate([transformer_feats, repeated_glove], axis=1)

print("Building 0.0421 Transformer...", flush=True)
transformer_embedder = build_0421_embedder(device=device)

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

for repeats in [2, 4, 6]:
    model_name = f"Hybrid_0421_GloVe_x{repeats}"
    print(f"\n--- Testing model: {model_name} ---", flush=True)
    embedder = WeightedGloVeEmbedder(transformer_embedder, glove_repeats=repeats)
    results = run_encoding(embedder, config, verbose=True)
    row = make_result_row(results, model_name, 1020 + 300 * repeats, f"0.0421 Transformer + GloVe (Repeated {repeats}x to reduce Ridge penalty)")
    upsert_overall_results([row], RESULTS_DIR)

