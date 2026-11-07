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

class HybridGloVeEmbedder(nn.Module):
    def __init__(self, transformer_embedder):
        super().__init__()
        self.transformer_embedder = transformer_embedder

    def forward(self, texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            transformer_feats = self.transformer_embedder(texts).numpy()
            
        B = len(texts)
        # Average (300) + Last (300) + Second Last (300)
        semantic_feats = np.zeros((B, 900), dtype=np.float32)
        
        for i, t in enumerate(texts):
            doc = nlp(t)
            words = [token for token in doc if not token.is_space]
            
            if doc.has_vector:
                semantic_feats[i, 0:300] = doc.vector # Average
                
            if len(words) > 0 and words[-1].has_vector:
                semantic_feats[i, 300:600] = words[-1].vector
                
            if len(words) > 1 and words[-2].has_vector:
                semantic_feats[i, 600:900] = words[-2].vector
                
        return np.concatenate([transformer_feats, semantic_feats], axis=1)

print("Building 0.0421 Transformer...", flush=True)
transformer_embedder = build_0421_embedder(device=device)
embedder = HybridGloVeEmbedder(transformer_embedder)

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

model_name = "Hybrid_0421_GloVe_Avg_And_Sequential"
print(f"\n--- Testing model: {model_name} ---", flush=True)
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 1020 + 900, "0.0421 Transformer + GloVe (Average + Last + Prev)")
upsert_overall_results([row], RESULTS_DIR)
