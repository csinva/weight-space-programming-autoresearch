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
# Download the large model which uses full GloVe vectors
try:
    nlp = spacy.load("en_core_web_lg")
except:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

class SemanticHybridEmbedder(nn.Module):
    def __init__(self, transformer_embedder):
        super().__init__()
        self.transformer_embedder = transformer_embedder

    def forward(self, texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            transformer_feats = self.transformer_embedder(texts).numpy()
            
        B = len(texts)
        semantic_feats = np.zeros((B, 300), dtype=np.float32)
        
        for i, t in enumerate(texts):
            # Using en_core_web_lg (300d GloVe)
            doc = nlp(t)
            if doc.has_vector:
                semantic_feats[i] = doc.vector
            
        return np.concatenate([transformer_feats, semantic_feats], axis=1)

print("Building 0.0421 Transformer...", flush=True)
transformer_embedder = build_0421_embedder(device=device)
embedder = SemanticHybridEmbedder(transformer_embedder)

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

model_name = "Hybrid_0421_GloVe_Semantic"
print(f"\n--- Testing model: {model_name} ---", flush=True)
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 1020 + 300, "0.0421 Transformer + 300-dim Spacy en_core_web_lg (GloVe) semantic blending")
upsert_overall_results([row], RESULTS_DIR)
