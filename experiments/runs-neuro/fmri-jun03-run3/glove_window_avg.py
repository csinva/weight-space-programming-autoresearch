import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import spacy
nlp = spacy.load("en_core_web_lg")

class GloVeWindowAvgEmbedder(nn.Module):
    def __init__(self, window_size=10):
        super().__init__()
        self.window_size = window_size

    def forward(self, texts: list[str]) -> np.ndarray:
        B = len(texts)
        D = 300
        semantic_feats = np.zeros((B, D), dtype=np.float32)
        
        for i, t in enumerate(texts):
            doc = nlp(t)
            words = [token for token in doc if not token.is_space]
            if not words:
                continue
                
            window_words = words[-self.window_size:] if self.window_size > 0 else words
            vecs = [w.vector for w in window_words if w.has_vector]
            
            if vecs:
                semantic_feats[i] = np.mean(vecs, axis=0)
                    
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

# Test how averaging over different window lengths of recent words works
# window=1 is just the last word
# window=3 is average of last 3 words
# window=10 is average of all 10 words
for window in [1, 3, 5, 10]:
    model_name = f"Pure_GloVe_Avg_Win_{window}"
    print(f"\n--- Testing model: {model_name} ---", flush=True)
    embedder = GloVeWindowAvgEmbedder(window_size=window)
    results = run_encoding(embedder, config, verbose=True)
    
    row = make_result_row(results, model_name, 300, f"Pure GloVe Average over last {window} words")
    upsert_overall_results([row], RESULTS_DIR)

