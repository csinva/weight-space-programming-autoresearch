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

class GloVeTemporalEmbedder(nn.Module):
    def __init__(self, mode="decay"):
        super().__init__()
        self.mode = mode

    def forward(self, texts: list[str]) -> np.ndarray:
        B = len(texts)
        # 300 dim Glove
        semantic_feats = np.zeros((B, 300), dtype=np.float32)
        
        for i, t in enumerate(texts):
            doc = nlp(t)
            words = [token for token in doc if not token.is_space]
            if not words:
                continue
                
            if self.mode == "decay":
                # Exponential decay weighting towards the end of the 10-gram
                decay_factor = 0.8
                weights = np.array([decay_factor**(len(words)-1-j) for j in range(len(words))])
                weights = weights / weights.sum()
                
                vec_sum = np.zeros(300)
                for j, w in enumerate(words):
                    if w.has_vector:
                        vec_sum += w.vector * weights[j]
                semantic_feats[i] = vec_sum
                
            elif self.mode == "diff":
                # Derivative (last - mean of previous)
                if len(words) > 1:
                    prev_words = words[:-1]
                    prev_vecs = [w.vector for w in prev_words if w.has_vector]
                    if prev_vecs and words[-1].has_vector:
                        prev_mean = np.mean(prev_vecs, axis=0)
                        semantic_feats[i] = words[-1].vector - prev_mean
                    elif words[-1].has_vector:
                        semantic_feats[i] = words[-1].vector
                elif words[-1].has_vector:
                    semantic_feats[i] = words[-1].vector
                    
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

for mode in ["decay", "diff"]:
    model_name = f"Pure_GloVe_{mode.capitalize()}"
    print(f"\n--- Testing model: {model_name} ---", flush=True)
    embedder = GloVeTemporalEmbedder(mode=mode)
    results = run_encoding(embedder, config, verbose=True)
    
    row = make_result_row(results, model_name, 300, f"Pure GloVe with {mode} temporal weighting")
    upsert_overall_results([row], RESULTS_DIR)

