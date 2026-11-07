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

class GloVeSequentialEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, texts: list[str]) -> np.ndarray:
        B = len(texts)
        # We'll take the 10 words, average them (300), AND concatenate the last word (300) 
        # and the second-to-last word (300) to give the ridge regression both general semantic blur 
        # and strict recent syntactic context.
        semantic_feats = np.zeros((B, 900), dtype=np.float32)
        
        for i, t in enumerate(texts):
            doc = nlp(t)
            words = [token for token in doc if not token.is_space]
            
            if doc.has_vector:
                semantic_feats[i, 0:300] = doc.vector # Average of all words
                
            if len(words) > 0 and words[-1].has_vector:
                semantic_feats[i, 300:600] = words[-1].vector
                
            if len(words) > 1 and words[-2].has_vector:
                semantic_feats[i, 600:900] = words[-2].vector
                
        return semantic_feats

embedder = GloVeSequentialEmbedder()

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

model_name = "Pure_GloVe_Semantic_Sequential"
print(f"\n--- Testing model: {model_name} ---", flush=True)
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 900, "GloVe features: 10-gram average + last word + second-to-last word")
upsert_overall_results([row], RESULTS_DIR)
