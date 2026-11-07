import torch
import torch.nn as nn
import numpy as np
import os
import sys
import urllib.request
import zipfile

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Download fasttext mini if we don't have a fasttext model
import spacy
nlp = spacy.load("en_core_web_md") # We'll just use spacy's vectors and call it a day if we can't easily get fasttext

# Let's try testing exact positional combinations of Word2Vec (GloVe) 
# instead since we already have it loaded in memory
class GloVePositionalEmbedder(nn.Module):
    def __init__(self, positions=[0, -1]):
        super().__init__()
        self.positions = positions

    def forward(self, texts: list[str]) -> np.ndarray:
        B = len(texts)
        D = 300
        semantic_feats = np.zeros((B, D * len(self.positions)), dtype=np.float32)
        
        for i, t in enumerate(texts):
            doc = nlp(t)
            words = [token for token in doc if not token.is_space]
            if not words:
                continue
                
            for j, pos in enumerate(self.positions):
                try:
                    if pos < 0:
                        word = words[pos]
                    else:
                        word = words[pos]
                        
                    if word.has_vector:
                        semantic_feats[i, j*D:(j+1)*D] = word.vector
                except IndexError:
                    pass
                    
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

for positions in [[-1], [-1, -2], [-1, -2, -3]]:
    model_name = f"Pure_Word2Vec_Pos_{'_'.join([str(p) for p in positions])}"
    print(f"\n--- Testing model: {model_name} ---", flush=True)
    embedder = GloVePositionalEmbedder(positions=positions)
    results = run_encoding(embedder, config, verbose=True)
    
    row = make_result_row(results, model_name, 300 * len(positions), f"Pure Word2Vec Exact Positions: {positions}")
    upsert_overall_results([row], RESULTS_DIR)

