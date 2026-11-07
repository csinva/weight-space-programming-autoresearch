import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
import spacy
nlp = spacy.load("en_core_web_lg")

class GloVeEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, texts: list[str]) -> np.ndarray:
        B = len(texts)
        semantic_feats = np.zeros((B, 300), dtype=np.float32)
        for i, t in enumerate(texts):
            doc = nlp(t)
            if doc.has_vector:
                semantic_feats[i] = doc.vector
        return semantic_feats

embedder = GloVeEmbedder()

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

model_name = "Pure_GloVe_Semantic"
print(f"\n--- Testing model: {model_name} ---", flush=True)
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 300, "Pure 300-dim Spacy en_core_web_lg (GloVe) semantic features")
upsert_overall_results([row], RESULTS_DIR)
