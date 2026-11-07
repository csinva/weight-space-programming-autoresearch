import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from final_model_0421 import build_embedder as build_0421_embedder
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class HybridDistilBERTEmbedder(nn.Module):
    def __init__(self, transformer_embedder, hf_model_name="distilbert-base-uncased"):
        super().__init__()
        self.transformer_embedder = transformer_embedder
        
        print(f"Loading {hf_model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.semantic_model = AutoModel.from_pretrained(hf_model_name).to(device)
        self.semantic_model.eval()

    def forward(self, texts: list[str]) -> np.ndarray:
        # Extract syntactic flow
        with torch.no_grad():
            transformer_feats = self.transformer_embedder(texts).numpy()
            
        # Extract contextual semantics
        batch_size = 128
        semantic_feats = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                outputs = self.semantic_model(**encoded)
                # Use [CLS] token (or mean pooling). DistilBERT [CLS] is index 0.
                cls_feats = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                semantic_feats.append(cls_feats)
                
        semantic_feats = np.concatenate(semantic_feats, axis=0)
        
        return np.concatenate([transformer_feats, semantic_feats], axis=1)

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

model_name = "Hybrid_0421_DistilBERT"
print(f"\n--- Testing model: {model_name} ---", flush=True)
embedder = HybridDistilBERTEmbedder(transformer_embedder, "distilbert-base-uncased")
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 1020 + 768, "0.0421 Transformer + DistilBERT [CLS] semantics")
upsert_overall_results([row], RESULTS_DIR)

