import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class WordHashMeanEmbedder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=1000, seed=42):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        np.random.seed(seed)
        # Random orthogonal embeddings
        emb = np.random.randn(vocab_size, embed_dim)
        if embed_dim <= vocab_size:
            q, r = np.linalg.qr(emb)
            emb = q
        else:
            q, r = np.linalg.qr(emb.T)
            emb = q.T
            
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embeddings.weight.data.copy_(torch.tensor(emb, dtype=torch.float32))
        self.embeddings.weight.requires_grad = False

    def forward(self, texts: list[str]) -> np.ndarray:
        B = len(texts)
        semantic_feats = np.zeros((B, self.embed_dim), dtype=np.float32)
        
        for i, t in enumerate(texts):
            words = t.lower().split()
            if not words:
                continue
                
            # Hash trick
            word_ids = [hash(w) % self.vocab_size for w in words]
            word_ids_tensor = torch.tensor(word_ids, dtype=torch.long).to(self.embeddings.weight.device)
            
            with torch.no_grad():
                vecs = self.embeddings(word_ids_tensor)
                semantic_feats[i] = vecs.mean(dim=0).cpu().numpy()
                
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

for dim in [300, 1000, 3000]:
    model_name = f"Pure_WordHash_Mean_D{dim}"
    print(f"\n--- Testing model: {model_name} ---", flush=True)
    embedder = WordHashMeanEmbedder(embed_dim=dim)
    results = run_encoding(embedder, config, verbose=True)
    
    row = make_result_row(results, model_name, dim, f"Random Word Hash Embeddings (Mean Pool)")
    upsert_overall_results([row], RESULTS_DIR)

