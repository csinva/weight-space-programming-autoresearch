import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from interpretable_transformer import ManualInterpretableTransformer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test wide random CNN
model = ManualInterpretableTransformer(n_layers=6, d_model=2048, n_heads=32, d_ff=8192).to(device)
model.eval()

class InterpretableEmbedder(nn.Module):
    def __init__(self, model, device='cpu'):
        super().__init__()
        self.model = model
        self.device = device
        chars = " abcdefghijklmnopqrstuvwxyz.,;:'\"!?-_=()[]{}0123456789"
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(self.vocab)

    @torch.no_grad()
    def forward(self, texts: list[str]) -> np.ndarray:
        B = len(texts)
        T = self.model.max_seq_len
        input_ids = torch.zeros((B, T), dtype=torch.long, device='cpu')
        
        for i, text in enumerate(texts):
            for j, char in enumerate(text[-T:]):
                input_ids[i, j] = self.vocab.get(char.lower(), 0)
                
        # Batch evaluation
        batch_size = 256
        features = []
        for i in range(0, B, batch_size):
            batch_ids = input_ids[i:i+batch_size].to(self.device)
            out = self.model(batch_ids)
            features.append(out[:, -1, :].cpu().numpy())
            
        return np.concatenate(features, axis=0)

embedder = InterpretableEmbedder(model, device=device)

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

model_name = "Deep_Wide_Random_CNN"
print(f"\n--- Testing model: {model_name} ---", flush=True)
results = run_encoding(embedder, config, verbose=True)

n_params = sum(p.numel() for p in model.parameters())
row = make_result_row(results, model_name, n_params, "Scaled random CNN (6L, 2048D, 32H)")
upsert_overall_results([row], RESULTS_DIR)
