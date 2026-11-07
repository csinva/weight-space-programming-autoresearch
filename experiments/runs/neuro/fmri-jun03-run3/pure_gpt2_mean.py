import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PureGPT2MeanEmbedder(nn.Module):
    def __init__(self, hf_model_name="gpt2"):
        super().__init__()
        
        print(f"Loading {hf_model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        # GPT2 doesn't have a pad token by default
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.semantic_model = AutoModel.from_pretrained(hf_model_name).to(device)
        self.semantic_model.eval()

    def forward(self, texts: list[str]) -> np.ndarray:
        batch_size = 128
        semantic_feats = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                outputs = self.semantic_model(**encoded)
                
                # Mean pooling over valid tokens
                attention_mask = encoded['attention_mask'].unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                mean_pooled = torch.sum(token_embeddings * attention_mask, dim=1) / torch.clamp(attention_mask.sum(1), min=1e-9)
                
                semantic_feats.append(mean_pooled.cpu().numpy())
                
        semantic_feats = np.concatenate(semantic_feats, axis=0)
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

model_name = "Pure_GPT2_Mean"
print(f"\n--- Testing model: {model_name} ---", flush=True)
embedder = PureGPT2MeanEmbedder("gpt2")
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 768, "Pure GPT-2 Mean Pool semantics (no Char Transformer)")
upsert_overall_results([row], RESULTS_DIR)

