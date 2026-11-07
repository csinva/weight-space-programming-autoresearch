import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LayerMeanEmbedder(nn.Module):
    def __init__(self, hf_model_name="gpt2-xl", layer_idx=24):
        super().__init__()
        
        print(f"Loading {hf_model_name} (Layer {layer_idx})...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.semantic_model = AutoModel.from_pretrained(hf_model_name, output_hidden_states=True).to(device)
        self.semantic_model.eval()
        self.layer_idx = layer_idx

    def forward(self, texts: list[str]) -> np.ndarray:
        batch_size = 32 # Reduced for XL
        semantic_feats = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                outputs = self.semantic_model(**encoded)
                
                # Get specific layer (index 0 is embeddings, 1 is layer 1, etc.)
                # If layer_idx is negative, it counts from the end
                layer_hidden_states = outputs.hidden_states[self.layer_idx]
                
                # Mean pooling over valid tokens
                attention_mask = encoded['attention_mask'].unsqueeze(-1)
                mean_pooled = torch.sum(layer_hidden_states * attention_mask, dim=1) / torch.clamp(attention_mask.sum(1), min=1e-9)
                
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

# Test early, middle, and late layers of GPT-2 XL (48 total layers)
# We already tested the final output (layer 48/last). The original baseline used layer 24 (middle).
for layer in [12, 24, 36]:
    model_name = f"Pure_gpt2_xl_Layer{layer}_Mean"
    print(f"\n--- Testing model: {model_name} ---", flush=True)
    embedder = LayerMeanEmbedder("gpt2-xl", layer_idx=layer)
    results = run_encoding(embedder, config, verbose=True)
    
    row = make_result_row(results, model_name, 1600, f"GPT-2 XL Layer {layer} Mean Pool semantics")
    upsert_overall_results([row], RESULTS_DIR)
    
    del embedder
    torch.cuda.empty_cache()

