import torch
import torch.nn as nn
import numpy as np
import os
import sys
import gc

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiScaleEmbedder(nn.Module):
    def __init__(self, layer_last, layer_mean, hf_model_name):
        super().__init__()
        self.layer_last = layer_last
        self.layer_mean = layer_mean
        print(f"Loading {hf_model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.semantic_model = AutoModel.from_pretrained(hf_model_name, output_hidden_states=True).to(device)
        self.semantic_model.eval()

    def forward(self, texts: list[str]) -> np.ndarray:
        batch_size = 12 # smaller batch size to avoid OOM with all hidden states
        all_feats = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                outputs = self.semantic_model(**encoded)
                
                hidden_states = outputs.hidden_states
                
                layer_m = hidden_states[self.layer_mean]
                attention_mask = encoded['attention_mask'].unsqueeze(-1)
                mean_pooled = torch.sum(layer_m * attention_mask, dim=1) / torch.clamp(attention_mask.sum(1), min=1e-9)
                
                layer_l = hidden_states[self.layer_last]
                seq_lengths = encoded['attention_mask'].sum(dim=1) - 1
                batch_idx = torch.arange(layer_l.shape[0], device=layer_l.device)
                last_token = layer_l[batch_idx, seq_lengths]
                
                combined = torch.cat([mean_pooled, last_token], dim=-1)
                all_feats.append(combined.cpu().numpy())
                
        return np.concatenate(all_feats, axis=0)

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

tasks = [
    ("gpt2-xl", "Hybrid_GPT2_XL_L48Mean_L12Last", 12, 48, 3200),
    ("gpt2-xl", "Hybrid_GPT2_XL_L48Mean_L36Last", 36, 48, 3200),
    ("gpt2-xl", "Hybrid_GPT2_XL_L48Mean_L48Last", 48, 48, 3200),
    ("Qwen/Qwen2.5-1.5B", "Hybrid_Qwen1.5B_L28Mean_L14Last", 14, 28, 3072),
]

for model_id, model_name, l_last, l_mean, dims in tasks:
    print(f"\n--- Testing model: {model_name} ---", flush=True)
    embedder = MultiScaleEmbedder(layer_last=l_last, layer_mean=l_mean, hf_model_name=model_id)
    results = run_encoding(embedder, config, verbose=True)

    row = make_result_row(results, model_name, dims, f"Concat of L{l_mean} Mean Pool and L{l_last} Last Token from {model_id}")
    upsert_overall_results([row], RESULTS_DIR)
    
    del embedder
    gc.collect()
    torch.cuda.empty_cache()
