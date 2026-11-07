import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DiffPoolEmbedder(nn.Module):
    def __init__(self, hf_model_name="gpt2"):
        super().__init__()
        
        print(f"Loading {hf_model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Need right padding for difference pooling so the last token is easily identifiable 
        # relative to the sequence. Or we can just handle the mask properly.
        self.tokenizer.padding_side = 'right'
            
        self.semantic_model = AutoModel.from_pretrained(hf_model_name).to(device)
        self.semantic_model.eval()

    def forward(self, texts: list[str]) -> np.ndarray:
        batch_size = 64
        semantic_feats = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                outputs = self.semantic_model(**encoded)
                
                attention_mask = encoded['attention_mask']
                token_embeddings = outputs.last_hidden_state
                
                # Difference pooling: Last token - Mean of previous tokens
                diff_feats = []
                for b_idx in range(len(batch_texts)):
                    seq_len = int(attention_mask[b_idx].sum().item())
                    
                    if seq_len > 1:
                        last_token = token_embeddings[b_idx, seq_len-1, :]
                        prev_tokens_mean = torch.mean(token_embeddings[b_idx, :seq_len-1, :], dim=0)
                        diff = last_token - prev_tokens_mean
                    elif seq_len == 1:
                        diff = token_embeddings[b_idx, 0, :]
                    else:
                        # Edge case of completely empty string (shouldn't happen)
                        diff = torch.zeros_like(token_embeddings[b_idx, 0, :])
                        
                    diff_feats.append(diff.cpu().numpy())
                    
                semantic_feats.extend(diff_feats)
                
        semantic_feats = np.array(semantic_feats)
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

hf_model = "gpt2"
model_name = f"Pure_GPT2_Diff"
print(f"\n--- Testing model: {model_name} ---", flush=True)
embedder = DiffPoolEmbedder(hf_model)
results = run_encoding(embedder, config, verbose=True)

row = make_result_row(results, model_name, 768, "GPT-2 Last Token - Mean of Previous (Diff Pooling)")
upsert_overall_results([row], RESULTS_DIR)

