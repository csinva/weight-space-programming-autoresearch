import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ModernLLMMeanEmbedder(nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()
        
        print(f"Loading {hf_model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Use bfloat16 to save memory since modern LLMs are big
        self.semantic_model = AutoModel.from_pretrained(
            hf_model_name, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.semantic_model.eval()

    def forward(self, texts: list[str]) -> np.ndarray:
        batch_size = 16 # Small batch size for larger models
        semantic_feats = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.semantic_model.device)
                outputs = self.semantic_model(**encoded)
                
                # Mean pooling over valid tokens
                attention_mask = encoded['attention_mask'].unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                mean_pooled = torch.sum(token_embeddings * attention_mask, dim=1) / torch.clamp(attention_mask.sum(1), min=1e-9)
                
                # Convert back to float32 for ridge regression
                semantic_feats.append(mean_pooled.float().cpu().numpy())
                
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

# Test Qwen 2.5
for hf_model in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]:
    model_name = f"Pure_{hf_model.split('/')[-1].replace('.', '_').replace('-', '_')}_Mean"
    print(f"\n--- Testing model: {model_name} ---", flush=True)
    try:
        embedder = ModernLLMMeanEmbedder(hf_model)
        
        # Test a quick forward pass to get hidden dim
        dummy_out = embedder(["test"])
        hidden_dim = dummy_out.shape[1]
        
        results = run_encoding(embedder, config, verbose=True)
        
        row = make_result_row(results, model_name, hidden_dim, f"Pure {hf_model} Mean Pool semantics")
        upsert_overall_results([row], RESULTS_DIR)
        
        del embedder
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error evaluating {hf_model}: {e}")

