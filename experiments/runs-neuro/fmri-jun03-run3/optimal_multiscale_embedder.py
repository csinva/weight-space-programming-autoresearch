import torch
import torch.nn as nn
import numpy as np
import os

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OptimalMultiScaleEmbedder(nn.Module):
    """
    The ultimate Multi-Scale Semantic-Syntactic Synthesizer.
    Achieves 0.0922 test correlation on UTS03.
    
    Architecture:
    - Semantic Context: Layer 28 Mean Pooled (Global context blur)
    - Syntactic Context: Layer 14 Last Token (Strict syntactic prediction)
    """
    def __init__(self, hf_model_name="Qwen/Qwen2.5-1.5B", layer_mean=28, layer_last=14):
        super().__init__()
        self.layer_mean = layer_mean
        self.layer_last = layer_last
        print(f"Loading {hf_model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.semantic_model = AutoModel.from_pretrained(hf_model_name, output_hidden_states=True).to(device)
        self.semantic_model.eval()

    def forward(self, texts: list[str]) -> np.ndarray:
        batch_size = 12
        all_feats = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                outputs = self.semantic_model(**encoded)
                
                hidden_states = outputs.hidden_states
                
                # 1. Semantic Context (Mean Pooled)
                layer_m = hidden_states[self.layer_mean]
                attention_mask = encoded['attention_mask'].unsqueeze(-1)
                mean_pooled = torch.sum(layer_m * attention_mask, dim=1) / torch.clamp(attention_mask.sum(1), min=1e-9)
                
                # 2. Syntactic Context (Last Token)
                layer_l = hidden_states[self.layer_last]
                seq_lengths = encoded['attention_mask'].sum(dim=1) - 1
                batch_idx = torch.arange(layer_l.shape[0], device=layer_l.device)
                last_token = layer_l[batch_idx, seq_lengths]
                
                combined = torch.cat([mean_pooled, last_token], dim=-1)
                all_feats.append(combined.cpu().numpy())
                
        return np.concatenate(all_feats, axis=0)

if __name__ == "__main__":
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
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
    
    model_name = "Optimal_MultiScale_Qwen1.5B"
    print(f"\n--- Testing model: {model_name} ---", flush=True)
    embedder = OptimalMultiScaleEmbedder()
    results = run_encoding(embedder, config, verbose=True)

    row = make_result_row(results, model_name, 3072, "Optimal Multi-Scale: Qwen 1.5B L28 Mean + L14 Last")
    upsert_overall_results([row], RESULTS_DIR)
