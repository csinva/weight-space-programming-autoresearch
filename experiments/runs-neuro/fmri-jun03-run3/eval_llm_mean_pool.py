import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class MeanPoolEmbedder(nn.Module):
    def __init__(self, layer_mean, hf_model_name):
        super().__init__()
        self.layer_mean = layer_mean
        print(f"Loading {hf_model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.semantic_model = AutoModel.from_pretrained(
            hf_model_name, 
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.semantic_model.eval()

    @torch.no_grad()
    def forward(self, ngrams: list[str]) -> np.ndarray:
        cleaned_ngrams = [str(n) for n in ngrams]
        B = 32
        out = []
        for i in range(0, len(cleaned_ngrams), B):
            batch_texts = cleaned_ngrams[i : i + B]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self.semantic_model.device) for k, v in inputs.items()}
            
            outputs = self.semantic_model(**inputs)
            hidden_states = outputs.hidden_states
            
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states[self.layer_mean].size()).float()
            mean_pooled_repr = (hidden_states[self.layer_mean] * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

            out.append(mean_pooled_repr.float().cpu().numpy())
            
        return np.vstack(out)

if __name__ == "__main__":
    hf_model_name = "Qwen/Qwen2.5-1.5B"
    model_shorthand_name = "Qwen_1.5B_MeanPool_L14"
    model_description = "Qwen-2.5-1.5B single scale semantic representation. Uses exactly mean pooling of Layer 14 (middle layer) to capture the broad contextual 'gist' of the sequence, ignoring exact syntax."
    
    print(f"\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    embedder = MeanPoolEmbedder(layer_mean=14, hf_model_name=hf_model_name)
    
    config = EncodingConfig()
    print("Running encoding...")
    r = run_encoding(embedder, config)

    row = make_result_row(
        r,
        model_shorthand_name=model_shorthand_name,
        n_params=1_500_000_000,
        description=model_description
    )
    upsert_overall_results([row], RESULTS_DIR)
