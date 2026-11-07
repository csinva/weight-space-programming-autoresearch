import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class TripleScaleEmbedder(nn.Module):
    def __init__(self, layer_last, layer_mean_mid, layer_mean_late, hf_model_name):
        super().__init__()
        self.layer_last = layer_last
        self.layer_mean_mid = layer_mean_mid
        self.layer_mean_late = layer_mean_late
        
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
        cleaned_ngrams = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]

        B = 2
        out = []
        for i in range(0, len(cleaned_ngrams), B):
            batch_texts = cleaned_ngrams[i : i + B]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self.semantic_model.device) for k, v in inputs.items()}
            
            outputs = self.semantic_model(**inputs)
            hidden_states = outputs.hidden_states
            
            # Syntax / Local context (Last Token)
            seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
            batch_indices = torch.arange(inputs['attention_mask'].shape[0], device=self.semantic_model.device)
            last_token_repr = hidden_states[self.layer_last][batch_indices, seq_lengths, :] 
            
            # Semantic Gist Mid-Level (Mean-Pooled)
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states[self.layer_mean_mid].size()).float()
            mean_mid_repr = (hidden_states[self.layer_mean_mid] * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

            # Semantic Gist Late-Level (Mean-Pooled)
            mean_late_repr = (hidden_states[self.layer_mean_late] * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

            # Combine them
            combined = torch.cat([last_token_repr, mean_mid_repr, mean_late_repr], dim=-1)
            out.append(combined.float().cpu().numpy())
            
        return np.vstack(out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    args = parser.parse_args()

    hf_model_name = "Qwen/Qwen2.5-14B"
    model_shorthand_name = "Qwen14B_TripleScale_L48M_L24M_L12L"
    model_description = "Qwen-2.5-14B triple-scale. Layer 12 Last Token + Layer 24 Mean + Layer 48 Mean."
    
    print(f"\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    embedder = TripleScaleEmbedder(layer_last=12, layer_mean_mid=24, layer_mean_late=48, hf_model_name=hf_model_name)
    
    config = EncodingConfig()
    config.subject = args.subject
    config.num_train = args.num_train
    config.num_test = args.num_test

    print("Running encoding...", flush=True)
    r = run_encoding(embedder, config)
    print(f"Finished encoding, saving results dict with length {len(r)}", flush=True)

    row = make_result_row(
        r,
        model_shorthand_name=model_shorthand_name,
        n_params=14_000_000_000,
        description=model_description
    )
    upsert_overall_results([row], RESULTS_DIR)
