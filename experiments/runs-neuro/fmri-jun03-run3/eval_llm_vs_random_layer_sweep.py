import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel, AutoConfig

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class LayerSweepEmbedder(nn.Module):
    def __init__(self, layer_idx, hf_model_name, is_random=False, use_mean=False):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_random = is_random
        self.use_mean = use_mean
        
        print(f"Loading {hf_model_name} (Random={is_random})...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if is_random:
            config = AutoConfig.from_pretrained(hf_model_name)
            self.semantic_model = AutoModel.from_config(config).to(torch.bfloat16).to("cuda:1")
        else:
            self.semantic_model = AutoModel.from_pretrained(
                hf_model_name, 
                output_hidden_states=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda:1"
            )
        self.semantic_model.eval()

    @torch.no_grad()
    def forward(self, ngrams: list[str]) -> np.ndarray:
        cleaned_ngrams = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]

        B = 64
        out = []
        for i in range(0, len(cleaned_ngrams), B):
            batch_texts = cleaned_ngrams[i : i + B]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self.semantic_model.device) for k, v in inputs.items()}
            
            outputs = self.semantic_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            if self.use_mean:
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states[self.layer_idx].size()).float()
                repr = (hidden_states[self.layer_idx] * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
            else:
                seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
                batch_indices = torch.arange(inputs['attention_mask'].shape[0], device=self.semantic_model.device)
                repr = hidden_states[self.layer_idx][batch_indices, seq_lengths, :] 
            
            out.append(repr.float().cpu().numpy())
            
        return np.vstack(out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--mean", action="store_true")
    args = parser.parse_args()

    hf_model_name = "Qwen/Qwen2.5-1.5B"
    state_str = "Random" if args.random else "Trained"
    pool_str = "Mean" if args.mean else "Last"
    model_shorthand_name = f"Qwen1.5B_{state_str}_L{args.layer}{pool_str}"
    model_description = f"Qwen-2.5-1.5B {state_str}. Layer {args.layer} {pool_str} pooling. Layer sweep analysis."
    
    print(f"\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    embedder = LayerSweepEmbedder(layer_idx=args.layer, hf_model_name=hf_model_name, is_random=args.random, use_mean=args.mean)
    
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
        n_params=1_500_000_000,
        description=model_description
    )
    upsert_overall_results([row], RESULTS_DIR)
