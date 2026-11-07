import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class QwenMultiScaleEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"Loading Qwen-2.5-1.5B...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModel.from_pretrained(
            "Qwen/Qwen2.5-1.5B", 
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0"
        )
        self.model.eval()

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
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states
            
            # L14 Last Token
            seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
            batch_indices = torch.arange(inputs['attention_mask'].shape[0], device=self.model.device)
            last_token = hidden_states[14][batch_indices, seq_lengths, :] 
            
            # L28 Mean Pool
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states[28].size()).float()
            mean_pool = (hidden_states[28] * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

            combined = torch.cat([last_token, mean_pool], dim=-1)
            out.append(combined.float().cpu().numpy())
            
        return np.vstack(out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    args = parser.parse_args()

    embedder = QwenMultiScaleEmbedder()
    
    # Sweep context lengths
    context_lengths = [5, 10, 20, 50, 100]
    
    for clen in context_lengths:
        model_shorthand_name = f"Qwen1.5B_Context_{clen}"
        model_description = f"Qwen-2.5-1.5B (L14Last+L28Mean) evaluated with ngram_size={clen}."
        
        print(f"\n--- Testing model: {model_shorthand_name} ---")
        print(model_description)
        
        config = EncodingConfig()
        config.subject = args.subject
        config.num_train = args.num_train
        config.num_test = args.num_test
        config.ngram_size = clen

        print(f"Running encoding with ngram_size={clen}...")
        r = run_encoding(embedder, config)
        print(f"Finished encoding, saving results dict with length {len(r)}")

        row = make_result_row(
            r,
            model_shorthand_name=model_shorthand_name,
            n_params=1_500_000_000,
            description=model_description
        )
        upsert_overall_results([row], RESULTS_DIR)
