import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class EnsembleEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        
        print(f"Loading Qwen2.5-1.5B...", flush=True)
        self.qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
        self.qwen_model = AutoModel.from_pretrained(
            "Qwen/Qwen2.5-1.5B", 
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:1"
        )
        self.qwen_model.eval()

        print(f"Loading GPT-2 XL...", flush=True)
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.gpt2_model = AutoModel.from_pretrained(
            "gpt2-xl", 
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:1"
        )
        self.gpt2_model.eval()

    @torch.no_grad()
    def forward(self, ngrams: list[str]) -> np.ndarray:
        cleaned_ngrams = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]

        B = 64
        out = []
        for i in range(0, len(cleaned_ngrams), B):
            batch_texts = cleaned_ngrams[i : i + B]
            
            # Qwen Features (L14 Last, L28 Mean)
            q_inputs = self.qwen_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            q_inputs = {k: v.to(self.qwen_model.device) for k, v in q_inputs.items()}
            q_outputs = self.qwen_model(**q_inputs).hidden_states
            
            q_seq_len = q_inputs['attention_mask'].sum(dim=1) - 1
            q_batch_idx = torch.arange(q_inputs['attention_mask'].shape[0], device=self.qwen_model.device)
            q_last = q_outputs[14][q_batch_idx, q_seq_len, :]
            
            q_mask = q_inputs['attention_mask'].unsqueeze(-1).expand(q_outputs[28].size()).float()
            q_mean = (q_outputs[28] * q_mask).sum(1) / torch.clamp(q_mask.sum(1), min=1e-9)

            # GPT2 Features (L24 Last, L48 Mean)
            g_inputs = self.gpt2_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            g_inputs = {k: v.to(self.gpt2_model.device) for k, v in g_inputs.items()}
            g_outputs = self.gpt2_model(**g_inputs).hidden_states
            
            g_seq_len = g_inputs['attention_mask'].sum(dim=1) - 1
            g_batch_idx = torch.arange(g_inputs['attention_mask'].shape[0], device=self.gpt2_model.device)
            g_last = g_outputs[24][g_batch_idx, g_seq_len, :]
            
            g_mask = g_inputs['attention_mask'].unsqueeze(-1).expand(g_outputs[48].size()).float()
            g_mean = (g_outputs[48] * g_mask).sum(1) / torch.clamp(g_mask.sum(1), min=1e-9)

            # Combine all 4 scales
            combined = torch.cat([q_last, q_mean, g_last, g_mean], dim=-1)
            out.append(combined.float().cpu().numpy())
            
        return np.vstack(out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    args = parser.parse_args()

    model_shorthand_name = "Ensemble_Qwen1.5B_GPT2XL_QuadScale"
    model_description = "Ensemble of Qwen-2.5-1.5B (L14Last + L28Mean) and GPT-2-XL (L24Last + L48Mean)."
    
    print(f"\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    embedder = EnsembleEmbedder()
    
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
        n_params=3_000_000_000,
        description=model_description
    )
    upsert_overall_results([row], RESULTS_DIR)
