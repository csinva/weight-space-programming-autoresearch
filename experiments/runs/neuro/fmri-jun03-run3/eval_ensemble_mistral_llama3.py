import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class EnsembleMistralLlama(nn.Module):
    def __init__(self):
        super().__init__()
        
        print(f"Loading mistralai/Mistral-7B-v0.1...", flush=True)
        self.m_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        if self.m_tokenizer.pad_token is None:
            self.m_tokenizer.pad_token = self.m_tokenizer.eos_token
        self.m_model = AutoModel.from_pretrained(
            "mistralai/Mistral-7B-v0.1", 
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0"
        )
        self.m_model.eval()

        print(f"Loading NousResearch/Meta-Llama-3-8B...", flush=True)
        self.l_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
        if self.l_tokenizer.pad_token is None:
            self.l_tokenizer.pad_token = self.l_tokenizer.eos_token
        self.l_model = AutoModel.from_pretrained(
            "NousResearch/Meta-Llama-3-8B", 
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:1"
        )
        self.l_model.eval()

    @torch.no_grad()
    def forward(self, ngrams: list[str]) -> np.ndarray:
        cleaned_ngrams = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]

        B = 32
        out = []
        for i in range(0, len(cleaned_ngrams), B):
            batch_texts = cleaned_ngrams[i : i + B]
            
            # Mistral Features (L16 Last, L32 Mean)
            m_inputs = self.m_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            m_inputs = {k: v.to(self.m_model.device) for k, v in m_inputs.items()}
            m_outputs = self.m_model(**m_inputs).hidden_states
            
            m_seq_len = m_inputs['attention_mask'].sum(dim=1) - 1
            m_batch_idx = torch.arange(m_inputs['attention_mask'].shape[0], device=self.m_model.device)
            m_last = m_outputs[16][m_batch_idx, m_seq_len, :]
            
            m_mask = m_inputs['attention_mask'].unsqueeze(-1).expand(m_outputs[32].size()).float()
            m_mean = (m_outputs[32] * m_mask).sum(1) / torch.clamp(m_mask.sum(1), min=1e-9)

            # Llama Features (L16 Last, L32 Mean)
            l_inputs = self.l_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            l_inputs = {k: v.to(self.l_model.device) for k, v in l_inputs.items()}
            l_outputs = self.l_model(**l_inputs).hidden_states
            
            l_seq_len = l_inputs['attention_mask'].sum(dim=1) - 1
            l_batch_idx = torch.arange(l_inputs['attention_mask'].shape[0], device=self.l_model.device)
            l_last = l_outputs[16][l_batch_idx, l_seq_len, :]
            
            l_mask = l_inputs['attention_mask'].unsqueeze(-1).expand(l_outputs[32].size()).float()
            l_mean = (l_outputs[32] * l_mask).sum(1) / torch.clamp(l_mask.sum(1), min=1e-9)

            # Combine all 4 scales
            combined = torch.cat([m_last, m_mean, l_last.to("cuda:0"), l_mean.to("cuda:0")], dim=-1)
            out.append(combined.float().cpu().numpy())
            
        return np.vstack(out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    args = parser.parse_args()

    model_shorthand_name = "Ensemble_Mistral7B_Llama3_QuadScale"
    model_description = "Ensemble of Mistral-7B (L16Last + L32Mean) and Llama-3-8B (L16Last + L32Mean)."
    
    print(f"\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    embedder = EnsembleMistralLlama()
    
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
        n_params=15_000_000_000,
        description=model_description
    )
    upsert_overall_results([row], RESULTS_DIR)
