import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class EnsembleMistralLlamaGPT2(nn.Module):
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
            device_map="cuda:2"
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
            device_map="cuda:2"
        )
        self.l_model.eval()

        print(f"Loading GPT-2 XL...", flush=True)
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.gpt2_model = AutoModel.from_pretrained(
            "gpt2-xl", 
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:3"
        )
        self.gpt2_model.eval()

    @torch.no_grad()
    def forward(self, ngrams: list[str]) -> np.ndarray:
        cleaned_ngrams = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]

        B = 16
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

            # GPT2 Features (L24 Last, L48 Mean)
            g_inputs = self.gpt2_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            g_inputs = {k: v.to(self.gpt2_model.device) for k, v in g_inputs.items()}
            g_outputs = self.gpt2_model(**g_inputs).hidden_states
            
            g_seq_len = g_inputs['attention_mask'].sum(dim=1) - 1
            g_batch_idx = torch.arange(g_inputs['attention_mask'].shape[0], device=self.gpt2_model.device)
            g_last = g_outputs[24][g_batch_idx, g_seq_len, :]
            
            g_mask = g_inputs['attention_mask'].unsqueeze(-1).expand(g_outputs[48].size()).float()
            g_mean = (g_outputs[48] * g_mask).sum(1) / torch.clamp(g_mask.sum(1), min=1e-9)


            # Combine all 6 scales
            combined = torch.cat([m_last, m_mean, l_last.to("cuda:2"), l_mean.to("cuda:2"), g_last.to("cuda:2"), g_mean.to("cuda:2")], dim=-1)
            out.append(combined.float().cpu().numpy())
            
        return np.vstack(out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    args = parser.parse_args()

    model_shorthand_name = "Ensemble_Mistral_Llama_GPT2_HexaScale"
    model_description = "Ensemble of Mistral-7B, Llama-3-8B, and GPT-2-XL DualScales."
    
    print(f"\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    embedder = EnsembleMistralLlamaGPT2()
    
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
        n_params=16_500_000_000,
        description=model_description
    )
    upsert_overall_results([row], RESULTS_DIR)
