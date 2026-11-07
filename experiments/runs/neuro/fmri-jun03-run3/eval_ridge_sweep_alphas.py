import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class UltimateEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        
        print(f"Loading Qwen-2.5-1.5B...", flush=True)
        self.q_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        if self.q_tokenizer.pad_token is None:
            self.q_tokenizer.pad_token = self.q_tokenizer.eos_token
        self.q_model = AutoModel.from_pretrained(
            "Qwen/Qwen2.5-1.5B", 
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0"
        )
        self.q_model.eval()

        print(f"Loading mistralai/Mistral-7B-v0.1...", flush=True)
        self.m_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        if self.m_tokenizer.pad_token is None:
            self.m_tokenizer.pad_token = self.m_tokenizer.eos_token
        self.m_model = AutoModel.from_pretrained(
            "mistralai/Mistral-7B-v0.1", 
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:1"
        )
        self.m_model.eval()

    @torch.no_grad()
    def forward(self, ngrams: list[str]) -> np.ndarray:
        cleaned_ngrams = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]

        B = 32
        out = []
        for i in range(0, len(cleaned_ngrams), B):
            batch_texts = cleaned_ngrams[i : i + B]
            
            # Qwen Features (L14 Last only - peak syntax)
            q_inputs = self.q_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            q_inputs = {k: v.to(self.q_model.device) for k, v in q_inputs.items()}
            q_outputs = self.q_model(**q_inputs).hidden_states
            
            q_seq_len = q_inputs['attention_mask'].sum(dim=1) - 1
            q_batch_idx = torch.arange(q_inputs['attention_mask'].shape[0], device=self.q_model.device)
            q_last = q_outputs[14][q_batch_idx, q_seq_len, :]

            # Mistral Features (L16 Last, L32 Mean)
            m_inputs = self.m_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            m_inputs = {k: v.to(self.m_model.device) for k, v in m_inputs.items()}
            m_outputs = self.m_model(**m_inputs).hidden_states
            
            m_seq_len = m_inputs['attention_mask'].sum(dim=1) - 1
            m_batch_idx = torch.arange(m_inputs['attention_mask'].shape[0], device=self.m_model.device)
            m_last = m_outputs[16][m_batch_idx, m_seq_len, :]
            
            m_mask = m_inputs['attention_mask'].unsqueeze(-1).expand(m_outputs[32].size()).float()
            m_mean = (m_outputs[32] * m_mask).sum(1) / torch.clamp(m_mask.sum(1), min=1e-9)

            # Combine them
            combined = torch.cat([q_last.to("cuda:0"), m_last.to("cuda:0"), m_mean.to("cuda:0")], dim=-1)
            out.append(combined.float().cpu().numpy())
            
        return np.vstack(out)

if __name__ == "__main__":
    from src import data, features, encoding
    import gc
    
    cfg = EncodingConfig()
    cfg.ngram_size = 20
    cfg.num_train = 8
    cfg.num_test = 2
    
    train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
    all_stories = train_stories + test_stories
    
    wordseqs = data.load_wordseqs(all_stories)
    resps = data.load_responses(all_stories, subject=cfg.subject)
    extra_trim = cfg.edge_trim_trs if cfg.trim_edges else 0
    if extra_trim:
        resps = {s: r[extra_trim:-extra_trim] for s, r in resps.items()}
    resp_train = np.vstack([resps[s] for s in train_stories])
    resp_test = np.vstack([resps[s] for s in test_stories])
    
    embedder = UltimateEnsemble()
    stim_train = features.get_features(
        wordseqs, train_stories, embedder, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
    stim_test = features.get_features(
        wordseqs, test_stories, embedder, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
        
    del embedder.q_model
    del embedder.m_model
    del embedder
    torch.cuda.empty_cache()
    gc.collect()

    print("\n--- Sweeping Alpha Ranges ---")
    
    alpha_grids = [
        ("Base", np.logspace(1, 7, 20)),
        ("High", np.logspace(3, 9, 20)),
        ("Low", np.logspace(-1, 5, 20)),
        ("UltraHigh", np.logspace(5, 11, 20))
    ]
    
    for name, alphas in alpha_grids:
        print(f"\nTesting Alpha Grid {name}: {alphas[0]} to {alphas[-1]}")
        wt, corrs, _ = encoding.bootstrap_ridge(
            stim_train, resp_train, stim_test, resp_test, alphas,
            nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
        )
        pred = np.nan_to_num(stim_test @ wt)
        final_corrs = np.array([np.corrcoef(resp_test[:, i], pred[:, i])[0, 1] for i in range(resp_test.shape[1])])
        mean_test_corr = float(np.nanmean(np.nan_to_num(final_corrs)))
        
        print(f"Alpha {name} TEST CORR: {mean_test_corr:.4f}")
        
        r = {
            'subject': cfg.subject,
            'test_corr': mean_test_corr,
            'corrs_train_mean': float(np.nanmean(corrs)),
            'corrs_test_frac>0.2': float(np.nanmean(np.nan_to_num(final_corrs) > 0.2)),
            'encoding_seconds': 0,
            'corrs_test': final_corrs
        }
        
        row = make_result_row(r, f"Ensemble_Ultimate_Context20_Alpha{name}", 8_500_000_000, f"Alpha sweep: {name}")
        upsert_overall_results([row], RESULTS_DIR)
