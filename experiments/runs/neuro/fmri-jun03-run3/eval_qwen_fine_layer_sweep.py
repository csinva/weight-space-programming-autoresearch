import torch
import torch.nn as nn
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel

from src import data, features, encoding
from src.eval import EncodingConfig, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class SingleModelEmbedder(nn.Module):
    def __init__(self, layer):
        super().__init__()
        hf_name = "Qwen/Qwen2.5-1.5B"
        print(f"Loading {hf_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            hf_name, output_hidden_states=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()
        self.layer = layer

    @torch.no_grad()
    def forward(self, ngrams: list[str]) -> np.ndarray:
        cleaned = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]
        B = 32
        out = []
        for i in range(0, len(cleaned), B):
            batch = cleaned[i:i+B]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            hs = outputs.hidden_states
            seq_len = inputs['attention_mask'].sum(dim=1) - 1
            b_idx = torch.arange(inputs['attention_mask'].shape[0], device=self.model.device)
            out.append(hs[self.layer][b_idx, seq_len, :].float().cpu().numpy())
        return np.vstack(out)

if __name__ == "__main__":
    cfg = EncodingConfig()
    cfg.ngram_size = 20
    
    train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
    all_stories = train_stories + test_stories
    
    wordseqs = data.load_wordseqs(all_stories)
    resps = data.load_responses(all_stories, subject=cfg.subject)
    extra_trim = cfg.edge_trim_trs if cfg.trim_edges else 0
    if extra_trim:
        resps = {s: r[extra_trim:-extra_trim] for s, r in resps.items()}
    resp_train = np.vstack([resps[s] for s in train_stories])
    resp_test = np.vstack([resps[s] for s in test_stories])
    
    alphas = np.logspace(1, 7, 20)
    
    for layer in [11, 12, 13, 14, 15, 16, 17]:
        print(f"\n--- Testing Qwen Layer {layer} Last ---")
        embedder = SingleModelEmbedder(layer)
        
        stim_train = features.get_features(
            wordseqs, train_stories, embedder, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
        stim_test = features.get_features(
            wordseqs, test_stories, embedder, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
            
        wt, corrs, _ = encoding.bootstrap_ridge(
            stim_train, resp_train, stim_test, resp_test, alphas,
            nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
        )
        pred = np.nan_to_num(stim_test @ wt)
        final_corrs = np.array([np.corrcoef(resp_test[:, i], pred[:, i])[0, 1] for i in range(resp_test.shape[1])])
        mean_test_corr = float(np.nanmean(np.nan_to_num(final_corrs)))
        
        print(f"Layer {layer} TEST CORR: {mean_test_corr:.4f}")
        
        r = {
            'subject': cfg.subject,
            'test_corr': mean_test_corr,
            'corrs_train_mean': float(np.nanmean(corrs)),
            'corrs_test_frac>0.2': float(np.nanmean(np.nan_to_num(final_corrs) > 0.2)),
            'encoding_seconds': 0,
            'corrs_test': final_corrs
        }
        
        row = make_result_row(r, f"Qwen1.5B_Context20_L{layer}Last", 1_500_000_000, f"Qwen-2.5-1.5B Layer {layer} Last Token.")
        upsert_overall_results([row], RESULTS_DIR)
        
        del embedder.model
        del embedder
        import gc
        torch.cuda.empty_cache()
        gc.collect()
