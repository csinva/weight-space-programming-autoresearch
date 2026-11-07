import torch
import torch.nn as nn
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel, AutoConfig

from src import data, features, encoding
from src.eval import EncodingConfig, make_result_row, upsert_overall_results, POPULAR_ROIS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class Qwen14BEmbedder(nn.Module):
    def __init__(self, target_layer):
        super().__init__()
        self.target_layer = target_layer
        hf_name = "Qwen/Qwen2.5-14B"
        print(f"Loading {hf_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModel.from_pretrained(
            hf_name,
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:1"
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, ngrams: list[str]) -> np.ndarray:
        cleaned = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]
        B = 16 
        out = []
        for i in range(0, len(cleaned), B):
            batch = cleaned[i:i+B]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            hs = outputs.hidden_states
            seq_len = inputs['attention_mask'].sum(dim=1) - 1
            b_idx = torch.arange(inputs['attention_mask'].shape[0], device=self.model.device)
            out.append(hs[self.target_layer][b_idx, seq_len, :].float().cpu().numpy())
        return np.vstack(out)

if __name__ == "__main__":
    cfg = EncodingConfig()
    cfg.ngram_size = 20
    # USE FULL TRAIN/TEST SPLIT
    cfg.num_train = 8
    cfg.num_test = 2
    cfg.ndelays = 4
    
    train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
    all_stories = train_stories + test_stories
    
    wordseqs = data.load_wordseqs(all_stories)
    resps = data.load_responses(all_stories, subject=cfg.subject)
    extra_trim = cfg.edge_trim_trs if cfg.trim_edges else 0
    if extra_trim:
        resps = {s: r[extra_trim:-extra_trim] for s, r in resps.items()}
    resp_train = np.vstack([resps[s] for s in train_stories])
    resp_test = np.vstack([resps[s] for s in test_stories])
    
    for layer in [20, 22, 24, 26, 28]:
        print(f"\n--- Testing Qwen2.5-14B Layer {layer} ---")
        t0 = time.time()
        
        embedder = Qwen14BEmbedder(target_layer=layer)
        
        stim_train = features.get_features(
            wordseqs, train_stories, embedder, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
        stim_test = features.get_features(
            wordseqs, test_stories, embedder, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
            
        del embedder.model
        del embedder
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        alphas = np.logspace(3, 10, 20)
        wt, corrs, _ = encoding.bootstrap_ridge(
            stim_train, resp_train, stim_test, resp_test, alphas,
            nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
        )
        pred = np.nan_to_num(stim_test @ wt)
        final_corrs = np.array([np.corrcoef(resp_test[:, i], pred[:, i])[0, 1] for i in range(resp_test.shape[1])])
        final_corrs = np.nan_to_num(final_corrs)
        mean_test_corr = float(np.nanmean(final_corrs))
        
        print(f"Layer {layer} TEST CORR: {mean_test_corr:.4f}")
        
        r = {
            'subject': cfg.subject,
            'test_corr': mean_test_corr,
            'corrs_train_mean': float(np.nanmean(corrs)),
            'corrs_test_frac>0.2': float(np.nanmean(final_corrs > 0.2)),
            'encoding_seconds': time.time() - t0,
            'corrs_test': final_corrs
        }
        
        rois = data.load_rois(cfg.subject)
        roi_corrs = {}
        for roi_name in POPULAR_ROIS:
            if roi_name in rois:
                idx = rois[roi_name]
                idx = idx[idx < len(final_corrs)]
                roi_corrs[roi_name] = float(np.nanmean(final_corrs[idx])) if len(idx) else float('nan')
        r['roi_corrs'] = roi_corrs
        
        row = make_result_row(r, f"Qwen14B_Context20_L{layer}Last_8_2", 14_000_000_000, f"Qwen-2.5-14B Layer {layer} Last Token with full split.")
        upsert_overall_results([row], RESULTS_DIR)

