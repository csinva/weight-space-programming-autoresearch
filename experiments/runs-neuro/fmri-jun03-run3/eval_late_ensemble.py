import torch
import torch.nn as nn
import numpy as np
import os
import gc
import time
from transformers import AutoTokenizer, AutoModel

from src import data, features, encoding
from src.eval import EncodingConfig, make_result_row, upsert_overall_results, POPULAR_ROIS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class SingleModelEmbedder(nn.Module):
    def __init__(self, hf_name, last_layer, mean_layer):
        super().__init__()
        self.hf_name = hf_name
        self.last_layer = last_layer
        self.mean_layer = mean_layer
        
        print(f"Loading {hf_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            hf_name, output_hidden_states=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

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
            
            features_to_cat = []
            if self.last_layer is not None:
                last_repr = hs[self.last_layer][b_idx, seq_len, :]
                features_to_cat.append(last_repr)
            
            if self.mean_layer is not None:
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hs[self.mean_layer].size()).float()
                mean_repr = (hs[self.mean_layer] * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
                features_to_cat.append(mean_repr)
            
            combined = torch.cat(features_to_cat, dim=-1)
            out.append(combined.float().cpu().numpy())
        return np.vstack(out)


if __name__ == "__main__":
    t0 = time.time()
    cfg = EncodingConfig()
    cfg.ngram_size = 20  # the breakthrough context length
    
    train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
    all_stories = train_stories + test_stories
    
    print("Loading fMRI data...")
    wordseqs = data.load_wordseqs(all_stories)
    resps = data.load_responses(all_stories, subject=cfg.subject)
    
    extra_trim = cfg.edge_trim_trs if cfg.trim_edges else 0
    if extra_trim:
        resps = {s: r[extra_trim:-extra_trim] for s, r in resps.items()}
    resp_train = np.vstack([resps[s] for s in train_stories])
    resp_test = np.vstack([resps[s] for s in test_stories])
    
    models_to_run = [
        ("Qwen/Qwen2.5-1.5B", 14, None),
        ("mistralai/Mistral-7B-v0.1", 16, 32),
    ]
    
    all_preds = []
    alphas = np.logspace(1, 7, 20)
    
    for hf_name, last_layer, mean_layer in models_to_run:
        print(f"\n======================================")
        print(f"--- Processing {hf_name} ---")
        embedder = SingleModelEmbedder(hf_name, last_layer, mean_layer)
        
        print("Extracting features...")
        stim_train = features.get_features(wordseqs, train_stories, embedder, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
        stim_test = features.get_features(wordseqs, test_stories, embedder, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
        
        del embedder.model
        del embedder
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Fitting ridge for {hf_name}...")
        wt, _, _ = encoding.bootstrap_ridge(
            stim_train, resp_train, stim_test, resp_test, alphas,
            nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
        )
        
        pred = np.nan_to_num(stim_test @ wt)
        all_preds.append(pred)
        
    print("\n======================================")
    print("--- Ensembling Predictions (Late Ensemble) ---")
    avg_pred = np.mean(all_preds, axis=0)
    
    final_corrs = np.array([np.corrcoef(resp_test[:, i], avg_pred[:, i])[0, 1] for i in range(resp_test.shape[1])])
    final_corrs = np.nan_to_num(final_corrs)
    
    mean_test_corr = float(np.nanmean(final_corrs))
    frac_above_2 = float(np.nanmean(final_corrs > 0.2))
    
    print(f"LATE ENSEMBLE TEST CORR: {mean_test_corr:.4f}")
    print(f"LATE ENSEMBLE FRAC > 0.2: {frac_above_2:.4f}")
    
    r = {
        'subject': cfg.subject,
        'test_corr': mean_test_corr,
        'corrs_train_mean': 0.0,
        'corrs_test_frac>0.2': frac_above_2,
        'encoding_seconds': time.time() - t0,
        'corrs_test': final_corrs
    }
    
    rois = data.load_rois(cfg.subject)
    roi_corrs = {}
    for name in POPULAR_ROIS:
        if name in rois:
            idx = rois[name]
            idx = idx[idx < len(final_corrs)]
            roi_corrs[name] = float(np.nanmean(final_corrs[idx])) if len(idx) else float('nan')
    r['roi_corrs'] = roi_corrs

    model_shorthand_name = "Late_Ensemble_Optimal_Context20"
    model_description = "Prediction-space average (Late Ensemble) of optimal Qwen1.5B + Mistral7B (Context=20). Tests late integration."
    
    row = make_result_row(r, model_shorthand_name, 8_500_000_000, model_description)
    upsert_overall_results([row], RESULTS_DIR)
