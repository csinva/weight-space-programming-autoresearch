import torch
import torch.nn as nn
import numpy as np
import os
import gc
import time
from transformers import AutoTokenizer, AutoModel

from src import data, features, encoding
from src.eval import EncodingConfig, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class SingleModelEmbedder(nn.Module):
    def __init__(self, hf_name, last_layer, mean_layer):
        super().__init__()
        print(f"Loading {hf_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            hf_name, output_hidden_states=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()
        self.last_layer = last_layer
        self.mean_layer = mean_layer

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
                features_to_cat.append(hs[self.last_layer][b_idx, seq_len, :])
            if self.mean_layer is not None:
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hs[self.mean_layer].size()).float()
                mean_repr = (hs[self.mean_layer] * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
                features_to_cat.append(mean_repr)
            
            out.append(torch.cat(features_to_cat, dim=-1).float().cpu().numpy())
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
    
    # Extract features BEFORE delays
    embedder1 = SingleModelEmbedder("Qwen/Qwen2.5-1.5B", 14, None)
    
    downsampled_train1 = {}
    downsampled_test1 = {}
    for story in train_stories:
        ws = wordseqs[story]
        ngrams = features.get_ngrams(list(ws.data), ngram_size=cfg.ngram_size)
        downsampled_train1[story] = features.lanczos_downsample(
            embedder1(ngrams), oldtime=ws.data_times, newtime=ws.tr_times)
    for story in test_stories:
        ws = wordseqs[story]
        ngrams = features.get_ngrams(list(ws.data), ngram_size=cfg.ngram_size)
        downsampled_test1[story] = features.lanczos_downsample(
            embedder1(ngrams), oldtime=ws.data_times, newtime=ws.tr_times)
            
    del embedder1.model
    del embedder1
    torch.cuda.empty_cache()
    gc.collect()

    embedder2 = SingleModelEmbedder("mistralai/Mistral-7B-v0.1", 16, 32)
    
    downsampled_train2 = {}
    downsampled_test2 = {}
    for story in train_stories:
        ws = wordseqs[story]
        ngrams = features.get_ngrams(list(ws.data), ngram_size=cfg.ngram_size)
        downsampled_train2[story] = features.lanczos_downsample(
            embedder2(ngrams), oldtime=ws.data_times, newtime=ws.tr_times)
    for story in test_stories:
        ws = wordseqs[story]
        ngrams = features.get_ngrams(list(ws.data), ngram_size=cfg.ngram_size)
        downsampled_test2[story] = features.lanczos_downsample(
            embedder2(ngrams), oldtime=ws.data_times, newtime=ws.tr_times)
            
    del embedder2.model
    del embedder2
    torch.cuda.empty_cache()
    gc.collect()
    
    # Combine features before delays
    downsampled_train = {}
    for story in train_stories:
        downsampled_train[story] = np.hstack([downsampled_train1[story], downsampled_train2[story]])
    downsampled_test = {}
    for story in test_stories:
        downsampled_test[story] = np.hstack([downsampled_test1[story], downsampled_test2[story]])
        
    feats_train = features.trim_and_zscore(downsampled_train, extra_trim=extra_trim)
    feats_test = features.trim_and_zscore(downsampled_test, extra_trim=extra_trim)
    
    alphas = np.logspace(1, 7, 20)
    
    for ndelays in [1, 2, 3, 4, 5, 6, 8]:
        stim_train = features.make_delayed(feats_train, ndelays=ndelays)
        stim_test = features.make_delayed(feats_test, ndelays=ndelays)
        
        # Responses need to be aligned with the delayed stimulus!
        # wait, `make_delayed` in features.py does NOT trim responses.
        # Let's check features.py: `make_delayed` just shifts the matrix. Wait! 
        # Oh no! The original features.py `make_delayed` returns a matrix of shape (N, ndelays * D) where N is unchanged!
        # It sets the first `delay` rows to 0. 
        # Let's verify features.py
        
        print(f"\n--- Testing ndelays={ndelays} ---")
        wt, corrs, _ = encoding.bootstrap_ridge(
            stim_train, resp_train, stim_test, resp_test, alphas,
            nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
        )
        pred = np.nan_to_num(stim_test @ wt)
        final_corrs = np.array([np.corrcoef(resp_test[:, i], pred[:, i])[0, 1] for i in range(resp_test.shape[1])])
        mean_test_corr = float(np.nanmean(np.nan_to_num(final_corrs)))
        frac_above_2 = float(np.nanmean(np.nan_to_num(final_corrs) > 0.2))
        
        print(f"NDELAYS {ndelays} TEST CORR: {mean_test_corr:.4f}")
        
        r = {
            'subject': cfg.subject,
            'test_corr': mean_test_corr,
            'corrs_train_mean': float(np.nanmean(corrs)),
            'corrs_test_frac>0.2': frac_above_2,
            'encoding_seconds': 0,
            'corrs_test': final_corrs
        }
        
        row = make_result_row(r, f"Ensemble_Ultimate_Context20_ndelays{ndelays}", 8_500_000_000, f"Qwen+Mistral with ndelays={ndelays}")
        upsert_overall_results([row], RESULTS_DIR)
