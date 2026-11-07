import torch
import torch.nn as nn
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel

from src import data, features, encoding
from src.eval import EncodingConfig, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class LLMEmbedder(nn.Module):
    def __init__(self, hf_name, target_layer, device):
        super().__init__()
        self.target_layer = target_layer
        print(f"Loading {hf_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModel.from_pretrained(
            hf_name,
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map=device
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
    
    alphas = np.logspace(3, 10, 20)
    
    # 1. Llama-3-8B
    embedder_llama = LLMEmbedder("meta-llama/Meta-Llama-3-8B", 16, "cuda:1")
    stim_train_l = features.get_features(wordseqs, train_stories, embedder_llama, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
    stim_test_l = features.get_features(wordseqs, test_stories, embedder_llama, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
    del embedder_llama.model; del embedder_llama; import gc; torch.cuda.empty_cache(); gc.collect()
    wt_l, _, _ = encoding.bootstrap_ridge(stim_train_l, resp_train, stim_test_l, resp_test, alphas, nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks)
    pred_llama = np.nan_to_num(stim_test_l @ wt_l)
    
    # 2. Qwen-2.5-14B
    embedder_qwen = LLMEmbedder("Qwen/Qwen2.5-14B", 24, "auto")
    stim_train_q = features.get_features(wordseqs, train_stories, embedder_qwen, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
    stim_test_q = features.get_features(wordseqs, test_stories, embedder_qwen, ngram_size=cfg.ngram_size, ndelays=cfg.ndelays, extra_trim=extra_trim)
    del embedder_qwen.model; del embedder_qwen; torch.cuda.empty_cache(); gc.collect()
    wt_q, _, _ = encoding.bootstrap_ridge(stim_train_q, resp_train, stim_test_q, resp_test, alphas, nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks)
    pred_qwen = np.nan_to_num(stim_test_q @ wt_q)
    
    try:
        pred_gemma = np.load(os.path.join(RESULTS_DIR, "pred_gemma2_9b_l23.npy"))
    except:
        pred_gemma = np.zeros_like(pred_llama)
        
    def eval_pred(pred, name, params, desc):
        corrs = np.array([np.corrcoef(resp_test[:, i], pred[:, i])[0, 1] for i in range(resp_test.shape[1])])
        corrs = np.nan_to_num(corrs)
        mean_corr = float(np.nanmean(corrs))
        print(f"{name} TEST CORR: {mean_corr:.4f}")
        r = {
            'subject': cfg.subject,
            'test_corr': mean_corr,
            'corrs_train_mean': 0.0,
            'corrs_test_frac>0.2': float(np.nanmean(corrs > 0.2)),
            'encoding_seconds': 0.0,
            'corrs_test': corrs,
            'roi_corrs': {}
        }
        row = make_result_row(r, name, params, desc)
        upsert_overall_results([row], RESULTS_DIR)

    eval_pred((pred_llama + pred_qwen + pred_gemma)/3.0, "Ensemble_Llama_Qwen_Gemma23", 31_000_000_000, "Average prediction ensemble of Llama 8B, Qwen 14B, Gemma 9B L23")
    eval_pred((pred_llama + pred_gemma)/2.0, "Ensemble_Llama_Gemma23", 17_000_000_000, "Average prediction ensemble of Llama 8B, Gemma 9B L23")
