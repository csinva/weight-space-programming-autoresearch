import torch
import torch.nn as nn
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel

from src import data, features, encoding
from src.eval import EncodingConfig, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class SingleLayerEmbedder(nn.Module):
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
    
    embedder = SingleLayerEmbedder("meta-llama/Meta-Llama-3-8B", 16, "cuda:0")
    
    context_sizes = [5, 20, 50]
    train_feats = []
    test_feats = []
    
    for c in context_sizes:
        print(f"Extracting features for Context={c}...")
        s_train = features.get_features(wordseqs, train_stories, embedder, c, cfg.ndelays, extra_trim)
        s_test = features.get_features(wordseqs, test_stories, embedder, c, cfg.ndelays, extra_trim)
        train_feats.append(s_train)
        test_feats.append(s_test)
        
    del embedder.model; del embedder; import gc; torch.cuda.empty_cache(); gc.collect()
    
    print("Concatenating Multi-Scale Embeddings...")
    stim_train_super = np.hstack(train_feats)
    stim_test_super = np.hstack(test_feats)
    
    print(f"Multi-scale embedding shape: {stim_train_super.shape}")
    
    print("Training Ridge regression on Multi-Scale Embedding...")
    wt, corrs, _ = encoding.bootstrap_ridge(
        stim_train_super, resp_train, stim_test_super, resp_test, alphas,
        nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
    )
    pred = np.nan_to_num(stim_test_super @ wt)
    
    corrs_final = np.array([np.corrcoef(resp_test[:, i], pred[:, i])[0, 1] for i in range(resp_test.shape[1])])
    corrs_final = np.nan_to_num(corrs_final)
    mean_corr = float(np.nanmean(corrs_final))
    print(f"Multi-Scale Super-Embedding TEST CORR: {mean_corr:.4f}")
    
    r = {
        'subject': cfg.subject,
        'test_corr': mean_corr,
        'corrs_train_mean': 0.0,
        'corrs_test_frac>0.2': float(np.nanmean(corrs_final > 0.2)),
        'encoding_seconds': 0.0,
        'corrs_test': corrs_final,
        'roi_corrs': {}
    }
    row = make_result_row(r, "MultiScale_Llama3_8B_L16_C5_20_50", 8_000_000_000, "Concatenated multi-scale contexts (5, 20, 50) of Llama-3-8B L16.")
    upsert_overall_results([row], RESULTS_DIR)
