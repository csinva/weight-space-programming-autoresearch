import torch
import torch.nn as nn
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

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
        B = 2 
        out = []
        for i in range(0, len(cleaned), B):
            batch = cleaned[i:i+B]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1500)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            hs = outputs.hidden_states
            seq_len = inputs['attention_mask'].sum(dim=1) - 1
            b_idx = torch.arange(inputs['attention_mask'].shape[0], device=self.model.device)
            out.append(hs[self.target_layer][b_idx, seq_len, :].float().cpu().numpy())
        return np.vstack(out)

def evaluate_model(X_train, X_test, subject, model_name, n_params, desc, cfg):
    print(f"\nEvaluating {model_name} on {subject}...")
    train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
    all_stories = train_stories + test_stories
    
    resps = data.load_responses(all_stories, subject=subject)
    extra_trim = cfg.edge_trim_trs if cfg.trim_edges else 0
    if extra_trim:
        resps = {s: r[extra_trim:-extra_trim] for s, r in resps.items()}
    Y_train = np.vstack([resps[s] for s in train_stories])
    Y_test = np.vstack([resps[s] for s in test_stories])
    
    alphas = np.logspace(3, 10, 20)
    
    t0 = time.time()
    wt, corrs, _ = encoding.bootstrap_ridge(
        X_train, Y_train, X_test, Y_test, alphas,
        nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
    )
    pred = np.nan_to_num(X_test @ wt)
    
    corrs_final = np.array([np.corrcoef(Y_test[:, i], pred[:, i])[0, 1] for i in range(Y_test.shape[1])])
    corrs_final = np.nan_to_num(corrs_final)
    mean_corr = float(np.nanmean(corrs_final))
    print(f"{subject} {model_name} TEST CORR: {mean_corr:.4f}")
    
    r = {
        'subject': subject,
        'test_corr': mean_corr,
        'corrs_train_mean': 0.0,
        'corrs_test_frac>0.2': float(np.nanmean(corrs_final > 0.2)),
        'encoding_seconds': time.time() - t0,
        'corrs_test': corrs_final,
    }
    
    # Calculate ROI corrs
    rois = data.load_rois(subject)
    roi_corrs = {}
    from src.eval import POPULAR_ROIS
    for name in POPULAR_ROIS:
        if name in rois:
            idx = rois[name]
            idx = idx[idx < len(corrs_final)]
            roi_corrs[name] = float(np.nanmean(corrs_final[idx])) if len(idx) else float('nan')
    r['roi_corrs'] = roi_corrs

    row = make_result_row(r, model_name, n_params, desc, "success")
    upsert_overall_results([row], RESULTS_DIR)

if __name__ == "__main__":
    cfg = EncodingConfig()
    cfg.ngram_size = 150
    cfg.num_train = 8
    cfg.num_test = 2
    cfg.ndelays = 4
    
    train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
    all_stories = train_stories + test_stories
    wordseqs = data.load_wordseqs(all_stories)
    extra_trim = cfg.edge_trim_trs if cfg.trim_edges else 0

    print("\n--- Extracting Context 150 Semantic Features ---")
    embedder1 = SingleLayerEmbedder("meta-llama/Meta-Llama-3-8B", 16, "cuda:0")
    stim_train_1 = features.get_features(wordseqs, train_stories, embedder1, cfg.ngram_size, cfg.ndelays, extra_trim)
    stim_test_1 = features.get_features(wordseqs, test_stories, embedder1, cfg.ngram_size, cfg.ndelays, extra_trim)
    del embedder1.model; del embedder1; import gc; torch.cuda.empty_cache(); gc.collect()
    
    embedder2 = SingleLayerEmbedder("Qwen/Qwen2.5-14B", 24, "cuda:1")
    stim_train_2 = features.get_features(wordseqs, train_stories, embedder2, cfg.ngram_size, cfg.ndelays, extra_trim)
    stim_test_2 = features.get_features(wordseqs, test_stories, embedder2, cfg.ngram_size, cfg.ndelays, extra_trim)
    del embedder2.model; del embedder2; torch.cuda.empty_cache(); gc.collect()
    
    embedder3 = SingleLayerEmbedder("google/gemma-2-9b", 23, "auto")
    stim_train_3 = features.get_features(wordseqs, train_stories, embedder3, cfg.ngram_size, cfg.ndelays, extra_trim)
    stim_test_3 = features.get_features(wordseqs, test_stories, embedder3, cfg.ngram_size, cfg.ndelays, extra_trim)
    del embedder3.model; del embedder3; torch.cuda.empty_cache(); gc.collect()

    # PCA Compression
    n_comps = 500
    pca1 = PCA(n_components=n_comps, random_state=42)
    st1_pca = pca1.fit_transform(stim_train_1)
    st1_test_pca = pca1.transform(stim_test_1)
    
    pca2 = PCA(n_components=n_comps, random_state=42)
    st2_pca = pca2.fit_transform(stim_train_2)
    st2_test_pca = pca2.transform(stim_test_2)
    
    pca3 = PCA(n_components=n_comps, random_state=42)
    st3_pca = pca3.fit_transform(stim_train_3)
    st3_test_pca = pca3.transform(stim_test_3)
    
    stim_train_semantic = np.hstack([st1_pca, st2_pca, st3_pca])
    stim_test_semantic = np.hstack([st1_test_pca, st2_test_pca, st3_test_pca])
    
    sem_name = f"SuperEmbedding_PCA_500_context150_LlamaQwenGemma"
    sem_desc = "Testing Context 150 integration limit across subjects."

    evaluate_model(stim_train_semantic, stim_test_semantic, "UTS01", sem_name, 31_000_000_000, sem_desc, cfg)
    evaluate_model(stim_train_semantic, stim_test_semantic, "UTS02", sem_name, 31_000_000_000, sem_desc, cfg)
