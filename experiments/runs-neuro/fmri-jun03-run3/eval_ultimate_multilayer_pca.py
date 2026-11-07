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
        print(f"Loading {hf_name} (Layer {target_layer})...", flush=True)
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
            
            layer_hidden = outputs.hidden_states[self.target_layer]
            
            # Use last token
            seq_len = inputs['attention_mask'].sum(dim=1) - 1
            b_idx = torch.arange(inputs['attention_mask'].shape[0], device=self.model.device)
            feats = layer_hidden[b_idx, seq_len, :].float().cpu().numpy()
            out.append(feats)
        return np.vstack(out)

def get_layer_features(hf_name, layer, device, wordseqs, train_stories, test_stories, cfg, extra_trim):
    embedder = SingleLayerEmbedder(hf_name, layer, device)
    stim_train = features.get_features(wordseqs, train_stories, embedder, cfg.ngram_size, cfg.ndelays, extra_trim)
    stim_test = features.get_features(wordseqs, test_stories, embedder, cfg.ngram_size, cfg.ndelays, extra_trim)
    del embedder.model; del embedder; import gc; torch.cuda.empty_cache(); gc.collect()
    return stim_train, stim_test

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
    
    t0 = time.time()
    
    model_configs = [
        ("meta-llama/Meta-Llama-3-8B", [12, 16, 20], "cuda:0"),
        ("Qwen/Qwen2.5-14B", [16, 24, 32], "cuda:1"),
        ("google/gemma-2-9b", [15, 23, 31], "auto")
    ]
    
    all_stim_train = []
    all_stim_test = []
    
    for hf_name, layers, device in model_configs:
        for layer in layers:
            tr, te = get_layer_features(hf_name, layer, device, wordseqs, train_stories, test_stories, cfg, extra_trim)
            all_stim_train.append(tr)
            all_stim_test.append(te)
            
    for n_comps in [150, 250]:
        print(f"\n--- Ultimate Multi-Layer PCA Super-Embedding (n_components per layer = {n_comps}) ---")
        
        train_pcas = []
        test_pcas = []
        
        for tr, te in zip(all_stim_train, all_stim_test):
            pca = PCA(n_components=n_comps, random_state=42)
            train_pcas.append(pca.fit_transform(tr))
            test_pcas.append(pca.transform(te))
            
        stim_train_super = np.hstack(train_pcas)
        stim_test_super = np.hstack(test_pcas)
        print(f"Super-embedding shape after PCA: {stim_train_super.shape}")
        
        wt, corrs, _ = encoding.bootstrap_ridge(
            stim_train_super, resp_train, stim_test_super, resp_test, alphas,
            nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
        )
        pred = np.nan_to_num(stim_test_super @ wt)
        
        corrs_final = np.array([np.corrcoef(resp_test[:, i], pred[:, i])[0, 1] for i in range(resp_test.shape[1])])
        corrs_final = np.nan_to_num(corrs_final)
        mean_corr = float(np.nanmean(corrs_final))
        print(f"PCA {n_comps} Ultimate Multi-Layer Super-Embedding TEST CORR: {mean_corr:.4f}")
        
        r = {
            'subject': cfg.subject,
            'test_corr': mean_corr,
            'corrs_train_mean': 0.0,
            'corrs_test_frac>0.2': float(np.nanmean(corrs_final > 0.2)),
            'encoding_seconds': time.time() - t0,
            'corrs_test': corrs_final,
        }
        
        # Calculate ROI corrs
        rois = data.load_rois(cfg.subject)
        roi_corrs = {}
        from src.eval import POPULAR_ROIS
        for name in POPULAR_ROIS:
            if name in rois:
                idx = rois[name]
                idx = idx[idx < len(corrs_final)]
                roi_corrs[name] = float(np.nanmean(corrs_final[idx])) if len(idx) else float('nan')
        r['roi_corrs'] = roi_corrs

        model_name = f"Ultimate_MultiLayer_PCA_{n_comps}_LlamaQwenGemma"
        desc = f"PCA dimension reduction ({n_comps} components per layer) across 9 layers total (3 from Llama, 3 from Qwen, 3 from Gemma) before super-embedding Ridge."
        row = make_result_row(r, model_name, 31_000_000_000, desc, "success")
        upsert_overall_results([row], RESULTS_DIR)
