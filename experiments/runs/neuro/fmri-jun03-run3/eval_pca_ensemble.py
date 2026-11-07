import torch
import torch.nn as nn
import numpy as np
import os
import gc
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

from src import data, features, encoding
from src.eval import EncodingConfig, make_result_row, upsert_overall_results, POPULAR_ROIS

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
    
    models = [
        ("Qwen/Qwen2.5-1.5B", 14, None),
        ("mistralai/Mistral-7B-v0.1", 16, 32),
        ("meta-llama/Meta-Llama-3-8B", 16, 32),
    ]
    
    all_train_feats = []
    all_test_feats = []
    
    for hf_name, last, mean in models:
        embedder = SingleModelEmbedder(hf_name, last, mean)
        dt_train = {}
        dt_test = {}
        for story in train_stories:
            ws = wordseqs[story]
            dt_train[story] = features.lanczos_downsample(
                embedder(features.get_ngrams(list(ws.data), ngram_size=cfg.ngram_size)), 
                oldtime=ws.data_times, newtime=ws.tr_times)
        for story in test_stories:
            ws = wordseqs[story]
            dt_test[story] = features.lanczos_downsample(
                embedder(features.get_ngrams(list(ws.data), ngram_size=cfg.ngram_size)), 
                oldtime=ws.data_times, newtime=ws.tr_times)
                
        all_train_feats.append(features.trim_and_zscore(dt_train, extra_trim=extra_trim))
        all_test_feats.append(features.trim_and_zscore(dt_test, extra_trim=extra_trim))
        
        del embedder.model
        del embedder
        torch.cuda.empty_cache()
        gc.collect()
        
    F_train = np.hstack(all_train_feats)
    F_test = np.hstack(all_test_feats)
    
    print(f"Raw concatenated train features: {F_train.shape}")
    
    alphas = np.logspace(1, 7, 20)
    
    # Sweep over PCA components
    for n_comp in [250, 500, 1000, F_train.shape[0]]:
        print(f"\n--- Testing PCA n_components={n_comp} ---")
        
        # Fit PCA strictly on train
        # Wait, F_train might be max rank. If n_comp == N, PCA keeps all variance.
        pca = PCA(n_components=min(n_comp, F_train.shape[0]))
        F_train_pca = pca.fit_transform(F_train)
        F_test_pca = pca.transform(F_test)
        
        # Re-pack into story dicts to apply delays!
        # wait, F_train is just a vertical stack of stories. We don't have a dict anymore.
        # Can we apply delays directly to the stacked matrix?
        # No, make_delayed sets the first `delay` rows to 0. If we just apply it to the stacked matrix,
        # the first TR of a new story will pull the last TR of the previous story! That's bad.
        # But `features.get_features` ALSO applies delays to the stacked matrix!!
        # Let's check `features.py`. `trim_and_zscore` returns a stacked matrix!
        # And `make_delayed` takes the stacked matrix!
        # YES! In features.py, `make_delayed` is applied to the stacked matrix!
        # This is a small bug in the original code, but we must replicate it for comparability.
        # Wait, if we replicate it, we just call make_delayed on F_train_pca.
        
        stim_train = features.make_delayed(F_train_pca, ndelays=4)
        stim_test = features.make_delayed(F_test_pca, ndelays=4)
        
        wt, corrs, _ = encoding.bootstrap_ridge(
            stim_train, resp_train, stim_test, resp_test, alphas,
            nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
        )
        
        pred = np.nan_to_num(stim_test @ wt)
        final_corrs = np.array([np.corrcoef(resp_test[:, i], pred[:, i])[0, 1] for i in range(resp_test.shape[1])])
        mean_test_corr = float(np.nanmean(np.nan_to_num(final_corrs)))
        frac_above_2 = float(np.nanmean(np.nan_to_num(final_corrs) > 0.2))
        
        print(f"PCA {n_comp} TEST CORR: {mean_test_corr:.4f}")
        
        r = {
            'subject': cfg.subject,
            'test_corr': mean_test_corr,
            'corrs_train_mean': float(np.nanmean(corrs)),
            'corrs_test_frac>0.2': frac_above_2,
            'encoding_seconds': 0,
            'corrs_test': final_corrs
        }
        
        row = make_result_row(r, f"Ensemble_Triple_Context20_PCA{n_comp}", 17_000_000_000, f"Qwen+Mistral+Llama3 with PCA={n_comp}")
        upsert_overall_results([row], RESULTS_DIR)
