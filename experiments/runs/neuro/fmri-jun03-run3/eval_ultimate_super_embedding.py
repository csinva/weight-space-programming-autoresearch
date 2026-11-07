import torch
import torch.nn as nn
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel

from src import data, features, encoding
from src.eval import EncodingConfig, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class MultiLayerEmbedder(nn.Module):
    def __init__(self, hf_name, target_layers, device):
        super().__init__()
        self.target_layers = target_layers
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
            
            layer_feats = []
            for layer in self.target_layers:
                layer_feats.append(hs[layer][b_idx, seq_len, :].float().cpu().numpy())
            
            # Concatenate along the hidden dimension (axis=1)
            out.append(np.hstack(layer_feats))
            
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
        for s in all_stories:
            resps[s] = resps[s][extra_trim:-extra_trim]
    
    models_to_run = [
        ("meta-llama/Meta-Llama-3-8B", [14, 16, 18], "cuda:0"),
        ("Qwen/Qwen2.5-14B", [22, 24, 26], "cuda:0"),
        ("google/gemma-2-9b", [21, 23, 25], "cuda:0"),
        ("mistralai/Mistral-Nemo-Base-2407", [18, 20, 22], "cuda:0")
    ]
    
    stim_train_all = []
    stim_test_all = []
    
    for hf_name, layers, dev in models_to_run:
        embedder = MultiLayerEmbedder(hf_name, layers, dev)
        
        X_train = features.get_features(wordseqs, train_stories, embedder, cfg.ngram_size, cfg.ndelays, extra_trim)
        X_test = features.get_features(wordseqs, test_stories, embedder, cfg.ngram_size, cfg.ndelays, extra_trim)
        
        stim_train_all.append(X_train)
        stim_test_all.append(X_test)
        
        # Free memory aggressively
        del embedder.model
        del embedder
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print("Concatenating all multi-layer models...")
    X_train_super = np.hstack(stim_train_all)
    X_test_super = np.hstack(stim_test_all)
    
    Y_train = np.vstack([resps[s] for s in train_stories])
    Y_test = np.vstack([resps[s] for s in test_stories])
    
    print(f"Super-embedding shape: {X_train_super.shape}")
    
    alphas = np.logspace(3, 10, 20)
    start_t = time.time()
    wt, corrs, _ = encoding.bootstrap_ridge(
        X_train_super, Y_train, X_test_super, Y_test, alphas,
        nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
    )
    pred = np.nan_to_num(X_test_super @ wt)
    corr = np.array([np.corrcoef(Y_test[:, i], pred[:, i])[0, 1] for i in range(Y_test.shape[1])])
    corr = np.nan_to_num(corr)
    end_t = time.time()
    
    test_corr = np.nanmean(corr)
    print(f"Ultimate 4-Model Multi-Layer Super-Embedding TEST CORR: {test_corr:.4f}")
    
    r = {
        'subject': cfg.subject,
        'test_corr': test_corr,
        'corrs_train_mean': 0.0,
        'corrs_test_frac>0.2': float(np.nanmean(corr > 0.2)),
        'encoding_seconds': end_t - start_t,
    }
    
    # Calculate ROI corrs
    rois = data.load_rois(cfg.subject)
    roi_corrs = {}
    from src.eval import POPULAR_ROIS
    for name in POPULAR_ROIS:
        if name in rois:
            idx = rois[name]
            idx = idx[idx < len(corr)]
            roi_corrs[name] = float(np.nanmean(corr[idx])) if len(idx) else float('nan')
    r['roi_corrs'] = roi_corrs

    row = make_result_row(
        r=r,
        status="success",
        model_shorthand_name="Ultimate_SuperEmbedding_4Models_MultiLayer",
        n_params=4e10,
        description="Concatenated features of Llama, Qwen, Gemma, Mistral, 3 peak layers each.",
    )
    
    upsert_overall_results([row], RESULTS_DIR)
    print("Saved!")
