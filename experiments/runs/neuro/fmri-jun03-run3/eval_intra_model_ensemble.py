import torch
import torch.nn as nn
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel

from src import data, features, encoding
from src.eval import EncodingConfig, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class LLMMultiLayerEmbedder(nn.Module):
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
    def forward(self, ngrams: list[str]) -> dict:
        cleaned = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]
        B = 16 
        outs = {layer: [] for layer in self.target_layers}
        for i in range(0, len(cleaned), B):
            batch = cleaned[i:i+B]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            hs = outputs.hidden_states
            seq_len = inputs['attention_mask'].sum(dim=1) - 1
            b_idx = torch.arange(inputs['attention_mask'].shape[0], device=self.model.device)
            for layer in self.target_layers:
                outs[layer].append(hs[layer][b_idx, seq_len, :].float().cpu().numpy())
        
        for layer in self.target_layers:
            outs[layer] = np.vstack(outs[layer])
        return outs

def get_multi_layer_features(wordseqs, stories, embedder, ngram_size, ndelays, extra_trim):
    from src.features import get_ngrams, lanczos_downsample, trim_and_zscore, make_delayed
    
    downsampled = {layer: {} for layer in embedder.target_layers}
    for story in stories:
        ws = wordseqs[story]
        ngrams = get_ngrams(list(ws.data), ngram_size=ngram_size)
        out_dict = embedder(ngrams)
        for layer in embedder.target_layers:
            downsampled[layer][story] = lanczos_downsample(
                out_dict[layer], oldtime=ws.data_times, newtime=ws.tr_times
            )
            
    final_stim = {}
    for layer in embedder.target_layers:
        feats = trim_and_zscore(downsampled[layer], extra_trim=extra_trim)
        final_stim[layer] = make_delayed(feats, ndelays=ndelays)
    return final_stim

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
    
    # Target layers around midpoint
    target_layers = [14, 15, 16, 17, 18]
    embedder = LLMMultiLayerEmbedder("meta-llama/Meta-Llama-3-8B", target_layers, "cuda:1")
    
    stim_train_dict = get_multi_layer_features(wordseqs, train_stories, embedder, cfg.ngram_size, cfg.ndelays, extra_trim)
    stim_test_dict = get_multi_layer_features(wordseqs, test_stories, embedder, cfg.ngram_size, cfg.ndelays, extra_trim)
    
    del embedder.model; del embedder; import gc; torch.cuda.empty_cache(); gc.collect()
    
    preds = {}
    for layer in target_layers:
        print(f"Training Ridge for Layer {layer}...")
        wt, _, _ = encoding.bootstrap_ridge(
            stim_train_dict[layer], resp_train, stim_test_dict[layer], resp_test, 
            alphas, nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
        )
        pred = np.nan_to_num(stim_test_dict[layer] @ wt)
        preds[layer] = pred
        
        # Single layer eval just to check
        corrs = np.array([np.corrcoef(resp_test[:, i], pred[:, i])[0, 1] for i in range(resp_test.shape[1])])
        print(f"Layer {layer} TEST CORR: {np.nanmean(corrs):.4f}")
        
    def eval_pred(pred, name, desc):
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
        row = make_result_row(r, name, 8_000_000_000, desc)
        upsert_overall_results([row], RESULTS_DIR)

    # 1. Ensemble all 5 layers
    avg_pred_all = np.mean(list(preds.values()), axis=0)
    eval_pred(avg_pred_all, "Ensemble_Llama8B_Intra_14to18", "Average of Llama-3-8B Layers 14,15,16,17,18")
    
    # 2. Ensemble 15, 16, 17
    avg_pred_15_17 = np.mean([preds[15], preds[16], preds[17]], axis=0)
    eval_pred(avg_pred_15_17, "Ensemble_Llama8B_Intra_15to17", "Average of Llama-3-8B Layers 15,16,17")
