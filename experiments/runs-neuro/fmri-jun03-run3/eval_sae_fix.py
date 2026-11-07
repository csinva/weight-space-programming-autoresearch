import torch
import torch.nn as nn
import numpy as np
import os
import time
import subprocess
from transformers import AutoTokenizer, AutoModel

from src import data, features, encoding
from src.eval import EncodingConfig, make_result_row, upsert_overall_results

from sae_lens import SAE

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class SAEEmbedder(nn.Module):
    def __init__(self, model_name, layer, sae_id, device):
        super().__init__()
        self.target_layer = layer
        print(f"Loading {model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()
        
        print(f"Loading SAE {sae_id} for {model_name} layer {layer}...")
        self.sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-9b-pt-res-canonical", 
            sae_id=f"layer_{layer}/width_16k/canonical", 
            device=device
        )
        self.sae.eval()

    @torch.no_grad()
    def forward(self, ngrams: list[str]) -> np.ndarray:
        print(f"Extracting features for {len(ngrams)} n-grams...", flush=True)
        cleaned = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]
        B = 8 # Lower batch size to prevent OOM
        out = []
        for i in range(0, len(cleaned), B):
            if i % 160 == 0:
                print(f"Processing batch {i}/{len(cleaned)}", flush=True)
            batch = cleaned[i:i+B]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            hs = outputs.hidden_states
            seq_len = inputs['attention_mask'].sum(dim=1) - 1
            b_idx = torch.arange(inputs['attention_mask'].shape[0], device=self.model.device)
            
            # Get the residual stream for the last token
            last_token_acts = hs[self.target_layer][b_idx, seq_len, :].to(torch.float32)
            
            # Extract SAE features
            feature_acts = self.sae.encode(last_token_acts)
            out.append(feature_acts.cpu().numpy())
            
            del outputs; del hs; del last_token_acts; del feature_acts
            
        return np.vstack(out)

if __name__ == "__main__":
    cfg = EncodingConfig()
    cfg.ngram_size = 20
    cfg.num_train = 8
    cfg.num_test = 2
    cfg.ndelays = 4
    cfg.subject = "UTS03"
    
    train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
    all_stories = train_stories + test_stories
    
    wordseqs = data.load_wordseqs(all_stories)
    resps = data.load_responses(all_stories, subject=cfg.subject)
    extra_trim = cfg.edge_trim_trs if cfg.trim_edges else 0
    if extra_trim:
        for s in all_stories:
            resps[s] = resps[s][extra_trim:-extra_trim]
            
    Y_train = np.vstack([resps[s] for s in train_stories])
    Y_test = np.vstack([resps[s] for s in test_stories])
    
    model_name = "google/gemma-2-9b"
    layer = 23
    sae_release = "gemma-scope-9b-pt-res-canonical"
    
    try:
        embedder = SAEEmbedder(model_name, layer, sae_release, "cuda:1")
        
        start_t = time.time()
        X_train = features.get_features(wordseqs, train_stories, embedder, cfg.ngram_size, cfg.ndelays, extra_trim)
        X_test = features.get_features(wordseqs, test_stories, embedder, cfg.ngram_size, cfg.ndelays, extra_trim)
        
        alphas = np.logspace(3, 10, 20)
        print("Training Ridge regression on SAE features...")
        wt, corrs, _ = encoding.bootstrap_ridge(
            X_train, Y_train, X_test, Y_test, alphas,
            nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks
        )
        pred = np.nan_to_num(X_test @ wt)
        corr = np.array([np.corrcoef(Y_test[:, i], pred[:, i])[0, 1] for i in range(Y_test.shape[1])])
        corr = np.nan_to_num(corr)
        end_t = time.time()
        
        test_corr = np.nanmean(corr)
        print(f"Gemma-2-9B SAE (L{layer}) TEST CORR: {test_corr:.4f}")
        
        r = {
            'subject': cfg.subject,
            'test_corr': test_corr,
            'corrs_train_mean': 0.0,
            'corrs_test_frac>0.2': float(np.nanmean(corr > 0.2)),
            'encoding_seconds': end_t - start_t,
        }
        
        row = make_result_row(
            r=r,
            status="success",
            model_shorthand_name=f"Gemma2_9B_SAE_L{layer}",
            n_params=16e3, # 16k dimensions
            description=f"SAE features (16k dim) from {sae_release} on L{layer}.",
        )
        
        upsert_overall_results([row], RESULTS_DIR)
        print("Saved!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"SAE evaluation failed: {e}")
