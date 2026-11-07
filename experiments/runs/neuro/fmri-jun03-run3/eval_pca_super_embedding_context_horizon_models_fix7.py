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
            torch_dtype=torch.float16,
            device_map=device
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, ngrams: list[str]) -> np.ndarray:
        cleaned = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]
        B = 2 # Extreme contexts require tiny batch size
        out = []
        for i in range(0, len(cleaned), B):
            batch = cleaned[i:i+B]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1500)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Use last token
            lens = inputs['attention_mask'].sum(dim=1) - 1
            hidden = outputs.hidden_states[self.target_layer]
            
            vecs = []
            for j in range(len(batch)):
                vecs.append(hidden[j, lens[j], :].float().cpu().numpy())
            out.append(np.stack(vecs))
            
        return np.concatenate(out, axis=0)

models_to_test = [
    ("meta-llama/Meta-Llama-3-8B", 16),
    ("Qwen/Qwen2.5-14B", 24),
    ("google/gemma-2-9b", 23)
]

def run(idx):
    cfg = EncodingConfig()
    cfg.num_train = 8
    cfg.num_test = 2
    cfg.ndelays = 4
    
    train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
    all_stories = train_stories + test_stories
    
    wordseqs = data.load_wordseqs(all_stories)
    responses = data.load_responses(all_stories, subject=cfg.subject)
    
    extra_trim = cfg.edge_trim_trs if cfg.trim_edges else 0

    Y_train_list = []
    for s in train_stories:
        resp = responses[s]
        if extra_trim > 0:
            resp = resp[extra_trim:-extra_trim]
        Y_train_list.append(resp)
    Y_train = np.concatenate(Y_train_list, axis=0)

    Y_test_list = []
    for s in test_stories:
        resp = responses[s]
        if extra_trim > 0:
            resp = resp[extra_trim:-extra_trim]
        Y_test_list.append(resp)
    Y_test = np.concatenate(Y_test_list, axis=0)

    hf_name, layer = models_to_test[idx]
    name_short = hf_name.split("/")[-1]
    print(f"\n--- Testing {name_short} at ctx150 ---")
    
    embedder = SingleLayerEmbedder(hf_name, layer, "cuda") 
    
    # We must use `features.get_features` to exactly replicate the trim shapes!
    X_train = features.get_features(
        wordseqs, train_stories, embedder, ngram_size=150,
        ndelays=0, extra_trim=extra_trim)
        
    X_test = features.get_features(
        wordseqs, test_stories, embedder, ngram_size=150,
        ndelays=0, extra_trim=extra_trim)
    
    # Cleanup model to save VRAM
    del embedder
    torch.cuda.empty_cache()
    
    pca = PCA(n_components=150, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    from src.encoding import fit_encoding
    
    # Create delays manually AFTER PCA (essential to not explode dimensions to PCA)
    X_train_d = features.make_delayed(X_train_pca, cfg.ndelays)
    X_test_d = features.make_delayed(X_test_pca, cfg.ndelays)
    
    print(f"X_train_d: {X_train_d.shape}, Y_train: {Y_train.shape}")
    assert X_train_d.shape[0] == Y_train.shape[0], "Shape mismatch train!"
    
    res = fit_encoding(
        X_train_d, Y_train, X_test_d, Y_test,
        nboots=5, chunklen=40, nchunks=20
    )
    
    train_corr_mean = res['corrs_train_mean']
    test_corr_mean = res['corrs_test_mean']
    frac_over_02 = res['corrs_test_frac>0.2']
    
    print(f"[{name_short}] train_corr: {train_corr_mean:.4f}, test_corr: {test_corr_mean:.4f}")
    
    row = f"SuperEmbedding_{name_short}_ctx150,{train_corr_mean:.5f},{test_corr_mean:.5f},{frac_over_02:.5f}\n"
    with open("runs-neuro/fmri-jun03-run3/results/single_model_ctx150_results.csv", "a") as f:
        f.write(row)

import sys
model_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
run(model_idx)
