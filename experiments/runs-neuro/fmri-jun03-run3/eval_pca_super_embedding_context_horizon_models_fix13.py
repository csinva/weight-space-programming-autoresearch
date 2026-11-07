import torch
import torch.nn as nn
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

from src import data, features, encoding
from src.eval import EncodingConfig

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

def run():
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

    from src.encoding import fit_and_eval
    
    print("\n--- Testing Qwen2.5-14B + Gemma-2-9B + Llama-3-8B Absolute Pipeline Re-Validation ---")
    
    X_train_comb = {}
    X_test_comb = {}
    
    for story in train_stories:
        X_train_comb[story] = []
    for story in test_stories:
        X_test_comb[story] = []
        
    for hf_name, layer in models_to_test:
        embedder = SingleLayerEmbedder(hf_name, layer, "cuda")
        
        for story in train_stories:
            ws = wordseqs[story]
            ngrams = features.get_ngrams(list(ws.data), ngram_size=150)
            X = embedder(ngrams)
            ds = features.lanczos_downsample(X, oldtime=ws.data_times, newtime=ws.tr_times)
            X_train_comb[story].append(ds)
            
        for story in test_stories:
            ws = wordseqs[story]
            ngrams = features.get_ngrams(list(ws.data), ngram_size=150)
            X = embedder(ngrams)
            ds = features.lanczos_downsample(X, oldtime=ws.data_times, newtime=ws.tr_times)
            X_test_comb[story].append(ds)
            
        del embedder
        torch.cuda.empty_cache()
    
    # Concatenate and Trim
    train_dict = {}
    for story in train_stories:
        train_dict[story] = np.concatenate(X_train_comb[story], axis=1)
    X_train = features.trim_and_zscore(train_dict, trim=5, extra_trim=extra_trim)
    
    test_dict = {}
    for story in test_stories:
        test_dict[story] = np.concatenate(X_test_comb[story], axis=1)
    X_test = features.trim_and_zscore(test_dict, trim=5, extra_trim=extra_trim)
    
    pca = PCA(n_components=150*3, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    train_corr_mean, test_corr_mean, frac_over_02 = fit_and_eval(
        X_train_pca, Y_train, X_test_pca, Y_test, n_delays=cfg.ndelays
    )
    
    print(f"[All 3] train_corr: {train_corr_mean:.4f}, test_corr: {test_corr_mean:.4f}")
    
    row = f"SuperEmbedding_TripleReval_ctx150,{train_corr_mean:.5f},{test_corr_mean:.5f},{frac_over_02:.5f}\n"
    with open("runs-neuro/fmri-jun03-run3/results/single_model_ctx150_results.csv", "a") as f:
        f.write(row)

run()
