import torch
import torch.nn as nn
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

from src import data, features, encoding

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

def extract_features(idx):
    hf_name, layer = models_to_test[idx]
    name_short = hf_name.split("/")[-1]
    
    out_dir = "runs-neuro/fmri-jun03-run3/features_single_ctx150"
    os.makedirs(out_dir, exist_ok=True)
    
    train_stories, test_stories = data.get_story_names(8, 2)
    all_stories = train_stories + test_stories
    wordseqs = data.load_wordseqs(all_stories)
    
    embedder = SingleLayerEmbedder(hf_name, layer, "cuda") 
    
    for story in all_stories:
        out_path = os.path.join(out_dir, f"{name_short}_{story}.npy")
        if os.path.exists(out_path):
            print(f"Skipping {out_path}")
            continue
            
        print(f"Extracting {story}...")
        ngrams = features.get_ngrams(wordseqs[story].data, ngram_size=150)
        X = embedder(ngrams)
        ds = features.lanczos_downsample(X, wordseqs[story].data_times, wordseqs[story].tr_times)
        
        # We will zscore per story exactly like we do inline
        ds = features._zscore(ds)
        np.save(out_path, ds)

import sys
model_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
extract_features(model_idx)
