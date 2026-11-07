import torch
import torch.nn as nn
import numpy as np
import os
import sys
from sklearn.decomposition import PCA

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PCA_GPT2_Embedder(nn.Module):
    def __init__(self, hf_model_name="gpt2-xl", n_components=300):
        super().__init__()
        
        print(f"Loading {hf_model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.semantic_model = AutoModel.from_pretrained(hf_model_name).to(device)
        self.semantic_model.eval()
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.is_fit = False

    def get_raw_features(self, texts: list[str]) -> np.ndarray:
        batch_size = 32
        semantic_feats = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                outputs = self.semantic_model(**encoded)
                
                # Mean pooling
                attention_mask = encoded['attention_mask'].unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                mean_pooled = torch.sum(token_embeddings * attention_mask, dim=1) / torch.clamp(attention_mask.sum(1), min=1e-9)
                
                semantic_feats.append(mean_pooled.cpu().numpy())
                
        return np.concatenate(semantic_feats, axis=0)

    def forward(self, texts: list[str]) -> np.ndarray:
        raw_feats = self.get_raw_features(texts)
        if not self.is_fit:
            print(f"Fitting PCA on {len(texts)} samples...", flush=True)
            self.pca.fit(raw_feats)
            self.is_fit = True
        return self.pca.transform(raw_feats)

config = EncodingConfig(
    subject="UTS03",
    num_train=8,
    num_test=2,
    ngram_size=10,
    ndelays=4,
    nboots=5,
    chunklen=40,
    nchunks=20,
    trim_edges=True
)

for dims in [100, 300, 800]:
    model_name = f"Pure_gpt2_xl_Mean_PCA_{dims}"
    print(f"\n--- Testing model: {model_name} ---", flush=True)
    embedder = PCA_GPT2_Embedder(n_components=dims)
    
    # We must fit PCA on all texts first. The easiest way is to let get_features call it.
    # The evaluation harness calls embedder(ngrams) twice (once train, once test). 
    # This implies PCA will be fit ONLY on train. This is mathematically correct.
    
    results = run_encoding(embedder, config, verbose=True)
    
    row = make_result_row(results, model_name, dims, f"GPT-2 XL Mean Pool semantics projected to {dims} dims via PCA")
    upsert_overall_results([row], RESULTS_DIR)
    
    del embedder
    torch.cuda.empty_cache()

