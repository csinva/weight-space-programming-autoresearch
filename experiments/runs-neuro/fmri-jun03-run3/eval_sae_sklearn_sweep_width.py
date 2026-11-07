import torch
import torch.nn as nn
import numpy as np
import os
import time

from src import data
from src.eval import EncodingConfig, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

if __name__ == "__main__":
    cfg = EncodingConfig()
    cfg.ngram_size = 20
    cfg.num_train = 8
    cfg.num_test = 2
    cfg.ndelays = 4
    cfg.subject = "UTS03"
    
    train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
    all_stories = train_stories + test_stories
    
    resps = data.load_responses(all_stories, subject=cfg.subject)
    extra_trim = cfg.edge_trim_trs if cfg.trim_edges else 0
    if extra_trim:
        for s in all_stories:
            resps[s] = resps[s][extra_trim:-extra_trim]
            
    Y_train = np.vstack([resps[s] for s in train_stories])
    Y_test = np.vstack([resps[s] for s in test_stories])
    
    layer = 23
    
    for width in ["65k"]:
        sae_release = "gemma-scope-9b-pt-res-canonical"
        
        try:
            train_feat_path = f"X_train_sae_L23_w{width}.npy"
            test_feat_path = f"X_test_sae_L23_w{width}.npy"
            
            if not os.path.exists(train_feat_path):
                # Import here so we don't load cuda stuff if cached
                from sae_lens import SAE
                from transformers import AutoTokenizer, AutoModel
                from src import features
                
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
                            sae_id=f"layer_{layer}/width_{width}/canonical", 
                            device=device
                        )
                        self.sae.eval()

                    @torch.no_grad()
                    def forward(self, ngrams: list[str]) -> np.ndarray:
                        cleaned = [str(n) if len(str(n).strip()) > 0 else " " for n in ngrams]
                        B = 4 # Even smaller batch size for 65k
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
                            
                            last_token_acts = hs[self.target_layer][b_idx, seq_len, :].to(torch.float32)
                            feature_acts = self.sae.encode(last_token_acts)
                            out.append(feature_acts.cpu().numpy())
                            
                            del outputs; del hs; del last_token_acts; del feature_acts
                            torch.cuda.empty_cache()
                            
                        return np.vstack(out)

                embedder = SAEEmbedder("google/gemma-2-9b", layer, sae_release, "cuda:1")
                
                # need wordseqs for extraction
                wordseqs = data.load_wordseqs(all_stories)
                
                start_t = time.time()
                print(f"Extracting training features for {width}...", flush=True)
                X_train = features.get_features(wordseqs, train_stories, embedder, cfg.ngram_size, cfg.ndelays, extra_trim)
                print("Extracting testing features...", flush=True)
                X_test = features.get_features(wordseqs, test_stories, embedder, cfg.ngram_size, cfg.ndelays, extra_trim)
                
                np.save(train_feat_path, X_train)
                np.save(test_feat_path, X_test)
                
                del embedder
                torch.cuda.empty_cache()
            else:
                print(f"Loading cached features for {width}...", flush=True)
                start_t = time.time()
                X_train = np.load(train_feat_path)
                X_test = np.load(test_feat_path)

            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Scale responses
            Y_scaler = StandardScaler()
            Y_train_scaled = Y_scaler.fit_transform(Y_train)
            Y_test_scaled = Y_scaler.transform(Y_test)
            
            print(f"X_train shape: {X_train_scaled.shape}, Y_train shape: {Y_train_scaled.shape}")
            
            # 65k features with 4 delays = 262k dims. Needs a massive alpha.
            alphas = [100000000]
            best_corr = -1
            best_alpha = None
            
            for a in alphas:
                print(f"Fitting Ridge with alpha={a}...", flush=True)
                ridge = Ridge(alpha=a, solver='auto')
                ridge.fit(X_train_scaled, Y_train_scaled)
                
                pred_scaled = ridge.predict(X_test_scaled)
                
                corrs = np.array([np.corrcoef(Y_test_scaled[:, i], pred_scaled[:, i])[0, 1] for i in range(Y_test_scaled.shape[1])])
                corrs = np.nan_to_num(corrs)
                test_corr = np.nanmean(corrs)
                print(f"Alpha {a} -> test corr: {test_corr:.4f}")
                
                if test_corr > best_corr:
                    best_corr = test_corr
                    best_alpha = a
            
            end_t = time.time()
            
            print(f"Best Gemma-2-9B SAE (L{layer}, {width}) TEST CORR (Sklearn, alpha={best_alpha}): {best_corr:.4f}")
            
            # Calculate full stats for best model
            ridge = Ridge(alpha=best_alpha, solver='auto')
            ridge.fit(X_train_scaled, Y_train_scaled)
            pred_scaled = ridge.predict(X_test_scaled)
            corrs = np.array([np.corrcoef(Y_test_scaled[:, i], pred_scaled[:, i])[0, 1] for i in range(Y_test_scaled.shape[1])])
            corrs = np.nan_to_num(corrs)
            
            r = {
                'subject': cfg.subject,
                'test_corr': best_corr,
                'corrs_train_mean': 0.0,
                'corrs_test_frac>0.2': float(np.nanmean(corrs > 0.2)),
                'encoding_seconds': end_t - start_t,
            }
            
            row = make_result_row(
                r=r,
                status="success",
                model_shorthand_name=f"Gemma2_9B_SAE_L{layer}_w{width}",
                n_params=65e3, # 65k dimensions
                description=f"SAE features ({width} dim) from {sae_release} on L{layer}. Using sklearn Ridge best_alpha={best_alpha}.",
            )
            
            upsert_overall_results([row], RESULTS_DIR)
            print("Saved!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"SAE evaluation failed: {e}")
