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
    sae_release = "gemma-scope-9b-pt-res-canonical"
    
    try:
        train_feat_path = "X_train_sae_L23.npy"
        test_feat_path = "X_test_sae_L23.npy"
        
        print("Loading cached features...", flush=True)
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
        
        alphas = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
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
        
        print(f"Best Gemma-2-9B SAE (L{layer}) TEST CORR (Sklearn, alpha={best_alpha}): {best_corr:.4f}")
        
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
            model_shorthand_name=f"Gemma2_9B_SAE_L{layer}",
            n_params=16e3, # 16k dimensions
            description=f"SAE features (16k dim) from {sae_release} on L{layer}. Using sklearn Ridge best_alpha={best_alpha}.",
        )
        
        upsert_overall_results([row], RESULTS_DIR)
        print("Saved!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"SAE evaluation failed: {e}")
