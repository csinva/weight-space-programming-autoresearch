import numpy as np
import os
import time
from sklearn.decomposition import PCA

from src import data, features, encoding

def run():
    out_dir = "runs-neuro/fmri-jun03-run3/features_ctx150"
    
    train_stories, test_stories = data.get_story_names(8, 2)
    all_stories = train_stories + test_stories
    
    responses = data.load_responses(all_stories, subject="UTS03")
    extra_trim = 5 # default from edge_trim_trs

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

    models_to_test = [
        "llama3",
        "qwen2.5",
        "gemma2"
    ]
    
    for name_short in models_to_test:
        print(f"\n--- Testing {name_short} from pre-cached ctx150 ---")
        
        X_train_list = []
        for story in train_stories:
            path = os.path.join(out_dir, f"{name_short}_{story}_ctx150.npy")
            X_train_list.append(np.load(path))
        X_train = np.concatenate(X_train_list, axis=0)
        
        X_test_list = []
        for story in test_stories:
            path = os.path.join(out_dir, f"{name_short}_{story}_ctx150.npy")
            X_test_list.append(np.load(path))
        X_test = np.concatenate(X_test_list, axis=0)
        
        pca = PCA(n_components=150, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Create delays manually
        X_train_delayed = features.make_delayed(X_train_pca, 4)
        X_test_delayed = features.make_delayed(X_test_pca, 4)
        
        res = encoding.fit_encoding(
            X_train_delayed, Y_train, X_test_delayed, Y_test,
            nboots=5,
            chunklen=40,
            nchunks=20
        )
        
        train_corr_mean = res['corrs_train_mean']
        test_corr_mean = res['corrs_test_mean']
        frac_over_02 = res['corrs_test_frac>0.2']
        
        print(f"[{name_short}] train_corr: {train_corr_mean:.4f}, test_corr: {test_corr_mean:.4f}")
        
        row = f"SuperEmbedding_{name_short}_ctx150_precached,{train_corr_mean:.5f},{test_corr_mean:.5f},{frac_over_02:.5f}\n"
        with open("runs-neuro/fmri-jun03-run3/results/single_model_ctx150_results.csv", "a") as f:
            f.write(row)

run()
