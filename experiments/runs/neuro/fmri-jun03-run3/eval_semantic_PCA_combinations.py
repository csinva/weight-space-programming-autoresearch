import numpy as np
import os
import json
import time

from src import data, features, encoding
from src.eval import EncodingConfig, make_result_row, upsert_overall_results

def test_model_combos():
    feature_dir = "runs-neuro/fmri-jun03-run3/features_ctx150"
    
    # Load all train/test stories
    train_stories, test_stories = data.get_story_names(num_train=8, num_test=2)
    
    # We test each single model independently to determine its max performance.
    models = ["llama3", "qwen2.5", "gemma2"]
    
    for model_name in models:
        print(f"\n--- Testing {model_name} at ctx150 ---")
        
        # Load train features
        train_feats = []
        for story in train_stories:
            path = os.path.join(feature_dir, f"{model_name}_{story}_ctx150.npy")
            train_feats.append(np.load(path))
        X_train = np.concatenate(train_feats, axis=0)
        
        # Load test features
        test_feats = []
        for story in test_stories:
            path = os.path.join(feature_dir, f"{model_name}_{story}_ctx150.npy")
            test_feats.append(np.load(path))
        X_test = np.concatenate(test_feats, axis=0)
        
        # We should run PCA on them.
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=150, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        config = EncodingConfig(
            model_shorthand=f"SuperEmbedding_{model_name}_ctx150_PCA150",
            model_description=f"Single model {model_name} at 150-word context. PCA 150.",
            n_delays=4,
            use_words=False,
            pca_components=-1 # we already did PCA
        )
        
        # We need response data
        print("loading responses...")
        responses = data.load_responses(train_stories + test_stories)
        
        Y_train_list = []
        for story in train_stories:
            Y_train_list.append(responses[story])
        Y_train = np.concatenate(Y_train_list, axis=0)
        
        Y_test_list = []
        for story in test_stories:
            Y_test_list.append(responses[story])
        Y_test = np.concatenate(Y_test_list, axis=0)
        
        print(f"X_train_pca: {X_train_pca.shape}, Y_train: {Y_train.shape}")
        
        from src.encoding import fit_and_eval
        train_corr, test_corr, frac_over_02 = fit_and_eval(X_train_pca, Y_train, X_test_pca, Y_test, n_delays=config.n_delays)
        
        print(f"[{model_name}] train_corr: {train_corr:.4f}, test_corr: {test_corr:.4f}")
        
        row = make_result_row(config, train_corr, test_corr, frac_over_02)
        upsert_overall_results(row, "runs-neuro/fmri-jun03-run3/results/overall_results.csv")
        
test_model_combos()
