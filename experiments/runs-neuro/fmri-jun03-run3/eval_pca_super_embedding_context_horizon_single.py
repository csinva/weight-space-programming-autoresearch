import numpy as np
from fmri_regression import run_regression
from sklearn.decomposition import PCA
import os
import glob
import json
import torch

def create_super_features(story, feature_dir, models):
    # This just creates a single PCA embedding
    all_feats = []
    
    for m_prefix in models:
        feat_path = os.path.join(feature_dir, f"{m_prefix}_{story}_ctx150.npy")
        if not os.path.exists(feat_path):
            raise Exception(f"Missing {feat_path}")
        arr = np.load(feat_path)
        all_feats.append(arr)
        
    f = np.concatenate(all_feats, axis=1)
    return f

def test_model_combos():
    feature_dir = "runs-neuro/fmri-jun03-run3/features_ctx150"
    if not os.path.exists(feature_dir):
        print(f"Directory {feature_dir} does not exist.")
        return
        
    models = ["llama3", "qwen2.5", "gemma2"]
    
    # Load all train/test stories
    with open("runs-neuro/fmri-jun03-run3/stimuli/stories.json") as f:
        stories = json.load(f)
    train_stories = ["itsabox", "odetostepfather", "inamoment", "hangtime", "ifthishaircouldtalk", "goingthelibertyway", "golfclubbing", "thetriangleshirtwaistconnection"]
    test_stories = ["fromboyhoodtofatherhood", "wheretheressmoke"]
    
    # Just to be safe, we will just use the pre-extracted features and run PCA over them per model combination
    print("Pre-extracted features are already concatenated inside eval_pca_super_embedding_context_horizon.py")
    
test_model_combos()
