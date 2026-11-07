import sys, importlib.util
import numpy as np
from src import data

# Multi-split independent validation to reduce single-split noise.
# Pool = TRAIN_STORIES[8:] : never used by the main fit set (TRAIN[:8]) nor by the
# test_corr selection metric (TEST_STORIES). Average test_corr over K disjoint folds,
# each fit on 8 stories and evaluated on 16 held-out stories (48 distinct eval stories).
POOL = data.TRAIN_STORIES[8:]
FOLDS = [
    (POOL[0:8],   POOL[8:24]),
    (POOL[24:32], POOL[32:48]),
    (POOL[48:56], POOL[56:72]),
]

_cur = {"train": None, "test": None}

def patched(num_train=None, num_test=1):
    return _cur["train"], _cur["test"]
data.get_story_names = patched

from src.eval import EncodingConfig, run_encoding

def load_model_module(path):
    spec = importlib.util.spec_from_file_location("modunder_eval", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

path = sys.argv[1]
m = load_model_module(path)
emb = m.build_embedder(device='cuda')

corrs = []
for i, (tr, te) in enumerate(FOLDS):
    _cur["train"], _cur["test"] = tr, te
    cfg = EncodingConfig(subject='UTS03', num_train=len(tr), num_test=len(te))
    r = run_encoding(emb, cfg, verbose=False)
    corrs.append(r['test_corr'])
    print(f"  fold{i}: test_corr={r['test_corr']:.6f}")
corrs = np.array(corrs)
print(f"VALK mean_test_corr: {corrs.mean():.6f}  std={corrs.std():.6f}  folds={list(np.round(corrs,6))}")
