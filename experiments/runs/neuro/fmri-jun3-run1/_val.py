import sys, importlib.util
from src import data

# Independent validation split: fit ridge on the SAME 8 training stories used in
# the main loop, but evaluate on 8 DIFFERENT training stories that were never part
# of the test_corr selection metric (truly held out from all hyperparameter tuning).
VAL_TRAIN = data.TRAIN_STORIES[:8]
VAL_TEST = data.TRAIN_STORIES[8:16]

def patched(num_train=None, num_test=1):
    return VAL_TRAIN, VAL_TEST
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
cfg = EncodingConfig(subject='UTS03', num_train=8, num_test=8)
r = run_encoding(emb, cfg, verbose=False)
print(f"VAL test_corr: {r['test_corr']:.6f}  train_corr={r['corrs_train_mean']:.6f}  median={r['corrs_test_median']:.6f}")
