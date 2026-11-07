"""Verify the cleaned LEAN-v17 produces BIT-IDENTICAL features to the original
run with --interact-set v3e (the config that scored 0.0778)."""
import sys, os, importlib.util
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


HERE = os.path.dirname(__file__)
orig = load("enc_orig", os.path.join(HERE, "enc_orig.py"))
clean = load("enc_clean", os.path.join(HERE, "enc_clean.py"))
from src import data

train, test = data.get_chunk_names(5, 5)
chunks = (train + test)[:4]
stim = data.load_stim(chunks); resp = data.load_responses(chunks)

emb_o = orig.build_embedder(device="cuda", cond_mode="none", interact_set="v3e")
emb_c = clean.build_embedder(device="cuda")          # defaults now v3e / none
win = int(getattr(emb_o, "win", 2048))
print(f"orig win={win}  clean defaults: cond={emb_c.cond_mode} interact={emb_c.interact_set}")

all_ok = True
for c in chunks:
    seg = stim[c]; T = min(data.seg_to_T(len(seg)), resp[c].shape[0])
    w = data.frame_windows(seg, win=win, n_frames=T)
    fo = np.asarray(emb_o(w)); fc = np.asarray(emb_c(w))
    eq = fo.shape == fc.shape and np.array_equal(fo, fc)
    maxd = float(np.abs(fo - fc).max()) if fo.shape == fc.shape else float("nan")
    print(f"  {c}: orig{fo.shape} clean{fc.shape} array_equal={eq} max_abs_diff={maxd:.2e}")
    all_ok = all_ok and eq
print("\n" + ("=== ALL BIT-IDENTICAL ===" if all_ok else "=== MISMATCH!! ==="))
