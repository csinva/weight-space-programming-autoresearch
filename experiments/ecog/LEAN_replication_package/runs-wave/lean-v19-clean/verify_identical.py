import sys, os, importlib.util
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
def load(name, path):
    spec=importlib.util.spec_from_file_location(name, path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
HERE=os.path.dirname(__file__)
o=load("enc_orig", os.path.join(HERE,"enc_orig.py"))
c=load("enc_clean", os.path.join(HERE,"enc_clean.py"))
from src import data
train,test=data.get_chunk_names(5,5); chunks=(train+test)[:4]
stim=data.load_stim(chunks); resp=data.load_responses(chunks)
eo=o.build_embedder(device="cuda"); ec=c.build_embedder(device="cuda")
win=int(getattr(eo,"win",2048)); ok=True
for ch in chunks:
    seg=stim[ch]; T=min(data.seg_to_T(len(seg)), resp[ch].shape[0])
    w=data.frame_windows(seg, win=win, n_frames=T)
    fo=np.asarray(eo(w)); fc=np.asarray(ec(w))
    eq=fo.shape==fc.shape and np.array_equal(fo,fc)
    print(f"  {ch}: {fo.shape} vs {fc.shape} equal={eq} maxdiff={np.abs(fo-fc).max() if fo.shape==fc.shape else 'NA'}")
    ok=ok and eq
print("=== ALL BIT-IDENTICAL ===" if ok else "=== MISMATCH ===")
