"""Runs /work/feature_space.py on the example audio exactly as the scoring harness frames it.
Checks shape, dtype, finite values, H <= 4000 and speed (< 30 s per segment). Contains no brain data and no scores."""
import glob, importlib.util, sys, time
sys.path.insert(0, '/work/pkgs')
import numpy as np

HOP, WINMAX = 1470, 4096


def frame_windows(seg, win):
    T = (len(seg) - WINMAX) // HOP + 1
    start0 = (WINMAX - win) // 2
    sw = np.lib.stride_tricks.sliding_window_view(seg, win)
    return np.ascontiguousarray(sw[start0 + np.arange(T) * HOP]), T


spec = importlib.util.spec_from_file_location('feature_space', '/work/feature_space.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
fs = mod.FeatureSpace(device='cpu')
win = int(getattr(fs, 'win', 2048))
assert 0 < win <= WINMAX, f'win={win} must be in (0, {WINMAX}]'
ok = True
for f in sorted(glob.glob('/work/examples/*.npy')):
    seg = np.load(f).astype(np.float32)
    w, T = frame_windows(seg, win)
    t0 = time.time()
    out = fs(w)
    dt = time.time() - t0
    out = np.asarray(out)
    msgs = []
    if out.ndim != 2 or out.shape[0] != T: msgs.append(f'shape {out.shape}, expected ({T}, H)')
    if out.dtype != np.float32: msgs.append(f'dtype {out.dtype}, expected float32')
    if out.ndim == 2 and out.shape[1] > 4000: msgs.append(f'H={out.shape[1]} > 4000')
    if not np.all(np.isfinite(out)): msgs.append('non-finite values')
    if dt > 30: msgs.append(f'{dt:.1f} s > 30 s')
    ok &= not msgs
    print(f'{f.split("/")[-1]}: T={T} H={out.shape[1] if out.ndim == 2 else "?"} {dt:.1f}s ' + ('OK' if not msgs else 'FAIL: ' + '; '.join(msgs)))
print('ALL CHECKS PASSED' if ok else 'CHECKS FAILED')
sys.exit(0 if ok else 1)
