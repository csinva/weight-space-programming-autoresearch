"""Paper audit (ii): a rebuilt model must not depend on the data it saw. Each frozen feature space is run on a
fixed clip (a patient1 training chunk not among any session's examples) twice, in fresh processes whose working
directory holds different decoy example audio (examples/clip*.npy, as in the sandbox); outputs must be identical."""
import json, os, random, shutil, subprocess, sys, tempfile
import numpy as np
BN = os.path.dirname(os.path.abspath(__file__))
PY = '/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python'
train = json.load(open('/path/to/ecog_data/circuit-building-autoresearch/runs-wave/newtest/train_frozen.json'))
used = set()
for s in range(1999, 2006):
    used |= set(random.Random(s).sample(train, 3))
pool = [c for c in train if c not in used]
fixed, decoy_a, decoy_b = pool[0], pool[1:4], pool[4:7]
SEG = '/path/to/scratch/patient1_wave_aligned/{}_wave.npy'
RUN = r'''
import importlib.util, sys, numpy as np
sys.path.insert(0, sys.argv[1] + '/pkgs')
spec = importlib.util.spec_from_file_location('fs', sys.argv[1] + '/feature_space.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fs = m.FeatureSpace(device='cpu'); win = int(getattr(fs, 'win', 2048))
seg = np.load(sys.argv[2]).astype(np.float32); T = (len(seg) - 4096) // 1470 + 1
sw = np.lib.stride_tricks.sliding_window_view(seg, win); w = np.ascontiguousarray(sw[(4096 - win) // 2 + np.arange(T) * 1470])
np.save(sys.argv[3], np.asarray(fs(w), dtype=np.float32))
'''
out = []
for name in sys.argv[1:]:
    fz = f'{BN}/frozen/{name}'
    res = []
    for decoys in (decoy_a, decoy_b):
        d = tempfile.mkdtemp(dir=os.environ.get('TMPDIR', '/tmp'))
        os.makedirs(f'{d}/examples')
        for i, c in enumerate(decoys):
            shutil.copy(SEG.format(c), f'{d}/examples/clip{i + 1}.npy')
        p = subprocess.run([PY, '-c', RUN, fz, SEG.format(fixed), f'{d}/out.npy'], cwd=d, capture_output=True, text=True)
        res.append(np.load(f'{d}/out.npy') if p.returncode == 0 else p.stderr[-300:])
        shutil.rmtree(d, ignore_errors=True)
    ok = all(isinstance(r, np.ndarray) for r in res) and res[0].shape == res[1].shape and np.array_equal(res[0], res[1])
    out.append(f'{name}: {"identical" if ok else "DIFFERS/ERROR"} ' + (str(res[0].shape) if isinstance(res[0], np.ndarray) else str(res[0])))
open(f'{BN}/INVARIANCE.md', 'w').write('# Audit (ii): outputs independent of the example data present\n\n' + '\n'.join(f'- {o}' for o in out) + '\n')
print('\n'.join(out))
sys.exit(0 if all('identical' in o for o in out) else 1)
