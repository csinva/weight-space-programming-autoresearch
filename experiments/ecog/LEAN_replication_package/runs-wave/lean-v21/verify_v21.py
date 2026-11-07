"""Check LEAN-v21 against LEAN-v20 on real chunks, same GPU, same process.
Every v21 column is mapped to its v20 source column; unchanged columns must match.
Groups whose values change by design (voicing gates, cross-rate energy) are reported."""
import importlib.util, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import data


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


V20 = load('lean_v20_enc', os.path.join(HERE, '..', 'varpart-v20', 'lean_v20_enc.py'))
V21 = load('lean_v21_enc', os.path.join(HERE, '..', 'varpart-v21', 'lean_v21_enc.py'))


def run(mod, hook, x, **kw):
    emb = mod.build_embedder(device='cuda', **kw)
    rec = []
    orig = getattr(emb, hook)
    def h(groups):
        rec.extend((n, t.detach().cpu().numpy()) for n, t in groups); return orig(groups)
    setattr(emb, hook, h)
    feat = emb(x)
    return dict(rec), feat


def quad_map(n_rates, s_keep, drop, rate_vals):
    out = []
    for r in range(n_rates):
        for s in s_keep:
            for b in range(3):
                for p in range(3):
                    if (rate_vals[r], s, p) in drop:
                        continue
                    out.append(((r * 5 + s) * 3 + b) * 3 + p)
    return out


QR = (2.0, 4.0, 8.0, 12.0)
q = quad_map(4, range(4), {(12.0, s, p) for s in (0, 3) for p in (0, 1)}, QR)
assert len(q) == 132
stmf_pairs = [i for i in range(17) if i not in (5, 9)]
slow_raw = quad_map(3, range(4), set(), (0.5, 1, 1.5))
slow_dyn = quad_map(4, range(4), set(), (0.5, 1, 1.5, 2))
n2 = [6 + k for k in range(9)]
s2 = [o + m for k, o in enumerate(np.cumsum([0] + n2[:-1])) for m in range(3, n2[k])]
assert len(s2) == 63
scat = s2 + [90 + i for i in s2] + list(range(180, 287))
MAP = {
    'stmf': [p * 20 + c for p in stmf_pairs for c in range(20)],
    'stmf_mean_gabor': stmf_pairs,
    'contrast_short': [i for i in range(24) if i % 6 != 0],
    'quad_modbank': q, 'quad_modbank_acausal': q,
    'quad_modbank_dyn': [b * 180 + i for b in range(3) for i in q],
    'slow_quad_modbank': slow_raw, 'slow_quad_modbank_dyn': slow_dyn,
    'scattering': scat,
}
BY_DESIGN = {'cochlea_voiced_gated', 'cochlea_voiced_gated_dyn', 'gate_signals', 'gate_signals_dyn',
             'quad_xrate_energy', 'quad_xrate_energy_dyn'}

os.environ.setdefault('ECOG_TEST_CHUNKS_FILE', os.path.join(HERE, '..', '..', 'evolve-wave', 'speech_rich_test_chunks.json'))
chunks = json.load(open(os.environ['ECOG_TEST_CHUNKS_FILE']))[:2]
worst = {}
for c in chunks:
    seg = data.load_stim([c])[c]; resp = data.load_responses([c])[c]
    T = min(data.seg_to_T(len(seg)), resp.shape[0])
    x = data.frame_windows(seg, win=4096, n_frames=T)
    g20, f20 = run(V20, '_condition', x, interact_set='v3e')
    g21, f21 = run(V21, '_concat_groups', x)
    g20['scattering'] = f20[:, -287:]; g21['scattering'] = f21[:, -233:]
    assert f21.shape[1] == 3392, f21.shape
    for name, a in g21.items():
        b = g20[name]
        if name in BY_DESIGN:
            if b.shape == a.shape:
                worst.setdefault(name, []).append(('by-design', float(np.abs(a - b).max())))
            else:
                worst.setdefault(name, []).append(('by-design', None))
            continue
        b = b[:, MAP[name]] if name in MAP else b
        assert a.shape == b.shape, (name, a.shape, b.shape)
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        scale = np.abs(b).max() + 1e-12
        worst.setdefault(name, []).append(('mapped' if name in MAP else 'unchanged', float(d.max() / scale)))
        if name == 'scattering':
            sub = [('S2m', 63), ('S2s', 63), ('S2peak', 9), ('joint', 8), ('contour', 9), ('s1_sub', 9),
                   ('s1_sub_dyn', 27), ('s1_peak', 9), ('if', 9), ('coh', 9), ('jitter', 9), ('S2_entropy', 9)]
            o = 0
            for sn, sd in sub:
                print(f'    scattering {sn:11s} max rel diff {d[:, o:o + sd].max() / scale:.3e}')
                o += sd
    gs = g21['gate_signals']
    print(f'{c}: v21 gate_signals frame-fraction > 0.5 at thresholds 0.15/0.3/0.6:',
          np.round((gs > 0.5).mean(0), 3).tolist(),
          '| v20 voicing gate min:', float(g20['gate_signals'][:, :3].min()))
print('\nper group: kind, max relative |v21 - v20| over chunks (0 = bit-identical)')
bad = []
for name, v in worst.items():
    kind = v[0][0]
    vals = [x for _, x in v if x is not None]
    m = max(vals) if vals else float('nan')
    flag = '' if kind == 'by-design' or m < 1e-5 else '  <-- MISMATCH'
    if flag: bad.append(name)
    print(f'  {name:28s} {kind:10s} {m:.3e}{flag}')
print('\nRESULT:', 'PASS' if not bad else f'FAIL {bad}')
