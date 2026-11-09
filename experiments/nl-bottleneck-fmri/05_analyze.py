"""Step 4: one row per original model with its performance and the mean over its rebuilds. Writes WORK/results.json
and prints a table. pred_sim: similarity of hidden-story predictions to the original's (see 04_evaluate.py)."""
import json
import numpy as np
from common import MODELS, WORK

rows = []
for it, name in MODELS:
    o = json.loads((WORK / 'originals_eval' / f'{name}.json').read_text())
    recs = [json.loads(p.read_text()) for p in sorted((WORK / 'recon_eval').glob(f'{name}-r*.json'))]
    ok = [r for r in recs if 'error' not in r]
    metas = [json.loads(p.read_text()) for p in sorted((WORK / 'recon').glob(f'{name}-r*/meta.json'))]
    mean = lambda k: float(np.mean([r[k] for r in ok])) if ok else None
    rows.append(dict(iteration=it, name=name, original={k: o[k] for k in ('visible', 'hidden', 'hidden_middle')},
                     rebuilt={k: mean(k) for k in ('visible', 'hidden', 'hidden_middle', 'pred_similarity', 'map_similarity')},
                     rebuilt_each=[{k: r.get(k) for k in ('rep', 'final', 'visible', 'hidden', 'pred_similarity', 'map_similarity', 'error')} for r in recs],
                     n_ok=len(ok), n=len(recs), cost_usd=sum(m.get('cost_usd') or 0 for m in metas),
                     experiments=[m.get('experiments') for m in metas]))
(WORK / 'results.json').write_text(json.dumps(rows, indent=1))
f = lambda v: '   -  ' if v is None else f'{v:.4f}'
print(f"{'iter':>4} {'model':32} | {'orig vis':>8} {'reb vis':>8} | {'orig hid':>8} {'reb hid':>8} | {'pred sim':>8} {'map sim':>8} | ok  cost")
for r in rows:
    o, b = r['original'], r['rebuilt']
    print(f"{r['iteration']:>4} {r['name'][:32]:32} | {f(o['visible']):>8} {f(b['visible']):>8} | {f(o['hidden']):>8} {f(b['hidden']):>8} | "
          f"{f(b['pred_similarity']):>8} {f(b['map_similarity']):>8} | {r['n_ok']}/{r['n']} ${r['cost_usd']:.2f}")
