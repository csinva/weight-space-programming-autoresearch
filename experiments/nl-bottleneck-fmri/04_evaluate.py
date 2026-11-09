"""Step 3: evaluate originals and rebuilds the way the paper's hidden-set table does: ridge regression fit on the
8 fitting stories (all TRs), evaluated on the 2 visible stories and on 10 hidden stories never shown to any agent.

  python 04_evaluate.py originals     -> WORK/originals_eval/<name>.json (+ .npz with hidden predictions)
  python 04_evaluate.py recon         -> WORK/recon_eval/<name>-r<k>.json

Similarity of a rebuild to its original: on the hidden stories, the per-voxel correlation between the two models'
predicted responses, averaged over the original's 10% best-predicted hidden voxels (pred_similarity); and the
correlation across all voxels between their per-voxel hidden performance maps (map_similarity)."""
import fcntl, json, sys
import numpy as np
from common import MODELS, WORK, fit_and_predict, load_embedder, split, voxel_corr


def evaluate(path):
    sp = split()
    r = fit_and_predict(load_embedder(path), sp['fit'], sp['visible'] + sp['new'])
    vis = np.concatenate([r['spans'][s] for s in sp['visible']])
    new = np.concatenate([r['spans'][s] for s in sp['new']])
    mid = np.concatenate([r['spans'][s][30:-30] for s in sp['new']])
    c_new = voxel_corr(r['Y'][new], r['pred'][new])
    return dict(visible=float(voxel_corr(r['Y'][vis], r['pred'][vis]).mean()), hidden=float(c_new.mean()),
                hidden_middle=float(voxel_corr(r['Y'][mid], r['pred'][mid]).mean()), train=r['train_corr'],
                n_features=r['n_features']), c_new, r['pred'][new]


def locked(fn, *a):
    with open(WORK / 'score.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return fn(*a)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def originals():
    out = WORK / 'originals_eval'; out.mkdir(exist_ok=True)
    for it, name in MODELS:
        if (out / f'{name}.json').exists():
            continue
        stats, c_new, pred = locked(evaluate, WORK / 'originals' / name / 'model.py')
        top = np.argsort(c_new)[-len(c_new) // 10:]
        np.savez_compressed(out / f'{name}.npz', corr_hidden=c_new.astype(np.float32), top=top, pred_top=pred[:, top].astype(np.float32))
        (out / f'{name}.json').write_text(json.dumps(dict(iteration=it, **stats), indent=1))
        print(it, name, {k: round(v, 4) for k, v in stats.items() if isinstance(v, float)}, flush=True)


def recon():
    out = WORK / 'recon_eval'; out.mkdir(exist_ok=True)
    for rd in sorted((WORK / 'recon').glob('*-r*')):
        meta = json.loads((rd / 'meta.json').read_text()) if (rd / 'meta.json').exists() else None
        if not meta or not meta.get('final') or (out / f'{rd.name}.json').exists():
            continue
        ref = np.load(WORK / 'originals_eval' / f"{meta['name']}.npz")
        try:
            stats, c_new, pred = locked(evaluate, rd / 'interpretable_transformers_lib' / f"{meta['final']}.py")
        except Exception as e:  # a rebuild that does not run counts as a failure
            (out / f'{rd.name}.json').write_text(json.dumps(dict(name=meta['name'], rep=meta['rep'], error=repr(e)[:500])))
            print(rd.name, 'error', repr(e)[:200], flush=True)
            continue
        p, q = pred[:, ref['top']], ref['pred_top']
        pz = (p - p.mean(0)) / (p.std(0) + 1e-9); qz = (q - q.mean(0)) / (q.std(0) + 1e-9)
        stats.update(name=meta['name'], rep=meta['rep'], final=meta['final'],
                     pred_similarity=float(np.nanmean((pz * qz).mean(0))),
                     map_similarity=float(np.corrcoef(c_new, ref['corr_hidden'])[0, 1]))
        (out / f'{rd.name}.json').write_text(json.dumps(stats, indent=1))
        print(rd.name, {k: round(v, 4) for k, v in stats.items() if isinstance(v, float)}, flush=True)


if __name__ == '__main__':
    {'originals': originals, 'recon': recon}[sys.argv[1]]()
