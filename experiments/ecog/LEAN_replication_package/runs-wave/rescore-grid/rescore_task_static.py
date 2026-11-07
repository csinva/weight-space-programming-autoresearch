"""Rescore trials: run each snapshot's own __main__ in a sandbox whose src/ is the patched
harness (src_patched), which streams the fit and saves per-chunk per-electrode test r.
usage: rescore_task.py --task i --ntasks N   |   rescore_task.py --only id1,id2,...  [--check]"""
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(REPO, '.venv', 'bin', 'python')
TOL = 0.00015  # logged scores are rounded to 4 dp


def acquire_gpu():
    import torch
    for k in range(8):
        try:
            torch.zeros(1, device='cuda'); torch.cuda.synchronize(); return
        except Exception as e:  # noqa: BLE001
            print(f'[gpu] busy {k + 1}/8: {e}', flush=True); time.sleep(5)
    jid = os.environ.get('SLURM_JOB_ID'); n = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    if jid and n < 15:
        os.system(f'scontrol requeue {jid}'); time.sleep(20)
    raise RuntimeError('no GPU')


def run_one(t):
    out = os.path.join(HERE, 'results', t['id'] + '.json')
    if os.path.exists(out):
        return json.load(open(out))
    if t.get('skip'):
        rec = dict(t, status='excluded', rc=None, seconds=0)
        json.dump(rec, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
        print(f"excluded {t['id']}: {t['skip']}", flush=True)
        return rec
    wd = os.path.join(HERE, 'work', t['id'])
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(os.path.join(wd, 'results'))
    os.symlink(os.path.join(HERE, 'src_patched'), os.path.join(wd, 'src'))
    shutil.copy(t['snapshot'], os.path.join(wd, 'wave_encoder.py'))
    npz = os.path.join(wd, 'rescore.npz')
    env = dict(os.environ, RESCORE_OUT=npz, WAVE_FEAT_CACHE=os.path.join(wd, 'cache'),
               ECOG_TEST_CHUNKS_FILE=os.path.join(REPO, 'evolve-wave', 'speech_rich_test_chunks.json'),
               ECOG_EXCLUDE_PREFIX='450a36e4', ECOG_RIDGE_GPU='1', PYTHONDONTWRITEBYTECODE='1')
    t0 = time.time()
    with open(os.path.join(HERE, 'logs', t['id'] + '.log'), 'w') as log:
        rc = subprocess.run([PY, 'wave_encoder.py', '--num-train', '500', '--num-test', '100'] + t.get('args', []),
                            cwd=wd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=4 * 3600).returncode
    rec = dict(t, rc=rc, seconds=round(time.time() - t0))
    if os.path.exists(npz):
        z = np.load(npz)
        rr = z['r_chunk_elec']; m = rr.mean(0)
        rec.update(all166=float(m.mean()), grid=float(m[:125].mean()), P=int(z['P']),
                   reproduced=bool(abs(float(m.mean()) - t['logged']) <= TOL))
        np.save(os.path.join(HERE, 'results', t['id'] + '_r.npy'), rr.astype(np.float32))
        rec['status'] = 'ok' if rec['reproduced'] else 'mismatch'
    else:
        rec['status'] = 'failed'
    json.dump(rec, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    shutil.rmtree(wd, ignore_errors=True)
    print(f"{rec['status']:8s} {t['id']:70s} logged={t['logged']:.4f} "
          f"all166={rec.get('all166', float('nan')):.5f} grid={rec.get('grid', float('nan')):.5f} "
          f"rc={rc} {rec['seconds']}s", flush=True)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int); ap.add_argument('--ntasks', type=int)
    ap.add_argument('--only'); ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    trials = json.load(open(os.path.join(HERE, 'trials.json')))
    if a.only:
        ids = a.only.split(','); mine = [t for t in trials if t['id'] in ids]
        assert len(mine) == len(ids), (ids, [t['id'] for t in mine])
    else:
        mine = [t for k, t in enumerate(trials) if k % a.ntasks == a.task]
    acquire_gpu()
    recs = [run_one(t) for t in mine]
    if a.check:
        bad = [r['id'] for r in recs if r['status'] != 'ok']
        print('PILOT', 'PASS' if not bad else f'FAIL {bad}')
        sys.exit(1 if bad else 0)
