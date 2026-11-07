"""Rerun every Figure 1 trial and score it on the old AND the new held-out test set.

Each trial's own __main__ runs in a sandbox whose src/ is newtest/src_patched: the fit is the
streamed copy of the harness ridge (training list asserted equal to train_frozen.json), the old
23-chunk test is scored as before, then the same weights are scored on the 493 new candidates.
A trial counts only if it reproduces its logged score on the old test (metric it was logged
with: all-166 for Jun 3-5, grid for Jun 6+) within TOL. Pre-Jun-5 trials that fail are retried
once with the NumPy 1.26 overlay (NEP 50 promotion broke some of them under NumPy 2).

usage: rescore_task_newtest.py --task i --ntasks N  |  --only id1,id2 [--check] [--trials extra_trials.json]
Array mode is work-stealing (atomic mkdir claims, stale-owner takeover, stop at < STOP_S left).
"""
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(REPO, '.venv', 'bin', 'python')
NP126 = os.path.join(REPO, 'runs-wave', 'rescore-grid', 'np126')
TOL = 0.00015
REINSTALL = '2026-06-05T12:08'   # venv numpy/torch reinstall, 08:08 EDT, in UTC log timestamps
STOP_S = 45 * 60
CLAIMS = os.path.join(HERE, 'claims')
PRIMARY = set(json.load(open(os.path.join(HERE, 'test_primary.json'))))


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


def attempt(t, pythonpath):
    wd = os.path.join(HERE, 'work', t['id'])
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(os.path.join(wd, 'results'))
    os.symlink(os.path.join(HERE, 'src_patched'), os.path.join(wd, 'src'))
    shutil.copy(t['snapshot'], os.path.join(wd, 'wave_encoder.py'))
    npz = os.path.join(wd, 'rescore.npz')
    env = dict(os.environ, RESCORE_OUT=npz, NEWTEST_DIR=HERE, WAVE_FEAT_CACHE=os.path.join(wd, 'cache'),
               ECOG_TEST_CHUNKS_FILE=os.path.join(REPO, 'evolve-wave', 'speech_rich_test_chunks.json'),
               ECOG_EXCLUDE_PREFIX='450a36e4', ECOG_RIDGE_GPU='1', PYTHONDONTWRITEBYTECODE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', **t.get('env', {}))
    env['TMPDIR'] = os.path.join(wd, 'tmp'); os.makedirs(env['TMPDIR'])
    if pythonpath:
        env['PYTHONPATH'] = pythonpath + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [PY, 'wave_encoder.py', '--num-train', t.get('num_train', '500'), '--num-test', '100'] + t.get('args', [])
    t0 = time.time()
    with open(os.path.join(HERE, 'logs', t['id'] + '.log'), 'a') as log:
        log.write(f"### {time.ctime()} pythonpath={pythonpath or '-'} cmd={' '.join(cmd[1:])}\n"); log.flush()
        try:
            rc = subprocess.run(cmd, cwd=wd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=6 * 3600).returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
    rec = dict(t, rc=rc, seconds=round(time.time() - t0), pythonpath_used=pythonpath or None)
    if os.path.exists(npz):
        z = np.load(npz)
        ro, m = z['r_chunk_elec'], z['r_chunk_elec'].mean(0)
        rec.update(old_all166=float(m.mean()), old_grid=float(m[:125].mean()), P=int(z['P']))
        rec['reproduced'] = bool(abs(rec['old_' + t['metric']] - t['logged']) <= TOL)
        np.save(os.path.join(HERE, 'results', t['id'] + '_rold.npy'), ro.astype(np.float32))
        if 'r_newtest' in z.files:
            rn, ch = z['r_newtest'], [str(c) for c in z['newtest_chunks']]
            prim = np.array([c in PRIMARY for c in ch])
            np.save(os.path.join(HERE, 'results', t['id'] + '_rnew.npy'), rn.astype(np.float32))
            rec.update(n_new=len(ch), n_primary=int(prim.sum()),
                       new_grid=float(rn[prim].mean(0)[:125].mean()), new_all166=float(rn[prim].mean(0).mean()),
                       newall_grid=float(rn.mean(0)[:125].mean()), seconds_newtest=float(z['seconds_newtest']))
        rec['status'] = 'ok' if rec['reproduced'] and 'n_new' in rec else 'mismatch'
    else:
        rec['status'] = 'failed'
    shutil.rmtree(wd, ignore_errors=True)
    return rec


def run_one(t):
    out = os.path.join(HERE, 'results', t['id'] + '.json')
    if os.path.exists(out):
        return json.load(open(out))
    rec = attempt(t, t.get('pythonpath'))
    if (rec['status'] != 'ok' and not t.get('pythonpath') and t.get('eval_ts')
            and t['eval_ts'] < REINSTALL):
        first = {k: rec.get(k) for k in ('status', 'rc', 'old_all166', 'old_grid')}
        rec = attempt(t, NP126)
        rec['first_attempt'] = first
    json.dump(rec, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    f = lambda k: rec.get(k, float('nan'))
    print(f"{rec['status']:8s} {t['id']:70s} logged={t['logged']:.4f} old={f('old_' + t['metric']):.5f} "
          f"new_grid={f('new_grid'):.5f} rc={rec['rc']} {rec['seconds']}s", flush=True)
    return rec


def _sq(args):
    return subprocess.run(['squeue', '-h'] + args, capture_output=True, text=True).stdout.strip()


def time_left_s():
    jid = os.environ.get('SLURM_JOB_ID')
    if not jid:
        return 1e12
    v = _sq(['-j', jid, '-o', '%L'])
    if not v or not v[0].isdigit():
        return 1e12
    d, _, hms = v.rpartition('-')
    parts = [int(x) for x in hms.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(d) if d else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def claim(t):
    d = os.path.join(CLAIMS, t['id'])
    me = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    for _ in range(2):
        try:
            os.mkdir(d)
            open(os.path.join(d, 'owner'), 'w').write(me)
            return True
        except FileExistsError:
            try:
                owner = open(os.path.join(d, 'owner')).read().strip()
            except OSError:
                return False
            if owner == me or _sq(['-j', owner, '-o', '%i']):
                return owner == me
            if open(os.path.join(d, 'owner')).read().strip() == owner:
                shutil.rmtree(d, ignore_errors=True)   # stale claim: owner job is gone
    return False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int); ap.add_argument('--ntasks', type=int)
    ap.add_argument('--only'); ap.add_argument('--check', action='store_true')
    ap.add_argument('--trials', default='trials.json')
    a = ap.parse_args()
    for d in ('results', 'logs', 'work', 'claims'):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    trials = json.load(open(os.path.join(HERE, a.trials)))
    if a.only:
        ids = a.only.split(','); mine = [t for t in trials if t['id'] in ids]
        assert len(mine) == len(ids), (ids, [t['id'] for t in mine])
        acquire_gpu()
        recs = []
        for t in mine:
            claim(t); recs.append(run_one(t))
        if a.check:
            bad = [r['id'] for r in recs if r['status'] != 'ok']
            print('PILOT', 'PASS' if not bad else f'FAIL {bad}', flush=True)
            sys.exit(1 if bad else 0)
        sys.exit(0)
    acquire_gpu()
    order = list(range(len(trials)))
    start = (a.task * len(trials)) // a.ntasks
    order = order[start:] + order[:start]
    n_done = 0
    for k in order:
        t = trials[k]
        if os.path.exists(os.path.join(HERE, 'results', t['id'] + '.json')):
            continue
        if time_left_s() < STOP_S:
            print(f'stopping: < {STOP_S // 60} min walltime left', flush=True)
            sys.exit(0)
        if not claim(t):
            continue
        run_one(t); n_done += 1
    print(f'pool exhausted; this task ran {n_done} trials', flush=True)
