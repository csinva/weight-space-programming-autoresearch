"""Rescore trials: run each snapshot's own __main__ in a sandbox whose src/ is the patched
harness (src_patched), which streams the fit and saves per-chunk per-electrode test r.
usage: rescore_task.py --task i --ntasks N   |   rescore_task.py --only id1,id2,...  [--check]

Array mode is WORK-STEALING: a task first takes its own static share (index % N == task), then
keeps claiming any unclaimed trial (atomic mkdir in claims/; flock is unreliable on this NFS)
until none are left or < STOP_S of walltime remains. A claim whose owner job is no longer in
the queue is stale and can be taken over. Shares of tasks that started before WS_SWITCH (old
static code, no claims) are left to them while they are still running."""
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
               ECOG_EXCLUDE_PREFIX='450a36e4', ECOG_RIDGE_GPU='1', PYTHONDONTWRITEBYTECODE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if t.get('pythonpath'):  # e.g. NumPy 1.26 overlay for trials that ran before the Jun 5 reinstall
        env['PYTHONPATH'] = t['pythonpath'] + os.pathsep + env.get('PYTHONPATH', '')
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


ARRAY = '29210855'
WS_SWITCH = '2026-11-05T15:23:07'   # tasks started before this run the old static-share code
STOP_S = 45 * 60
CLAIMS = os.path.join(HERE, 'claims')


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


_old = {'t': 0.0, 'ids': set()}


def old_running():
    """Array task ids still RUNNING that started before WS_SWITCH (static code, no claims)."""
    if time.time() - _old['t'] > 60:
        ids = set()
        for line in _sq(['-r', '-j', ARRAY, '-t', 'R', '-o', '%K %S']).splitlines():
            k, st = line.split()
            if st < WS_SWITCH:
                ids.add(int(k))
        _old.update(t=time.time(), ids=ids)
    return _old['ids']


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
                return False          # being created right now
            if owner == me or _sq(['-j', owner, '-o', '%i']):
                return owner == me    # live owner (or ourselves after a requeue)
            if open(os.path.join(d, 'owner')).read().strip() == owner:
                shutil.rmtree(d, ignore_errors=True)   # stale claim: owner job is gone
    return False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int); ap.add_argument('--ntasks', type=int)
    ap.add_argument('--only'); ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    trials = json.load(open(os.path.join(HERE, 'trials.json')))
    if a.only:
        ids = a.only.split(','); mine = [t for t in trials if t['id'] in ids]
        assert len(mine) == len(ids), (ids, [t['id'] for t in mine])
        acquire_gpu()
        recs = [run_one(t) for t in mine]
        if a.check:
            bad = [r['id'] for r in recs if r['status'] != 'ok']
            print('PILOT', 'PASS' if not bad else f'FAIL {bad}')
            sys.exit(1 if bad else 0)
        sys.exit(0)
    os.makedirs(CLAIMS, exist_ok=True)
    idx = list(range(len(trials)))
    own = [k for k in idx if k % a.ntasks == a.task]
    rest = [k for k in idx[a.task:] + idx[:a.task] if k % a.ntasks != a.task]
    acquire_gpu()
    n_done = 0
    for sweep in range(2):            # 2nd sweep picks up shares of old tasks that died
        for k in own + rest:
            t = trials[k]
            if os.path.exists(os.path.join(HERE, 'results', t['id'] + '.json')):
                continue
            if time_left_s() < STOP_S:
                print(f'stopping: < {STOP_S // 60} min walltime left', flush=True)
                sys.exit(0)
            if k % a.ntasks in old_running():
                continue
            if not t.get('skip') and not claim(t):
                continue
            run_one(t); n_done += 1
    print(f'pool exhausted; this task ran {n_done} trials', flush=True)
