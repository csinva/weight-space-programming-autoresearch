"""Rerun every Figure 1 trial on five more subjects (held-out session per subject).

Work item = (subject, trial). Each trial's own __main__ runs in a sandbox whose src/ is src_multi:
the subject's directories, electrode count and frozen train/test lists come from
<subject>/split.json (MS_SPLIT); the fit is the streamed harness ridge (rescore_stream), and the
held-out session is scored chunk by chunk. patient1VAL (original patient1 split: frozen train, old 23-chunk
test) runs only in the pilot and must reproduce the logged patient1 scores, which validates the
multi-subject harness end to end. Trials logged before the Jun 5 venv reinstall that fail are
retried once with the NumPy 1.26.4 overlay.

usage: rescore_task_multi.py --task i --ntasks N  |  --only subj:id,subj:id [--check]
Work-stealing: every task walks the same priority order (key models first, subjects interleaved);
atomic mkdir claims, stale-owner takeover (owner not in a cached squeue), stop at < STOP_S left.
"""
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NT = os.path.join(os.path.dirname(HERE), 'newtest')
PY = os.path.join(REPO, '.venv', 'bin', 'python')
NP126 = os.path.join(REPO, 'runs-wave', 'rescore-grid', 'np126')
SUBJECTS = ['patient2', 'patient3', 'patient4', 'patient5', 'patient6']
REINSTALL = '2026-06-05T12:08'
TOL = 0.00015
STOP_S = 45 * 60
CLAIMS = os.path.join(HERE, 'claims')
KEY = ['LEAN-v18.q45-SLOWQUAD5SCALES', 'LEAN-v35-AUTOLAGZ', 'LEAN-v16', 'INTERP-v3e-interact',
       'COMBINE-v5-S2s+s1dyn', 'LEAN-v18.q29-GATESIGS', 'LEAN-v31-LOGEPROD8', 'Gammatone-cochleagram',
       'Gammatone-log-adapt', 'Gammatone-3win-5tau', 'Gammatone-BEST+HL+modbank+STMF14']


def items():
    trials = json.load(open(os.path.join(NT, 'trials.json')))
    rank = {n: i for i, n in enumerate(KEY)}
    order = sorted(trials, key=lambda t: (rank.get(t['name'], len(KEY)), t['order']))
    return [(s, t) for t in order for s in SUBJECTS]


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


def attempt(subj, t, pythonpath):
    split = os.path.join(HERE, subj, 'split.json')
    wd = os.path.join(HERE, 'work', f'{subj}__{t["id"]}')
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(os.path.join(wd, 'results'))
    os.symlink(os.path.join(HERE, 'src_multi'), os.path.join(wd, 'src'))
    shutil.copy(t['snapshot'], os.path.join(wd, 'wave_encoder.py'))
    npz = os.path.join(wd, 'rescore.npz')
    env = dict(os.environ, RESCORE_OUT=npz, MS_SPLIT=split, WAVE_FEAT_CACHE=os.path.join(wd, 'cache'),
               ECOG_RIDGE_GPU='1', PYTHONDONTWRITEBYTECODE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               TMPDIR=os.path.join(wd, 'tmp'), **t.get('env', {}))
    env.pop('ECOG_TEST_CHUNKS_FILE', None); env.pop('ECOG_EXCLUDE_PREFIX', None)
    os.makedirs(env['TMPDIR'])
    if pythonpath:
        env['PYTHONPATH'] = pythonpath + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [PY, 'wave_encoder.py', '--num-train', t.get('num_train', '500'), '--num-test', '100'] + t.get('args', [])
    logd = os.path.join(HERE, 'logs', subj); os.makedirs(logd, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(logd, t['id'] + '.log'), 'a') as log:
        log.write(f"### {time.ctime()} pythonpath={pythonpath or '-'} cmd={' '.join(cmd[1:])}\n"); log.flush()
        try:
            rc = subprocess.run(cmd, cwd=wd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=6 * 3600).returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
    rec = dict(t, subject=subj, rc=rc, seconds=round(time.time() - t0), pythonpath_used=pythonpath or None)
    cfg = json.load(open(split))
    if os.path.exists(npz):
        z = np.load(npz)
        rr = z['r_chunk_elec']; m = rr.mean(0)
        rec.update(all_elec=float(m.mean()), grid=float(m[:cfg['n_grid']].mean()), P=int(z['P']),
                   n_test=int(rr.shape[0]), n_elec=int(rr.shape[1]))
        np.save(os.path.join(HERE, 'results', subj, t['id'] + '_r.npy'), rr.astype(np.float32))
        if subj == 'patient1VAL':
            got = rec['grid'] if t['metric'] == 'grid' else rec['all_elec']
            rec['reproduced'] = bool(abs(got - t['logged']) <= TOL)
            rec['status'] = 'ok' if rec['reproduced'] else 'mismatch'
        else:
            rec['status'] = 'ok'
    else:
        rec['status'] = 'failed'
    shutil.rmtree(wd, ignore_errors=True)
    return rec


def run_one(subj, t):
    os.makedirs(os.path.join(HERE, 'results', subj), exist_ok=True)
    out = os.path.join(HERE, 'results', subj, t['id'] + '.json')
    if os.path.exists(out):
        return json.load(open(out))
    rec = attempt(subj, t, t.get('pythonpath'))
    if rec['status'] != 'ok' and not t.get('pythonpath') and t.get('eval_ts') and t['eval_ts'] < REINSTALL:
        first = {k: rec.get(k) for k in ('status', 'rc', 'grid', 'all_elec')}
        rec = attempt(subj, t, NP126)
        rec['first_attempt'] = first
    json.dump(rec, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    print(f"{rec['status']:8s} {subj} {t['id']:66s} grid={rec.get('grid', float('nan')):.5f} "
          f"all={rec.get('all_elec', float('nan')):.5f} rc={rec['rc']} {rec['seconds']}s", flush=True)
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


_live = {'t': 0.0, 'ids': set()}


def live_jobs(force=False):
    """Unique job ids (%A: one per array element) of this user's queued/running jobs, cached 60 s."""
    if force or time.time() - _live['t'] > 60:
        _live.update(t=time.time(), ids=set(_sq(['-u', os.environ.get('USER', ''), '-o', '%A']).split()))
    return _live['ids']


def claim(key):
    d = os.path.join(CLAIMS, key)
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
            if owner == me:
                return True
            if not owner or owner in live_jobs() or owner in live_jobs(force=True):
                return False           # being written right now, or a live owner
            if open(os.path.join(d, 'owner')).read().strip() == owner:
                shutil.rmtree(d, ignore_errors=True)   # stale claim: owner job is gone
    return False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int); ap.add_argument('--ntasks', type=int)
    ap.add_argument('--only'); ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    for d in ('results', 'logs', 'work', 'claims'):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    by_id = {t['id']: t for t in json.load(open(os.path.join(NT, 'trials.json')))}
    if a.only:
        acquire_gpu()
        recs = []
        for it in a.only.split(','):
            subj, tid = it.split(':', 1)
            claim(f'{subj}__{tid}'); recs.append(run_one(subj, by_id[tid]))
        if a.check:
            bad = [f"{r['subject']}:{r['id']}" for r in recs if r['status'] != 'ok']
            print('PILOT', 'PASS' if not bad else f'FAIL {bad}', flush=True)
            sys.exit(1 if bad else 0)
        sys.exit(0)
    work = items()
    acquire_gpu()
    n_done = 0
    for k in range(len(work)):                     # all tasks walk the priority order; claims split it
        subj, t = work[k]
        if os.path.exists(os.path.join(HERE, 'results', subj, t['id'] + '.json')):
            continue
        if time_left_s() < STOP_S:
            print(f'stopping: < {STOP_S // 60} min walltime left', flush=True)
            sys.exit(0)
        if not claim(f'{subj}__{t["id"]}'):
            continue
        run_one(subj, t); n_done += 1
    print(f'pool exhausted; this task ran {n_done} items', flush=True)
