"""Score one frozen model on one subject exactly once (array index over 11 models x 6 subjects). Same wrapper
as the textbook baseline; harness = scoring/src_bn (src_multi + optional saving of grid predictions, patient1)."""
import json, os, shutil, subprocess, sys, time
import numpy as np
BN = '/path/to/ecog_data/lean_text_bottleneck'
MS = '/path/to/ecog_data/circuit-building-autoresearch/runs-wave/multisubj'
PY = '/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python'
MODELS = ['original'] + [f'rebuild_{k}' for k in range(1, 6)] + [f'control_{k}' for k in range(1, 6)]
SUBJ = ['patient1ALL', 'patient2', 'patient3', 'patient4', 'patient5', 'patient6']
idx = int(sys.argv[1]); m, s = MODELS[idx // len(SUBJ)], SUBJ[idx % len(SUBJ)]
out = f'{BN}/scoring/results/{m}_{s}'
if os.path.exists(out + '.json'):
    sys.exit(0)
wd = f'{BN}/scoring/work/{m}_{s}'
shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
os.symlink(f'{BN}/scoring/src_bn', f'{wd}/src')
shutil.copy(f'{BN}/scoring/tb_encoder.py', f'{wd}/wave_encoder.py')
npz = f'{wd}/rescore.npz'
env = dict(os.environ, RESCORE_OUT=npz, MS_SPLIT=f'{MS}/{s}/split.json', TB_AGENT_DIR=f'{BN}/frozen/{m}',
           ECOG_RIDGE_GPU='1', PYTHONDONTWRITEBYTECODE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
           TMPDIR=f'{wd}/tmp', OMP_NUM_THREADS=os.environ.get('SLURM_CPUS_PER_TASK', '8'))
if s == 'patient1ALL':
    env['SAVE_PRED'] = f'{BN}/scoring/preds/{m}_{s}.npz'
env.pop('ECOG_TEST_CHUNKS_FILE', None); env.pop('ECOG_EXCLUDE_PREFIX', None); env.pop('PYTHONPATH', None)
os.makedirs(env['TMPDIR'])
t0 = time.time()
with open(f'{BN}/scoring/logs/{m}_{s}.log', 'w') as log:
    rc = subprocess.run([PY, 'wave_encoder.py'], cwd=wd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
cfg = json.load(open(f'{MS}/{s}/split.json'))
rec = dict(model=m, subject=s, rc=rc, seconds=round(time.time() - t0))
if os.path.exists(npz):
    z = np.load(npz); rr = z['r_chunk_elec']; mm = rr.mean(0)
    rec.update(status='ok', grid=float(mm[:cfg['n_grid']].mean()), all_elec=float(mm.mean()), P=int(z['P']), n_test=int(rr.shape[0]))
    np.save(out + '_r.npy', rr.astype(np.float32))
else:
    rec['status'] = 'failed'
json.dump(rec, open(out + '.json', 'w'))
shutil.rmtree(wd, ignore_errors=True)
print(rec, flush=True)
