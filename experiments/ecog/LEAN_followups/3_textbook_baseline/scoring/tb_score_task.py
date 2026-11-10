"""Score frozen agent k on subject s exactly once (item = SLURM array index over agents x subjects)."""
import json, os, shutil, subprocess, sys, time
import numpy as np
B = '/path/to/ecog_data/lean_textbook_baseline'
MS = '/path/to/ecog_data/circuit-building-autoresearch/runs-wave/multisubj'
PY = '/path/to/ecog_data/circuit-building-autoresearch/.venv/bin/python'
SUBJ = ['patient1ALL', 'patient2', 'patient3', 'patient4', 'patient5', 'patient6']
idx = int(sys.argv[1]); k, s = idx // len(SUBJ) + 1, SUBJ[idx % len(SUBJ)]
out = f'{B}/scoring/results/agent{k}_{s}'
if os.path.exists(out + '.json'):
    sys.exit(0)
wd = f'{B}/scoring/work/agent{k}_{s}'
shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
os.symlink(f'{MS}/src_multi', f'{wd}/src')
shutil.copy(f'{B}/scoring/tb_encoder.py', f'{wd}/wave_encoder.py')
npz = f'{wd}/rescore.npz'
env = dict(os.environ, RESCORE_OUT=npz, MS_SPLIT=f'{MS}/{s}/split.json', TB_AGENT_DIR=f'{B}/frozen/agent_{k}',
           ECOG_RIDGE_GPU='1', PYTHONDONTWRITEBYTECODE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
           TMPDIR=f'{wd}/tmp', OMP_NUM_THREADS=os.environ.get('SLURM_CPUS_PER_TASK', '8'))
env.pop('ECOG_TEST_CHUNKS_FILE', None); env.pop('ECOG_EXCLUDE_PREFIX', None); env.pop('PYTHONPATH', None)
os.makedirs(env['TMPDIR'])
t0 = time.time()
with open(f'{B}/scoring/logs/agent{k}_{s}.log', 'w') as log:
    rc = subprocess.run([PY, 'wave_encoder.py'], cwd=wd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
cfg = json.load(open(f'{MS}/{s}/split.json'))
rec = dict(agent=k, subject=s, rc=rc, seconds=round(time.time() - t0))
if os.path.exists(npz):
    z = np.load(npz); rr = z['r_chunk_elec']; m = rr.mean(0)
    rec.update(status='ok', grid=float(m[:cfg['n_grid']].mean()), all_elec=float(m.mean()), P=int(z['P']),
               n_test=int(rr.shape[0]))
    np.save(out + '_r.npy', rr.astype(np.float32))
else:
    rec['status'] = 'failed'
json.dump(rec, open(out + '.json', 'w'))
shutil.rmtree(wd, ignore_errors=True)
print(rec, flush=True)
