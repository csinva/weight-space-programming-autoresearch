"""Copy the multi-subject rerun into LEAN_newtest_package/multisubject/ (real files, no symlinks) and build
compact per-trial arrays per subject: chunk-averaged r per electrode (trials x electrodes) and
grid-mean r per test chunk (trials x chunks), with the trial ids and chunk names."""
import glob
import json
import os
import shutil

import numpy as np

MS = os.path.dirname(os.path.abspath(__file__))
RW = os.path.dirname(MS)
REPO = os.path.dirname(RW)
OUT = os.path.join(REPO, 'LEAN_newtest_package', 'multisubject')
SUBJ = ['patient2', 'patient3', 'patient4', 'patient5', 'patient6']


def cp(src, rel):
    dst = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.realpath(src), dst)


shutil.rmtree(OUT, ignore_errors=True)
for f in ('check_subjects.py', 'make_splits.py', 'frozen_hashes.txt', 'extract_multi.py', 'run_extract.sh',
          'rescore_task_multi.py', 'run_pilot.sh', 'run_array.sh', 'pilot_items.txt', 'ms_analyses.py',
          'varpart_aggregate.py', 'run_ms_gpu.sh', 'run_ms_cpu.sh', 'summarize_multisubj.py', 'package_multisubj.py',
          'jobs.txt'):
    cp(os.path.join(MS, f), f'code/{f}')
for f in glob.glob(os.path.join(MS, 'src_multi', '*.py')):
    cp(f, f'code/src_multi/{os.path.basename(f)}')
for f in glob.glob(os.path.join(MS, 'summary', '*')):
    cp(f, f'summary/{os.path.basename(f)}')
for s in SUBJ + ['patient1VAL', 'patient1ALL']:
    cp(os.path.join(MS, s, 'split.json'), f'subjects/{s}/split.json')
    for f in ('RESULTS.md', 'results_family.csv', 'results_section.csv', 'results_group.csv', 'perelec_R2.npz'):
        p = os.path.join(MS, s, 'varpart', f)
        if os.path.exists(p):
            cp(p, f'subjects/{s}/varpart/{f}')
    for f in ('RESULTS.md', 'results.csv', 'perelec.npz', 'perchunk_r.npz'):
        p = os.path.join(MS, s, 'both', f)
        if os.path.exists(p):
            cp(p, f'subjects/{s}/both/{f}')
for p in glob.glob(os.path.join(MS, 'logs', 'pilot_*.out')):
    cp(p, f'validation/{os.path.basename(p)}')
for p in glob.glob(os.path.join(MS, 'logs', '*patient1VAL*.out')):
    cp(p, f'validation/{os.path.basename(p)}')
for p in glob.glob(os.path.join(MS, 'results', 'patient1VAL', '*.json')):
    cp(p, f'validation/patient1VAL_trials/{os.path.basename(p)}')

trials = json.load(open(os.path.join(RW, 'newtest', 'trials.json')))
for s in SUBJ:
    cfg = json.load(open(os.path.join(MS, s, 'split.json')))
    ids, E, C, recs = [], [], [], []
    for t in trials:
        r = json.load(open(os.path.join(MS, 'results', s, t['id'] + '.json')))
        recs.append({k: r.get(k) for k in ('id', 'run', 'name', 'order', 'phase', 'status', 'grid', 'all_elec', 'P',
                                            'n_test', 'n_elec', 'seconds', 'rc')})
        rr = np.load(os.path.join(MS, 'results', s, t['id'] + '_r.npy'))
        ids.append(t['id']); E.append(rr.mean(0)); C.append(rr[:, :cfg['n_grid']].mean(1))
    json.dump(recs, open(os.path.join(OUT, 'subjects', s, 'trial_results.json'), 'w'), indent=0)
    np.savez_compressed(os.path.join(OUT, 'subjects', s, 'trials_compact.npz'), ids=np.array(ids),
                        test_chunks=np.array(cfg['test']), r_elec=np.array(E, np.float32),
                        r_chunk_grid=np.array(C, np.float32), n_grid=cfg['n_grid'])
print('files', sum(len(f) for _, _, f in os.walk(OUT)))
