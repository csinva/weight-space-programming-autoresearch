"""Freeze per-subject train/test lists for the multi-subject rerun (before any scoring).

Same rule as patient1: one whole session held out (the paper's standard holdout per subject,
molmo_subjects.json), never used in training; train = the first 500 chunks of a seed-42 shuffle of
the remaining valid chunks (data._valid_chunks semantics). Valid = in master_list.json with audio,
high-gamma and Whisper features. Test = every valid chunk of the held-out session (no transcripts
exist for these subjects, so no speech filter). patient1VAL reproduces the original patient1 setup
(frozen train, old 23-chunk test) to validate the multi-subject harness against logged scores.
"""
import hashlib, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.join(os.path.dirname(HERE), 'newtest')
SUBJ = {  # held-out session, electrodes, grid rows (first N, grid_masks/*.npy)
    'patient2': ('236d910e', 235, 127), 'patient3': ('e3d31b95', 116, 63), 'patient4': ('2eeb170d', 155, 96),
    'patient5': ('5e416143', 188, 124), 'patient6': ('5d2842ef', 167, 127)}
lines = []
for s, (held, nel, ngrid) in SUBJ.items():
    root = f'/path/to/scratch/ondemand/{s}_Dayfiles/Dayfiles'
    wav, wh = f'{root}/wav_files', f'/path/to/ecog_data/whisper_features/{s}'
    ml = json.load(open(f'{root}/master_list.json'))
    ok = lambda c: all(os.path.exists(p) for p in (f'{wav}/{c}.wav', f'{wav}/{c}_ecog_hasson_highgamma.npy', f'{wh}/{c}_whisper.npy'))
    valid = [c for c in ml if ok(c)]
    test = [c for c in valid if c.startswith(held)]
    pool = list(valid); np.random.RandomState(42).shuffle(pool)
    train = [c for c in pool if not c.startswith(held)][:500]
    assert len(train) == 500 and not set(train) & set(test)
    os.makedirs(os.path.join(HERE, s, 'wave_aligned'), exist_ok=True)
    cfg = dict(subject=s, data_dir=wav, whisper_dir=wh, wave_dir=os.path.join(HERE, s, 'wave_aligned'),
               held_out_session=held, n_electrodes=nel, n_grid=ngrid, train=train, test=test)
    json.dump(cfg, open(os.path.join(HERE, s, 'split.json'), 'w'), indent=0)
    h = lambda v: hashlib.sha256(json.dumps(v).encode()).hexdigest()[:16]
    lines.append(f'{s}: valid {len(valid)}, train {len(train)} sha {h(train)}, test {len(test)} ({held}) sha {h(test)}, '
                 f'sessions in train {len(set(c[:8] for c in train))}')
# validation config: original patient1 setup through the multi-subject harness
os.makedirs(os.path.join(HERE, 'patient1VAL'), exist_ok=True)
json.dump(dict(subject='patient1VAL', data_dir='/path/to/ecog_data/wav_files',
               whisper_dir='/path/to/ecog_data/wav_files',
               wave_dir='/path/to/scratch/patient1_wave_aligned', held_out_session='450a36e4',
               n_electrodes=166, n_grid=125, train=json.load(open(f'{NT}/train_frozen.json')),
               test=json.load(open(f'{NT}/test_old.json'))), open(os.path.join(HERE, 'patient1VAL', 'split.json'), 'w'), indent=0)
open(os.path.join(HERE, 'frozen_hashes.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
