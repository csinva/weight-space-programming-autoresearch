"""Trial list for grid rescoring: every all-166-phase autoresearch trial with a recoverable
encoder snapshot (matched by the model_shorthand_name inside the snapshot file)."""
import csv, glob, hashlib, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); RW = os.path.dirname(HERE)
RUNS = ['wave-slurm-23628821', 'wave-trim', 'wave-lean', 'reset-A', 'reset-B', 'reset-C', 'reset-D',
        'reset-E', 'reset-F', 'reset-G', 'reset-H', 'reset-I', 'graft-lean', 'lean-interp-ling',
        'lean-interp', 'exceed-dense']
EX = re.compile(r'refer|MEL-CTRL|baseline', re.I)
NAME = re.compile(r'^model_shorthand_name\s*=\s*[\'"]([^\'"]+)[\'"]', re.M)
inner = {}
for run in RUNS:
    for p in sorted(glob.glob(os.path.join(RW, run, 'wave_encoders_lib', '*.py'))):
        m = NAME.search(open(p, errors='ignore').read())
        if m:
            inner.setdefault(m.group(1), []).append(p)
trials, seen, missing = [], set(), []
for run in RUNS:
    rows = list(csv.reader(open(os.path.join(RW, run, 'results', 'overall_results.csv'))))[1:]
    for r in rows:
        if len(r) < 7 or EX.search(r[6]) or (r[6], r[1]) in seen:
            continue
        seen.add((r[6], r[1]))
        name = r[6]
        local = [p for p in inner.get(name, []) if f'/{run}/' in p] or inner.get(name, [])
        fn = os.path.join(RW, run, 'wave_encoders_lib', f'{name}.py')
        snap = local[0] if local else (fn if os.path.exists(fn) else None)
        tid = f"{run}__{re.sub(r'[^A-Za-z0-9]+', '_', name)[:60]}_{hashlib.md5((run + name + r[1]).encode()).hexdigest()[:6]}"
        rec = dict(id=tid, run=run, name=name, logged=float(r[1]), snapshot=snap, order=len(trials) + len(missing))
        (trials if snap else missing).append(rec)
json.dump(trials, open(os.path.join(HERE, 'trials.json'), 'w'), indent=0)
json.dump(missing, open(os.path.join(HERE, 'trials_missing_snapshot.json'), 'w'), indent=0)
print(f'{len(trials)} trials with snapshots, {len(missing)} without')
