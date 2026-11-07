"""Trial list for the new-test rerun: every logged main-line trial of Figure 1, in Figure 1 order.

Rows are selected exactly as summary_figures/plot_progress_grid.py does (same runs, phase order,
exclusions and (name, score) dedup). Code source per row:
  1. the rescore-grid record, if that snapshot already reproduced its logged score (Nov 5);
  2. otherwise the replayed snapshot (replay_snapshots.py) of the eval that produced the row:
     the last eval of that name whose read-back score equals the logged one (4 dp), else the
     last non-crashed eval of that name.
Every trial must again reproduce its logged score on the old test set (rescore_task_newtest.py).
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RW, 'summary_figures'))
from plot_progress_grid import EXCLUDE, GRID_RUNS, PHASES  # noqa: E402

RESCORED = {}
for f in glob.glob(os.path.join(RW, 'rescore-grid', 'results', '*.json')):
    r = json.load(open(f))
    if r.get('status') == 'ok':
        RESCORED[(r['run'], r['name'], r['logged'])] = r
REPLAY = {json.load(open(f))['run']: json.load(open(f))['evals'] for f in glob.glob(os.path.join(HERE, 'replay_*.json'))}


def strip_std(args):
    out, i = [], 0
    while i < len(args):
        if args[i] in ('--num-train', '--num-test'):
            i += 2
            continue
        out.append(args[i]); i += 1
    return out


def num_train(args):
    return args[args.index('--num-train') + 1] if '--num-train' in args else '500'


trials, missing, seen = [], [], set()
for header, runs in PHASES:
    for run in runs:
        rows = list(csv.reader(open(os.path.join(RW, run, 'results', 'overall_results.csv'))))
        if rows and rows[0][0] == 'subject':
            rows = rows[1:]
        for r in rows:
            if len(r) < 7 or EXCLUDE.search(r[6]) or (r[6], r[1]) in seen:
                continue
            seen.add((r[6], r[1]))
            name, logged = r[6], float(r[1])
            tid = f"{run}__{re.sub(r'[^A-Za-z0-9]+', '_', name)[:60]}_{hashlib.md5((run + name + r[1]).encode()).hexdigest()[:6]}"
            rec = dict(id=tid, run=run, name=name, logged=logged, order=len(trials) + len(missing),
                       phase=header.replace('\n', ' '), metric='grid' if run in GRID_RUNS else 'all166')
            old = RESCORED.get((run, name, logged))
            if old:
                rec.update(snapshot=old['snapshot'], args=old.get('args', []), source='rescore-grid')
                if old.get('pythonpath'):
                    rec['pythonpath'] = old['pythonpath']
                trials.append(rec)
                continue
            ev = [e for e in REPLAY.get(run, []) if e.get('name') == name and e.get('snapshot')]
            hit = [e for e in ev if any(abs(s - logged) <= 6e-5 for s in e['scores'])]
            pick = (hit or [e for e in ev if not e['crashed']] or ev or [None])[-1]
            if pick is None:
                rec['why'] = 'no eval of this name in the logs'
                missing.append(rec)
                continue
            rec.update(snapshot=pick['snapshot'], args=strip_std(pick['args']), source='replay',
                       replay_k=pick['k'], replay_how=pick['how'], replay_verified=pick['verified'],
                       score_seen=bool(hit), eval_ts=pick['ts'], env=pick.get('env') or {})
            if num_train(pick['args']) != '500':
                rec['nonstandard'] = f"--num-train {num_train(pick['args'])}"
                rec['num_train'] = num_train(pick['args'])
            trials.append(rec)

json.dump(trials, open(os.path.join(HERE, 'trials.json'), 'w'), indent=0)
json.dump(missing, open(os.path.join(HERE, 'trials_missing.json'), 'w'), indent=0)
by = {}
for t in trials:
    by.setdefault(t['source'], 0); by[t['source']] += 1
print(f'{len(trials)} trials ({by}), {len(missing)} without code; '
      f"score read back for {sum(t.get('score_seen', False) for t in trials if t['source'] == 'replay')} replayed; "
      f"nonstandard {sum('nonstandard' in t for t in trials)}")
for m in missing[:10]:
    print('  missing:', m['run'], m['name'], m['why'])
