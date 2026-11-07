"""Assign each lean-interp trial an explicit harness file + CLI flags.

The lean-interp loop ran a fixed command and chose each variant by editing the harness's
argparse defaults (--interact-set / --cond-mode); its snapshot files were later overwritten
with newer harness states. So each trial = (a harness file that contains its branch) +
(--interact-set <set> [--cond-mode <mode>]). Every trial still has to reproduce its logged
score in rescore_task.py, so a wrong assignment is caught, not plotted."""
import glob, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); RW = os.path.dirname(HERE)
ORIG = os.path.join(RW, 'lean-v17-clean', 'enc_orig.py')
LIB = sorted(glob.glob(os.path.join(RW, 'lean-interp', 'wave_encoders_lib', '*.py')))
COND = {'v1a': 'rep_target60_k8', 'v1b': 'rep_target120_k16', 'v1c': 'rep_target30_k4',
        'v1d': 'rep_sqrt60', 'v1e': 'rep_small_only'}
FINAL = {'INTERP-final': 'v2g', 'INTERP-final2': 'v2m', 'INTERP-final3': 'v2z',
         'INTERP-final4': 'v2z', 'INTERP-final5': 'v3e', 'INTERP-final6': 'v3e'}
src = {p: open(p, errors='ignore').read() for p in [ORIG] + LIB}


def has_branch(path, s):
    txt = src[path]
    return bool(re.search(r"interact_set\s*==\s*'%s'" % s, txt) or
                re.search(r"interact_set\s+in\s*\([^)]*'%s'" % s, txt))


trials = json.load(open(os.path.join(HERE, 'trials.json')))
summary = []
for t in trials:
    if t['run'] != 'lean-interp':
        continue
    m = re.match(r'INTERP-(v\d[a-z]+)-(interact|cond)$', t['name'])
    if t['name'] in FINAL:
        iset, cond = FINAL[t['name']], 'none'
    elif m and m.group(2) == 'cond':
        iset, cond = 'none', COND[m.group(1)]
    elif m:
        iset, cond = m.group(1), 'none'
    else:
        summary.append((t['name'], 'UNPARSED')); continue
    own = t['snapshot']
    order = [own, ORIG] + [p for p in LIB if p != own]
    if iset == 'none':
        cands = [p for p in order if f"'{cond}'" in src[p]]
    else:
        cands = [p for p in order if has_branch(p, iset)]
    if not cands:
        summary.append((t['name'], 'NO-BRANCH')); t['snapshot'] = None; continue
    t['snapshot'] = cands[0]
    t['args'] = ['--interact-set', iset, '--cond-mode', cond]
    summary.append((t['name'], f"{os.path.basename(cands[0])} {' '.join(t['args'])}"))
trials = [t for t in trials if t['snapshot']]
json.dump(trials, open(os.path.join(HERE, 'trials.json'), 'w'), indent=0)
for s in summary:
    print(*s)
print(len(trials), 'trials total')
