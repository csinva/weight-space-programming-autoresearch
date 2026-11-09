"""Step 0: copy each selected model (and the snapshot files it loads) into WORK/originals/<name>/, and build
WORK/data_visible, a copy of the data folder holding only the fitting and visible stories, for the rebuild sandboxes."""
import json, re, shutil
from common import DATA, MODELS, RUN, WORK

LIB = RUN / 'interpretable_transformers_lib'


def deps(name, seen=None):
    seen = [] if seen is None else seen
    for d in re.findall(r"os\.path\.join\(_HERE,\s*'([^']+\.py)'", (LIB / name).read_text()):
        if d not in seen:
            seen.append(d)
            deps(d, seen)
    return seen


if __name__ == '__main__':
    sel = {}
    for it, name in MODELS:
        out = WORK / 'originals' / name
        shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
        shutil.copy(LIB / f'{name}.py', out / 'model.py')
        helpers = deps(f'{name}.py')
        if any('import interpretable_transformer' in (LIB / f).read_text() for f in [f'{name}.py'] + helpers):
            shutil.copy(RUN / 'interpretable_transformer.py', out / 'interpretable_transformer.py')  # the run's main file, imported for its word lists
            helpers.append('interpretable_transformer.py')
        for h in helpers:
            shutil.copy(LIB / h, out / h)
        sel[name] = dict(iteration=it, helpers=helpers)
        print(it, name, helpers)
    (WORK / 'selected.json').write_text(json.dumps(sel, indent=1))
    split = json.loads((DATA / 'split.json').read_text())
    vis = WORK / 'data_visible'
    shutil.rmtree(vis, ignore_errors=True)
    for sub in ('grids', 'resp'):
        (vis / sub).mkdir(parents=True)
    for s in split['fit'] + split['visible']:
        shutil.copy(DATA / 'grids' / f'{s}.TextGrid', vis / 'grids')
        shutil.copy(DATA / 'resp' / f'{s}.hf5', vis / 'resp')
    shutil.copy(DATA / 'respdict.json', vis)
    (vis / 'split.json').write_text(json.dumps(dict(fit=split['fit'], visible=split['visible'])))
    shutil.copytree(DATA / 'nltk', vis / 'nltk')
    print('visible data:', len(split['fit']), 'fitting +', len(split['visible']), 'visible stories')
