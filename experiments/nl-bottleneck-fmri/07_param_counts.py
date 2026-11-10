"""Step 7: parameter counts of the originals and their rebuilds.

The search counted parameters as the elements of embedder.model.parameters(), which gives 2 for the WordNet-based
models (iterations 238 to 242) because their weights are plain arrays and dictionaries, and most rebuilds have no
torch model at all. To compare originals and rebuilds the same way, we count every number a constructed embedder
stores: the elements of each torch tensor and numpy array, and each numeric value held in its dictionaries, lists
and own objects. Word lists (strings) and external resources such as WordNet are not counted. Where the search's
count applies it is reported alongside (search_n_params).

  python 07_param_counts.py   -> param_counts.json (next to results.json)
"""
import json
import numbers
from pathlib import Path

import numpy as np
import torch
from common import MODELS, WORK, load_embedder


def stored_numbers(obj, model_module, seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return obj.numel()
    if isinstance(obj, np.ndarray):
        return obj.size if obj.dtype.kind in 'biufc' else sum(stored_numbers(v, model_module, seen) for v in obj.flat)
    if isinstance(obj, bool) or isinstance(obj, (str, bytes)):
        return 0
    if isinstance(obj, numbers.Number):
        return 1
    if isinstance(obj, dict):
        return sum(stored_numbers(v, model_module, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(stored_numbers(v, model_module, seen) for v in obj)
    own = type(obj).__module__ == model_module or isinstance(obj, torch.nn.Module)
    if own and hasattr(obj, '__dict__'):
        return sum(stored_numbers(v, model_module, seen) for v in vars(obj).values())
    return 0  # external objects (corpus readers, tokenizers) are not the model's own parameters


def count(path):
    embedder = load_embedder(path)
    search = sum(p.numel() for p in embedder.model.parameters()) if hasattr(embedder, 'model') and hasattr(embedder.model, 'parameters') else None
    return dict(n_numbers=int(stored_numbers(embedder, type(embedder).__module__)), search_n_params=None if search is None else int(search))


def main():
    out = dict(original={}, rebuilt={})
    for it, name in MODELS:
        out['original'][name] = dict(iteration=it, **count(WORK / 'originals' / name / 'model.py'))
        print('original', it, name, out['original'][name], flush=True)
    for rd in sorted((WORK / 'recon').glob('*-r*')):
        meta = json.loads((rd / 'meta.json').read_text()) if (rd / 'meta.json').exists() else None
        if not meta or not meta.get('final'):
            continue
        out['rebuilt'][rd.name] = dict(name=meta['name'], rep=meta['rep'], **count(rd / 'interpretable_transformers_lib' / f"{meta['final']}.py"))
        print('rebuilt', rd.name, out['rebuilt'][rd.name], flush=True)
    (Path(__file__).resolve().parent / 'param_counts.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
