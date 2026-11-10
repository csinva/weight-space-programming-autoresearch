"""Derive the describer's model.py from LEAN-v20 (varpart-v20/lean_v20_enc.py) mechanically: drop comments,
docstrings, run-history strings (model_shorthand_name / model_description), the harness disk-cache glue
(_disk_features, _install_disk_features), harness imports (src.*) and the __main__ block; define SR = 44100.
Then verify the cleaned module produces features identical to the original on example audio."""
import ast, os, sys
import numpy as np
SRC = '/path/to/ecog_data/circuit-building-autoresearch/runs-wave/varpart-v20/lean_v20_enc.py'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared', 'model.py')
tree = ast.parse(open(SRC).read())
DROP_FUNCS = {'_disk_features', '_install_disk_features'}
body = []
for node in tree.body:
    if isinstance(node, ast.Expr):
        continue                                   # module docstring and harness calls (sys.path.insert, os.makedirs)
    if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('src'):
        continue
    if isinstance(node, ast.FunctionDef) and node.name in DROP_FUNCS:
        continue
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith('model_') or
                                            isinstance(t, ast.Name) and t.id in ('CACHE_DIR', 'RESULTS_DIR') for t in node.targets):
        continue
    if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and getattr(node.test.left, 'id', '') == '__name__':
        continue
    body.append(node)
for node in ast.walk(ast.Module(body=body, type_ignores=[])):     # strip nested docstrings
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and \
            isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant) and \
            isinstance(node.body[0].value.value, str):
        node.body = node.body[1:] or [ast.Pass()]
fut = [n for n in body if isinstance(n, ast.ImportFrom) and n.module == '__future__']
rest = [n for n in body if n not in fut]
mod = ast.Module(body=fut + [ast.parse('SR = 44100').body[0]] + rest, type_ignores=[])
code = ast.unparse(mod) + '\n'
for bad in ('patient1', '/gpfs', 'Dayfiles', 'user', 'lab', 'REVERTED', 'test_corr'):
    assert bad not in code, bad
open(OUT, 'w').write(code)
print('model.py lines', code.count('\n'), 'chars', len(code))
if len(sys.argv) > 1 and sys.argv[1] == 'verify':
    import importlib.util, torch
    sys.path.insert(0, os.path.dirname(SRC))                   # varpart-v20 has src -> evolve-wave/src
    import lean_v20_enc as orig
    spec = importlib.util.spec_from_file_location('clean', OUT); clean = importlib.util.module_from_spec(spec); spec.loader.exec_module(clean)
    seg = np.load('/path/to/ecog_data/lean_textbook_baseline/agent_1/work/examples/clip1.npy').astype(np.float32)
    def windows(win):
        T = (len(seg) - 4096) // 1470 + 1
        sw = np.lib.stride_tricks.sliding_window_view(seg, win)
        return np.ascontiguousarray(sw[(4096 - win) // 2 + np.arange(T) * 1470])
    eo, ec = orig.build_embedder(device='cpu', interact_set='v3e'), clean.build_embedder(device='cpu', interact_set='v3e')
    fo, fc = eo(windows(eo.win)), ec(windows(ec.win))
    print('shape', fo.shape, fc.shape, 'identical', bool(np.array_equal(fo, fc)), 'max|d|', float(np.abs(fo - fc).max()))
