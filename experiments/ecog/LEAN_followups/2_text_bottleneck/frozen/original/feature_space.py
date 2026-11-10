"""Adapter: the original LEAN-v20 (shared/model.py, bit-identical to varpart-v20/lean_v20_enc.py) behind the
same FeatureSpace interface the rebuilds implement, so it is scored by exactly the same path."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location('lean_v20_model', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.py'))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


class FeatureSpace:
    def __init__(self, device='cpu'):
        self.emb = _m.build_embedder(device=device, interact_set='v3e')
        self.win = int(self.emb.win)

    def __call__(self, windows):
        return self.emb(windows)
