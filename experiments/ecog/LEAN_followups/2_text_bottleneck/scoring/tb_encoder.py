"""Scores one frozen textbook-baseline feature space with the LEAN harness (runs-wave/multisubj/src_multi).
The agent's packages go first on the path, as PYTHONPATH=/work/pkgs did in its sandbox, and the feature space
is built with device='cpu', the setting the agent tested. Output is checked (finite float32, H <= 4000) on
every segment; the fit/score is the streamed harness ridge (RESCORE_OUT, MS_SPLIT)."""
import importlib.util
import os
import sys

AGENT = os.environ['TB_AGENT_DIR']
sys.path.insert(0, os.path.join(AGENT, 'pkgs'))
import numpy as np  # noqa: E402
from src.eval import EncodingConfig, run_encoding  # noqa: E402

spec = importlib.util.spec_from_file_location('feature_space', os.path.join(AGENT, 'feature_space.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Checked:
    def __init__(self, fs):
        self.fs = fs
        self.win = int(getattr(fs, 'win', 2048))
        assert 0 < self.win <= 4096

    def __call__(self, windows):
        out = np.asarray(self.fs(np.ascontiguousarray(windows, dtype=np.float32)))
        assert out.ndim == 2 and out.shape[0] == windows.shape[0] and out.shape[1] <= 4000, out.shape
        assert np.all(np.isfinite(out)), 'non-finite features'
        return out.astype(np.float32, copy=False)


if __name__ == '__main__':
    run_encoding(Checked(mod.FeatureSpace(device='cpu')), EncodingConfig(num_train=500, num_test=100))
