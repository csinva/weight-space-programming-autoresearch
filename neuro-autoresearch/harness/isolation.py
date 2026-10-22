"""OS-isolated image-only candidate execution; private scoring stays in the caller.

Bubblewrap supplies mount, PID, user and network namespaces. There is no fallback
that imports a candidate in the scorer. Installed Python packages are trusted
runtime inputs; datasets, run directories, HOME and the repository are not mounts.
"""
from __future__ import annotations

import json
import glob
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def runtime_mounts() -> list[str]:
    """Read-only interpreter/runtime mounts, never a whole home or checkout."""
    roots = {Path('/usr')}
    for prefix in {Path(sys.prefix), Path(sys.base_prefix)}:
        for name in ('bin', 'lib', 'lib64'):
            path = prefix / name
            if path.exists():
                roots.add(path)
    # Keep only outermost mounts.
    roots = {p for p in roots if not any(q != p and q in p.parents for q in roots)}
    argv = []
    for path in sorted(roots):
        argv += ['--ro-bind', str(path), str(path)]
    cfg = Path(sys.prefix) / 'pyvenv.cfg'
    if cfg.is_file():
        argv += ['--ro-bind', str(cfg), str(cfg)]
    for name in ('bin', 'sbin', 'lib', 'lib64'):
        path = Path('/') / name
        if path.is_symlink():
            argv += ['--symlink', os.readlink(path), str(path)]
        elif path.exists():
            argv += ['--ro-bind', str(path), str(path)]
    return argv


def sandbox_command() -> list[str]:
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise RuntimeError('Candidate isolation requires bubblewrap; refusing unsafe execution')
    return [bwrap, '--die-with-parent', '--unshare-all', '--new-session',
            '--cap-drop', 'ALL', *runtime_mounts(), '--proc', '/proc', '--dev', '/dev',
            '--tmpfs', '/tmp', '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin',
            '--setenv', 'HOME', '/tmp', '--setenv', 'OMP_NUM_THREADS', '1']


def check_isolation() -> None:
    """Probe namespaces before loading private data or launching candidates."""
    result = subprocess.run([*sandbox_command(), '/usr/bin/true'], capture_output=True, timeout=10)
    if result.returncode:
        raise RuntimeError('Candidate isolation unavailable: ' + result.stderr.decode(errors='replace')[-1000:])


class IsolatedModel:
    """Trusted proxy: candidate code receives images, returns arrays and plain JSON."""
    def __init__(self, path: str | Path):
        check_isolation()
        self.source = Path(path).read_bytes()
        self.name = Path(path).stem
        self.description = self.insight = self.idea = ''
        self.n_params = 0
        self.repair_log = ''

    def __call__(self, images):
        self.repair_log = ""
        import torch
        array = images.detach().cpu().numpy() if torch.is_tensor(images) else np.asarray(images)
        with tempfile.TemporaryDirectory(prefix='neuro-candidate-') as temp:
            root = Path(temp)
            (root / 'model.py').write_bytes(self.source)
            np.save(root / 'images.npy', array, allow_pickle=False)
            output = root / 'output'
            output.mkdir()
            worker = Path(__file__).with_name('candidate_worker.py')
            devices = []
            for device in sorted(glob.glob('/dev/nvidia*')):
                if Path(device).is_file() or not Path(device).is_dir():
                    devices += ['--dev-bind', device, device]
            if devices and Path('/sys').is_dir():
                devices += ['--ro-bind', '/sys', '/sys']
            argv = [*sandbox_command(), *devices,
                    '--setenv', 'NEURO_CANDIDATE_DEVICE', str(images.device) if torch.is_tensor(images) else 'cpu',
                    '--setenv', 'CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '0'), '--ro-bind', str(root / 'model.py'), '/model.py',
                    '--ro-bind', str(root / 'images.npy'), '/images.npy',
                    '--ro-bind', str(worker), '/worker.py',
                    '--bind', str(output), '/output', '--chdir', '/tmp',
                    sys.executable, '-I', '/worker.py']
            with (root / 'stderr.log').open('w+b') as errors:
                result = subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=errors,
                                        timeout=int(os.environ.get('NEURO_EVAL_TIMEOUT', '420')))
                if result.returncode:
                    errors.seek(max(0, errors.tell() - 3000))
                    self.repair_log = errors.read().decode(errors='replace')
                    raise RuntimeError('Isolated candidate failed: ' + self.repair_log)
            feature_path = output / 'features.npy'
            info_path = output / 'metadata.json'
            for path, limit in ((feature_path, 2_000_000_000), (info_path, 8192)):
                if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
                    raise ValueError('Invalid candidate output file')
            features = np.load(feature_path, allow_pickle=False, mmap_mode='r')
            if features.ndim != 2 or len(features) != len(array) or features.shape[1] >= 10_000:
                self.repair_log = f"Feature shape {features.shape}; expected ({len(array)}, D) with D < 10000"
            elif features.dtype.kind not in 'fiu':
                self.repair_log = f"Feature dtype {features.dtype}; expected real numeric values"
            elif not np.isfinite(features).all():
                self.repair_log = "Feature array contains NaN or infinite values"
            if self.repair_log:
                raise ValueError(self.repair_log)
            metadata = json.loads(info_path.read_text())
            reported = int(metadata['n_params'])
            if reported < 0:
                raise ValueError('Invalid candidate parameter count')
            self.n_params = max(self.n_params, reported)
            for key in ('name', 'description', 'insight', 'idea'):
                setattr(self, key, str(metadata.get(key, ''))[:600])
            return np.array(features, copy=True)
