"""OS-isolated image-only candidate execution; private scoring stays in the caller.

Bubblewrap supplies mount, PID, user and network namespaces. There is no fallback
that imports a candidate in the scorer. Installed Python packages are trusted
runtime inputs; datasets, run directories, HOME and the repository are not mounts.
"""

from __future__ import annotations

from pathlib import Path
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


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
        import numpy as np
        array = images.detach().cpu().numpy() if torch.is_tensor(images) else np.asarray(images)
        with tempfile.TemporaryDirectory(prefix='neuro-candidate-') as temp:
            root = Path(temp)
            (root / 'model.py').write_bytes(self.source)
            np.save(root / 'images.npy', array, allow_pickle=False)
            output = root / 'output'
            output.mkdir()
            worker = Path(__file__)
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


MAX_DOUT = 10_000
FORWARD_BATCH = 512
MAX_PARAMS = 100_000


def _parameter_limit() -> int:
    # Final evaluation can restore a saved run environment in the same process.
    # Read its cap at evaluation time, rather than keeping the import-time value.
    limit = int(os.environ.get("NEURO_MAX_PARAMS") or MAX_PARAMS)
    if limit < 0:
        raise ValueError("NEURO_MAX_PARAMS must be nonnegative")
    return limit


def _forward(model, X):
    """Run sample-independent model forwards in bounded batches."""
    import torch

    if not torch.is_tensor(X) or X.shape[0] <= FORWARD_BATCH:
        return model(X)
    return torch.cat([torch.as_tensor(model(X[i:i + FORWARD_BATCH]))
                      for i in range(0, X.shape[0], FORWARD_BATCH)], 0)


def count_params(model) -> int:
    """Count registered and loose tensors, deduplicated by identity.

    Call after forward to include lazy weights. Traverse nested containers and
    modules with cycle/depth guards; uninspectable values are skipped."""
    import torch

    seen_t: set[int] = set()      # tensor ids already counted
    seen_c: set[int] = set()      # container/module ids already visited (cycle guard)
    total = 0

    def add(t: "torch.Tensor") -> None:
        nonlocal total
        if id(t) not in seen_t:
            seen_t.add(id(t))
            total += int(t.numel())

    def walk(obj, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(obj, torch.Tensor):
            add(obj)
            return
        if id(obj) in seen_c:
            return
        if isinstance(obj, torch.nn.Module):
            seen_c.add(id(obj))
            for t in obj.parameters(recurse=True):
                add(t)
            for t in obj.buffers(recurse=True):
                add(t)
            for v in vars(obj).values():
                walk(v, depth + 1)
        elif isinstance(obj, dict):
            seen_c.add(id(obj))
            for v in obj.values():
                walk(v, depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            seen_c.add(id(obj))
            for v in obj:
                walk(v, depth + 1)

    try:
        if isinstance(model, torch.nn.Module):
            walk(model, 0)                                  # params/buffers/submodules + loose attrs
        for v in getattr(model, "__dict__", {}).values():  # loose tensors on any model (incl. plain classes)
            walk(v, 0)
    except Exception:  # noqa: BLE001  -- honest best-effort; never fail scoring over a count
        pass
    return total


def robust_n_params(model) -> int:
    """Use the larger of declared parameters and materialized tensor elements."""
    try:
        declared = int(getattr(model, "n_params", 0) or 0)
    except (TypeError, ValueError):
        declared = 0
    return max(declared, count_params(model))


def _validate_model_size(model, d_out: int) -> int:
    """Check materialized weights after forward and before expensive scoring."""
    if d_out >= MAX_DOUT:
        raise ValueError(f"representation D_out={d_out} >= {MAX_DOUT} cap")
    n_params = robust_n_params(model)
    limit = _parameter_limit()
    if n_params > limit:
        raise ValueError(f"n_params {n_params:,} exceeds the {limit:,} cap "
                         "(counts all materialized fixed weights)")
    return n_params


def _readout_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_ENV_KEYS = (
    "NEURO_SUBJECT", "NEURO_IMSIZE", "NEURO_MAX_PARAMS", "NEURO_DATA_DIR",
    "LAION_DATA_DIR", "LAION_FMRI_DIR", "TRIPLEN_DATA_DIR", "NSD_ROOT",
    "NSD_STIMULI_HDF5_PATH", "NSD_DATA_DIR", "NSD_DATA_FILE", "NSD_BETAS_DIR",
    "TVSD_DATA_DIR", "TVSD_STIMULI_HDF5_PATH", "LAION_NCSNR_MIN", "NSD_NCSNR_MIN",
    "TRIPLEN_MIN_RELIABILITY", "TVSD_ORACLE_MIN",
)


def evaluation_sha256() -> str:
    """Fingerprint fixed scientific code so a resumed queue cannot change its reward."""
    from pathlib import Path
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_versions() -> dict:
    """Track numerical dependencies as part of the reproducible run contract."""
    from importlib.metadata import version
    import platform
    return {"python": platform.python_version(), **{name: version(name) for name in (
        "torch", "numpy", "scipy", "h5py", "pillow", "nibabel", "xarray", "pandas")}}


def _count_loc(text: str) -> int:
    """Lines of code in a model source: non-blank, non-comment lines."""
    lines = (line.strip() for line in text.splitlines())
    return sum(bool(line) and not line.startswith("#") for line in lines)


_ATTR_RE = {
    "n_params": re.compile(r"^\s*n_params\s*=\s*([0-9_]+)\s*(?:#.*)?$", re.M),
    "model_loc": re.compile(r"^\s*model_loc\s*=\s*([0-9_]+)\s*(?:#.*)?$", re.M),
}


def model_complexity(path: str) -> dict:
    """Read declared n_params/model_loc without importing candidate code.

    Fall back to a source line count for model_loc; the evaluator measures
    materialized parameters after forward."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except (OSError, TypeError):
        return {"n_params": None, "model_loc": None}

    def declared(name):
        m = _ATTR_RE[name].search(text)
        return int(m.group(1).replace("_", "")) if m else None

    model_loc = declared("model_loc")
    return {"n_params": declared("n_params"),
            "model_loc": model_loc if model_loc is not None else _count_loc(text)}


_CHEAT_PATTERNS = (
    (r"from_pretrained|AutoModel|AutoTokenizer|\bGPT2\w*|\btransformers\b|huggingface|\btimm\b|"
     r"open_clip|sentence_transformers|torch\.hub\.load|torchvision\.models", "loads a pretrained model"),
    (r"\burllib\b|\brequests\b|https?://|socket\.|\bwget\b|\bcurl\b|snapshot_download", "network / download"),
    (r"neuro_eval\.(rsa|encoding|datasets|readers|_evaluation)|looped\.data|looped\.metrics|\bload_split\b|reward_region_mask", "reads the target/readout"),
    (r"corrcoef\s*=|evaluate_on_split\s*=|\bridge\.\w+\s*=|\breward\.\w+\s*=", "monkeypatches the readout"),
)


def scan_source(path):
    """Reason string if the model SOURCE contains an obvious reward-hack, else None."""
    try:
        src = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return None
    return next((why for pat, why in _CHEAT_PATTERNS if re.search(pat, src)), None)


def _candidate_worker():
    import importlib.util
    import json
    import os
    import resource

    import numpy as np
    import torch

    BATCH_TOLERANCE = 0.01  # Approximate independence check; allow 1% vector L2 roundoff.

    resource.setrlimit(resource.RLIMIT_FSIZE, (2_000_000_000, 2_000_000_000))
    spec = importlib.util.spec_from_file_location('candidate', '/model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MODEL
    if isinstance(model, torch.nn.Module):
        model.eval()
    device = os.environ['NEURO_CANDIDATE_DEVICE']
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable inside candidate isolation; refusing device fallback')
    images = torch.from_numpy(np.load('/images.npy', allow_pickle=False)).to(device)
    with torch.no_grad():
        output = model(images)
        # Fixed-seed, sample-independent representations must agree across batching.
        for index in range(min(3, len(images))):
            single = torch.as_tensor(model(images[index:index + 1]))
            batched = torch.as_tensor(output[index:index + 1])
            if batched.shape != single.shape:
                raise ValueError('Representation shape depends on batch composition')
            a, b = batched.double(), single.double()
            difference = torch.linalg.vector_norm(a - b)
            scale = torch.maximum(torch.linalg.vector_norm(a), torch.linalg.vector_norm(b))
            if not torch.isfinite(difference) or difference > BATCH_TOLERANCE * scale:
                raise ValueError('Representation depends on batch composition or mutable state')
    if torch.is_tensor(output):
        output = output.detach().cpu().numpy()
    output = np.asarray(output)
    np.save('/output/features.npy', output, allow_pickle=False)
    seen = set()
    def count(value, depth=0):
        if depth > 8 or id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            return value.numel()
        if isinstance(value, dict):
            return sum(count(v, depth + 1) for v in value.values())
        if isinstance(value, (tuple, list, set)):
            return sum(count(v, depth + 1) for v in value)
        if hasattr(value, '__dict__'):
            return count(vars(value), depth + 1)
        return 0
    metadata = {key: str(getattr(model, key, ''))[:600] for key in ('name', 'description', 'insight', 'idea')}
    metadata['n_params'] = max(int(getattr(model, 'n_params', 0)), count(model))
    with open('/output/metadata.json', 'w') as file:
        json.dump(metadata, file)


if __name__ == "__main__":
    _candidate_worker()
