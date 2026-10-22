"""Export a small image-only debugging interface for isolated search agents."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def export_public_inputs(run: str | Path, dataset, *, split=None) -> Path:
    """Write eight examples without exposing responses, metadata or source paths.

    Scaffolding uses deterministic synthetic pixels and never opens neural data.
    A trusted caller with an already loaded search split may instead supply it;
    only its protocol-filtered X_test images are exported. Existing examples are
    reused on resume, without loading data or regenerating their bytes.
    """
    from harness.datasets import get_dataset
    from harness.protocols import validate_search_dataset

    if isinstance(dataset, str):
        dataset = get_dataset(validate_search_dataset(dataset))
    name = validate_search_dataset(dataset.loader)
    if dataset.stage != "search" or dataset.metric != "rsa":
        raise ValueError("Public examples require a fixed RSA search dataset")
    root = Path(run) / "public"
    image_path, spec_path = root / "images.npy", root / "input_spec.json"
    kind = "synthetic_debug_images" if split is None else "permitted_search_images"
    contract = {"schema_version": 1, "dataset": name, "kind": kind,
                "layout": "flattened_HWC_RGB", "imsize": dataset.imsize,
                "dtype": "float32", "value_range": [0, 1],
                "output": "finite numeric (images, features) with fewer than 10000 features",
                "images_file": "images.npy"}
    if spec_path.exists():
        existing = json.loads(spec_path.read_text())
        if any(existing.get(key) != value for key, value in contract.items()):
            raise ValueError("Public input contract changed; use a fresh run")
        if (not image_path.is_file() or image_path.is_symlink()
                or hashlib.sha256(image_path.read_bytes()).hexdigest() != existing["images_sha256"]):
            raise ValueError("Public debug images changed after export")
        return root
    if split is None:
        images = np.random.default_rng(0).random((8, dataset.imsize ** 2 * 3), dtype=np.float32)
    else:
        values = split[1][:8]
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        images = np.array(values, dtype=np.float32, copy=True)
    if (images.ndim != 2 or images.shape[1] != dataset.imsize ** 2 * 3 or not len(images)
            or not np.isfinite(images).all() or np.any(images < 0) or np.any(images > 1)):
        raise ValueError("Debug images must satisfy the fixed flattened RGB input contract")
    root.mkdir(parents=True, exist_ok=True)
    with image_path.open("xb") as file:
        np.save(file, images, allow_pickle=False)
    contract.update(n_images=len(images), images_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest())
    with spec_path.open("x") as file:
        json.dump(contract, file, indent=2, allow_nan=False)
        file.write("\n")
    return root
