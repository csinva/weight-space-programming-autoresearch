"""Compatibility facade — the data loaders now live in ``harness/loaders/``.

The loading code is organized there one module per dataset family
(``core`` / ``neuralset`` / ``tvsd`` / ``nnn``); this module just re-exports the public
surface so existing ``from harness.data import ...`` imports keep working. New code should
import from ``harness.loaders`` (and ``harness.loaders.core`` for the shared helpers).
"""
from __future__ import annotations

from harness.loaders import load_split  # noqa: F401
from harness.loaders.core import (  # noqa: F401
    RawPixels,
    _study_path,
    _load_workers,
    _cache_dir,
    _build_cache_key,
    _build_cache_load,
    _build_cache_save,
    _ROI_DEFS,
    _S3_NSD,
    _nsd_subject,
    _ensure_roi_label_file,
    _roi_voxel_keep,
    _apply_roi,
    _collapse_repeats,
    _image_id,
    _canonical_test_ids,
)
