"""Manual fixed-cohort evaluation: RSA search or held-out encoding.

Examples::

    python -m harness.run --dataset laion --stage search --model model.py
    python -m harness.run --dataset nsd --stage final --model runs/frozen/model.py

Only the configured source dataset is opened. The stage is checked against the
dataset's immutable protocol, so final data cannot become a search reward.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _load_model(path: str):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("neuro_manual_model", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import model from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "MODEL"):
        raise ValueError(f"{path} must export MODEL")
    return module.MODEL


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=os.environ.get("NEURO_DATASET", "laion"))
    parser.add_argument("--stage", choices=("search", "final"))
    parser.add_argument("--model", default="model.py", help="Python file exporting MODEL")
    parser.add_argument("--output", type=Path, help="Save the complete JSON result")
    args = parser.parse_args(argv)

    os.environ["NEURO_DATASET"] = args.dataset
    from harness.datasets import get_dataset
    from harness.reward import _validate_contract, evaluate_subjects

    dataset = get_dataset(args.dataset)
    _validate_contract(dataset, args.stage)
    model = _load_model(args.model)
    row = evaluate_subjects(model, dataset, stage=args.stage)
    rendered = json.dumps(_json_safe(row), indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return row


if __name__ == "__main__":
    raise SystemExit(0 if main()["status"] == "ok" else 1)
