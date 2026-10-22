#!/usr/bin/env python3
"""Freeze an RSA-selected model, then evaluate its paired OOD dataset with encoding.

    python -m experiments.evaluation.final --run runs/laion__2026-10-21
    python -m experiments.evaluation.final --model model.py --search-dataset triplen

No encoding result is written to the search database. Model selection finishes
before any final-dataset responses are loaded.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from harness.protocols import SOURCE_REVISION, protocol_for, validate_search_dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_model(run: str | Path) -> dict:
    """Read only the highest RSA row from a compatible run, validating its bytes."""
    import hub
    run = Path(run).resolve()
    env = hub.search_contract(str(run))
    protocol = protocol_for(env["NEURO_DATASET"])
    from search.setup_run import harness_sha256
    if env.get("source_revision") != SOURCE_REVISION or env.get("harness_sha256") != harness_sha256():
        raise ValueError("Selected run used different scientific code; restore its harness before final evaluation")
    with sqlite3.connect((run / "search.db").as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM models WHERE status='ok' AND metric='rsa' AND stage='search' "
            "AND protocol_version=? AND dataset=? AND test_corr IS NOT NULL "
            "ORDER BY test_corr DESC, created_at, id LIMIT 1",
            (protocol.version, protocol.search_dataset),
        ).fetchone()
    if row is None or not math.isfinite(row["test_corr"]):
        raise ValueError("Run has no finite RSA-scored model under the current protocol")
    source = Path(row["path"]).resolve()
    if not source.is_relative_to(run / "models"):
        raise ValueError("Selected model is outside the run's immutable model archive")
    digest = sha256(source)
    if row["model_sha256"] != digest:
        raise ValueError("Selected model bytes changed after RSA scoring; score it again before final evaluation")
    return {"path": str(source), "sha256": digest, "model_id": row["id"],
            "search_score": row["test_corr"], "search_metric": "rsa", "run": str(run),
            "search_dataset": protocol.search_dataset, "environment": env,
            "selection_rule": "highest_search_rsa_then_created_at_then_id"}


def manual_model(path: str | Path, dataset: str) -> dict:
    """Explicitly identify a frozen manual model; never claim it was RSA-selected."""
    path = Path(path).resolve()
    dataset = validate_search_dataset(dataset)
    return {"path": str(path), "sha256": sha256(path), "model_id": None,
            "search_score": None, "search_metric": None, "run": None,
            "search_dataset": dataset, "environment": {}, "selection_rule": "explicit_manual_model"}


@contextlib.contextmanager
def final_environment(selection: dict):
    protocol = protocol_for(selection["search_dataset"])
    from search.setup_run import EVAL_ENV_KEYS
    previous = os.environ.copy()
    try:
        saved = selection["environment"]
        storage_keys = (("NSD_DATA_DIR", "NSD_DATA_FILE", "NSD_ROOT", "NSD_STIMULI_HDF5_PATH", "NSD_BETAS_DIR")
                        if protocol.final_dataset == "nsd" else ("TVSD_DATA_DIR", "TVSD_STIMULI_HDF5_PATH"))
        storage_overrides = {key: previous[key] for key in storage_keys if key in previous}
        if saved:
            for key in EVAL_ENV_KEYS:
                os.environ.pop(key, None)
            os.environ.update({k: str(saved[k]) for k in EVAL_ENV_KEYS if k in saved})
        os.environ.update(storage_overrides)
        os.environ.update(NEURO_DATASET=protocol.final_dataset, NEURO_STAGE="final",
                          NEURO_METRIC="encoding", NEURO_SUBJECT=",".join(protocol.final_subjects))
        os.environ.pop("LAB_RUN", None)
        os.environ.pop("LAB_PROPOSER", None)
        yield protocol
    finally:
        os.environ.clear()
        os.environ.update(previous)


def evaluate_frozen(selection: dict, frozen: Path) -> tuple[dict, dict]:
    """Load final data only after the caller has frozen the selected model."""
    import hub

    reason = hub._scan_source(str(frozen))
    if reason:
        raise ValueError(f"Model rejected: {reason}")
    with final_environment(selection) as protocol:
        from harness.datasets import get_dataset, with_subject
        from harness.reward import evaluate_subjects
        from harness.loaders.vision.looped import dataset_provenance
        from harness.isolation import IsolatedModel
        model = IsolatedModel(frozen)
        if hashlib.sha256(model.source).hexdigest() != selection["sha256"]:
            raise ValueError("Frozen model no longer matches the selected hash")
        dataset = get_dataset(protocol.final_dataset)
        result = evaluate_subjects(model, dataset, stage="final")
        if result.get("status") != "ok" or not math.isfinite(result.get("test_corr", float("nan"))):
            raise RuntimeError(f"Final encoding failed: {result.get('status')}")
        provenance = {subject: dataset_provenance(with_subject(dataset, subject))
                      for subject in protocol.final_subjects}
    return result, provenance


def run_final(selection: dict, output: str | Path | None = None) -> Path:
    protocol = protocol_for(selection["search_dataset"])
    output = Path(output) if output else (
        REPO / "results" / "final" / protocol.name / selection["sha256"][:16] / "evaluation.json")
    output = output.resolve()
    if output.name in {"model.py", "selection.json"}:
        raise ValueError("Final output must not replace the frozen model or selection manifest")
    if selection["run"] and output.is_relative_to(Path(selection["run"])):
        raise ValueError("Write final results outside the search run so agents cannot read OOD feedback")
    if output.exists():
        raise FileExistsError(f"Final artifact already exists: {output}; choose a new --output to rerun")
    source = Path(selection["path"])
    model_bytes = source.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != selection["sha256"]:
        raise ValueError("Model changed between selection and freezing")
    output.parent.mkdir(parents=True, exist_ok=True)
    frozen = output.parent / "model.py"
    if frozen.exists() and frozen.read_bytes() != model_bytes:
        raise FileExistsError(f"A different frozen model already exists at {frozen}")
    if not frozen.exists():
        with frozen.open("xb") as file:
            file.write(model_bytes)
    from search.setup_run import harness_sha256, runtime_versions
    manifest = {
        "schema_version": 1, "stage": "frozen_before_final",
        "model_sha256": selection["sha256"], "frozen_model": str(frozen),
        "selection": {k: v for k, v in selection.items() if k != "environment"},
        "search_environment": selection["environment"],
        "protocol": protocol.to_dict(), "source_revision": SOURCE_REVISION,
        "harness_sha256": harness_sha256(), "runtime_versions": runtime_versions(),
    }
    manifest_path = output.parent / "selection.json"
    payload = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(payload):
            raise FileExistsError(f"A different frozen selection already exists at {manifest_path}")
    else:
        with manifest_path.open("x") as file:
            file.write(payload)
    result, data = evaluate_frozen(selection, frozen)
    if sha256(frozen) != selection["sha256"]:
        raise ValueError("Frozen model changed during final evaluation")
    if harness_sha256() != manifest["harness_sha256"]:
        raise ValueError("Scientific code changed during final evaluation")
    document = {
        "schema_version": 1,
        "stage": "final",
        "metric": "encoding",
        "metric_detail": "ridge_5fold_per_unit_pearson_r",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol.to_dict(),
        "source_revision": SOURCE_REVISION,
        "harness_sha256": manifest["harness_sha256"],
        "runtime_versions": manifest["runtime_versions"],
        "selection_manifest": str(manifest_path),
        "selection": {k: v for k, v in selection.items() if k != "environment"},
        "search_environment": selection["environment"],
        "model_sha256": selection["sha256"],
        "frozen_model": str(frozen),
        "dataset": protocol.final_dataset,
        "encoding_score": result["test_corr"],
        "result": result,
        "data_provenance": data,
    }
    # Strict JSON rejects incomplete/nonfinite scores. Create once; never overwrite
    # a prior final evaluation or update the search database.
    payload = json.dumps(document, indent=2, allow_nan=False) + "\n"
    with output.open("x") as file:
        file.write(payload)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="RSA run from which to select the champion")
    source.add_argument("--model", help="explicit model file, without claiming a searched champion")
    parser.add_argument("--search-dataset", help="laion or triplen; required with --model")
    parser.add_argument("--output", help="final JSON path outside the search run")
    args = parser.parse_args(argv)
    if args.run:
        selection = select_model(args.run)
        if args.search_dataset and validate_search_dataset(args.search_dataset) != selection["search_dataset"]:
            parser.error("--search-dataset does not match --run")
    else:
        if not args.search_dataset:
            parser.error("--model requires --search-dataset")
        selection = manual_model(args.model, args.search_dataset)
    print(f"Frozen selection: {selection['sha256']} ({selection['search_dataset']})", flush=True)
    output = run_final(selection, args.output)
    print(f"Final encoding artifact: {output}")


if __name__ == "__main__":
    main()
