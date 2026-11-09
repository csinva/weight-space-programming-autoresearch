#!/usr/bin/env python3
"""Freeze a search-selected model, then evaluate it on held-out data with encoding.

    python -m final_eval --run artifacts/search/laion__visual__encoding__2026-11-07
    python -m final_eval --model model.py --search-dataset nsd --region visual

LAION and NSD champions are scored on the same dataset's held-out subjects and
held-out shared images (the headline), plus the three other subject x image
cells as diagnostics. Triple-N champions are scored on TVSD. No encoding result
is written to the search database. Model selection finishes before any held-out
responses are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
HEADLINE = "heldout_subjects__heldout_images"  # LAION/NSD cells: "<subjects>_subjects__<images>_images"

from neuro_eval.benchmarks import SOURCE_REVISION, protocol_for, validate_search_dataset
from neuro_eval.environment import final_environment


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_model(run: str | Path) -> dict:
    """Read only the highest objective score from a compatible run, validating its bytes."""
    run = Path(run).resolve()
    env = json.loads((run / "env.json").read_text())
    dataset = validate_search_dataset(env.get("NEURO_DATASET", ""))
    protocol = protocol_for(dataset, region=env.get("NEURO_REGION", ""),
                            metric=env.get("NEURO_METRIC", ""))
    if (env.get("NEURO_STAGE") != "search"
            or env.get("protocol") != json.loads(json.dumps(protocol.to_dict()))
            or env.get("NEURO_SUBJECT") != ",".join(protocol.search_subjects)):
        raise ValueError("Run is incompatible with the current search protocol")
    from neuro_eval.model import evaluation_sha256
    if env.get("source_revision") != SOURCE_REVISION or env.get("evaluation_sha256") != evaluation_sha256():
        raise ValueError("Selected run used different scientific code; restore its evaluation code before final evaluation")
    with sqlite3.connect((run / "search.db").as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM models WHERE status='ok' AND metric=? AND region=? AND stage='search' "
            "AND protocol_version=? AND dataset=? AND test_corr IS NOT NULL "
            "ORDER BY test_corr DESC, created_at, id LIMIT 1",
            (protocol.search_metric, protocol.search_region, protocol.version, protocol.search_dataset),
        ).fetchone()
    if row is None or not math.isfinite(row["test_corr"]):
        raise ValueError("Run has no finite scored model under the current protocol")
    source = Path(row["path"]).resolve()
    if not source.is_relative_to(run / "models"):
        raise ValueError("Selected model is outside the run's immutable model archive")
    digest = sha256(source)
    if row["model_sha256"] != digest:
        raise ValueError("Selected model bytes changed after search scoring; score it again before final evaluation")
    return {"path": str(source), "sha256": digest, "model_id": row["id"],
            "search_score": row["test_corr"], "search_metric": protocol.search_metric, "search_region": protocol.search_region, "run": str(run),
            "search_dataset": protocol.search_dataset, "environment": env,
            "selection_rule": "highest_search_score_then_created_at_then_id"}


def manual_model(path: str | Path, dataset: str, region: str) -> dict:
    """Explicitly identify a frozen manual model; never claim it was search-selected."""
    path = Path(path).resolve()
    dataset = validate_search_dataset(dataset)
    protocol = protocol_for(dataset, region=region)
    return {"path": str(path), "sha256": sha256(path), "model_id": None,
            "search_score": None, "search_metric": None, "search_region": protocol.search_region, "run": None,
            "search_dataset": dataset, "environment": {}, "selection_rule": "explicit_manual_model"}


def evaluate_frozen(selection: dict, frozen: Path) -> tuple[dict, dict]:
    """Load final data only after the caller has frozen the selected model.

    Returns {cell: result} and {cell: {subject: provenance}}.
    """
    from neuro_eval.model import scan_source

    reason = scan_source(str(frozen))
    if reason:
        raise ValueError(f"Model rejected: {reason}")
    with final_environment(selection) as protocol:
        from neuro_eval.datasets import get_dataset, with_subject, dataset_provenance
        from neuro_eval.encoding import evaluate_subjects
        from neuro_eval.model import IsolatedModel
        model = IsolatedModel(frozen)
        if hashlib.sha256(model.source).hexdigest() != selection["sha256"]:
            raise ValueError("Frozen model no longer matches the selected hash")
        dataset = get_dataset(protocol.final_dataset, region=protocol.search_region,
                              search_metric=protocol.search_metric)
        halves = ("heldout", "search") if protocol.final_dataset == protocol.search_dataset else ("heldout",)
        results, provenance = {}, {}
        for cohort, images in itertools.product(halves, halves):
            cell = replace(dataset, cohort=cohort, images=images)
            name = f"{cohort}_subjects__{images}_images"
            results[name] = evaluate_subjects(model, cell, stage="final")
            if results[name].get("status") != "ok" or not math.isfinite(results[name].get("test_corr", float("nan"))):
                raise RuntimeError(f"Final encoding failed on {name}: {results[name].get('status')}")
            subjects = protocol.final_subjects if cohort == "heldout" else protocol.search_subjects
            provenance[name] = {s: dataset_provenance(with_subject(cell, s)) for s in subjects}
    return results, provenance


def run_final(selection: dict, output: str | Path | None = None) -> Path:
    protocol = protocol_for(selection["search_dataset"], region=selection["search_region"],
                            metric=selection["search_metric"] or "rsa")
    output = Path(output) if output else (
        REPO / "artifacts" / "final" / protocol.name / selection["sha256"] / "evaluation.json")
    output = output.resolve()
    if output.name in {"model.py", "selection.json"}:
        raise ValueError("Final output must not replace the frozen model or selection manifest")
    if output.is_relative_to((REPO / "artifacts" / "search").resolve()) or (
            selection["run"] and output.is_relative_to(Path(selection["run"]).resolve())):
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
    from neuro_eval.model import evaluation_sha256, runtime_versions
    evaluation_scope = {"final_neural_data_usage": "final_only", "headline": HEADLINE}
    manifest = {
        "schema_version": 1, "stage": "frozen_before_final",
        "model_sha256": selection["sha256"], "frozen_model": str(frozen),
        "selection": {k: v for k, v in selection.items() if k != "environment"},
        "search_environment": selection["environment"],
        "protocol": protocol.to_dict(), "source_revision": SOURCE_REVISION,
        "evaluation_scope": evaluation_scope,
        "evaluation_sha256": evaluation_sha256(), "runtime_versions": runtime_versions(),
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
    if evaluation_sha256() != manifest["evaluation_sha256"]:
        raise ValueError("Scientific code changed during final evaluation")
    document = {
        "schema_version": 1,
        "stage": "final",
        "metric": "encoding",
        "metric_detail": "ridge_5fold_per_unit_pearson_r",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol.to_dict(),
        "source_revision": SOURCE_REVISION,
        "evaluation_sha256": manifest["evaluation_sha256"],
        "runtime_versions": manifest["runtime_versions"],
        "selection_manifest": str(manifest_path),
        "selection": {k: v for k, v in selection.items() if k != "environment"},
        "search_environment": selection["environment"],
        "model_sha256": selection["sha256"],
        "frozen_model": str(frozen),
        "dataset": protocol.final_dataset,
        "encoding_score": result[HEADLINE]["test_corr"],
        "cell_scores": {name: cell["test_corr"] for name, cell in result.items()},
        "result": result,
        "data_provenance": data,
        "evaluation_scope": manifest["evaluation_scope"],
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
    source.add_argument("--run", help="search run from which to select the champion")
    source.add_argument("--model", help="explicit model file, without claiming a searched champion")
    parser.add_argument("--search-dataset", help="laion, nsd or triplen; required with --model")
    parser.add_argument("--region", choices=("visual", "early", "ventral"), help="required with --model; inferred from --run")
    parser.add_argument("--output", help="final JSON path outside the search run")
    args = parser.parse_args(argv)
    if args.run:
        selection = select_model(args.run)
        if args.search_dataset and validate_search_dataset(args.search_dataset) != selection["search_dataset"]:
            parser.error("--search-dataset does not match --run")
        if args.region and args.region != selection["search_region"]:
            parser.error("--region does not match --run")
    else:
        if not args.search_dataset or not args.region:
            parser.error("--model requires --search-dataset and --region")
        selection = manual_model(args.model, args.search_dataset, args.region)
    print(f"Frozen selection: {selection['sha256']} ({selection['search_dataset']})", flush=True)
    output = run_final(selection, args.output)
    print(f"Final encoding artifact: {output}")


if __name__ == "__main__":
    main()
