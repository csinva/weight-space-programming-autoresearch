#!/usr/bin/env python3
"""Export current-protocol RSA champions; final encoding never affects selection."""
import argparse
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.evaluation.final import select_model
from harness.protocols import PROTOCOLS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", help="candidate search run; defaults to runs/*")
    args = parser.parse_args(argv)
    runs = [Path(path) for path in args.run] if args.run else sorted((REPO / "runs").glob("*__*"))
    best = {}
    identities = {}
    for run in runs:
        try:
            selected = select_model(run)
        except (ValueError, OSError, KeyError) as exc:
            print(f"Skipping {run.name}: {exc}", file=sys.stderr)
            continue
        dataset = selected["search_dataset"]
        env = selected["environment"]
        identity = {key: value for key, value in env.items() if key != "providers"}
        if dataset in identities and identity != identities[dataset]:
            raise ValueError(f"{dataset} runs have different evaluation contracts; use --run to choose a comparable set")
        identities[dataset] = identity
        if dataset not in best or selected["search_score"] > best[dataset]["search_score"]:
            best[dataset] = selected
    for dataset, selected in best.items():
        destination = REPO / "runs/best" / dataset
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selected["path"], destination / "model.py")
        record = {"protocol": PROTOCOLS[dataset].to_dict(), **selected}
        (destination / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{dataset}: RSA {selected['search_score']:.6f} → {destination / 'model.py'}")
    if not best:
        print("No current-protocol RSA champions found. Historical exports are unchanged.")


if __name__ == "__main__":
    main()
