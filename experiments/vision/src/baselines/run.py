"""Score baselines under the exact search protocol, or on its held-out final data.

    python -m baselines.run --dataset laion --region visual
    python -m baselines.run --dataset laion --region visual --final --only gabor,gist
    python -m baselines.run --final --only pixels --model champion=artifacts/final/.../model.py

Baselines are fixed a priori, so scoring them on final data selects nothing; results
go to artifacts/final/baselines/, never to artifacts/search/. Final NSD rows add named-area
diagnostics (each area's retained voxels within the pooled units), as in search.
"""
import argparse
import json
from pathlib import Path

from neuro_eval.comparison import table
from neuro_eval.model import load_model

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "final" / "baselines"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="laion", choices=("laion", "nsd", "triplen"))
    parser.add_argument("--region", default=None, choices=("visual", "early", "ventral"))
    parser.add_argument("--final", action="store_true", help="score held-out subjects and images (TVSD for triplen)")
    parser.add_argument("--only", help="comma-separated baseline names")
    parser.add_argument("--model", action="append", default=[], metavar="[NAME=]PATH",
                        help="also score a frozen model file (repeatable)")
    parser.add_argument("--out", type=Path, default=OUT, help="results directory")
    args = parser.parse_args(argv)

    validate_output(args.out)

    from baselines import load
    from neuro_eval.benchmarks import default_search_region, protocol_for
    from neuro_eval.comparison import compare_models

    region = args.region or default_search_region(args.dataset)
    models = load(args.only.split(",") if args.only else None) if args.only or not args.model else {}
    candidates_by_name = candidates(args.model)
    trusted = models.keys() - candidates_by_name.keys()
    models.update(candidates_by_name)
    rows = compare_models(models, dataset=args.dataset, region=region, final=args.final, trusted=trusted)
    dataset = protocol_for(args.dataset, region=region, metric="encoding").final_dataset if args.final else args.dataset
    suffix = "__final" if args.final else ""
    save(rows, args.out / f"{dataset}__{region}__encoding{suffix}.json")


def candidates(specs):
    """Frozen model files, imported in-process like the baselines; keyed by NAME (default: parent directory)."""
    return {name or Path(path).resolve().parent.name: load_model(path)
            for name, _, path in (spec.rpartition("=") for spec in specs)}


def validate_output(out):
    if Path(out).resolve().is_relative_to((REPO / "artifacts" / "search").resolve()):
        raise ValueError("Write comparison results outside search storage")


def save(rows, out):
    validate_output(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}
    results.update(rows)
    out.write_text(json.dumps(results, indent=2, default=str))
    out.with_suffix(".md").write_text(table(results))
    print(f"wrote {out} and {out.with_suffix('.md').name}")


if __name__ == "__main__":
    main()
