"""Save the stimulus shape and range used by exact simplification's equivalence check."""
import argparse
import json
from pathlib import Path

import torch

from harness.datasets import DATASET, Dataset
from harness.loaders import load_split


def capture_input_spec(dataset: Dataset = DATASET) -> dict:
    """Capture training inputs, or the search inputs when RSA has no train split."""
    X_train, X_test, *_ = load_split(dataset)
    X = X_train if len(X_train) else X_test
    if not len(X):
        raise ValueError(f"Cannot capture input specification for {dataset.study}: "
                         "both training and search/test stimulus splits are empty")
    sample = torch.as_tensor(X[:32])
    return {
        "stim_shape": list(sample.shape[1:]),
        "dtype": str(sample.dtype).removeprefix("torch."),
        "low": sample.float().min().item(),
        "high": sample.float().max().item(),
        "batch": len(sample),
        "seed": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    spec = capture_input_spec()
    args.output.write_text(json.dumps(spec))


if __name__ == "__main__":
    main()
