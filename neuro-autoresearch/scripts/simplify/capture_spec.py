"""Save the stimulus shape and range used by exact simplification's equivalence check."""
import json
import sys

import torch

from harness.datasets import DATASET
from harness.loaders import load_split


X, *_ = load_split(DATASET)
sample = torch.as_tensor(X)[:32]
spec = {
    "stim_shape": list(sample.shape[1:]),
    "dtype": str(sample.dtype).removeprefix("torch."),
    "low": sample.float().min().item(),
    "high": sample.float().max().item(),
    "batch": len(sample),
    "seed": 0,
}
with open(sys.argv[1], "w") as f:
    json.dump(spec, f)
