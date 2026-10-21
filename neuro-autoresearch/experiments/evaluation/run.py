#!/usr/bin/env python3
"""Canonical final evaluation CLI; see :mod:`experiments.evaluation.final`."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.evaluation.final import main

if __name__ == "__main__":
    main()
