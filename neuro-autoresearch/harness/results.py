"""Fixed result persistence. The agent should NOT edit this file.

Appends one scored row per evaluation to the per-study CSV ``results/<study>.csv``
(append-only history), so the agent's hill-climb trajectory is preserved across a
run. Scoring itself lives in ``harness.reward``.
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone

import pandas as pd

CORR_PREFIX = "corr_"

CORE_COLUMNS = [
    "iteration", "timestamp", "study", "model_name", "idea", "n_params",
    "n_train", "n_test", "n_targets", "test_corr", "train_corr", "gap",
    "frac_above_0.1", "best_so_far", "seconds", "status",
]


def _csv_path(study, results_dir: str, reward_roi: str = "nsdgeneral",
              run_tag: str = "") -> str:
    # Per-region loops get their own file (e.g. NsdSubj01__PPA.csv) so each region's
    # hill-climb history and best_so_far stay separate; "nsdgeneral" keeps the bare
    # study name. A run_tag (NEURO_RUN) suffixes it further so each RUN is unique.
    stem = study if reward_roi in (None, "nsdgeneral") else f"{study}__{reward_roi}"
    if run_tag:
        stem = f"{stem}__{run_tag}"
    return os.path.join(results_dir, f"{stem}.csv")


def frozen_slices(study, results_dir: str = "results",
                  reward_roi: str = "nsdgeneral", run_tag: str = "") -> list[str] | None:
    """The ``corr_<name>`` slice names already frozen in the region's results CSV.

    Returns ``None`` when the file does not exist yet (nothing is frozen, so the
    caller's freshly declared slices apply). The ``corr_`` column convention lives
    here, in the module that owns the result schema.
    """
    csv = _csv_path(study, results_dir, reward_roi, run_tag)
    if not os.path.exists(csv):
        return None
    cols = pd.read_csv(csv, nrows=0).columns
    return [c[len(CORR_PREFIX):] for c in cols if c.startswith(CORR_PREFIX)]


def save_results(row: dict, results_dir: str = "results") -> None:
    """Append ``row`` to the per-study CSV ``results/<study>.csv`` (append-only).

    One row per evaluation -- existing rows are never overwritten, so the agent's
    hill-climb history is preserved. The harness fills the ledger fields it owns:
    1-based ``iteration`` (count of existing rows + 1), ``timestamp``, and running
    ``best_so_far`` (max ``test_corr`` in the file). The agent-declared slice columns
    (``corr_<name>``) are FROZEN by the FIRST row written: later rows fill only those
    columns, newly declared slices are ignored with a warning, and the file never
    gains or loses columns.
    """
    os.makedirs(results_dir, exist_ok=True)
    csv = _csv_path(row["study"], results_dir,
                    row.get("reward_roi", "nsdgeneral"), row.get("run_tag", ""))
    df = pd.read_csv(csv) if os.path.exists(csv) else None

    row = dict(row)
    row["iteration"] = (len(df) if df is not None else 0) + 1
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prev_best = df["test_corr"].max() if df is not None and len(df) else None
    row["best_so_far"] = (
        row["test_corr"] if prev_best is None else max(prev_best, row["test_corr"])
    )

    if df is not None:                      # freeze: the existing slice columns are authoritative
        frozen = [c for c in df.columns if c.startswith(CORR_PREFIX)]
        ignored = [c for c in row if c.startswith(CORR_PREFIX) and c not in frozen]
        if ignored:
            warnings.warn(f"{csv}: slice columns frozen as {frozen or '[]'}; ignoring {ignored}")
        for c in ignored:
            row.pop(c)
        columns = CORE_COLUMNS + frozen
    else:
        columns = CORE_COLUMNS + [c for c in row if c.startswith(CORR_PREFIX)]

    new = pd.DataFrame([row], columns=columns)
    out = pd.concat([df, new], ignore_index=True) if df is not None else new
    out.reindex(columns=columns).to_csv(csv, index=False)
