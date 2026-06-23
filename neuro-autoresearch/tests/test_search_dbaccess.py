"""Smoke tests for the DB-backed open-ended search queue."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "search"))

import claim  # noqa: E402
import db  # noqa: E402
import dbaccess  # noqa: E402


def test_enqueue_claim_update_round_trip(tmp_path):
    db_path = os.path.join(tmp_path, "results.db")
    model_path = os.path.join(tmp_path, "model.py")
    open(model_path, "w").write("MODEL = object()\n")

    db.init(db_path)
    dbaccess.enqueue(db_path, "abc12", 1, "smoke", "nsd", "ventral", model_path)

    row = claim.claim_one(db_path)
    assert row == {
        "id": "abc12",
        "path": os.path.abspath(model_path),
        "dataset": "nsd",
        "roi": "ventral",
        "iter": 1,
    }

    assert claim.update_result(
        db_path,
        "abc12",
        {"test_corr": "0.25", "train_corr": "0.5", "gap": "nan"},
        "ok",
    ) == "ok"

    scored = dbaccess.top_n(db_path, n=1)
    assert scored[0]["id"] == "abc12"
    assert scored[0]["test_corr"] == 0.25
    assert scored[0]["code"] == "MODEL = object()\n"
