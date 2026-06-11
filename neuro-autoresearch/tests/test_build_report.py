"""Build the analyze HTML report from a synthetic fixture and assert its content.

Creates a couple of ``runs/<tag>/results/<study>.csv`` files (one with a
``metadata.json``, one with a non-ok row and a training-related idea that must be
excluded from the plots/writeup), runs the builder, and checks the produced HTML.

Run standalone (`python tests/test_build_report.py`) or via pytest.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analyze"))
import build_report  # noqa: E402

HEADER = (
    "iteration,timestamp,study,model_name,idea,summary,n_params,n_train,n_test,n_targets,"
    "test_corr,train_corr,gap,best_so_far,seconds,status"
)


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(HEADER + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


def _make_fixture(root):
    runs_dir = os.path.join(root, "runs")

    # Run A: nsd, improving reward, with metadata.
    a = os.path.join(runs_dir, "runA")
    _write_csv(os.path.join(a, "results", "nsd.csv"), [
        (1, "2026-01-01T00:00:00+00:00", "nsd", "identity", "raw pixels baseline", "returns pixels unchanged", 0, 800, 200, 50, 0.10, 0.20, 0.10, 0.10, 1.0, "ok"),
        (2, "2026-01-01T00:01:00+00:00", "nsd", "edges", "Sobel edge features", "Sobel gradient magnitude", 12, 800, 200, 50, 0.18, 0.25, 0.07, 0.18, 1.0, "ok"),
        (3, "2026-01-01T00:02:00+00:00", "nsd", "gabor", "Gabor filter bank", "multi-orientation Gabor energy", 30, 800, 200, 50, 0.31, 0.40, 0.09, 0.31, 1.0, "ok"),
    ])
    with open(os.path.join(a, "metadata.json"), "w") as fh:
        fh.write('{"dataset": "NSD subj1", "model": "gpt", "effort": "high", "note": "fixture"}')

    # Run B: text, includes a failed row and a training-related idea (both excluded).
    b = os.path.join(runs_dir, "runB")
    _write_csv(os.path.join(b, "results", "text.csv"), [
        (1, "2026-01-02T00:00:00+00:00", "text", "wordlen", "word length feature", "char count per token", 2, 500, 100, 30, 0.05, 0.09, 0.04, 0.05, 0.5, "ok"),
        (2, "2026-01-02T00:01:00+00:00", "text", "bigfit", "train a small MLP on the stimulus", "fits an MLP on stimulus", 99, 500, 100, 30, 0.40, 0.90, 0.50, 0.05, 0.5, "ok"),
        (3, "2026-01-02T00:02:00+00:00", "text", "boom", "spectral features", "FFT magnitude bins", 8, 500, 100, 30, 0.12, 0.15, 0.03, 0.12, 0.5, "error"),
    ])

    return runs_dir


def test_report_built_and_contains_expected_content():
    with tempfile.TemporaryDirectory() as root:
        runs_dir = _make_fixture(root)
        out = os.path.join(root, "report.html")

        runs = build_report.discover_runs(runs_dir, repo_root=root)
        assert len(runs) == 2, f"expected 2 run/study files, got {len(runs)}"

        result_path = build_report.build_report(runs_dir=runs_dir, out=out, repo_root=root)
        assert os.path.exists(result_path)
        html = open(result_path).read()
        assert len(html) > 2000

        # Structure: summary + per-run plots (offline Plotly), without the giant
        # append-only attempts ledger in the human report.
        assert html.count('Plotly.newPlot(') >= 3
        assert "cdn.plot.ly" not in html
        assert "plotly.js" in html.lower()
        assert 'id="attempts"' not in html
        assert "Attempts</h2>" not in html

        # Labels from metadata + study fallback.
        assert "NSD subj1" in html and "high" in html
        assert "runA" in html and "runB" in html

        # Ideas surface in the writeups / run details.
        assert "Gabor filter bank" in html
        assert "word length feature" in html

        # Failures are summarized by status without requiring the raw attempt table.
        assert "failed" in html
        assert "✕ 1/3" in html

        # Writeup mentions an improving idea.
        assert "best_so_far" in html


def test_exclusion_logic():
    assert build_report.mentions_training("train a small MLP")
    assert build_report.mentions_training("Training-based readout")
    assert not build_report.mentions_training("Gabor filter bank")
    assert build_report.is_ok("ok")
    assert not build_report.is_ok("error")


if __name__ == "__main__":
    test_report_built_and_contains_expected_content()
    test_exclusion_logic()
    print("OK: report built and validated")
