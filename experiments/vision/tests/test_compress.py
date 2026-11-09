"""Description validation: word budget and code-identifier leaks trigger one corrective retry."""
import pytest

import compress as run

CODE = "def _surface_geom_map(x):\n    return x\nMODEL = _surface_geom_map\n"


def test_describe_retries_then_accepts(monkeypatch):
    replies = iter([("uses _surface_geom_map on " + "word " * 10, 0.01), ("Hessian shape index of luminance.", 0.01)])
    prompts = []
    monkeypatch.setattr(run, "claude", lambda prompt: (prompts.append(prompt), next(replies))[1])
    text, cost = run.describe(CODE, words=8)
    assert text == "Hessian shape index of luminance." and cost == pytest.approx(0.02)
    assert "_surface_geom_map" in prompts[1].split("rejected because")[1] and "words (limit 8)" in prompts[1]


def test_describe_gives_up(monkeypatch):
    monkeypatch.setattr(run, "claude", lambda prompt: ("```python\nx\n```", 0.0))
    with pytest.raises(ValueError):
        run.describe(CODE, words=200)


@pytest.mark.parametrize("final", [False, True])
def test_compression_scores_models_directly(tmp_path, monkeypatch, final):
    import baselines
    from neuro_eval import comparison, model

    original = tmp_path / "model.py"
    original.write_text("MODEL = None\n")
    monkeypatch.setattr(run, "REPO", tmp_path)
    monkeypatch.setattr(run, "describe", lambda code, words: ("Gabor energy.", 0.0))
    def rebuild(description, path):
        path.write_text("MODEL = None\n")
        return True, 0.0, []
    monkeypatch.setattr(run, "rebuild", rebuild)
    monkeypatch.setattr(model, "load_model", lambda path: str(path))
    monkeypatch.setattr(baselines, "load", lambda names: {run.FLOOR: "reference"})
    calls = []
    def compare(models, **kwargs):
        calls.append((models, kwargs))
        return {name: {"test_corr": 0.1} for name in models}
    monkeypatch.setattr(comparison, "compare_models", compare)
    monkeypatch.setattr(run, "cka", lambda paths: {})
    reports = []
    monkeypatch.setattr(run, "report", lambda *args: reports.append(args))
    run.main(["--model", str(original), "--builders", "1"] + (["--final"] if final else []))
    assert [kwargs["final"] for _, kwargs in calls] == ([False, True] if final else [False])
    assert all(kwargs["trusted"] == (run.FLOOR,) for _, kwargs in calls)
    assert all(set(models) == {"original", "rebuild_200w_0", run.FLOOR} for models, _ in calls)
    assert len(reports[0][2]) == len(calls)
    assert len(list(reports[0][0].glob("laion__visual__encoding*.json"))) == len(calls)
