"""Turn autoresearch run outputs into a single self-contained HTML report.

Scans per-study append-only result CSVs (``runs/<tag>/results/<study>.csv`` and the
repo-root ``results/<study>.csv``), optionally labelled by ``runs/<tag>/metadata.json``,
and writes one white-background, interactive HTML file with:

  * top summary plot(s) overlaying every run's reward (``test_corr``) vs iteration,
    faceted by dataset (the ``study`` column);
  * per-run ``test_corr`` / ``train_corr`` vs iteration with a running-max line;
  * an auto-generated writeup of which ideas were tried and which improved
    ``best_so_far``.

Rows whose ``status`` is not ``ok`` or whose ``idea`` mentions training are flagged and
excluded from the plots and the writeup (the proposed model is never trained), while
failure and exclusion counts remain visible in the report.

Usage::

    python analyze/build_report.py [--runs-dir runs] [--repo-root .] [--out analyze/report.html]
"""
from __future__ import annotations

import argparse
import html
import json
import math
import numbers
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from itertools import count

import pandas as pd

# Canonical left-to-right ordering of NSD regions (early visual -> ventral category).
REGION_ORDER = ["V1", "V2", "V3", "hV4", "early", "ventral", "PPA", "EBA", "FFA", "VWFA"]
REGION_NAMES = {
    "V1": "primary visual cortex (V1)",
    "V2": "secondary visual cortex (V2)",
    "V3": "visual area V3",
    "hV4": "color/shape area hV4",
    "early": "early visual stream",
    "ventral": "ventral visual stream",
    "PPA": "place/scene area (PPA)",
    "EBA": "body-selective area (EBA)",
    "FFA": "face-selective area (FFA)",
    "VWFA": "visual word-form area (VWFA)",
}
# Frozen ImageNet AlexNet fc7, read out with the same ridge evaluator. These are
# documented in prompts/datasets.md and are shown only as references; no rerun occurs.
ALEXNET_BASELINES = {
    "V1": 0.342,
    "V2": 0.299,
    "V3": 0.321,
    "hV4": 0.289,
    "FFA": 0.396,
    "PPA": 0.403,
    "EBA": 0.436,
    "VWFA": 0.321,
    "ventral": 0.284,
    "overall": 0.324,
}
_TRAIN_RE = re.compile(r"\btrain", re.IGNORECASE)
_TAG_REGION_RE = re.compile(r"^nsd-([a-z0-9]+)-")
_PLOT_COUNTER = count(1)
_PLOTLY_BUNDLE = os.path.join(os.path.dirname(__file__), "vendor", "plotly-cartesian.min.js")
_PLOTLY_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "displayModeBar": False,
    "topojsonURL": "",
}
_PALETTE = [
    "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
    "#0891b2", "#be123c", "#4f46e5", "#65a30d", "#c2410c",
]


def mentions_training(idea: object) -> bool:
    """True when the idea text refers to training the model (which is never allowed)."""
    return isinstance(idea, str) and bool(_TRAIN_RE.search(idea))


def is_ok(status: object) -> bool:
    return isinstance(status, str) and status.strip().lower() == "ok"


def canonical_region(value: object) -> str | None:
    """Map loose region spellings (tag fragments, filenames) to canonical labels."""
    if not isinstance(value, str):
        return None
    low = value.strip().lower()
    aliases = {rg.lower(): rg for rg in REGION_ORDER}
    aliases.update({"v4": "hV4", "nsdgeneral": "overall"})
    return aliases.get(low)


def _valid_n_params(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals[vals.notna() & (vals > 0)]


def has_usable_n_params(df: pd.DataFrame) -> bool:
    """Only show model-size plots when n_params is substantially logged."""
    if "n_params" not in df.columns:
        return False
    vals = _valid_n_params(df["n_params"])
    if len(vals) < 2:
        return False
    return vals.nunique() > 1


def failure_count(df: pd.DataFrame) -> int:
    """Rows that actually failed evaluation: status is present and not ok."""
    if "status" not in df.columns:
        return 0
    return int((~df["status"].apply(is_ok)).sum())


def failure_rate(df: pd.DataFrame) -> float:
    return failure_count(df) / len(df) if len(df) else 0.0


def _clean_detail_echo(value: object) -> str:
    """Remove old HTML <details> extraction artifacts like 'truncated…full text'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if "…" not in text:
        return text
    head, tail = text.split("…", 1)
    head_words = re.sub(r"\s+", " ", head).strip()
    tail_words = re.sub(r"\s+", " ", tail).strip()
    if len(head_words) >= 24 and tail_words.startswith(head_words[:24]):
        return tail.strip()
    return text


def run_label_map(runs: list["Run"]) -> dict[str, str]:
    """Readable run labels, with simple numbering only when a label repeats."""
    bases = [r.display_label for r in runs]
    counts = Counter(bases)
    seen: defaultdict[str, int] = defaultdict(int)
    out = {}
    for run in runs:
        base = run.display_label
        if counts[base] > 1:
            seen[base] += 1
            out[run.csv_path] = f"{base} #{seen[base]}"
        else:
            out[run.csv_path] = base
    return out


def display_source_path(path: str) -> str:
    """Avoid leaking temporary reconstruction paths into the static report."""
    if not path:
        return "result ledger"
    norm = os.path.abspath(path)
    cwd = os.path.abspath(os.getcwd())
    if norm.startswith(cwd + os.sep):
        return os.path.relpath(norm, cwd)
    if norm.startswith(("/tmp/", "/private/tmp/", "/var/folders/")):
        return "reconstructed result ledger"
    return path


@dataclass
class Run:
    """One result CSV: a (tag, study) pair with its rows and optional metadata."""

    tag: str
    study: str
    csv_path: str
    df: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        meta = self.metadata
        bits = [meta.get("model"), meta.get("effort"), meta.get("note")]
        suffix = " · ".join(str(b) for b in bits if b)
        dataset = meta.get("dataset") or self.study
        head = f"{self.tag} — {dataset}"
        return f"{head} ({suffix})" if suffix else head

    @property
    def region(self) -> str | None:
        """Reward region inferred from metadata, filename, or tag."""
        for key in ("roi", "region", "reward_region"):
            val = self.metadata.get(key)
            if isinstance(val, str) and val:
                return canonical_region(val)
        stem = os.path.splitext(os.path.basename(self.csv_path))[0]
        for bit in (stem, self.tag):
            for rg in REGION_ORDER:
                if re.search(rf"(?:^|__|-){re.escape(rg)}(?:$|__|-)", bit, re.IGNORECASE):
                    return rg
            m = _TAG_REGION_RE.search(bit)
            if m:
                return canonical_region(m.group(1))
        return None

    @property
    def display_label(self) -> str:
        rg = self.region
        if rg:
            return f"{rg} run"
        return self.study if self.tag == "results" else self.tag

    @property
    def region_label(self) -> str:
        rg = self.region
        return REGION_NAMES.get(rg, rg or self.study)

    @cached_property
    def included(self) -> pd.DataFrame:
        """Rows kept for plots/writeup: status ok and the idea is not about training."""
        return self.df[~self.df["excluded"]]


def _read_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "iteration" not in df.columns:
        return None
    for col in ("iteration", "test_corr", "train_corr", "best_so_far", "seconds", "n_params", "gap"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "status" not in df.columns:
        df["status"] = "ok"
    if "idea" not in df.columns:
        df["idea"] = ""
    df["idea"] = df["idea"].fillna("").map(_clean_detail_echo)
    if "summary" in df.columns:
        df["summary"] = df["summary"].fillna("").map(_clean_detail_echo)
    df["excluded"] = df.apply(
        lambda r: (not is_ok(r["status"])) or mentions_training(r["idea"]), axis=1
    )
    return df.sort_values("iteration").reset_index(drop=True)


def _load_metadata(run_dir: str) -> dict:
    path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _csvs_in(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")
    )


def _make_run(tag: str, csv: str, metadata: dict) -> Run | None:
    """Build a Run from one result CSV, or None if it is unreadable/empty."""
    df = _read_csv(csv)
    if df is None:
        return None
    study = df["study"].dropna().iloc[0] if df["study"].notna().any() \
        else os.path.splitext(os.path.basename(csv))[0]
    return Run(tag, str(study), csv, df, metadata)


def discover_runs(runs_dir: str, repo_root: str, include_root_results: bool = True) -> list[Run]:
    """Find every result CSV under ``runs/<tag>/results/`` and ``<repo_root>/results/``.

    When ``include_root_results`` is False the stray repo-root ``results/`` ledger
    (the harness's current-model evals) is skipped, so the report shows only the
    per-run search under ``runs/``.
    """
    runs: list[Run] = []

    if os.path.isdir(runs_dir):
        for tag in sorted(os.listdir(runs_dir)):
            run_dir = os.path.join(runs_dir, tag)
            if not os.path.isdir(run_dir):
                continue
            metadata = _load_metadata(run_dir)
            csvs = _csvs_in(os.path.join(run_dir, "results")) or _csvs_in(run_dir)
            runs.extend(r for csv in csvs if (r := _make_run(tag, csv, metadata)))

    root_results = os.path.join(repo_root, "results")
    if include_root_results and os.path.abspath(root_results) != os.path.abspath(os.path.join(runs_dir, "results")):
        runs.extend(r for csv in _csvs_in(root_results) if (r := _make_run("results", csv, {})))

    return runs


def _load_plotly_bundle() -> str:
    if not os.path.exists(_PLOTLY_BUNDLE):
        raise FileNotFoundError(
            f"Plotly bundle not found at {_PLOTLY_BUNDLE}; vendor "
            "analyze/vendor/plotly-cartesian.min.js before building the report."
        )
    with open(_PLOTLY_BUNDLE, encoding="utf-8") as fh:
        bundle = fh.read()
    # The cartesian bundle carries Plotly's default topojson CDN URL for geo
    # traces. This report never emits geo traces, configures that URL to empty,
    # and must not contain any CDN endpoint in the self-contained HTML.
    bundle = re.sub(r"https://cdn\.plot\.ly/?", "", bundle)
    if "cdn.plot.ly" in bundle:
        raise RuntimeError("vendored Plotly bundle still references cdn.plot.ly")
    return bundle


def _plot_color(index: int) -> str:
    return _PALETTE[index % len(_PALETTE)]


def _sanitize_json(value):
    """Convert pandas/numpy scalars and non-finite numbers into Plotly-safe JSON."""
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(v) for v in value]
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        f = float(value)
        return f if math.isfinite(f) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json(value) -> str:
    return json.dumps(_sanitize_json(value), allow_nan=False, separators=(",", ":"))


def _merge_layout(base: dict, updates: dict | None) -> dict:
    out = dict(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_layout(out[key], value)
        else:
            out[key] = value
    return out


def _plotly_layout(layout: dict) -> dict:
    base = {
        "autosize": True,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {
            "size": 13,
            "family": "-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
        },
        "margin": {"l": 72, "r": 34, "t": 58, "b": 58},
        "hovermode": "closest",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        "xaxis": {
            "automargin": True,
            "gridcolor": "#e5e7eb",
            "zerolinecolor": "#d1d5db",
            "linecolor": "#d1d5db",
        },
        "yaxis": {
            "automargin": True,
            "gridcolor": "#e5e7eb",
            "zerolinecolor": "#d1d5db",
            "linecolor": "#d1d5db",
        },
    }
    return _merge_layout(base, layout)


def _plotly_div(data: list[dict], layout: dict, *, height: int | None = None,
                caption: str | None = None) -> str:
    plot_id = f"plot-{next(_PLOT_COUNTER)}"
    style = f' style="height:{int(height)}px"' if height else ""
    caption_html = f'<p class="chart-caption">{html.escape(caption)}</p>' if caption else ""
    return (
        '<div class="chart-card">\n'
        f'<div id="{plot_id}" class="chart-plot"{style}></div>\n'
        "<script>\n"
        f'Plotly.newPlot("{plot_id}",{_json(data)},{_json(_plotly_layout(layout))},{_json(_PLOTLY_CONFIG)});'
        "\n</script>\n"
        f"{caption_html}\n</div>\n"
    )


def summary_plot(runs: list[Run]) -> str | None:
    """Overlay every run's reward vs iteration, one facet per dataset (study)."""
    labels = run_label_map(runs)
    by_study: dict[str, list[Run]] = {}
    for run in runs:
        if not run.included.empty:
            by_study.setdefault(run.study, []).append(run)
    if not by_study:
        return None

    charts = []
    for study in sorted(by_study):
        data = []
        for idx, run in enumerate(by_study[study]):
            inc = run.included.dropna(subset=["test_corr"]).sort_values("iteration")
            if inc.empty:
                continue
            label = labels.get(run.csv_path, run.display_label)
            data.append({
                "type": "scatter",
                "mode": "lines+markers",
                "x": inc["iteration"].tolist(),
                "y": inc["test_corr"].tolist(),
                "name": label,
                "line": {"width": 1.8, "color": _plot_color(idx)},
                "marker": {"size": 6, "color": _plot_color(idx)},
                "hovertemplate": (
                    html.escape(label)
                    + "<br>Iteration %{x}<br>Held-out correlation %{y:.4f}<extra></extra>"
                ),
            })
        if not data:
            continue
        charts.append(_plotly_div(
            data,
            {
                "title": {"text": f"{study}: reward across proposed models"},
                "xaxis": {"title": {"text": "Iteration"}},
                "yaxis": {"title": {"text": "Held-out correlation (Pearson r)"}},
            },
            height=420,
            caption="Each trace is one run; use the legend to toggle runs.",
        ))
    return "".join(charts) if charts else None


def per_run_plot(run: Run, label: str | None = None) -> str | None:
    inc = run.included
    if inc.empty:
        return None
    inc = inc.sort_values("iteration")
    x = inc["iteration"].tolist()
    data = [{
        "type": "scatter",
        "mode": "lines+markers",
        "x": x,
        "y": inc["test_corr"].tolist(),
        "name": "test_corr",
        "line": {"width": 1.8, "color": "#2563eb"},
        "marker": {"size": 6, "color": "#2563eb"},
        "hovertemplate": "Iteration %{x}<br>test_corr %{y:.4f}<extra></extra>",
    }]
    if inc["train_corr"].notna().any():
        data.append({
            "type": "scatter",
            "mode": "lines+markers",
            "x": x,
            "y": inc["train_corr"].tolist(),
            "name": "train_corr",
            "line": {"width": 1.4, "color": "#ea580c"},
            "marker": {"size": 6, "symbol": "square", "color": "#ea580c"},
            "hovertemplate": "Iteration %{x}<br>train_corr %{y:.4f}<extra></extra>",
        })
    running_max = inc["test_corr"].cummax()
    data.append({
        "type": "scatter",
        "mode": "lines",
        "x": x,
        "y": running_max.tolist(),
        "name": "running max (test_corr)",
        "line": {"width": 2, "color": "#16a34a", "dash": "dash", "shape": "hv"},
        "hovertemplate": "Iteration %{x}<br>running max %{y:.4f}<extra></extra>",
    })
    layout = {
        "title": {"text": f"{label or run.display_label}: reward trajectory"},
        "xaxis": {"title": {"text": "Iteration (one proposed model per row)"}},
        "yaxis": {"title": {"text": "Correlation (Pearson r)"}},
    }
    if has_usable_n_params(inc):
        npar = pd.to_numeric(inc["n_params"], errors="coerce")
        npar = npar.where(npar > 0).tolist()
        data.append({
            "type": "scatter",
            "mode": "lines+markers",
            "x": x,
            "y": npar,
            "name": "n_params",
            "yaxis": "y2",
            "line": {"width": 1.4, "color": "#9333ea", "dash": "dot"},
            "marker": {"size": 5, "color": "#9333ea"},
            "hovertemplate": "Iteration %{x}<br>n_params %{y:,}<extra></extra>",
        })
        layout["yaxis2"] = {
            "title": {"text": "Logged parameters (log)"},
            "type": "log",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "automargin": True,
            "color": "#9333ea",
        }
        layout["margin"] = {"l": 70, "r": 82, "t": 58, "b": 58}
    return _plotly_div(data, layout, height=420)


def _truncate(text: str, n: int = 90) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def run_writeup(run: Run) -> str:
    """Prose summary: ideas tried and which raised ``best_so_far`` (included rows only).

    Each improvement also reports the gain (Δ), the cumulative wall-clock at that point,
    and the model's parameter/feature count, tying the search trace to its cost.
    """
    inc = run.included
    total = len(run.df)
    excluded = int(run.df["excluded"].sum())
    if inc.empty:
        return (
            f"<p>No usable attempts (all {total} row(s) excluded: non-ok status or "
            f"training-related idea).</p>"
        )

    has_sec = "seconds" in inc.columns and inc["seconds"].notna().any()
    cum_h = (inc["seconds"].fillna(0).cumsum() / 3600.0) if has_sec else None

    improvements = []
    best = float("-inf")
    for pos, (_, row) in enumerate(inc.iterrows()):
        tc = row["test_corr"]
        if pd.notna(tc) and tc > best:
            delta = (float(tc) - best) if best != float("-inf") else None
            best = float(tc)
            npar = row.get("n_params")
            npar_i = None
            try:
                npar_i = int(float(npar)) if pd.notna(npar) and float(npar) > 0 else None
            except (TypeError, ValueError):
                npar_i = None
            improvements.append((
                int(row["iteration"]), str(row["model_name"]), str(row["idea"]), best, delta,
                (float(cum_h.iloc[pos]) if has_sec else None),
                npar_i,
            ))

    total_h = float(inc["seconds"].fillna(0).sum() / 3600.0) if has_sec else None
    parts = [
        f"<p><b>{len(inc)}</b> scored attempt(s)"
        + (f", <b>{excluded}</b> excluded (non-ok status or training-related idea)" if excluded else "")
        + f". Best <code>test_corr</code> = <b>{best:.4f}</b>"
        + (f" reached in <b>{total_h:.1f}h</b> of wall-clock" if total_h else "")
        + ".</p>"
    ]
    if improvements:
        items = []
        for it, name, idea, val, delta, h, npar in improvements:
            meta = []
            if delta is not None:
                meta.append(f"+{delta:.4f}")
            if h is not None:
                meta.append(f"@{h:.1f}h")
            if npar is not None:
                meta.append(f"{npar:,}p")
            tail = f" <span class='meta'>({', '.join(meta)})</span>" if meta else ""
            items.append(
                f"<li>iter <b>{it}</b> — <i>{html.escape(name)}</i>: "
                f"{html.escape(_truncate(idea))} → <code>test_corr={val:.4f}</code>{tail}</li>"
            )
        parts.append(f"<p>Ideas that improved <code>best_so_far</code>:</p><ul>{''.join(items)}</ul>")
    else:
        parts.append("<p>No attempt improved on the first scored idea.</p>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Mechanistic analysis: what the best model in each run actually computes.
# --------------------------------------------------------------------------- #

def best_row(run: Run) -> pd.Series | None:
    """The included attempt with the highest ``test_corr`` (the model that was found)."""
    inc = run.included
    if inc.empty or inc["test_corr"].isna().all():
        return None
    return inc.loc[inc["test_corr"].idxmax()]


def identity_row(run: Run) -> pd.Series | None:
    """First included identity/raw-pixel baseline row for a run, when logged."""
    inc = run.included
    if inc.empty or "model_name" not in inc.columns:
        return None
    mask = inc["model_name"].astype(str).str.lower().eq("identity")
    if not mask.any():
        return None
    return inc.loc[mask].iloc[0]


def baseline_comparison(runs: list[Run]) -> str:
    """Compare each region winner with logged identity and documented AlexNet baselines."""
    by_region = {}
    for run in runs:
        rg = run.region
        br = best_row(run)
        if br is None or not rg:
            continue
        ident = identity_row(run)
        identity = float(ident["test_corr"]) if ident is not None and pd.notna(ident.get("test_corr")) else None
        alex = ALEXNET_BASELINES.get(rg)
        best = float(br["test_corr"]) if pd.notna(br.get("test_corr")) else None
        if best is None:
            continue
        candidate = (rg, run.display_label, run.region_label, best, identity, alex)
        if rg not in by_region or best > by_region[rg][3]:
            by_region[rg] = candidate
    rows = list(by_region.values())
    if not rows:
        return "<p class='meta'>No logged identity rows or region-tagged NSD runs found for baseline comparison.</p>"

    order = {rg: i for i, rg in enumerate(REGION_ORDER)}
    rows.sort(key=lambda r: order.get(r[0], 999))
    names = [r[2] for r in rows]
    best_vals = [r[3] for r in rows]
    identity_vals = [None if r[4] is None else r[4] for r in rows]
    alex_vals = [None if r[5] is None else r[5] for r in rows]
    best_colors = [
        "#16a34a" if alex is not None and best >= alex else "#2563eb"
        for (_rg, _label, _name, best, _ident, alex) in rows
    ]
    best_delta = [
        "" if alex is None else f"{best - alex:+.3f}"
        for (_rg, _label, _name, best, _ident, alex) in rows
    ]
    data = [
        {
            "type": "bar",
            "orientation": "h",
            "name": "identity / raw pixels",
            "y": names,
            "x": identity_vals,
            "marker": {"color": "#9ca3af"},
            "text": ["" if v is None else f"{v:.3f}" for v in identity_vals],
            "textposition": "outside",
            "cliponaxis": False,
            "hovertemplate": "identity / raw pixels<br>%{y}<br>%{x:.3f}<extra></extra>",
        },
        {
            "type": "bar",
            "orientation": "h",
            "name": "best untrained search model",
            "y": names,
            "x": best_vals,
            "customdata": best_delta,
            "marker": {"color": best_colors},
            "text": [f"{v:.3f}" for v in best_vals],
            "textposition": "outside",
            "cliponaxis": False,
            "hovertemplate": (
                "best untrained search model<br>%{y}<br>"
                "%{x:.3f} (%{customdata} vs AlexNet)<extra></extra>"
            ),
        },
        {
            "type": "bar",
            "orientation": "h",
            "name": "trained AlexNet fc7",
            "y": names,
            "x": alex_vals,
            "marker": {"color": "#dc8a00"},
            "text": ["" if v is None else f"{v:.3f}" for v in alex_vals],
            "textposition": "outside",
            "cliponaxis": False,
            "hovertemplate": "trained AlexNet fc7<br>%{y}<br>%{x:.3f}<extra></extra>",
        },
    ]
    chart = _plotly_div(
        data,
        {
            "title": {"text": "Baselines by reward region"},
            "barmode": "group",
            "xaxis": {"title": {"text": "Held-out correlation (mean Pearson r; higher is better)"}},
            "yaxis": {
                "title": {"text": "Reward region"},
                "categoryorder": "array",
                "categoryarray": names,
            },
            "margin": {"l": 210, "r": 48, "t": 58, "b": 58},
        },
        height=max(380, 62 * len(rows) + 150),
        caption="Green untrained bars meet or beat the trained AlexNet reference for that region.",
    )

    body = []
    for rg, _label, name, best, ident, alex in rows:
        delta_identity = "" if ident is None else f"{best - ident:+.4f}"
        delta_alex = "" if alex is None else f"{best - alex:+.4f}"
        body.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td class='num'>{'' if ident is None else f'{ident:.4f}'}</td>"
            f"<td class='num'>{best:.4f}</td>"
            f"<td class='num'>{'' if alex is None else f'{alex:.4f}'}</td>"
            f"<td class='num'>{delta_identity}</td>"
            f"<td class='num {'' if alex is None else ('good' if best >= alex else 'bad')}'>{delta_alex}</td>"
            "</tr>"
        )
    table = (
        "<table class='baseline'><thead><tr><th>reward region</th><th>identity</th>"
        "<th>best untrained</th><th>trained AlexNet</th><th>Δ vs identity</th>"
        "<th>Δ vs AlexNet</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>"
    )
    return (
        "<p class='meta'>Identity is shown only when that row is already present in the log. "
        "AlexNet is the documented frozen ImageNet fc7 baseline; this report does not rerun it.</p>"
        + chart
        + table
    )


def _run_dir(run: Run) -> str:
    """The ``runs/<tag>`` directory holding a run's model source, from its CSV path."""
    d = os.path.dirname(run.csv_path)
    return os.path.dirname(d) if os.path.basename(d) == "results" else d


def find_model_source(run: Run, model_name: object) -> tuple[str | None, str | None]:
    """Locate the ``.py`` whose ``Model.name`` equals ``model_name`` in the run dir.

    Searches ``model.py`` then ``models_lib/*.py``; returns ``(path, source)`` or
    ``(None, None)`` when no file matches (e.g. test fixtures with CSVs only).
    """
    if not isinstance(model_name, str) or not model_name:
        return None, None
    run_dir = _run_dir(run)
    candidates = []
    top = os.path.join(run_dir, "model.py")
    if os.path.isfile(top):
        candidates.append(top)
    lib = os.path.join(run_dir, "models_lib")
    if os.path.isdir(lib):
        candidates += [os.path.join(lib, f) for f in sorted(os.listdir(lib)) if f.endswith(".py")]
    pat = re.compile(r"""name\s*=\s*["']""" + re.escape(model_name) + r"""["']""")
    for path in candidates:
        try:
            src = open(path).read()
        except Exception:
            continue
        if pat.search(src):
            return path, src
    return None, None


def cross_region_heatmap(runs: list[Run]) -> tuple[str | None, list[str], list[str]]:
    """Heatmap of each run's best model scored against every region (``corr_<region>``).

    Returns ``(plot_html_or_None, row_labels, col_regions)``. ``None`` when no run
    exposes ``corr_<region>`` columns (so test fixtures simply skip the section).
    """
    rows, labels = [], []
    label_map = run_label_map(runs)
    present: list[str] = []
    for run in runs:
        br = best_row(run)
        if br is None:
            continue
        cols = [rg for rg in REGION_ORDER if f"corr_{rg}" in run.df.columns and pd.notna(br.get(f"corr_{rg}"))]
        if not cols:
            continue
        for rg in cols:
            if rg not in present:
                present.append(rg)
    present = [rg for rg in REGION_ORDER if rg in present]
    if not present:
        return None, [], []

    for run in runs:
        br = best_row(run)
        if br is None:
            continue
        vals = [br.get(f"corr_{rg}") for rg in present]
        if all(pd.isna(v) for v in vals):
            continue
        labels.append(label_map.get(run.csv_path, run.display_label))
        rows.append([float(v) if pd.notna(v) else None for v in vals])

    if not rows:
        return None, [], []

    text = [["" if v is None else f"{v:.2f}" for v in row] for row in rows]
    chart = _plotly_div(
        [{
            "type": "heatmap",
            "x": present,
            "y": labels,
            "z": rows,
            "text": text,
            "texttemplate": "%{text}",
            "colorscale": "Viridis",
            "colorbar": {"title": {"text": "held-out corr"}},
            "hovertemplate": "%{y} × %{x} → corr %{z:.4f}<extra></extra>",
        }],
        {
            "title": {"text": "Cross-region predictivity of each run's best model"},
            "xaxis": {"title": {"text": "Brain region scored"}},
            "yaxis": {"title": {"text": "Run (optimized region)"}},
            "margin": {"l": 170, "r": 80, "t": 58, "b": 70},
        },
        height=max(360, 42 * len(rows) + 160),
        caption="Rows are run winners; columns are held-out predictivity in each region.",
    )
    return chart, labels, present


def _code_block(source: str | None, path: str | None) -> str:
    if not source:
        return ""
    rel = html.escape(path or "")
    return (
        f'<details class="code"><summary>Show model source <span class="meta">({rel})</span></summary>'
        f"<pre><code>{html.escape(source)}</code></pre></details>"
    )


def _feature_note(n_params: object) -> str:
    """Human note on a model's parameter / feature-dimension count."""
    try:
        n = int(float(n_params))
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M fixed/expansion values (large random-feature lift, never trained)"
    return f"{n:,} fixed values (never trained)"


def mechanics_section(runs: list[Run]) -> str:
    """One card per run for the best model it found: idea, summary, reward, source."""
    cards = []
    ranked = []
    labels = run_label_map(runs)
    for run in runs:
        br = best_row(run)
        if br is None:
            continue
        ranked.append((float(br["test_corr"]) if pd.notna(br["test_corr"]) else float("-inf"), run, br))
    ranked.sort(key=lambda t: t[0], reverse=True)

    for score, run, br in ranked:
        name = str(br.get("model_name", "") or "")
        path, src = find_model_source(run, name)
        idea = str(br.get("idea", "") or "")
        summary = str(br.get("summary", "") or "")
        note = _feature_note(br.get("n_params"))
        target = run.region_label
        rg = run.region
        # Compact cross-region line for this model.
        xreg = []
        for rg in REGION_ORDER:
            c = f"corr_{rg}"
            if c in run.df.columns and pd.notna(br.get(c)):
                xreg.append(f"{rg}={float(br[c]):.2f}")
        xreg_html = (
            f"<p class='meta'>across regions: {html.escape(' · '.join(xreg))}</p>" if xreg else ""
        )
        identity = identity_row(run)
        identity_html = ""
        if identity is not None and pd.notna(identity.get("test_corr")):
            identity_html = f" · identity {float(identity['test_corr']):.4f}"
        alex = ALEXNET_BASELINES.get(run.region or "")
        alex_html = f" · AlexNet {alex:.4f}" if alex is not None else ""
        source_html = _code_block(src, path)
        cards.append(
            f'<div class="card">'
            f"<h3>{html.escape(labels.get(run.csv_path, run.display_label))} — best model for {html.escape(target)} "
            f'<span class="reward">test_corr {score:.4f}</span></h3>'
            f"<p class='meta'><code>{html.escape(name)}</code>"
            + (f" · {html.escape(note)}" if note else "")
            + identity_html + alex_html
            + "</p>"
            + (f"<p><b>Idea.</b> {html.escape(idea)}</p>" if idea else "")
            + (f"<p><b>Computation.</b> {html.escape(summary)}</p>" if summary else "")
            + xreg_html
            + source_html
            + "</div>"
        )
    if not cards:
        return "<p class='meta'>No scored models to analyze.</p>"
    return "".join(cards)


# --------------------------------------------------------------------------- #
# Search process & computational-principle attribution.
# --------------------------------------------------------------------------- #

_PRINCIPLES = {
    "Gabor / oriented energy": r"gabor|orient|edge|complex.?cell|simple.?cell|\benergy\b|quadrature",
    "Divisive normalization": r"divnorm|divis|normali|gain.?control|naka|rushton|surround|whiten",
    "Colour opponency": r"colou?r|opponent|chroma|r-g|b-y|\brgb\b|\bhue\b|\blab\b",
    "Multiscale / spatial frequency": r"multiscale|multi-scale|pyramid|\bscale|wavelength|coarse|\bfine|spatial.?freq|\bsf\b|nyquist|\bband",
    "Scattering / second order": r"scatter|2nd.?order|second.?order|second.?layer|two.?stage|\bhier|cascade|layer.?2",
    "Texture statistics (Gram/GIST/PS)": r"texture|\bgram\b|gist|portilla|simoncelli|co-?occur|moment|covar",
    "Random features (RFF / random conv)": r"random|\brff\b|fourier|\bjl\b|projection|codebook|conjunction|arc-?cos",
    "Pooling / retinotopy": r"\bpool|fovea|retino|spatial.?layout|\bgrid\b|histogram|position",
}


def _improvement_events(run: Run) -> list[tuple[str, float, float | None]]:
    """``(idea, new_best, delta)`` for every attempt that raised ``best_so_far``."""
    events = []
    best = float("-inf")
    for _, row in run.included.iterrows():
        tc = row["test_corr"]
        if pd.notna(tc) and tc > best:
            delta = (float(tc) - best) if best != float("-inf") else None
            best = float(tc)
            events.append((str(row.get("idea", "") or ""), best, delta))
    return events


def principle_attribution(runs: list[Run]) -> tuple[str | None, str]:
    """Sum, across runs, the ``best_so_far`` gains whose idea text names each principle.

    Attribution is overlapping (one idea can invoke several principles); the chart shows
    which computational motifs are associated with the most reward gain over the search.
    """
    totals = {k: 0.0 for k in _PRINCIPLES}
    counts = {k: 0 for k in _PRINCIPLES}
    for run in runs:
        for idea, _val, delta in _improvement_events(run):
            if not delta or delta <= 0:
                continue
            low = idea.lower()
            for name, pat in _PRINCIPLES.items():
                if re.search(pat, low):
                    totals[name] += delta
                    counts[name] += 1
    items = [(k, v) for k, v in sorted(totals.items(), key=lambda kv: kv[1], reverse=True) if v > 0]
    if not items:
        return None, ""

    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    chart = _plotly_div(
        [{
            "type": "bar",
            "orientation": "h",
            "y": names,
            "x": vals,
            "customdata": [counts[name] for name in names],
            "marker": {"color": "#2563eb"},
            "text": [f"{v:.3f}" for v in vals],
            "textposition": "outside",
            "cliponaxis": False,
            "hovertemplate": (
                "%{y}<br>Σ Δ test_corr %{x:.4f}<br>"
                "%{customdata} improving step(s)<extra></extra>"
            ),
        }],
        {
            "title": {"text": "Which computational principles moved the reward (all runs)"},
            "xaxis": {
                "title": {
                    "text": "Sum of held-out-correlation gain from improving steps"
                }
            },
            "yaxis": {"title": {"text": "Computational principle"}},
            "showlegend": False,
            "margin": {"l": 230, "r": 45, "t": 58, "b": 58},
        },
        height=max(340, 46 * len(items) + 150),
        caption="Bars sum overlapping attributions from ideas that improved best_so_far.",
    )

    body = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{v:.4f}</td>"
        f"<td class='num'>{counts[k]}</td></tr>"
        for k, v in items
    )
    table = (
        "<table class='principles'><thead><tr><th>principle</th>"
        "<th>Σ Δ test_corr</th><th>#&nbsp;steps</th></tr></thead><tbody>"
        + body + "</tbody></table>"
    )
    return chart, table


def search_dashboard(runs: list[Run]) -> str | None:
    """Four-panel view of the search, adapting to whichever columns were logged."""
    labels = run_label_map(runs)

    series = []
    for run in runs:
        inc = run.included.dropna(subset=["test_corr"]).sort_values("iteration")
        if not run.df.empty:
            series.append((run, inc))
    if not series:
        return None

    colors = {run.csv_path: _plot_color(i) for i, (run, _) in enumerate(series)}

    any_sec = any(
        "seconds" in inc.columns and inc["seconds"].notna().any()
        for _, inc in series
    )
    progress_data = []
    for run, inc in series:
        if inc.empty:
            continue
        if "seconds" in inc.columns and inc["seconds"].notna().any():
            x = inc["seconds"].fillna(0).cumsum() / 3600.0
        else:
            x = inc["iteration"]
        label = labels.get(run.csv_path, run.display_label)
        progress_data.append({
            "type": "scatter",
            "mode": "lines+markers",
            "x": x.tolist(),
            "y": inc["test_corr"].cummax().tolist(),
            "name": label,
            "line": {"width": 1.8, "color": colors[run.csv_path]},
            "marker": {"size": 5, "color": colors[run.csv_path]},
            "hovertemplate": (
                html.escape(label)
                + ("<br>Cumulative hours %{x:.3f}" if any_sec else "<br>Iteration %{x}")
                + "<br>best_so_far %{y:.4f}<extra></extra>"
            ),
        })
    progress_chart = _plotly_div(
        progress_data,
        {
            "title": {"text": "Search progress: best reward so far"},
            "xaxis": {"title": {"text": "Cumulative wall-clock (hours)" if any_sec else "Iteration"}},
            "yaxis": {"title": {"text": "Best held-out correlation so far"}},
        },
        height=380,
    )

    any_n_params = any(not inc.empty and has_usable_n_params(inc) for _, inc in series)
    if any_n_params:
        size_data = []
        for run, inc in series:
            if inc.empty or not has_usable_n_params(inc):
                continue
            label = labels.get(run.csv_path, run.display_label)
            npar = pd.to_numeric(inc["n_params"], errors="coerce").where(lambda s: s > 0)
            color = colors[run.csv_path]
            size_data.append({
                "type": "scatter",
                "mode": "markers",
                "x": npar.tolist(),
                "y": inc["test_corr"].tolist(),
                "name": label,
                "marker": {"size": 7, "color": color, "opacity": 0.62},
                "hovertemplate": (
                    html.escape(label)
                    + "<br>Logged parameters %{x:,}<br>Held-out correlation %{y:.4f}<extra></extra>"
                ),
            })
            bi = inc["test_corr"].idxmax()
            best_n = pd.to_numeric(pd.Series([inc.loc[bi, "n_params"]]), errors="coerce").iloc[0]
            if pd.notna(best_n) and best_n > 0:
                size_data.append({
                    "type": "scatter",
                    "mode": "markers",
                    "x": [float(best_n)],
                    "y": [float(inc.loc[bi, "test_corr"])],
                    "name": f"{label} best",
                    "showlegend": False,
                    "marker": {
                        "symbol": "star",
                        "size": 16,
                        "color": color,
                        "line": {"color": "#111827", "width": 1},
                    },
                    "hovertemplate": (
                        html.escape(label)
                        + " best<br>Logged parameters %{x:,}<br>"
                        "Held-out correlation %{y:.4f}<extra></extra>"
                    ),
                })
        size_chart = _plotly_div(
            size_data,
            {
                "title": {"text": "Does logged model size explain reward? (star = run best)"},
                "xaxis": {"title": {"text": "Logged parameters (log)"}, "type": "log"},
                "yaxis": {"title": {"text": "Held-out correlation (Pearson r)"}},
            },
            height=380,
        )
    else:
        order = sorted(range(len(series)), key=lambda i: failure_rate(series[i][0].df))
        rates = [100.0 * failure_rate(series[i][0].df) for i in order]
        y_labels = [labels.get(series[i][0].csv_path, series[i][0].display_label) for i in order]
        size_chart = _plotly_div(
            [{
                "type": "bar",
                "orientation": "h",
                "x": rates,
                "y": y_labels,
                "marker": {"color": [colors[series[i][0].csv_path] for i in order]},
                "text": [
                    f"{failure_count(series[i][0].df)}/{len(series[i][0].df)} ({rate:.0f}%)"
                    for i, rate in zip(order, rates)
                ],
                "textposition": "outside",
                "cliponaxis": False,
                "hovertemplate": "%{y}<br>Failed rows %{text}<extra></extra>",
            }],
            {
                "title": {"text": "Failure rate by run"},
                "xaxis": {"title": {"text": "Failed rows (% of total iterations)"}},
                "yaxis": {"title": {"text": "Run"}},
                "showlegend": False,
                "margin": {"l": 150, "r": 45, "t": 58, "b": 58},
            },
            height=380,
        )

    gap_data = []
    for run, inc in series:
        if not inc.empty and inc["train_corr"].notna().any():
            label = labels.get(run.csv_path, run.display_label)
            gap_data.append({
                "type": "scatter",
                "mode": "lines",
                "x": inc["iteration"].tolist(),
                "y": (inc["train_corr"] - inc["test_corr"]).tolist(),
                "name": label,
                "line": {"width": 1.6, "color": colors[run.csv_path]},
                "hovertemplate": (
                    html.escape(label)
                    + "<br>Iteration %{x}<br>train_corr - test_corr %{y:.4f}<extra></extra>"
                ),
            })
    if gap_data:
        gap_chart = _plotly_div(
            gap_data,
            {
                "title": {"text": "Readout generalization gap over the search"},
                "xaxis": {"title": {"text": "Iteration"}},
                "yaxis": {"title": {"text": "train_corr - test_corr"}},
                "shapes": [{
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": 0,
                    "y1": 0,
                    "line": {"color": "#6b7280", "width": 1, "dash": "dot"},
                }],
            },
            height=380,
        )
    else:
        gap_chart = (
            '<div class="chart-card note-card">'
            '<p class="meta">No train_corr values available</p>'
            "</div>"
        )

    order = sorted(range(len(series)), key=lambda i: (
        series[i][1]["seconds"].fillna(0).sum() / 3600.0
        if any_sec and "seconds" in series[i][1].columns else len(series[i][0].df)
    ))
    y_labels = [labels.get(series[i][0].csv_path, series[i][0].display_label) for i in order]
    if any_sec:
        hours = [
            float(series[i][1]["seconds"].fillna(0).sum() / 3600.0)
            if "seconds" in series[i][1].columns else 0.0
            for i in order
        ]
        compute_chart = _plotly_div(
            [{
                "type": "bar",
                "orientation": "h",
                "x": hours,
                "y": y_labels,
                "customdata": [len(series[i][0].df) for i in order],
                "marker": {"color": [colors[series[i][0].csv_path] for i in order]},
                "text": [f"{h:.1f}h / {len(series[i][0].df)} total it" for h, i in zip(hours, order)],
                "textposition": "outside",
                "cliponaxis": False,
                "hovertemplate": "%{y}<br>Total wall-clock %{x:.3f}h<br>%{customdata} total iterations<extra></extra>",
            }],
            {
                "title": {"text": "Compute spent per run"},
                "xaxis": {"title": {"text": "Total wall-clock (hours)"}},
                "yaxis": {"title": {"text": "Run"}},
                "showlegend": False,
                "margin": {"l": 150, "r": 45, "t": 58, "b": 58},
            },
            height=380,
        )
    else:
        scored = [len(series[i][1]) for i in order]
        failed = [failure_count(series[i][0].df) for i in order]
        compute_chart = _plotly_div(
            [
                {
                    "type": "bar",
                    "orientation": "h",
                    "name": "scored ok",
                    "x": scored,
                    "y": y_labels,
                    "marker": {"color": "#2563eb"},
                    "hovertemplate": "%{y}<br>Scored rows %{x}<extra></extra>",
                },
                {
                    "type": "bar",
                    "orientation": "h",
                    "name": "failed",
                    "x": failed,
                    "y": y_labels,
                    "marker": {"color": "#dc2626"},
                    "hovertemplate": "%{y}<br>Failed rows %{x}<extra></extra>",
                },
            ],
            {
                "title": {"text": "Scored vs failed iterations"},
                "barmode": "stack",
                "xaxis": {"title": {"text": "Iterations"}},
                "yaxis": {"title": {"text": "Run"}},
                "margin": {"l": 150, "r": 45, "t": 58, "b": 58},
            },
            height=380,
        )

    return (
        '<div class="dashboard-grid">'
        f"{progress_chart}{size_chart}{gap_chart}{compute_chart}"
        "</div>"
    )


_PRINCIPLES_INTRO = """
<p>Each run proposes fixed feature extractors, then the ridge readout asks:
<i>which hand-written computation makes brain activity easier to predict?</i> The bars
below sum the reward gains from improving steps that mention each principle. A single
step can count for multiple principles because real models combine ingredients.</p>
<ul>
  <li><b>Gabor / oriented energy</b> — quadrature filter pairs collapsed to
      <code>√(cos²+sin²)</code> give phase-invariant edge/contrast detectors (V1 complex cells).</li>
  <li><b>Divisive normalization</b> — dividing each unit by a pooled neighbourhood is the
      canonical cortical gain-control that decorrelates and equalises contrast (Heeger).</li>
  <li><b>Colour opponency</b> — luminance / R−G / B−Y channels mirror retinal–LGN coding.</li>
  <li><b>Multiscale / spatial frequency</b> — a bank of wavelengths covers coarse layout
      through fine detail; different regions weight the bands differently.</li>
  <li><b>Scattering / second order</b> — a Gabor layer applied to the first layer's energy
      captures the nonlinear pooling V2/V3 perform on V1 output (translation-stable).</li>
  <li><b>Texture statistics</b> — Gram / GIST / Portilla–Simoncelli summaries describe
      material and scene structure, which drives PPA and the ventral stream.</li>
  <li><b>Random features</b> — fixed random projections (RFF, random convs, codebooks) lift
      features into a high-dimensional kernel space without any training.</li>
  <li><b>Pooling / retinotopy</b> — foveal weighting and spatial grids encode where things
      are, not just what.</li>
</ul>
"""

_PROCESS_INTRO = """
<p>How did the search actually progress? These views read the ledger's iterations,
reward values, train–test gap, and status field to characterize the search dynamics
rather than any single model.</p>
<ul>
  <li><b>Progress</b> — best reward as a function of iteration, or wall-clock when
      reliable seconds are present.</li>
  <li><b>Failure rate</b> — non-<code>ok</code> status rows as a percentage of all iterations.</li>
  <li><b>Generalization gap</b> — <code>train_corr − test_corr</code> tracks how much the
      ridge readout overfits as features grow richer.</li>
  <li><b>Iteration accounting</b> — scored rows versus failed rows when compute timing is
      not available in the report data.</li>
</ul>
"""


_FINDINGS_INTRO = """
<p>This loop evolves a <b>hand-written, never-trained</b> feature extractor
(<code>model.py</code>): a stimulus image is mapped to features, a ridge readout is
fit against recorded fMRI (the only thing fit — that fit <i>is</i> the evaluation),
and held-out <code>test_corr</code> on NSD subject 01 is the reward. Each run rewards
a single visual region, and the report compares the winner to both the logged
identity/raw-pixel baseline and the documented trained-AlexNet reference.</p>

<p><b>The architecture the search converged on (in every region).</b> Without being
told to, the loop rediscovered the classical early-visual front-end:</p>
<ul>
  <li><b>Quadrature Gabor banks</b> (orientations × spatial wavelengths) reduced to
      <b>complex-cell energy</b> <code>√(cos² + sin²)</code> — V1 simple→complex cells;</li>
  <li><b>colour opponency</b> (luminance, R−G, B−Y) — retinal/LGN channels;</li>
  <li><b>log contrast compression</b> followed by <b>divisive normalization</b> with a
      spatial-surround pool — Heeger's canonical cortical gain control;</li>
  <li>raw pixels concatenated in as a low-level anchor.</li>
</ul>
<p>On top of this shared core, each region added a distinct elaboration: a second
<b>scattering</b> layer (Gabor-on-energy) for V2/V3, a wider normalization pool for V3's
larger receptive fields, <b>texture Gram matrices + GIST + random Fourier features</b> for
scene-selective PPA, <b>colour histograms</b> for EBA, and heterogeneous random conv banks
for the face/word regions.</p>

<p><b>Key findings.</b></p>
<ol>
  <li><b>V1 is the most predictable region and the simplest model wins it</b> — a pure
      Gabor-energy model with a 3×3 surround norm. Predictivity then falls along the
      cortical hierarchy V1 &gt; V2 &gt; V3 &gt; hV4 &gt; category-selective regions.</li>
  <li><b>Every region's best model still predicts V1 best</b> (~0.40 in the column below,
      regardless of which region was the reward). The region-specific machinery buys only
      modest gains on its own target — the models remain fundamentally low-level/V1-like
      and do not yet capture high-level <b>category selectivity</b> (FFA/PPA/EBA/VWFA stay
      ≈0.20–0.29).</li>
  <li>The baseline chart makes the main boundary clear: untrained encoders can match or
      beat AlexNet in early/mid visual cortex, but still trail the learned network in
      high-level category regions where semantics matter.</li>
</ol>
"""


_CSS = """
* { box-sizing: border-box; }
body { background:#ffffff; color:#1a1a1a; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       margin:0; padding:0; line-height:1.5; }
.page { max-width:1150px; margin:0 auto; padding:2rem 1.25rem 3rem; }
h1 { margin-top:0; } h2 { border-bottom:2px solid #eee; padding-bottom:.3rem; margin-top:2.4rem; scroll-margin-top:4.5rem; }
h3 { margin-bottom:.4rem; }
.toc { position:sticky; top:0; z-index:10; margin:1.2rem 0 1.8rem; padding:.65rem .8rem;
       display:flex; flex-wrap:wrap; gap:.45rem .85rem; align-items:center;
       background:rgba(255,255,255,.96); border:1px solid #e5e7eb; border-radius:10px;
       box-shadow:0 1px 8px rgba(15,23,42,.06); backdrop-filter:blur(6px); }
.toc b { color:#374151; margin-right:.25rem; }
.toc a { color:#0a58ca; text-decoration:none; font-size:.9rem; }
.toc a:hover { text-decoration:underline; }
.chart-card { width:100%; margin:1rem 0 1.2rem; padding:.75rem .75rem .55rem;
       border:1px solid #e5e7eb; border-radius:10px; background:#fff;
       box-shadow:0 1px 4px rgba(15,23,42,.04); }
.chart-plot { width:100%; min-height:360px; }
.chart-caption { margin:.25rem .25rem 0; color:#666; font-size:.85rem; }
.dashboard-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:1rem; align-items:stretch; }
.dashboard-grid .chart-card { margin:0; }
.note-card { min-height:380px; display:flex; align-items:center; justify-content:center; text-align:center; }
.run-block { display:block; margin:1.0rem 0 1.2rem; padding:0.9rem 1.1rem; border:1px solid #eee; border-radius:8px; background:#fafafa; }
.run-block summary { cursor:pointer; list-style:none; }
.run-block summary::-webkit-details-marker { display:none; }
.run-block summary::before { content:"▸"; display:inline-block; width:1rem; color:#666; }
.run-block[open] summary::before { content:"▾"; }
.run-title { font-weight:700; font-size:1rem; margin-right:.4rem; }
.fail-badge { display:inline-block; margin-left:.55rem; padding:.05rem .45rem; border-radius:999px;
        background:#fee2e2; color:#b42318; font-weight:700; font-size:.78rem; }
.writeup { font-size:.95rem; }
.writeup ul { margin:.3rem 0; } .writeup code { background:#eef; padding:0 .25rem; border-radius:3px; }
table { border-collapse:collapse; width:100%; background:#fff; table-layout:fixed; font-size:.86rem; }
th, td { border:1px solid #e3e3e3; padding:.4rem .55rem; text-align:left; vertical-align:top;
         overflow-wrap:anywhere; word-break:break-word; }
th { background:#f4f4f7; user-select:none; }
.good { color:#1a7f37; font-weight:700; } .bad { color:#b42318; font-weight:700; }
.meta { color:#666; font-size:.85rem; }
.findings ul, .findings ol { margin:.4rem 0 .8rem 1.2rem; } .findings li { margin:.25rem 0; }
.card { margin:1.2rem 0; padding:1rem 1.2rem; border:1px solid #e3e3e3; border-left:4px solid #2563eb;
        border-radius:8px; background:#fafbff; }
.card h3 { margin:.1rem 0 .5rem; }
.card .reward { float:right; font-size:.85rem; font-weight:600; color:#fff; background:#2563eb;
        padding:.1rem .5rem; border-radius:999px; }
.card p { margin:.35rem 0; }
details.code summary { cursor:pointer; color:#0a58ca; margin-top:.5rem; }
pre { background:#0d1117; color:#e6edf3; padding:.9rem 1rem; border-radius:6px; overflow:auto;
      font-size:.8rem; line-height:1.45; max-height:30rem; }
pre code { background:none; color:inherit; padding:0; white-space:pre; }
table.principles { table-layout:auto; width:auto; min-width:28rem; max-width:46rem; font-size:.9rem; }
table.principles th { position:static; }
table.baseline { table-layout:auto; font-size:.9rem; margin-top:1rem; }
table.baseline th { position:static; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
@media (max-width: 820px) {
  .dashboard-grid { grid-template-columns:1fr; }
  .page { padding:1rem .75rem 2rem; }
  .toc { position:static; }
}
"""


def build_html(runs: list[Run]) -> str:
    global _PLOT_COUNTER
    _PLOT_COUNTER = count(1)
    plotly_bundle = _load_plotly_bundle()
    n_attempts = sum(len(r.df) for r in runs)
    n_excluded = sum(int(r.df["excluded"].sum()) for r in runs)
    n_failed = sum(failure_count(r.df) for r in runs)
    fail_pct = (100.0 * n_failed / n_attempts) if n_attempts else 0.0
    studies = sorted({r.study for r in runs})
    labels = run_label_map(runs)

    summary = summary_plot(runs)
    summary_html = (
        summary
        if summary
        else "<p class='meta'>No included attempts to plot.</p>"
    )

    run_blocks = []
    for idx, run in enumerate(runs):
        label = labels.get(run.csv_path, run.display_label)
        plot = per_run_plot(run, label=label)
        plot_html = plot if plot else "<p class='meta'>No included attempts to plot.</p>"
        fail_count = failure_count(run.df)
        fail_pct_run = 100.0 * fail_count / len(run.df) if len(run.df) else 0.0
        fail_badge = (
            f"<span class='fail-badge' title='non-ok status rows'>✕ {fail_count}/{len(run.df)} "
            f"({fail_pct_run:.0f}%)</span>"
            if fail_count else ""
        )
        run_blocks.append(
            f'<details class="run-block"><summary>'
            f'<span class="run-title">{html.escape(label)}</span> '
            f'<span class="meta">{html.escape(run.region_label)}</span>{fail_badge}</summary>'
            f'<p class="meta">run: {html.escape(run.label)} · '
            f'source: {html.escape(display_source_path(run.csv_path))}</p>'
            f'<div class="writeup">{run_writeup(run)}</div>{plot_html}'
            f'</details>'
        )

    overview = (
        f"<p class='meta'>{len(runs)} run/study file(s) · {n_attempts} attempt(s) · "
        f"{n_failed} failed ({fail_pct:.0f}%) · {n_excluded} omitted from plots · "
        f"datasets: {', '.join(html.escape(s) for s in studies) or '—'}</p>"
    )

    heatmap, _, _ = cross_region_heatmap(runs)
    cross_html = (
        f"{heatmap}"
        "<p class='meta'>Rows: each run's best model. Columns: the region it is scored against. "
        "Note how the V1 column stays high across <em>every</em> run.</p>"
        if heatmap
        else "<p class='meta'>No per-region (corr_&lt;region&gt;) columns found.</p>"
    )

    principle_chart, principle_table = principle_attribution(runs)
    principles_html = (
        f'<div class="findings">{_PRINCIPLES_INTRO}</div>'
        + (f"{principle_chart}{principle_table}"
           if principle_chart else "<p class='meta'>No improving steps to attribute.</p>")
    )

    dashboard = search_dashboard(runs)
    process_html = (
        f'<div class="findings">{_PROCESS_INTRO}</div>'
        + (dashboard if dashboard else "<p class='meta'>Not enough data to plot the search process.</p>")
    )

    sections = [
        ("findings", "What the search found"),
        ("baselines", "Baselines"),
        ("principles", "Computational principles"),
        ("process", "The search process"),
        ("cross-region", "Cross-region predictivity"),
        ("mechanics", "Model mechanics"),
        ("summary", "Reward across all runs"),
        ("per-run", "Per-run trajectories"),
    ]
    toc = (
        '<nav class="toc" aria-label="Report sections"><b>Sections</b>'
        + "".join(f'<a href="#{sid}">{html.escape(title)}</a>' for sid, title in sections)
        + "</nav>"
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autoresearch report</title>
<script>{plotly_bundle}</script>
<style>{_CSS}</style></head>
<body>
<main class="page">
<h1>neuro-autoresearch report</h1>
{overview}
{toc}
<h2 id="findings">What the search found</h2>
<div class="findings">{_FINDINGS_INTRO}</div>
<h2 id="baselines">Baselines</h2>
{baseline_comparison(runs)}
<h2 id="principles">Computational principles</h2>
{principles_html}
<h2 id="process">The search process</h2>
{process_html}
<h2 id="cross-region">Cross-region predictivity</h2>
<p class="meta">Each row is the best model found for one reward region; each column is a
brain region scored after the same ridge readout. This makes it easier to see whether
a model is genuinely region-specific or mostly predicts early visual cortex.</p>
{cross_html}
<h2 id="mechanics">Model mechanics — what each run's best model computes</h2>
<p class="meta">Cards are ordered by reward. Each card names the target brain region,
describes the computation directly, and includes source code when the original run folder is available.</p>
{mechanics_section(runs)}
<h2 id="summary">Reward across all runs</h2>
{summary_html}
<h2 id="per-run">Per-run trajectories</h2>
<p class="meta">Per run: test/train correlation and the running-max reward. Logged
<code>n_params</code> appears only when enough positive, varying values are present to make
the axis meaningful. Expand a run when you want its full trajectory details.</p>
{''.join(run_blocks) if run_blocks else "<p class='meta'>No runs found.</p>"}
</main>
</body></html>
"""


def _render_to(runs: list[Run], out: str) -> str:
    """Write the HTML report for already-discovered ``runs`` to ``out``."""
    html_text = build_html(runs)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(html_text)
    return out


def build_report(runs_dir: str = "runs", out: str = "analyze/report.html", repo_root: str = ".",
                 include_root_results: bool = True) -> str:
    return _render_to(discover_runs(runs_dir, repo_root, include_root_results), out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="runs", help="directory of per-run output folders (default: runs)")
    parser.add_argument("--repo-root", default=".", help="repo root, scanned for results/<study>.csv (default: .)")
    parser.add_argument("--out", default="analyze/report.html", help="output HTML path (default: analyze/report.html)")
    parser.add_argument("--runs-only", action="store_true",
                        help="ignore the repo-root results/ ledger; report only runs/<tag> (used for the hosted page)")
    args = parser.parse_args()

    runs = discover_runs(args.runs_dir, args.repo_root, include_root_results=not args.runs_only)
    out = _render_to(runs, args.out)
    print(f"wrote {out} ({len(runs)} run/study file(s), {sum(len(r.df) for r in runs)} attempt(s))")


if __name__ == "__main__":
    main()
