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
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
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
# One fixed color per canonical region, used everywhere a region appears (scoreboard,
# summary traces, dashboard, mechanics cards, run blocks) so the eye can track a region
# across the whole report. Ordered roughly cool->warm along the cortical hierarchy.
REGION_COLORS = {
    "V1": "#2563eb",
    "V2": "#0891b2",
    "V3": "#0d9488",
    "hV4": "#16a34a",
    "early": "#65a30d",
    "ventral": "#ca8a04",
    "PPA": "#ea580c",
    "EBA": "#dc2626",
    "FFA": "#be123c",
    "VWFA": "#9333ea",
}


def region_color(region: str | None, fallback_index: int = 0) -> str:
    """The region's fixed color, or a palette color for region-less runs."""
    return REGION_COLORS.get(region or "", _PALETTE[fallback_index % len(_PALETTE)])


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


def _load_asset(*relpath: str) -> str:
    """Read a static UI asset (CSS / HTML prose) from ``analyze/assets``.

    The report's static markup lives in flat files (mirroring the vendored Plotly
    bundle) so this module stays logic-only; the content is inlined at build time.
    """
    with open(os.path.join(_ASSETS_DIR, *relpath), encoding="utf-8") as fh:
        return fh.read()


def _section_intro(name: str) -> str:
    """Prose intro HTML for a report section (``assets/sections/<name>.html``)."""
    return _load_asset("sections", f"{name}.html")


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
            "size": 13.5,
            "color": "#2b2f36",
            "family": "Seravek, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
        },
        "title": {
            "font": {
                "size": 17,
                "color": "#16181d",
                "family": "Charter, Iowan Old Style, Palatino Linotype, Georgia, serif",
            },
            "x": 0,
            "xanchor": "left",
        },
        "margin": {"l": 72, "r": 34, "t": 80, "b": 58},
        "hovermode": "closest",
        "hoverlabel": {
            "bgcolor": "white",
            "bordercolor": "#d8d5cd",
            "font": {"size": 12.5, "color": "#0f172a"},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 12},
        },
        "xaxis": {
            "automargin": True,
            "gridcolor": "#efede7",
            "zerolinecolor": "#d8d5cd",
            "linecolor": "#d8d5cd",
            "title": {"font": {"size": 13, "color": "#6b7280"}},
        },
        "yaxis": {
            "automargin": True,
            "gridcolor": "#efede7",
            "zerolinecolor": "#d8d5cd",
            "linecolor": "#d8d5cd",
            "title": {"font": {"size": 13, "color": "#6b7280"}},
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
            color = region_color(run.region, idx)
            data.append({
                "type": "scatter",
                "mode": "lines+markers",
                "x": inc["iteration"].tolist(),
                "y": inc["test_corr"].tolist(),
                "name": label,
                "line": {"width": 1.8, "color": color},
                "marker": {"size": 6, "color": color},
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
                "legend": {"orientation": "v", "x": 1.01, "xanchor": "left",
                           "y": 1, "yanchor": "top"},
                "margin": {"l": 72, "r": 165, "t": 80, "b": 58},
            },
            height=440,
            caption="Each trace is one run, in its region's color; click legend entries to toggle runs.",
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
        layout["margin"] = {"l": 70, "r": 82, "t": 80, "b": 58}
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


def region_winners(runs: list[Run]) -> list[tuple]:
    """Best run per reward region, in canonical region order.

    Each item is ``(region, run_label, region_name, best, identity, alexnet)``;
    shared by the header scoreboard and the baseline section.
    """
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
    order = {rg: i for i, rg in enumerate(REGION_ORDER)}
    rows.sort(key=lambda r: order.get(r[0], 999))
    return rows


def region_scoreboard(runs: list[Run]) -> str:
    """Compact card row: each region's best ``test_corr`` vs the trained-AlexNet bar."""
    rows = region_winners(runs)
    if not rows:
        return ""
    cards = []
    for rg, _label, name, best, _ident, alex in rows:
        color = region_color(rg)
        if alex is None:
            delta_html = ""
        else:
            beats = best >= alex
            cls = "good" if beats else "bad"
            sign = "▲" if beats else "▼"
            delta_html = (
                f"<span class='sb-delta {cls}' title='best untrained minus trained AlexNet'>"
                f"{sign} {best - alex:+.3f}</span>"
            )
        alex_html = (
            f"<span class='sb-alex'>AlexNet {alex:.3f}</span>" if alex is not None else ""
        )
        cards.append(
            f"<div class='sb-card' style='border-top-color:{color}' title='{html.escape(name)}'>"
            f"<span class='sb-region' style='color:{color}'>{html.escape(rg)}</span>"
            f"<span class='sb-best'>{best:.3f}</span>"
            f"{alex_html}{delta_html}</div>"
        )
    return (
        "<div class='scoreboard'>"
        + "".join(cards)
        + "</div>"
        "<p class='scoreboard-caption'>Best untrained model per reward region (held-out "
        "Pearson r) vs the trained-AlexNet reference — green beats AlexNet, red trails it.</p>"
    )


def _takeaway(text: str) -> str:
    """A highlighted one-paragraph insight box; callers pass already-escaped HTML."""
    if not text:
        return ""
    return f"<div class='takeaway'><span class='takeaway-tag'>Takeaway</span> {text}</div>"


_EARLY_REGIONS = {"V1", "V2", "V3", "hV4", "early"}
_CATEGORY_REGIONS = {"FFA", "PPA", "EBA", "VWFA", "ventral"}


def _baseline_takeaway(rows: list[tuple]) -> str:
    """Computed verdict on the beat-AlexNet goal, from the region winners."""
    scored = [r for r in rows if r[5] is not None]
    if not scored:
        return ""
    beat = [r for r in scored if r[3] >= r[5]]
    trail = [r for r in scored if r[3] < r[5]]
    bits = []
    if beat:
        names = ", ".join(r[0] for r in beat)
        top = max(beat, key=lambda r: r[3] - r[5])
        bits.append(
            f"Untrained models <b>beat trained AlexNet in {len(beat)} of {len(scored)} "
            f"regions</b> ({names}); the biggest win is {top[0]} ({top[3] - top[5]:+.3f})."
        )
    else:
        bits.append(f"No region winner beats trained AlexNet yet (0 of {len(scored)}).")
    if trail:
        worst = min(trail, key=lambda r: r[3] - r[5])
        names = ", ".join(r[0] for r in trail)
        bits.append(
            f"They still trail in {names}; the largest deficit is {worst[0]} "
            f"({worst[3] - worst[5]:+.3f})."
        )
        if beat and all(r[0] in _EARLY_REGIONS for r in beat) and \
                all(r[0] in _CATEGORY_REGIONS for r in trail):
            bits.append(
                "The boundary is clean: hand-written features match learned ones in "
                "early/mid visual cortex but not yet in category-selective regions, "
                "where semantics matter."
            )
    return _takeaway(" ".join(bits))


def baseline_comparison(runs: list[Run]) -> str:
    """Compare each region winner with logged identity and documented AlexNet baselines."""
    rows = region_winners(runs)
    if not rows:
        return "<p class='meta'>No logged identity rows or region-tagged NSD runs found for baseline comparison.</p>"

    # Dumbbell chart: per region, a connector from the trained-AlexNet mark to the best
    # untrained model — direction and colour answer "who wins, and by how much" at a glance.
    names = [r[2] for r in rows]
    data = []
    for rg, _label, name, best, _ident, alex in rows:
        if alex is None:
            continue
        beats = best >= alex
        data.append({
            "type": "scatter",
            "mode": "lines",
            "x": [alex, best],
            "y": [name, name],
            "line": {"color": "#a7cdaa" if beats else "#e6b2b2", "width": 6},
            "showlegend": False,
            "hoverinfo": "skip",
        })
    alex_pts = [(r[2], r[5]) for r in rows if r[5] is not None]
    if alex_pts:
        data.append({
            "type": "scatter",
            "mode": "markers",
            "name": "trained AlexNet fc7",
            "x": [v for _, v in alex_pts],
            "y": [n for n, _ in alex_pts],
            "marker": {"symbol": "diamond", "size": 11, "color": "#4b5563"},
            "hovertemplate": "trained AlexNet fc7<br>%{y}<br>%{x:.3f}<extra></extra>",
        })
    ident_pts = [(r[2], r[4]) for r in rows if r[4] is not None]
    if ident_pts:
        data.append({
            "type": "scatter",
            "mode": "markers",
            "name": "identity / raw pixels",
            "x": [v for _, v in ident_pts],
            "y": [n for n, _ in ident_pts],
            "marker": {"symbol": "circle-open", "size": 9, "color": "#9ca3af",
                       "line": {"width": 1.5}},
            "hovertemplate": "identity / raw pixels<br>%{y}<br>%{x:.3f}<extra></extra>",
        })
    deltas = ["" if r[5] is None else f"{r[3] - r[5]:+.3f}" for r in rows]
    data.append({
        "type": "scatter",
        "mode": "markers+text",
        "name": "best untrained model",
        "x": [r[3] for r in rows],
        "y": names,
        "customdata": deltas,
        "marker": {"size": 13, "color": [region_color(r[0]) for r in rows],
                   "line": {"color": "#ffffff", "width": 1.5}},
        "text": deltas,
        "textposition": [
            "middle right" if r[5] is None or r[3] >= r[5] else "middle left"
            for r in rows
        ],
        "textfont": {
            "size": 11.5,
            "color": ["#166534" if r[5] is not None and r[3] >= r[5] else "#b91c1c"
                      for r in rows],
        },
        "cliponaxis": False,
        "hovertemplate": (
            "best untrained model<br>%{y}<br>%{x:.3f} (%{customdata} vs AlexNet)<extra></extra>"
        ),
    })
    chart = _plotly_div(
        data,
        {
            "title": {"text": "Best untrained model vs trained AlexNet, by region"},
            "xaxis": {"title": {"text": "Held-out correlation (mean Pearson r; higher is better)"}},
            "yaxis": {
                "categoryorder": "array",
                "categoryarray": names,
                "autorange": "reversed",
                "gridcolor": "#f3f1eb",
            },
            "margin": {"l": 210, "r": 70, "t": 80, "b": 58},
        },
        height=max(380, 48 * len(rows) + 170),
        caption="Each row: the gap between the search's best untrained model (●, labelled with "
                "the difference) and frozen trained AlexNet fc7 (◆). Green connector = untrained "
                "wins; ○ = logged raw-pixel baseline. Regions run early → category down the page.",
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
        _baseline_takeaway(rows)
        + "<p class='meta'>Identity is shown only when that row is already present in the log. "
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


def cross_region_heatmap(runs: list[Run]) -> tuple[str | None, str]:
    """Heatmap of each run's best model scored against every region (``corr_<region>``).

    Returns ``(plot_html, takeaway_html)``, or ``("" , "")`` parts when no run
    exposes ``corr_<region>`` columns (so test fixtures simply skip the section).
    """
    rows, labels, targets = [], [], []
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
        return None, ""

    for run in runs:
        br = best_row(run)
        if br is None:
            continue
        vals = [br.get(f"corr_{rg}") for rg in present]
        if all(pd.isna(v) for v in vals):
            continue
        labels.append(label_map.get(run.csv_path, run.display_label))
        targets.append(run.region)
        rows.append([float(v) if pd.notna(v) else None for v in vals])

    if not rows:
        return None, ""

    # Mark the region each model predicts best — the report's central question is
    # whether that ever moves away from early visual cortex.
    text = [["" if v is None else f"{v:.2f}" for v in row] for row in rows]
    argmax = []
    for i, row in enumerate(rows):
        j = max((j for j, v in enumerate(row) if v is not None),
                key=lambda j: row[j], default=None)
        argmax.append(None if j is None else present[j])
        if j is not None:
            text[i][j] += " ★"

    takeaway = ""
    n = sum(1 for a in argmax if a is not None)
    if n:
        top_counts = Counter(a for a in argmax if a is not None)
        top_region, top_n = top_counts.most_common(1)[0]
        on_target = sum(1 for a, t in zip(argmax, targets) if a is not None and a == t)
        bits = [
            f"<b>{top_n} of {n} run winners predict {top_region} best</b> (★), regardless "
            f"of which region the search was rewarded on."
        ]
        if on_target < n:
            bits.append(
                f"Only {on_target} of {n} models predict their own target region best — "
                "the discovered features remain fundamentally early-visual, and the "
                "region-specific machinery buys only modest gains on its own target."
            )
        takeaway = _takeaway(" ".join(bits))

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
            "yaxis": {"title": {"text": "Run (optimized region)"}, "autorange": "reversed"},
            "margin": {"l": 170, "r": 80, "t": 80, "b": 70},
        },
        height=max(360, 42 * len(rows) + 160),
        caption="Rows are run winners; columns are held-out predictivity in each region. "
                "★ marks the region each model predicts best.",
    )
    return chart, takeaway


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
        # Compact cross-region profile: small bars instead of a number dump, so the
        # eye picks up the shape (V1-heavy vs target-specific) instantly.
        profile = [
            (reg, float(br[f"corr_{reg}"]))
            for reg in REGION_ORDER
            if f"corr_{reg}" in run.df.columns and pd.notna(br.get(f"corr_{reg}"))
        ]
        xreg_html = ""
        if profile:
            vmax = max(v for _, v in profile) or 1.0
            bars = "".join(
                f"<div class='xreg-item{' xreg-target' if reg == run.region else ''}'>"
                f"<span class='xreg-label'>{html.escape(reg)}</span>"
                f"<div class='xreg-bar'><div class='xreg-fill' "
                f"style='width:{100 * v / vmax:.0f}%;background:{region_color(reg)}'></div></div>"
                f"<span class='xreg-val'>{v:.2f}</span></div>"
                for reg, v in profile
            )
            xreg_html = (
                "<div class='xreg'><span class='xreg-title'>cross-region predictivity "
                "<span class='meta'>(reward region outlined)</span></span>"
                f"{bars}</div>"
            )
        identity = identity_row(run)
        identity_html = ""
        if identity is not None and pd.notna(identity.get("test_corr")):
            identity_html = f" · identity {float(identity['test_corr']):.4f}"
        alex = ALEXNET_BASELINES.get(run.region or "")
        alex_html = f" · AlexNet {alex:.4f}" if alex is not None else ""
        source_html = _code_block(src, path)
        accent = region_color(run.region)
        cards.append(
            f'<div class="card" style="border-left-color:{accent}">'
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
    return f'<div class="cards-grid">{"".join(cards)}</div>'


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
            "margin": {"l": 230, "r": 45, "t": 80, "b": 58},
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

    total_gain = sum(v for _, v in items)
    top_name, top_val = items[0]
    takeaway = ""
    if total_gain > 0:
        head = (
            f"<b>{html.escape(top_name)}</b> is associated with the most reward gain "
            f"({top_val:.3f}, {100 * top_val / total_gain:.0f}% of attributed gain"
        )
        if len(items) > 1:
            second_name, second_val = items[1]
            head += f"), followed by <b>{html.escape(second_name)}</b> ({second_val:.3f})"
        takeaway = _takeaway(
            head + ". Attribution overlaps — one improving idea can invoke several "
            "principles — so read the bars as <i>which motifs travel with progress</i>, "
            "not as exclusive credit."
        )
    return takeaway + chart, table


def search_takeaways(runs: list[Run]) -> tuple[str, str]:
    """``(process, summary)`` insight boxes computed from the search ledger."""
    n_scored = 0
    n_improve = 0
    halfway = []
    for run in runs:
        inc = run.included.dropna(subset=["test_corr"]).sort_values("iteration")
        n_scored += len(inc)
        n_improve += len(_improvement_events(run))
        if len(inc) >= 4:
            cm = inc["test_corr"].cummax()
            first, final = float(cm.iloc[0]), float(cm.iloc[-1])
            if final > first:
                mid = float(cm.iloc[len(cm) // 2 - 1])
                halfway.append((mid - first) / (final - first))
    process = ""
    if halfway:
        pct = 100.0 * sum(halfway) / len(halfway)
        process = _takeaway(
            f"Progress is front-loaded: by the midpoint of a run the search had already "
            f"captured <b>{pct:.0f}%</b> of its eventual reward gain, on average. Later "
            "iterations mostly explore variations that do not beat the incumbent — "
            "watch the running-max curves flatten."
        )
    summary = ""
    if n_scored:
        rate = 100.0 * n_improve / n_scored
        lead = (
            f"Improvements are rare: only <b>{n_improve} of {n_scored}</b> scored "
            f"attempts ({rate:.0f}%) raised a run's best."
            if rate < 25 else
            f"<b>{n_improve} of {n_scored}</b> scored attempts ({rate:.0f}%) raised "
            "their run's best at the time they were tried."
        )
        summary = _takeaway(
            lead + " The rest were sideways steps — the attempts that landed are "
            "itemized in the per-run writeups below."
        )
    return process, summary


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

    colors = {run.csv_path: region_color(run.region, i) for i, (run, _) in enumerate(series)}

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
            "showlegend": False,
        },
        height=380,
        caption="One line per run, in its region's color (see the scoreboard); hover for the run name.",
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
                "showlegend": False,
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
                "margin": {"l": 150, "r": 45, "t": 80, "b": 58},
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
                "showlegend": False,
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
                "margin": {"l": 150, "r": 45, "t": 80, "b": 58},
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
                "margin": {"l": 150, "r": 45, "t": 80, "b": 58},
            },
            height=380,
        )

    return (
        '<div class="dashboard-grid">'
        f"{progress_chart}{size_chart}{gap_chart}{compute_chart}"
        "</div>"
    )


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
        accent = region_color(run.region, idx)
        run_blocks.append(
            f'<details class="run-block" style="border-left:4px solid {accent}"><summary>'
            f'<span class="run-dot" style="background:{accent}"></span>'
            f'<span class="run-title">{html.escape(label)}</span> '
            f'<span class="meta">{html.escape(run.region_label)}</span>{fail_badge}</summary>'
            f'<p class="meta">run: {html.escape(run.label)} · '
            f'source: {html.escape(display_source_path(run.csv_path))}</p>'
            f'<div class="writeup">{run_writeup(run)}</div>{plot_html}'
            f'</details>'
        )

    # Header stat cards: the numbers a reader wants before any chart.
    winners = region_winners(runs)
    if winners:
        best_rg, _bl, _bn, best_val, _bi, _ba = max(winners, key=lambda r: r[3])
        best_stat = (
            f"<div class='stat'><span class='stat-value' style='color:{region_color(best_rg)}'>"
            f"{best_val:.3f}</span><span class='stat-label'>best test_corr ({html.escape(best_rg)})</span></div>"
        )
    else:
        bests = [float(br["test_corr"]) for r in runs
                 if (br := best_row(r)) is not None and pd.notna(br["test_corr"])]
        best_stat = (
            f"<div class='stat'><span class='stat-value'>{max(bests):.3f}</span>"
            "<span class='stat-label'>best test_corr</span></div>"
        ) if bests else ""
    datasets_txt = ", ".join(html.escape(s) for s in studies) or "—"
    header = (
        "<header class='report-header'>"
        "<h1>neuro-autoresearch report</h1>"
        "<p class='subtitle'>An AlphaEvolve-style search over <b>hand-written, never-trained</b> "
        "encoding models: each proposed model is read out with ridge regression against recorded "
        "brain activity, and held-out correlation is the reward.</p>"
        "<div class='stat-row'>"
        f"<div class='stat'><span class='stat-value'>{len(runs)}</span><span class='stat-label'>run/study file(s)</span></div>"
        f"<div class='stat'><span class='stat-value'>{n_attempts}</span><span class='stat-label'>attempt(s)</span></div>"
        f"<div class='stat'><span class='stat-value'>{n_failed} <small>({fail_pct:.0f}%)</small></span><span class='stat-label'>failed</span></div>"
        f"<div class='stat'><span class='stat-value'>{n_excluded}</span><span class='stat-label'>omitted from plots</span></div>"
        f"{best_stat}"
        f"<div class='stat'><span class='stat-value stat-text'>{datasets_txt}</span><span class='stat-label'>datasets</span></div>"
        "</div>"
        f"{region_scoreboard(runs)}"
        "</header>"
    )

    heatmap, heat_takeaway = cross_region_heatmap(runs)
    cross_html = (
        f"{heat_takeaway}{heatmap}"
        if heatmap
        else "<p class='meta'>No per-region (corr_&lt;region&gt;) columns found.</p>"
    )

    principle_chart, principle_table = principle_attribution(runs)
    principles_html = (
        f'<div class="findings">{_section_intro("principles")}</div>'
        + (f"{principle_chart}{principle_table}"
           if principle_chart else "<p class='meta'>No improving steps to attribute.</p>")
    )

    process_takeaway, summary_takeaway = search_takeaways(runs)
    dashboard = search_dashboard(runs)
    process_html = (
        process_takeaway
        + f'<div class="findings">{_section_intro("process")}</div>'
        + (dashboard if dashboard else "<p class='meta'>Not enough data to plot the search process.</p>")
    )

    sections = [
        ("recap", "Project recap"),
        ("setup", "Current setup &amp; logged data"),
        ("findings", "What the search found"),
        ("baselines", "Did it beat trained AlexNet?"),
        ("principles", "Which computations mattered?"),
        ("process", "How did the search behave?"),
        ("cross-region", "Are the models region-specific?"),
        ("mechanics", "What does each winner compute?"),
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
<style>{_load_asset("report.css")}</style></head>
<body>
<main class="page">
{header}
{toc}
<div class="howto"><b>How to read this report.</b>
<ul>
<li>Models are <b>never trained</b> — weights are written directly in code. Only a ridge
readout is fit against recorded brain activity, and that fit <i>is</i> the evaluation.</li>
<li>The score, <code>test_corr</code>, is the held-out correlation between predicted and
measured responses. Each run rewards <b>one brain region</b>; the search mutates the model
to raise that region's score. Differences under ~0.005 are noise.</li>
<li>The bar to clear is a <b>trained ImageNet AlexNet</b> read out the same way — the
question throughout is where hand-written computation matches learned features.</li>
</ul></div>
<h2 id="recap">Project recap</h2>
<p class="section-sub">A presentation-friendly summary of the project's arc — the previous
hierarchical multi-agent setup, the move to a unified data interface, and the headline
result so far.</p>
<div class="findings">{_section_intro("recap")}</div>
<h2 id="setup">Current setup &amp; logged data</h2>
<p class="section-sub">What every evaluation records in the append-only database, and the
exact columns the agent reads back as its feedback signal.</p>
<div class="findings">{_section_intro("setup")}</div>
<h2 id="findings">What the search found</h2>
<p class="section-sub">A written synthesis of what the search converged on and why.</p>
<div class="findings">{_section_intro("findings")}</div>
<h2 id="baselines">Did it beat trained AlexNet?</h2>
<p class="section-sub">Each region's best untrained model against the logged identity
baseline and the documented trained-AlexNet reference.</p>
{baseline_comparison(runs)}
<h2 id="principles">Which computations mattered?</h2>
<p class="section-sub">Hand-written computational motifs ranked by how much reward gain
they travelled with, summed over every improving step.</p>
{principles_html}
<h2 id="process">How did the search behave?</h2>
<p class="section-sub">Search dynamics read straight from the ledger: progress, model
size, generalization gap, and compute.</p>
{process_html}
<h2 id="cross-region">Are the models region-specific?</h2>
<p class="section-sub">Each row is the best model found for one reward region; each column is a
brain region scored after the same ridge readout. If a model were truly region-specific,
its ★ (best-predicted region) would sit on its own target.</p>
{cross_html}
<h2 id="mechanics">What does each winner compute?</h2>
<p class="section-sub">Cards are ordered by reward. Each card names the target brain region,
describes the computation directly, shows the model's cross-region profile as bars, and
includes source code when the original run folder is available.</p>
{mechanics_section(runs)}
<h2 id="summary">Reward across all runs</h2>
<p class="section-sub">Every scored attempt's held-out correlation vs iteration, one trace
per run, faceted by dataset.</p>
{summary_takeaway}
{summary_html}
<h2 id="per-run">Per-run trajectories</h2>
<p class="section-sub">Per run: test/train correlation and the running-max reward. Logged
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
    with open(out, "w", encoding="utf-8") as fh:
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
