"""Turn autoresearch run outputs into a single self-contained HTML report.

Scans per-study append-only result CSVs (``runs/<tag>/results/<study>.csv`` and the
repo-root ``results/<study>.csv``), optionally labelled by ``runs/<tag>/metadata.json``,
and writes one white-background, interactive HTML file with:

  * top summary plot(s) overlaying every run's reward (``test_corr``) vs iteration,
    faceted by dataset (the ``study`` column);
  * per-run ``test_corr`` / ``train_corr`` vs iteration with a running-max line;
  * an interactive, sortable table of attempts with expandable ``idea`` text;
  * an auto-generated writeup of which ideas were tried and which improved
    ``best_so_far``.

Rows whose ``status`` is not ``ok`` or whose ``idea`` mentions training are flagged and
excluded from the plots and the writeup (the proposed model is never trained), but are
still listed in the table so nothing is silently dropped.

Usage::

    python analyze/build_report.py [--runs-dir runs] [--repo-root .] [--out analyze/report.html]
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
from dataclasses import dataclass, field
from functools import cached_property

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TABLE_COLUMNS = [
    "iteration", "model_name", "idea", "summary", "test_corr", "train_corr", "best_so_far", "status",
]
# Canonical left-to-right ordering of NSD regions (early visual -> ventral category).
REGION_ORDER = ["V1", "V2", "V3", "hV4", "early", "ventral", "PPA", "EBA", "FFA", "VWFA"]
_TRAIN_RE = re.compile(r"\btrain", re.IGNORECASE)


def mentions_training(idea: object) -> bool:
    """True when the idea text refers to training the model (which is never allowed)."""
    return isinstance(idea, str) and bool(_TRAIN_RE.search(idea))


def is_ok(status: object) -> bool:
    return isinstance(status, str) and status.strip().lower() == "ok"


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
    df["idea"] = df["idea"].fillna("")
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


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_tag(b64: str, alt: str) -> str:
    return f'<img class="plot" alt="{html.escape(alt)}" src="data:image/png;base64,{b64}">'


def summary_plot(runs: list[Run]) -> str | None:
    """Overlay every run's reward vs iteration, one facet per dataset (study)."""
    by_study: dict[str, list[Run]] = {}
    for run in runs:
        if not run.included.empty:
            by_study.setdefault(run.study, []).append(run)
    if not by_study:
        return None

    studies = sorted(by_study)
    fig, axes = plt.subplots(
        1, len(studies), figsize=(6.2 * len(studies), 4.4), squeeze=False, facecolor="white"
    )
    for ax, study in zip(axes[0], studies):
        for run in by_study[study]:
            inc = run.included
            ax.plot(inc["iteration"], inc["test_corr"], marker="o", ms=3, lw=1.4, label=run.label)
        ax.set_title(f"{study}: test_corr vs iteration")
        ax.set_xlabel("iteration")
        ax.set_ylabel("test_corr")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return _fig_to_base64(fig)


def per_run_plot(run: Run) -> str | None:
    inc = run.included
    if inc.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="white")
    ax.plot(inc["iteration"], inc["test_corr"], marker="o", ms=3, lw=1.4, color="#1f77b4", label="test_corr")
    if inc["train_corr"].notna().any():
        ax.plot(inc["iteration"], inc["train_corr"], marker="s", ms=3, lw=1.0, color="#ff7f0e", alpha=0.8, label="train_corr")
    running_max = inc["test_corr"].cummax()
    ax.plot(inc["iteration"], running_max, drawstyle="steps-post", lw=1.6, color="#2ca02c", ls="--", label="running max (test_corr)")
    ax.set_title(run.label, fontsize=9)
    ax.set_xlabel("iteration")
    ax.set_ylabel("correlation")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if "n_params" in inc.columns and inc["n_params"].notna().any():
        ax2 = ax.twinx()
        ax2.plot(inc["iteration"], inc["n_params"].clip(lower=1), color="#9467bd", alpha=0.55,
                 lw=1.1, ls=":", marker=".", ms=3, label="n_params")
        ax2.set_yscale("log")
        ax2.set_ylabel("n_params (log)", color="#9467bd")
        ax2.tick_params(axis="y", labelcolor="#9467bd")
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax.legend(handles, labels, fontsize=8, loc="best")
    fig.tight_layout()
    return _fig_to_base64(fig)


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
            improvements.append((
                int(row["iteration"]), str(row["model_name"]), str(row["idea"]), best, delta,
                (float(cum_h.iloc[pos]) if has_sec else None),
                (int(npar) if pd.notna(npar) else None),
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


def _idea_cell(idea: str) -> str:
    idea = idea or ""
    safe_full = html.escape(idea)
    if len(idea) <= 90:
        return f'<td class="idea">{safe_full}</td>'
    return (
        '<td class="idea"><details><summary>'
        f"{html.escape(_truncate(idea))}</summary>{safe_full}</details></td>"
    )


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return "" if pd.isna(value) else f"{value:.4f}"
    if value is None:
        return ""
    return html.escape(str(value))


def attempts_table(runs: list[Run]) -> str:
    rows_html = []
    for run in runs:
        for _, row in run.df.iterrows():
            excluded = bool(row["excluded"])
            cells = [f'<td class="run">{html.escape(run.tag)}</td>', f'<td>{html.escape(run.study)}</td>']
            for col in TABLE_COLUMNS:
                value = row.get(col, "")
                if col in ("idea", "summary"):
                    cells.append(_idea_cell(str(value)))
                elif col in ("test_corr", "train_corr", "best_so_far"):
                    cells.append(f'<td class="num">{_fmt(value)}</td>')
                else:
                    cells.append(f"<td>{_fmt(value)}</td>")
            flag = '<td class="flag">excluded</td>' if excluded else '<td class="flag ok">ok</td>'
            cls = ' class="excluded"' if excluded else ""
            rows_html.append(f"<tr{cls}>" + flag + "".join(cells) + "</tr>")

    headers = ["flag", "run", "study"] + TABLE_COLUMNS
    head_html = "".join(
        f'<th onclick="sortTable({i})">{html.escape(h)} <span class="arrow"></span></th>'
        for i, h in enumerate(headers)
    )
    return (
        '<table id="attempts"><thead><tr>'
        + head_html
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )


# --------------------------------------------------------------------------- #
# Mechanistic analysis: what the best model in each run actually computes.
# --------------------------------------------------------------------------- #

def best_row(run: Run) -> pd.Series | None:
    """The included attempt with the highest ``test_corr`` (the model that was found)."""
    inc = run.included
    if inc.empty or inc["test_corr"].isna().all():
        return None
    return inc.loc[inc["test_corr"].idxmax()]


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

    Returns ``(base64_png_or_None, row_labels, col_regions)``. ``None`` when no run
    exposes ``corr_<region>`` columns (so test fixtures simply skip the section).
    """
    rows, labels = [], []
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
        target = run.metadata.get("dataset") or run.study
        labels.append(f"{run.tag}")
        rows.append([float(v) if pd.notna(v) else float("nan") for v in vals])

    import numpy as np
    mat = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(1.0 * len(present) + 3, 0.5 * len(rows) + 2.2), facecolor="white")
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("region scored (corr)")
    ax.set_title("Best model per run — predictivity across all regions")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < (np.nanmax(mat) * 0.6) else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    return _fig_to_base64(fig), labels, present


def _code_block(source: str | None, path: str | None) -> str:
    if not source:
        return "<p class='meta'>Model source not available in this run folder.</p>"
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
    if n == 0:
        return "parameter-free (filters/projections are fixed or seeded random)"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M fixed/expansion values (large random-feature lift, never trained)"
    return f"{n:,} fixed values (never trained)"


def mechanics_section(runs: list[Run]) -> str:
    """One card per run for the best model it found: idea, summary, reward, source."""
    cards = []
    ranked = []
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
        target = run.metadata.get("dataset") or run.study
        # Compact cross-region line for this model.
        xreg = []
        for rg in REGION_ORDER:
            c = f"corr_{rg}"
            if c in run.df.columns and pd.notna(br.get(c)):
                xreg.append(f"{rg}={float(br[c]):.2f}")
        xreg_html = (
            f"<p class='meta'>across regions: {html.escape(' · '.join(xreg))}</p>" if xreg else ""
        )
        cards.append(
            f'<div class="card">'
            f"<h3>{html.escape(run.tag)} — <code>{html.escape(name)}</code> "
            f'<span class="reward">test_corr {score:.4f}</span></h3>'
            f"<p class='meta'>target region: {html.escape(str(target))}"
            + (f" · {html.escape(note)}" if note else "")
            + "</p>"
            + (f"<p><b>Idea.</b> {html.escape(idea)}</p>" if idea else "")
            + (f"<p><b>What it computes.</b> {html.escape(summary)}</p>" if summary else "")
            + xreg_html
            + _code_block(src, path)
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

    fig, ax = plt.subplots(figsize=(8.2, 0.5 * len(items) + 1.7), facecolor="white")
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    ax.barh(names, vals, color="#2563eb")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Σ gain in best test_corr from improving steps mentioning the principle")
    ax.set_title("Which computational principles moved the reward (all runs)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

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
    return _fig_to_base64(fig), table


def search_dashboard(runs: list[Run]) -> str | None:
    """Four-panel view of the search: efficiency, complexity, overfitting, compute."""
    import numpy as np

    series = []
    for run in runs:
        inc = run.included.dropna(subset=["test_corr"]).sort_values("iteration")
        if not inc.empty:
            series.append((run, inc))
    if not series:
        return None

    cmap = plt.get_cmap("tab10" if len(series) <= 10 else "tab20")
    colors = {run.csv_path: cmap(i % cmap.N) for i, (run, _) in enumerate(series)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")
    a, b, c, d = axes.ravel()

    any_sec = False
    for run, inc in series:
        col = colors[run.csv_path]
        if "seconds" in inc.columns and inc["seconds"].notna().any():
            x = inc["seconds"].fillna(0).cumsum() / 3600.0
            any_sec = True
        else:
            x = inc["iteration"]
        a.plot(x, inc["test_corr"].cummax(), lw=1.5, color=col, label=run.tag)
    a.set_xlabel("cumulative wall-clock (h)" if any_sec else "iteration")
    a.set_ylabel("best_so_far (test_corr)")
    a.set_title("Search efficiency: best reward vs wall-clock")
    a.grid(True, alpha=0.3)
    a.legend(fontsize=6, ncol=2)

    for run, inc in series:
        if "n_params" not in inc.columns:
            continue
        col = colors[run.csv_path]
        npar = inc["n_params"].clip(lower=1)
        b.scatter(npar, inc["test_corr"], s=14, color=col, alpha=0.55, edgecolors="none")
        bi = inc["test_corr"].idxmax()
        b.scatter(max(float(inc.loc[bi, "n_params"]) if pd.notna(inc.loc[bi, "n_params"]) else 1, 1),
                  inc.loc[bi, "test_corr"], marker="*", s=130, color=col,
                  edgecolors="black", linewidths=0.4)
    b.set_xscale("log")
    b.set_xlabel("n_params (log)")
    b.set_ylabel("test_corr")
    b.set_title("Does complexity pay? reward vs model size (★ = run best)")
    b.grid(True, alpha=0.3)

    for run, inc in series:
        if inc["train_corr"].notna().any():
            c.plot(inc["iteration"], inc["train_corr"] - inc["test_corr"],
                   lw=1.2, color=colors[run.csv_path])
    c.axhline(0, color="gray", lw=0.8, ls=":")
    c.set_xlabel("iteration")
    c.set_ylabel("train_corr − test_corr")
    c.set_title("Readout generalization gap over the search")
    c.grid(True, alpha=0.3)

    order = list(np.argsort([(inc["seconds"].fillna(0).sum() / 3600.0
                              if "seconds" in inc.columns else 0.0) for _, inc in series]))
    ypos = np.arange(len(order))
    hours = [float(series[i][1]["seconds"].fillna(0).sum() / 3600.0)
             if "seconds" in series[i][1].columns else 0.0 for i in order]
    iters = [len(series[i][1]) for i in order]
    d.barh(ypos, hours, color=[colors[series[i][0].csv_path] for i in order])
    for i, (h, n) in enumerate(zip(hours, iters)):
        d.text(h, i, f" {h:.1f}h / {n} it", va="center", fontsize=7)
    d.set_yticks(ypos)
    d.set_yticklabels([series[i][0].tag for i in order], fontsize=7)
    d.set_xlabel("total wall-clock (h)")
    d.set_title("Compute spent per run")
    d.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    return _fig_to_base64(fig)


_PRINCIPLES_INTRO = """
<p>The point of the search is not just the winning model but the <b>computational
principles</b> it keeps rediscovering. Each principle below is a known ingredient of
biological vision; the chart sums, across all runs, the reward gained by every improving
step whose idea invokes it (a step can invoke several, so the bars overlap).</p>
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
<p>How did the search actually progress? These views read the ledger's
<code>iteration</code>, wall-clock <code>seconds</code>, <code>n_params</code> and the
train–test gap to characterise the search dynamics rather than any single model.</p>
<ul>
  <li><b>Efficiency</b> — best reward as a function of cumulative wall-clock: most gains
      arrive early, then improvements become small and expensive.</li>
  <li><b>Does complexity pay?</b> — reward vs model size (log axis). Bigger is not better:
      the strongest models (e.g. V1) are among the smallest; large random-feature lifts
      add parameters without proportional reward.</li>
  <li><b>Generalization gap</b> — <code>train_corr − test_corr</code> tracks how much the
      ridge readout overfits as features grow richer.</li>
  <li><b>Compute spent</b> — total wall-clock and number of iterations per run.</li>
</ul>
"""


_FINDINGS_INTRO = """
<p>This loop evolves a <b>hand-written, never-trained</b> feature extractor
(<code>model.py</code>): a stimulus image is mapped to features, a ridge readout is
fit against recorded fMRI (the only thing fit — that fit <i>is</i> the evaluation),
and held-out <code>test_corr</code> on NSD subject 01 is the reward. Each run rewards
a single visual region.</p>

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
  <li>These untrained encoders are <b>competitive with a frozen ImageNet AlexNet</b>
      (~0.32 on the ventral stream as documented in <code>AGENTS.md</code>) — beating it on
      V1, lagging on category regions where learned semantics matter.</li>
</ol>
"""


_CSS = """
* { box-sizing: border-box; }
body { background:#ffffff; color:#1a1a1a; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       margin:0; padding:2rem; line-height:1.5; }
h1 { margin-top:0; } h2 { border-bottom:2px solid #eee; padding-bottom:.3rem; margin-top:2.4rem; }
h3 { margin-bottom:.4rem; }
.plot { max-width:100%; height:auto; border:1px solid #eee; border-radius:6px; background:#fff; }
.run-block { margin:1.4rem 0 2.2rem; padding:1rem 1.2rem; border:1px solid #eee; border-radius:8px; background:#fafafa; }
.writeup { font-size:.95rem; }
.writeup ul { margin:.3rem 0; } .writeup code { background:#eef; padding:0 .25rem; border-radius:3px; }
table { border-collapse:collapse; width:100%; background:#fff; table-layout:fixed; font-size:.86rem; }
th, td { border:1px solid #e3e3e3; padding:.4rem .55rem; text-align:left; vertical-align:top;
         overflow-wrap:anywhere; word-break:break-word; }
th { background:#f4f4f7; cursor:pointer; user-select:none; position:sticky; top:0; }
th:hover { background:#e9e9f2; }
.idea { max-width:30rem; }
.idea details summary { cursor:pointer; color:#0a58ca; }
.idea details[open] summary { margin-bottom:.4rem; }
tr.excluded { background:#fff5f5; color:#7a3030; }
.flag { font-weight:600; white-space:nowrap; } .flag.ok { color:#1a7f37; } tr.excluded .flag { color:#c0392b; }
.run { white-space:nowrap; } .arrow { font-size:.7rem; color:#888; }
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
"""

_JS = """
function sortTable(col){
  var t=document.getElementById('attempts'), tb=t.tBodies[0];
  var rows=Array.prototype.slice.call(tb.rows);
  var dir=t.getAttribute('data-sort-col')==String(col) && t.getAttribute('data-sort-dir')=='asc' ? 'desc':'asc';
  t.setAttribute('data-sort-col',col); t.setAttribute('data-sort-dir',dir);
  rows.sort(function(a,b){
    var x=a.cells[col].innerText.trim(), y=b.cells[col].innerText.trim();
    var nx=parseFloat(x), ny=parseFloat(y);
    var bothNum=!isNaN(nx)&&!isNaN(ny)&&x!==''&&y!=='';
    var cmp = bothNum ? nx-ny : x.localeCompare(y);
    return dir=='asc'?cmp:-cmp;
  });
  rows.forEach(function(r){tb.appendChild(r);});
  var ths=t.tHead.rows[0].cells;
  for(var i=0;i<ths.length;i++){var s=ths[i].querySelector('.arrow'); if(s) s.textContent = i==col ? (dir=='asc'?'▲':'▼') : '';}
}
"""


def build_html(runs: list[Run]) -> str:
    n_attempts = sum(len(r.df) for r in runs)
    n_excluded = sum(int(r.df["excluded"].sum()) for r in runs)
    studies = sorted({r.study for r in runs})

    summary = summary_plot(runs)
    summary_html = (
        f'<h2>Reward across all runs</h2>{_img_tag(summary, "reward vs iteration, faceted by dataset")}'
        if summary
        else "<h2>Reward across all runs</h2><p class='meta'>No included attempts to plot.</p>"
    )

    run_blocks = []
    for run in runs:
        plot = per_run_plot(run)
        plot_html = _img_tag(plot, run.label) if plot else "<p class='meta'>No included attempts to plot.</p>"
        run_blocks.append(
            f'<div class="run-block"><h3>{html.escape(run.label)}</h3>'
            f'<p class="meta">{html.escape(run.csv_path)}</p>'
            f'<div class="writeup">{run_writeup(run)}</div>{plot_html}</div>'
        )

    overview = (
        f"<p class='meta'>{len(runs)} run/study file(s) · {n_attempts} attempt(s) · "
        f"{n_excluded} excluded · datasets: {', '.join(html.escape(s) for s in studies) or '—'}</p>"
    )

    heatmap, _, _ = cross_region_heatmap(runs)
    cross_html = (
        f'{_img_tag(heatmap, "best model per run scored across all regions")}'
        "<p class='meta'>Rows: each run's best model. Columns: the region it is scored against. "
        "Note how the V1 column stays high across <em>every</em> run.</p>"
        if heatmap
        else "<p class='meta'>No per-region (corr_&lt;region&gt;) columns found.</p>"
    )

    principle_img, principle_table = principle_attribution(runs)
    principles_html = (
        f'<div class="findings">{_PRINCIPLES_INTRO}</div>'
        + (f'{_img_tag(principle_img, "reward gain attributed to each computational principle")}{principle_table}'
           if principle_img else "<p class='meta'>No improving steps to attribute.</p>")
    )

    dashboard = search_dashboard(runs)
    process_html = (
        f'<div class="findings">{_PROCESS_INTRO}</div>'
        + (_img_tag(dashboard, "search process dashboard") if dashboard
           else "<p class='meta'>Not enough data to plot the search process.</p>")
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autoresearch report</title>
<style>{_CSS}</style></head>
<body>
<h1>neuro-autoresearch report</h1>
{overview}
<h2>What the search found</h2>
<div class="findings">{_FINDINGS_INTRO}</div>
<h2>Computational principles</h2>
{principles_html}
<h2>The search process</h2>
{process_html}
<h2>Cross-region predictivity</h2>
{cross_html}
<h2>Model mechanics — what each run's best model computes</h2>
<p class="meta">Cards are ordered by reward. Each shows the winning model's idea, what it
computes, its cross-region profile, and the full (untrained) source.</p>
{mechanics_section(runs)}
{summary_html}
<h2>Per-run trajectories</h2>
<p class="meta">Per run: test/train correlation and the running-max reward (left axis) with
model size <code>n_params</code> on the right axis, plus which ideas raised
<code>best_so_far</code> (with gain, wall-clock and parameter count).</p>
{''.join(run_blocks) if run_blocks else "<p class='meta'>No runs found.</p>"}
<h2>Attempts</h2>
<p class="meta">Click a header to sort. Expand an idea to read it in full. Excluded rows
(non-ok status or training-related idea) are highlighted and omitted from the plots.</p>
{attempts_table(runs)}
<script>{_JS}</script>
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
