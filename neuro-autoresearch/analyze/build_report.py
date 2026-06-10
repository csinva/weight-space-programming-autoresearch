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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CORE_COLUMNS = [
    "iteration", "timestamp", "study", "model_name", "idea", "n_params",
    "n_train", "n_test", "n_targets", "test_corr", "train_corr", "gap",
    "frac_above_0.1", "best_so_far", "seconds", "status",
]
TABLE_COLUMNS = [
    "iteration", "model_name", "idea", "test_corr", "train_corr", "best_so_far", "status",
]
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

    @property
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
    for col in ("iteration", "test_corr", "train_corr", "best_so_far"):
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


def discover_runs(runs_dir: str, repo_root: str) -> list[Run]:
    """Find every result CSV under ``runs/<tag>/results/`` and ``<repo_root>/results/``."""
    runs: list[Run] = []

    if os.path.isdir(runs_dir):
        for tag in sorted(os.listdir(runs_dir)):
            run_dir = os.path.join(runs_dir, tag)
            if not os.path.isdir(run_dir):
                continue
            metadata = _load_metadata(run_dir)
            csvs = _csvs_in(os.path.join(run_dir, "results")) or _csvs_in(run_dir)
            for csv in csvs:
                df = _read_csv(csv)
                if df is None:
                    continue
                study = df["study"].dropna().iloc[0] if df["study"].notna().any() else os.path.splitext(os.path.basename(csv))[0]
                runs.append(Run(tag, str(study), csv, df, metadata))

    root_results = os.path.join(repo_root, "results")
    if os.path.abspath(root_results) != os.path.abspath(os.path.join(runs_dir, "results")):
        for csv in _csvs_in(root_results):
            df = _read_csv(csv)
            if df is None:
                continue
            study = df["study"].dropna().iloc[0] if df["study"].notna().any() else os.path.splitext(os.path.basename(csv))[0]
            runs.append(Run("results", str(study), csv, df, {}))

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
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _truncate(text: str, n: int = 90) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def run_writeup(run: Run) -> str:
    """Prose summary: ideas tried and which raised ``best_so_far`` (included rows only)."""
    inc = run.included
    total = len(run.df)
    excluded = int(run.df["excluded"].sum())
    if inc.empty:
        return (
            f"<p>No usable attempts (all {total} row(s) excluded: non-ok status or "
            f"training-related idea).</p>"
        )

    improvements = []
    best = float("-inf")
    for _, row in inc.iterrows():
        tc = row["test_corr"]
        if pd.notna(tc) and tc > best:
            best = float(tc)
            improvements.append((int(row["iteration"]), str(row["model_name"]), str(row["idea"]), best))

    parts = [
        f"<p><b>{len(inc)}</b> scored attempt(s)"
        + (f", <b>{excluded}</b> excluded (non-ok status or training-related idea)" if excluded else "")
        + f". Best <code>test_corr</code> = <b>{best:.4f}</b>.</p>"
    ]
    if improvements:
        items = "".join(
            f"<li>iter <b>{it}</b> — <i>{html.escape(name)}</i>: "
            f"{html.escape(_truncate(idea))} → <code>test_corr={val:.4f}</code></li>"
            for it, name, idea, val in improvements
        )
        parts.append(f"<p>Ideas that improved <code>best_so_far</code>:</p><ul>{items}</ul>")
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
                if col == "idea":
                    cells.append(_idea_cell(str(value)))
                elif col in ("test_corr", "train_corr", "best_so_far"):
                    cells.append(f'<td class="num">{_fmt(float(value) if pd.notna(value) else value)}</td>')
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

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autoresearch report</title>
<style>{_CSS}</style></head>
<body>
<h1>neuro-autoresearch report</h1>
{overview}
{summary_html}
<h2>Per-run trajectories</h2>
{''.join(run_blocks) if run_blocks else "<p class='meta'>No runs found.</p>"}
<h2>Attempts</h2>
<p class="meta">Click a header to sort. Expand an idea to read it in full. Excluded rows
(non-ok status or training-related idea) are highlighted and omitted from the plots.</p>
{attempts_table(runs)}
<script>{_JS}</script>
</body></html>
"""


def build_report(runs_dir: str = "runs", out: str = "analyze/report.html", repo_root: str = ".") -> str:
    runs = discover_runs(runs_dir, repo_root)
    html_text = build_html(runs)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(html_text)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="runs", help="directory of per-run output folders (default: runs)")
    parser.add_argument("--repo-root", default=".", help="repo root, scanned for results/<study>.csv (default: .)")
    parser.add_argument("--out", default="analyze/report.html", help="output HTML path (default: analyze/report.html)")
    args = parser.parse_args()

    out = build_report(args.runs_dir, args.out, args.repo_root)
    runs = discover_runs(args.runs_dir, args.repo_root)
    print(f"wrote {out} ({len(runs)} run/study file(s), {sum(len(r.df) for r in runs)} attempt(s))")


if __name__ == "__main__":
    main()
