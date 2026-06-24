#!/usr/bin/env python3
"""Turn the proposer-pool search DBs into ONE self-contained, interactive HTML report.

A *search run* lives in ``runs/<dataset>__<date>/`` -- a single flat ``models`` table in
``search.db`` (owned by ``lab.py``) plus a sandbox per proposer::

    search.db                      one row per model: id, idea, summary, proposer, dataset,
                                   status, test_corr, train_corr, gap, n_params, seconds,
                                   created_at, path
    models/<id>.py                 the immutable, never-trained source for each row
    proposers/pNN/direction.txt    that proposer's self-chosen specialization (one line)
    proposers/pNN/proposer.log     its full agent trace -- reasoning + the commands it ran

The reward is one dataset-level ``test_corr`` and the live dimension is ``proposer``: ten headless
agents searching the SAME space in parallel, each chasing a different front. This report makes that
process explorable:

  * header + global trajectory (running-best reward over the search);
  * whether capacity (``n_params``) or overfit (the train/test ``gap``) pays;
  * a **proposer roster** -- what front each agent is working and how it is doing;
  * an **activity timeline** -- every model as a dot on a per-proposer lane over wall-clock,
    coloured by reward, so you can watch the parallel search unfold;
  * a **proposer explorer** -- toggle through the agents to read each one's models (idea + score +
    source) and its activity log rendered as a readable, filterable timeline.

The output HTML is dependency-free: the shared CSS and the vendored Plotly cartesian bundle are
inlined, charts are inline Plotly specs, and the only scripting is a few lines of vanilla JS for the
tabs and the log filter.

Usage::

    python analyze/report.py                       # every runs/*/search.db -> analyze/report.html
    python analyze/report.py --run runs/nsd__2026-06-24
    python analyze/report.py --out docs/index.html -k 6 --top-src 5
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import sqlite3
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_PLOTLY_BUNDLE = os.path.join(HERE, "vendor", "plotly-cartesian.min.js")
_CSS = os.path.join(HERE, "assets", "report.css")

# Stable, colour-blind-friendly palette for proposers / runs.
PALETTE = ["#1e40af", "#b45309", "#166534", "#9d174d", "#5b21b6", "#0e7490",
           "#9a3412", "#3f6212", "#a21caf", "#1d4ed8", "#854d0e", "#0f766e"]
_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


# --------------------------------------------------------------------------- helpers
def esc(s) -> str:
    return html.escape("" if s is None else str(s))


def fnum(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "–"


def color_of(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _pretty_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%b %d %Y, %H:%M")
    except (TypeError, ValueError):
        return iso or "–"


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, TypeError):
        return ""


# --------------------------------------------------------------------------- log parsing
_SESSION_RE = re.compile(r"^\[proposer\]\s+(\S+)\s+session start\s+(\S+)(?:\s+\(([^)]*)\))?")


def parse_log(text: str) -> str:
    """Render a proposer's raw agent log as a compact, readable timeline.

    The log interleaves three things: ``[proposer] ... session start`` markers, ``● Title (kind)``
    tool calls (with ``│`` command lines and a ``└ N lines…`` output stub -- the real output is
    elided), and free prose (the agent's reasoning). We group consecutive prose into paragraphs and
    fold each tool call into a ``<details>`` so the trace reads like a story but stays scannable.
    Every entry carries its text in ``data-q`` so the client-side filter can hide non-matches."""
    if not text.strip():
        return '<p class="meta">No log recorded for this proposer.</p>'
    lines = text.splitlines()
    out: list[str] = []
    prose: list[str] = []
    i, n = 0, len(lines)

    def flush_prose():
        if prose:
            joined = " ".join(s.strip() for s in prose if s.strip())
            if joined:
                out.append(f'<p class="log-prose" data-q="{esc(joined.lower())}">{esc(joined)}</p>')
            prose.clear()

    while i < n:
        line = lines[i]
        m = _SESSION_RE.match(line.strip())
        if m:
            flush_prose()
            ts, meta = m.group(2), m.group(3) or ""
            out.append(f'<div class="log-session" data-q="session start">'
                       f'<span class="ls-dot"></span>session start '
                       f'<span class="meta">{esc(ts)}{" · " + esc(meta) if meta else ""}</span></div>')
            i += 1
            continue
        if line.startswith("● "):
            flush_prose()
            title = line[2:].strip()
            cmd: list[str] = []
            outstub = ""
            i += 1
            while i < n and (lines[i].startswith("  │") or lines[i].startswith("  └")
                             or lines[i].startswith("│") or lines[i].startswith("└")):
                s = lines[i].lstrip()
                if s.startswith("│"):
                    cmd.append(s[1:].strip())
                elif s.startswith("└"):
                    outstub = s[1:].strip()
                i += 1
            cmd_txt = "\n".join(cmd).strip()
            q = (title + " " + cmd_txt).lower()
            body = (f'<pre class="log-cmd"><code>{esc(cmd_txt)}</code></pre>' if cmd_txt else "")
            stub = f'<span class="log-out">{esc(outstub)}</span>' if outstub else ""
            if body:
                out.append(f'<details class="log-tool" data-q="{esc(q)}">'
                           f'<summary>{esc(title)} {stub}</summary>{body}</details>')
            else:
                out.append(f'<div class="log-tool log-tool-bare" data-q="{esc(q)}">'
                           f'{esc(title)} {stub}</div>')
            continue
        prose.append(line)
        i += 1
    flush_prose()
    return "\n".join(out)


# --------------------------------------------------------------------------- data model
class Proposer:
    """One agent in the pool: its self-chosen ``direction``, the models it proposed (scored, in
    order), and its parsed activity log."""

    def __init__(self, pid: str, run_dir: str, ok_rows: list[dict], all_rows: list[dict]):
        self.pid = pid
        self.dir = os.path.join(run_dir, "proposers", pid)
        self.direction = _read_text(os.path.join(self.dir, "direction.txt")).strip()
        self.log_text = _read_text(os.path.join(self.dir, "proposer.log"))
        self.ok = ok_rows                       # scored, ordered by created_at
        self.all = all_rows                     # every row it wrote (any status)

    @property
    def scores(self) -> list[float]:
        return [float(r["test_corr"]) for r in self.ok]

    @property
    def best(self) -> float | None:
        return max(self.scores) if self.scores else None

    @property
    def mean(self) -> float | None:
        return sum(self.scores) / len(self.scores) if self.scores else None

    @property
    def errors(self) -> int:
        return sum(1 for r in self.all if r.get("status") == "error")


class Run:
    """One ``search.db`` + its sandbox: rows, the scored subset in proposal order, and proposers."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.dir = os.path.dirname(db_path)
        self.name = os.path.basename(self.dir) or "run"
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            self.rows = [dict(r) for r in con.execute("SELECT * FROM models")]
        finally:
            con.close()
        self.dataset = next((r.get("dataset") for r in self.rows if r.get("dataset")), "")
        self.ok = sorted(
            (r for r in self.rows if r.get("status") == "ok" and r.get("test_corr") is not None),
            key=lambda r: (r.get("created_at") or "", r.get("id") or ""))
        running = 0.0
        for r in self.ok:
            running = max(running, float(r["test_corr"]))
            r["_best_so_far"] = running
        self.proposers = self._build_proposers()

    def _build_proposers(self) -> list[Proposer]:
        pids = sorted({(r.get("proposer") or "?") for r in self.rows})
        props = []
        for pid in pids:
            ok = [r for r in self.ok if (r.get("proposer") or "?") == pid]
            allr = [r for r in self.rows if (r.get("proposer") or "?") == pid]
            props.append(Proposer(pid, self.dir, ok, allr))
        props.sort(key=lambda p: (p.best if p.best is not None else -1), reverse=True)
        return props

    @property
    def tally(self) -> dict:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.get("status") or "?"] = out.get(r.get("status") or "?", 0) + 1
        return out

    @property
    def champion(self) -> dict | None:
        return max(self.ok, key=lambda r: r["test_corr"]) if self.ok else None

    @property
    def span(self) -> str:
        stamps = sorted(r.get("created_at") or "" for r in self.rows if r.get("created_at"))
        if not stamps:
            return "–"
        try:
            secs = (datetime.fromisoformat(stamps[-1]) - datetime.fromisoformat(stamps[0])).total_seconds()
            return f"{secs / 3600:.1f} h" if secs >= 3600 else f"{secs / 60:.0f} min"
        except ValueError:
            return "–"


def discover_runs(runs_dir: str, only: str | None) -> list[Run]:
    if only:
        db = only if only.endswith(".db") else os.path.join(only, "search.db")
        dbs = [db] if os.path.exists(db) else []
    else:
        dbs = sorted(glob.glob(os.path.join(runs_dir, "*", "search.db")))
    return [r for r in (Run(db) for db in dbs) if r.rows]


# --------------------------------------------------------------------------- plotly
def _load_plotly() -> str:
    bundle = _read_text(_PLOTLY_BUNDLE)
    bundle = re.sub(r"https://cdn\.plot\.ly/?", "", bundle)
    if "cdn.plot.ly" in bundle:
        raise RuntimeError("vendored Plotly still references the CDN; revendor the offline bundle")
    return bundle


_LAYOUT_BASE = {
    "paper_bgcolor": "#ffffff", "plot_bgcolor": "#ffffff",
    "font": {"family": "Seravek, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
             "size": 12, "color": "#2b2f36"},
    "margin": {"l": 60, "r": 24, "t": 30, "b": 48},
    "xaxis": {"gridcolor": "#efede7", "zerolinecolor": "#e4e2dc", "linecolor": "#e4e2dc"},
    "yaxis": {"gridcolor": "#efede7", "zerolinecolor": "#e4e2dc", "linecolor": "#e4e2dc"},
    "legend": {"font": {"size": 11}, "bgcolor": "rgba(255,255,255,.6)"},
    "hoverlabel": {"font": {"family": "ui-monospace, Menlo, monospace", "size": 11}},
}


def _merge(layout: dict) -> dict:
    out = json.loads(json.dumps(_LAYOUT_BASE))
    for k, v in layout.items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


_PLOT_SEQ = [0]


def plot(data: list[dict], layout: dict, *, height: int = 380, caption: str = "") -> str:
    _PLOT_SEQ[0] += 1
    pid = f"plot{_PLOT_SEQ[0]}"
    cap = f'<div class="chart-caption">{caption}</div>' if caption else ""
    return (f'<div class="chart-card"><div id="{pid}" class="chart-plot" '
            f'style="height:{height}px"></div>{cap}</div>'
            f'<script>Plotly.newPlot("{pid}",{json.dumps(data)},'
            f'{json.dumps(_merge(layout))},{json.dumps(_PLOTLY_CONFIG)});</script>')


def sparkline(scores: list[float], lo: float, hi: float, color: str) -> str:
    """A tiny inline-SVG sparkline for a roster card (no Plotly, cheap): faint = each model's
    reward, bold = the running best."""
    if not scores:
        return '<svg class="spark" viewBox="0 0 120 28"></svg>'
    run, best = [], 0.0
    for s in scores:
        best = max(best, s)
        run.append(best)
    w, h, span = 120, 28, (hi - lo) or 1.0

    def xy(seq):
        return " ".join(f"{(j / max(len(seq) - 1, 1)) * w:.1f},"
                        f"{h - ((v - lo) / span) * (h - 4) - 2:.1f}" for j, v in enumerate(seq))
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polyline points="{xy(scores)}" fill="none" stroke="{color}" stroke-width="0.6" '
            f'stroke-opacity="0.35"/>'
            f'<polyline points="{xy(run)}" fill="none" stroke="{color}" stroke-width="1.6"/></svg>')


# --------------------------------------------------------------------------- charts
def trajectory_chart(run: Run) -> str:
    """Reward per scored model in proposal order, coloured by proposer, with a black running-best
    line. Click a proposer in the legend to isolate its points."""
    data: list[dict] = []
    for i, p in enumerate(run.proposers):
        pts = [(j, r) for j, r in enumerate(run.ok, 1) if (r.get("proposer") or "?") == p.pid]
        if not pts:
            continue
        data.append({
            "type": "scatter", "mode": "markers", "name": p.pid,
            "x": [j for j, _ in pts], "y": [r["test_corr"] for _, r in pts],
            "text": [f"{r['id']} · {(r.get('idea') or '')[:70]}" for _, r in pts],
            "hovertemplate": "%{text}<br>test_corr=%{y:.4f}<extra>" + p.pid + "</extra>",
            "marker": {"size": 6, "color": color_of(i), "opacity": 0.8},
        })
    data.append({
        "type": "scatter", "mode": "lines", "name": "best so far",
        "x": list(range(1, len(run.ok) + 1)), "y": [r["_best_so_far"] for r in run.ok],
        "line": {"color": "#16181d", "width": 2.5}, "hoverinfo": "skip",
    })
    layout = {"xaxis": {"title": {"text": "scored model (proposal order)"}},
              "yaxis": {"title": {"text": "test_corr (reward)"}},
              "legend": {"orientation": "h", "y": -0.18}}
    return plot(data, layout, height=420,
                caption="Each dot is one evaluated model, coloured by the agent that proposed it; the "
                        "bold line is the best reward reached so far. Click a proposer in the legend to isolate it.")


def complexity_chart(run: Run) -> str:
    """Reward vs model *source size* (``model_loc`` — lines of code defining the model, captured by
    the proposer at propose time and stored in the DB). The declared ``n_params`` is 0 for these
    parameter-free / random-but-fixed transforms, so the complexity that actually varies is how much
    code each agent wrote — and it tracks reward."""
    rows = [r for r in run.ok if r.get("model_loc") is not None]
    xs = [int(r["model_loc"]) for r in rows]
    ys = [r["test_corr"] for r in rows]
    note = ""
    if len(xs) > 2:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5
        sy = sum((y - my) ** 2 for y in ys) ** 0.5
        if sx and sy:
            note = f" Reward rises with code size (r≈{cov / (sx * sy):.2f}) — more feature families, not trained capacity."
    data = [{
        "type": "scatter", "mode": "markers", "name": "models",
        "x": xs, "y": ys,
        "text": [f"{r['id']} · {(r.get('idea') or '')[:60]}" for r in rows],
        "hovertemplate": "%{text}<br>source=%{x} lines<br>test_corr=%{y:.4f}<extra></extra>",
        "marker": {"size": 6, "color": "#1e40af", "opacity": 0.55},
    }]
    layout = {"xaxis": {"title": {"text": "model_loc (lines of code defining the model)"}},
              "yaxis": {"title": {"text": "test_corr"}}, "showlegend": False}
    return plot(data, layout, height=340,
                caption="Reward vs model code size (<code>model_loc</code>; the declared "
                        "<code>n_params</code> stays 0 — these transforms are parameter-free)." + note)


def generalization_chart(run: Run) -> str:
    rows = [r for r in run.ok if r.get("train_corr") is not None]
    xs = [r["train_corr"] for r in rows]
    ys = [r["test_corr"] for r in rows]
    lo = min(xs + ys) if xs else 0.0
    hi = max(xs + ys) if xs else 1.0
    pad = max((hi - lo) * 0.05, 0.01)
    data = [{
        "type": "scatter", "mode": "markers", "name": "models", "x": xs, "y": ys,
        "text": [r["id"] for r in rows],
        "hovertemplate": "%{text}<br>train=%{x:.4f}<br>test=%{y:.4f}<extra></extra>",
        "marker": {"size": 6, "color": "#9a3412", "opacity": 0.55},
    }, {
        "type": "scatter", "mode": "lines", "name": "train = test",
        "x": [lo - pad, hi + pad], "y": [lo - pad, hi + pad],
        "line": {"color": "#9aa1ab", "width": 1, "dash": "dot"}, "hoverinfo": "skip",
    }]
    layout = {"xaxis": {"title": {"text": "train_corr"}},
              "yaxis": {"title": {"text": "test_corr"}}, "showlegend": False}
    return plot(data, layout, height=340,
                caption="Distance below the dotted diagonal is the train/test gap. Points hugging it generalize cleanly.")


def activity_timeline(run: Run) -> str:
    """Every scored model as a dot: x = wall-clock, y = proposer lane, colour = reward. Watch ten
    agents search the same space in parallel and see where the good models surfaced."""
    order = [p.pid for p in run.proposers]
    data = [{
        "type": "scatter", "mode": "markers",
        "x": [r.get("created_at") for r in run.ok],
        "y": [(r.get("proposer") or "?") for r in run.ok],
        "text": [f"{r['id']} · {(r.get('idea') or '')[:70]}" for r in run.ok],
        "customdata": [fnum(r["test_corr"]) for r in run.ok],
        "hovertemplate": "%{text}<br>%{x}<br>test_corr=%{customdata}<extra>%{y}</extra>",
        "marker": {
            "size": 8, "color": [r["test_corr"] for r in run.ok], "colorscale": "Viridis",
            "showscale": True, "colorbar": {"title": {"text": "test_corr"}, "thickness": 12},
            "line": {"width": 0.4, "color": "#ffffff"}, "opacity": 0.9,
        },
    }]
    layout = {"xaxis": {"title": {"text": "wall-clock"}, "type": "date"},
              "yaxis": {"categoryorder": "array", "categoryarray": order[::-1],
                        "title": {"text": "proposer"}},
              "showlegend": False}
    return plot(data, layout, height=max(320, 34 * len(order) + 90),
                caption="One dot per scored model on its proposer's lane, brighter = higher reward. "
                        "The decentralized pool, unfolding over time.")


# --------------------------------------------------------------------------- html building blocks
def stat(value: str, label: str) -> str:
    return (f'<div class="stat"><div class="stat-value">{value}</div>'
            f'<div class="stat-label">{esc(label)}</div></div>')


def _read_source_cached(path: str, _cache: dict = {}) -> str:
    if path not in _cache:
        _cache[path] = _read_text(path) or "# source unavailable"
    return _cache[path]


def model_row(r: dict, rank: int, with_src: bool, champ_id: str | None) -> str:
    idea = esc(r.get("idea") or "(no idea recorded)")
    cls = " good" if r["id"] == champ_id else ""
    np_extra = f" · {esc(r['n_params'])} params" if r.get("n_params") else ""
    loc = r.get("model_loc")
    size = f"{loc} lines" if loc is not None else "– lines"
    meta = (f"{size}{np_extra} · gap {fnum(r.get('gap'))} · "
            f"{fnum(r.get('seconds'), 1)}s · {_pretty_date(r.get('created_at'))}")
    head = (f"<span class='m-rank'>#{rank}</span>"
            f"<span class='m-score{cls}'>{fnum(r['test_corr'])}</span>"
            f"<span class='m-id'>{esc(r['id'])}</span><span class='m-idea'>{idea}</span>")
    meta_div = f"<div class='m-meta meta'>{meta}</div>"
    if with_src:
        src = _read_source_cached(r.get("path") or "")
        return (f"<details class='m-item code'><summary><div class='m-head'>{head}</div>{meta_div}"
                f"</summary><pre><code>{esc(src)}</code></pre></details>")
    return f"<div class='m-item'><div class='m-head'>{head}</div>{meta_div}</div>"


def proposer_roster(run: Run, ri: int) -> str:
    """A card per agent: its direction, headline stats, and a running-best sparkline. Click to jump
    into that agent in the explorer below."""
    scores_all = [s for p in run.proposers for s in p.scores]
    lo, hi = (min(scores_all), max(scores_all)) if scores_all else (0.0, 1.0)
    champ = run.champion
    cards = []
    for i, p in enumerate(run.proposers):
        c = color_of(i)
        is_champ = champ and (champ.get("proposer") or "?") == p.pid
        direction = esc(p.direction) if p.direction else "<span class='meta'>(no stated direction)</span>"
        crown = ' <span class="lead-star" title="run champion">★</span>' if is_champ else ""
        cards.append(
            f'<button class="prop-card" style="border-top-color:{c}" '
            f'onclick="pickTab(\'r{ri}\',\'{p.pid}\')">'
            f'<div class="pc-head"><span class="pc-id" style="color:{c}">{esc(p.pid)}{crown}</span>'
            f'<span class="pc-best">{fnum(p.best)}</span></div>'
            f'<div class="pc-dir">{direction}</div>'
            f'{sparkline(p.scores, lo, hi, c)}'
            f'<div class="pc-foot meta">{len(p.ok)} scored · mean {fnum(p.mean)}'
            f'{" · " + str(p.errors) + " err" if p.errors else ""}</div></button>')
    return f'<div class="roster">{"".join(cards)}</div>'


def proposer_explorer(run: Run, ri: int, top_src: int) -> str:
    """The toggle: a tab per agent. The selected panel shows the agent's direction, its models
    (idea + score, top-N with expandable source), and its activity log as a filterable timeline."""
    champ = run.champion
    champ_id = champ["id"] if champ else None
    tabs, panels = [], []
    for i, p in enumerate(run.proposers):
        c = color_of(i)
        active = " active" if i == 0 else ""
        is_champ = champ and (champ.get("proposer") or "?") == p.pid
        star = ' <span class="lead-star">★</span>' if is_champ else ""
        tabs.append(
            f'<button class="tab{active}" id="tab-r{ri}-{p.pid}" '
            f'style="--tabc:{c}" onclick="pickTab(\'r{ri}\',\'{p.pid}\')">'
            f'{esc(p.pid)}{star} <span class="tab-best">{fnum(p.best)}</span></button>')

        ranked = sorted(p.ok, key=lambda r: r["test_corr"], reverse=True)
        model_html = "".join(
            model_row(r, rank, with_src=(rank <= top_src), champ_id=champ_id)
            for rank, r in enumerate(ranked, 1)) or "<p class='meta'>No scored models yet.</p>"

        direction = esc(p.direction) if p.direction else "no stated direction"
        stats = (f"{len(p.ok)} scored · best {fnum(p.best)} · mean {fnum(p.mean)} · "
                 f"{p.errors} errors · {len(p.all)} total proposals")
        log_html = parse_log(p.log_text)
        panels.append(
            f'<div class="tab-panel{active}" id="panel-r{ri}-{p.pid}">'
            f'<div class="panel-direction" style="border-left-color:{c}">'
            f'<span class="pd-tag">direction</span>{direction}</div>'
            f'<p class="meta">{stats}</p>'
            f'<div class="panel-cols">'
            f'<section class="panel-models"><h3>Models proposed '
            f'<span class="meta">(top {top_src} expand to source)</span></h3>'
            f'<div class="m-scroll">{model_html}</div></section>'
            f'<section class="panel-log"><div class="log-toolbar"><h3>Activity log</h3>'
            f'<input type="search" class="log-filter" placeholder="filter log…" '
            f'oninput="filterLog(this,\'r{ri}\',\'{p.pid}\')"></div>'
            f'<div class="log-timeline" id="log-r{ri}-{p.pid}">{log_html}</div></section>'
            f'</div></div>')
    return (f'<div class="explorer"><div class="tabs" role="tablist">{"".join(tabs)}</div>'
            f'<div class="tab-panels">{"".join(panels)}</div></div>')


def champion_cards(run: Run, k: int) -> str:
    top = sorted(run.ok, key=lambda r: r["test_corr"], reverse=True)[:k]
    cards = []
    for r in top:
        src = _read_source_cached(r.get("path") or "")
        np_extra = f" · {esc(r['n_params'])} params" if r.get("n_params") else ""
        loc = r.get("model_loc")
        size = f"{loc} lines" if loc is not None else "– lines"
        meta = (f"proposer {esc(r.get('proposer') or '?')} · {size}"
                f"{np_extra} · train {fnum(r.get('train_corr'))} · gap {fnum(r.get('gap'))} · "
                f"{fnum(r.get('seconds'), 1)} s")
        summary = f"<p>{esc(r['summary'])}</p>" if r.get("summary") else ""
        cards.append(
            f'<div class="card"><span class="reward">{fnum(r["test_corr"])}</span>'
            f'<h3>{esc(r.get("idea") or "(no idea recorded)")}</h3>{summary}'
            f'<p class="meta">{meta}</p>'
            f'<details class="code"><summary>model.py — {esc(r["id"])}</summary>'
            f'<pre><code>{esc(src)}</code></pre></details></div>')
    return f'<div class="cards-grid">{"".join(cards)}</div>'


# --------------------------------------------------------------------------- run section
def header_section(runs: list[Run]) -> str:
    n_models = sum(len(r.rows) for r in runs)
    n_ok = sum(len(r.ok) for r in runs)
    n_prop = len({p.pid for r in runs for p in r.proposers})
    champ = max((r.champion for r in runs if r.champion), key=lambda c: c["test_corr"], default=None)
    datasets = sorted({r.dataset for r in runs if r.dataset})
    fails = sum(v for r in runs for k, v in r.tally.items() if k == "error")
    stats = [stat(str(len(runs)), "search runs"), stat(str(n_models), "models proposed"),
             stat(str(n_ok), "scored ok"), stat(str(n_prop), "proposers"),
             stat(str(fails), "errors")]
    if champ:
        stats.append(stat(f"{fnum(champ['test_corr'])} <small>{esc(champ['id'])}</small>",
                          "champion reward"))
    sub = ("Ten headless agents propose never-trained models in parallel; each is scored by a ridge "
           "read-out against recorded brain activity, and held-out <code>test_corr</code> is the "
           "reward. Coverage: " + (", ".join(esc(d) for d in datasets) or "–") + ".")
    return (f'<header class="report-header"><h1>Autoresearch search report</h1>'
            f'<p class="subtitle">{sub}</p><div class="stat-row">{"".join(stats)}</div></header>')


def run_section(run: Run, ri: int, k: int, top_src: int, multi: bool) -> str:
    parts = []
    if multi:
        champ = run.champion
        best = fnum(champ["test_corr"]) if champ else "–"
        parts.append(f'<h2 id="run-{ri}">{esc(run.name)}</h2>'
                     f'<p class="section-sub">champion {best} · {len(run.ok)} scored · {run.span} · '
                     f'{len(run.proposers)} proposers</p>')
    if not run.ok:
        parts.append('<p class="meta">No scored models in this run yet.</p>')
        return "".join(parts)

    def sec(hid, title, sub):
        return f'<h2 id="{hid}-{ri}">{title}</h2><p class="section-sub">{sub}</p>'

    parts.append(sec("trajectory", "The search trajectory",
                     "Reward of every scored model in proposal order, coloured by agent, with the running best."))
    parts.append(trajectory_chart(run))
    parts.append(sec("roster", "What each proposer is exploring",
                     "Ten agents, ten different fronts. Each card is one agent's specialization and how it is doing — click to open it below."))
    parts.append(proposer_roster(run, ri))
    parts.append(sec("timeline", "When the search happened",
                     "Every scored model on its agent's lane over wall-clock, brighter = higher reward."))
    parts.append(activity_timeline(run))
    parts.append(sec("pays", "Does complexity or overfit pay?",
                     "Reward against model code size (<code>model_loc</code>, set by the proposer; the declared "
                     "<code>n_params</code> stays 0 for these parameter-free transforms) and against the train/test gap."))
    parts.append(f'<div class="dashboard-grid">{complexity_chart(run)}{generalization_chart(run)}</div>')
    parts.append(sec("explorer", "Proposer explorer",
                     "Toggle through the agents: read each one's models (top expand to full source) and its activity log as a filterable timeline."))
    parts.append(proposer_explorer(run, ri, top_src))
    parts.append(sec("champions", "Champion models",
                     "The highest-reward models across the whole run, with full never-trained source."))
    parts.append(champion_cards(run, k))
    return "".join(parts)


# --------------------------------------------------------------------------- assemble
_JS = """
function pickTab(scope, pid){
  document.querySelectorAll('[id^="tab-'+scope+'-"]').forEach(function(b){
    b.classList.toggle('active', b.id==='tab-'+scope+'-'+pid); });
  document.querySelectorAll('[id^="panel-'+scope+'-"]').forEach(function(p){
    p.classList.toggle('active', p.id==='panel-'+scope+'-'+pid); });
  var el=document.getElementById('tab-'+scope+'-'+pid);
  if(el){ el.scrollIntoView({block:'nearest', inline:'center', behavior:'smooth'}); }
}
function filterLog(input, scope, pid){
  var q=input.value.trim().toLowerCase();
  var root=document.getElementById('log-'+scope+'-'+pid);
  if(!root) return;
  root.querySelectorAll('[data-q]').forEach(function(n){
    n.style.display = (!q || n.getAttribute('data-q').indexOf(q)>=0) ? '' : 'none'; });
}
"""


def _toc(runs: list[Run], multi: bool) -> str:
    if multi:
        links = "".join(f'<a href="#run-{i}">{esc(r.name)}</a>' for i, r in enumerate(runs))
    else:
        links = ('<a href="#trajectory-0">Trajectory</a><a href="#roster-0">Proposers</a>'
                 '<a href="#timeline-0">Timeline</a><a href="#pays-0">What pays</a>'
                 '<a href="#explorer-0">Explorer</a><a href="#champions-0">Champions</a>')
    return f'<nav class="toc"><b>Report</b>{links}</nav>'


def build(runs: list[Run], k: int, top_src: int) -> str:
    css = _read_text(_CSS)
    plotly = _load_plotly()
    if not runs:
        body = ('<header class="report-header"><h1>Autoresearch search report</h1>'
                '<p class="subtitle">No <code>runs/*/search.db</code> found yet — launch a search first.</p></header>')
        return _shell(css, plotly, body)
    multi = len(runs) > 1
    runs = sorted(runs, key=lambda r: r.name, reverse=True)
    parts = [header_section(runs), _toc(runs, multi)]
    for ri, run in enumerate(runs):
        parts.append(run_section(run, ri, k, top_src, multi))
    return _shell(css, plotly, "".join(parts))


def _shell(css: str, plotly: str, body: str) -> str:
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Autoresearch search report</title>"
            f"<style>{css}</style><script>{plotly}</script></head>"
            f"<body><main class='page'>{body}</main><script>{_JS}</script></body></html>")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the autoresearch search report from runs/*/search.db.")
    p.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    p.add_argument("--run", default=None,
                   help="a single run dir or search.db (default: every runs/*/search.db)")
    p.add_argument("--out", default=os.path.join(HERE, "report.html"))
    p.add_argument("-k", "--champions", type=int, default=4, help="champion cards per run")
    p.add_argument("--top-src", type=int, default=3,
                   help="per proposer, how many top models embed expandable source")
    a = p.parse_args(argv)

    runs = discover_runs(a.runs_dir, a.run)
    html_out = build(runs, a.champions, a.top_src)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html_out)
    size = len(html_out) / 1024
    tally = ", ".join(f"{r.name}: {len(r.ok)} scored / {len(r.proposers)} proposers" for r in runs) or "no runs"
    print(f"wrote {a.out}  ({size:.0f} KB; {tally})")


if __name__ == "__main__":
    main()
