#!/usr/bin/env python3
"""Render benchmark result JSONs into a single self-contained HTML dashboard.

Pure stdlib only (json, glob, os, argparse, html). No torch, no CDN, no JS.
Reads scripts/profile_readout/results/*.json and writes results/index.html.

Usage:
    python scripts/profile_readout/render_html.py
    python scripts/profile_readout/render_html.py --out path.html --results-dir dir/
"""
import argparse
import glob
import html
import json
import os

# ---------------------------------------------------------------------------
# The warm scholarly theme, embedded verbatim (extracted from the repo docs/).
# ---------------------------------------------------------------------------
STYLE = """
:root{--paper:#f3eee2;--paper-deep:#ece4d3;--card:#fbf8f0;--card-2:#f7f1e4;--ink:#211d17;--ink-soft:#5a5249;--ink-faint:#8b8173;--line:#ddd3bf;--line-soft:#e7ddca;--brand:#1f4fc4;--warm:#b04527;--win:#1f7a4d;--c-V1:#2563eb;--c-VWFA:#9333ea;--serif:'Charter','Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif;--sans:'Seravek','Gill Sans Nova',Ubuntu,Calibri,'Segoe UI',system-ui,sans-serif;--mono:'JetBrains Mono','SF Mono','Cascadia Code',Menlo,Consolas,monospace;}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);background:var(--paper);font-family:var(--sans);font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased;background-image:radial-gradient(var(--line) 1px,transparent 1.4px);background-size:26px 26px;background-attachment:fixed;}
.wrap{position:relative;max-width:1080px;margin:0 auto;padding:0 22px;}
header{padding:34px 0 0;}
.brand{font-family:var(--serif);font-weight:700;font-size:25px;letter-spacing:-.01em;}
.brand .dot{color:var(--brand);}
.brand .sub{color:var(--ink-faint);font-weight:400;}
.spectrum-rule{height:4px;border-radius:3px;margin:16px 0 0;background:linear-gradient(90deg,var(--c-V1),#0891b2,#0d9488,#16a34a,#ca8a04,#ea580c,#dc2626,#be123c,var(--c-VWFA));}
main{padding:14px 0 90px;}
.section{margin:48px 0 0;padding:22px;background:var(--card);border:1px solid var(--line);border-radius:16px;}
h2.sec{font-family:var(--serif);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.16em;color:var(--brand);margin:0 0 4px;}
h3.head{font-family:var(--serif);font-size:26px;line-height:1.15;font-weight:700;letter-spacing:-.015em;margin:.1em 0 .35em;}
.small{font-size:13px;color:var(--ink-faint);}
code{font-family:var(--mono);font-size:.86em;background:var(--card-2);border:1px solid var(--line-soft);padding:1px 5px;border-radius:5px;}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0;}
.chip{background:var(--card-2);border:1px solid var(--line);border-radius:12px;padding:10px 15px;display:flex;flex-direction:column;min-width:110px;}
.chip .v{font-family:var(--serif);font-weight:700;font-size:22px;line-height:1;}
.chip .k{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--ink-faint);margin-top:5px;}
table.ledger{width:100%;border-collapse:collapse;margin-top:14px;font-size:13.5px;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;}
table.ledger th,table.ledger td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line-soft);}
table.ledger th{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-faint);background:var(--card-2);}
table.ledger td:first-child{font-family:var(--mono);font-size:12.5px;white-space:nowrap;}
table.ledger td.num{font-family:var(--mono);text-align:right;color:var(--ink-soft);}
table.ledger tr:last-child td{border-bottom:0;font-weight:700;}
.figure{margin-top:18px;}
.figure .ftitle{font-family:var(--serif);font-size:18px;font-weight:700;}
.bar-chart{margin-top:14px;}
.bar-row{display:flex;align-items:center;gap:12px;margin-bottom:10px;}
.bar-label{font-family:var(--mono);font-size:12px;min-width:110px;text-align:right;color:var(--ink);}
.bar-container{flex:1;height:22px;background:var(--card-2);border-radius:6px;border:1px solid var(--line-soft);overflow:hidden;}
.bar-fill{height:100%;background:linear-gradient(90deg,var(--brand),var(--c-VWFA));}
.bar-value{font-family:var(--mono);font-size:11px;color:var(--ink-soft);min-width:64px;text-align:right;}
.callout{border-left:3px solid var(--warm);background:var(--card-2);border-radius:0 12px 12px 0;padding:14px 18px;margin-top:16px;}
.callout h4{font-family:var(--serif);font-size:17px;margin:0 0 6px;}
.callout p{margin:0;color:var(--ink-soft);font-size:14.5px;}
footer{border-top:1px solid var(--line);margin-top:50px;padding:24px 0 40px;color:var(--ink-faint);font-size:13px;}
"""


# ---------------------------------------------------------------------------
# Small formatting helpers — all defensive against missing / non-numeric data.
# ---------------------------------------------------------------------------
def esc(x):
    """HTML-escape any value, coercing to str first."""
    return html.escape(str(x))


def fnum(x, decimals=3, dash="—"):
    """Format a number with `decimals` places; return dash for None/non-numeric."""
    try:
        if x is None:
            return dash
        return f"{float(x):.{decimals}f}"
    except (TypeError, ValueError):
        return dash


def fcorr(x):
    """Format a correlation to 4 decimals."""
    return fnum(x, 4)


def fint(x, dash="—"):
    """Format an integer-ish value with thousands separators."""
    try:
        if x is None:
            return dash
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return esc(x) if x is not None else dash


# ---------------------------------------------------------------------------
# Card rendering — one benchmark result -> one HTML <section>.
# ---------------------------------------------------------------------------
def render_card(res):
    """Render a single result dict into an HTML section string."""
    # --- header bits (every field via .get with fallbacks) ---------------
    dataset = res.get("dataset", "?")
    subject = res.get("subject", "?")
    timestamp = res.get("timestamp", "?")
    procs = res.get("procs", "?")
    method = res.get("method")  # may be absent in older files
    shapes = res.get("shapes") or {}
    model_src = res.get("model_src", "—")
    contention_ratio = res.get("contention_ratio")

    header_parts = [
        f"{esc(dataset)} · subj {esc(subject)}",
        f"<span class='small'>{esc(timestamp)}</span>",
    ]
    head_html = " &nbsp;·&nbsp; ".join(header_parts)

    # method shown only when present
    method_chip = ""
    if method:
        method_chip = (
            f"<div class='chip'><span class='v'>{esc(method)}</span>"
            f"<span class='k'>method</span></div>"
        )

    # --- metadata chip row ----------------------------------------------
    chips = [
        f"<div class='chip'><span class='v'>{fint(procs)}</span><span class='k'>procs</span></div>",
        f"<div class='chip'><span class='v'>{fint(shapes.get('n_train'))}</span><span class='k'>n_train</span></div>",
        f"<div class='chip'><span class='v'>{fint(shapes.get('n_test'))}</span><span class='k'>n_test</span></div>",
        f"<div class='chip'><span class='v'>{fint(shapes.get('n_targets'))}</span><span class='k'>voxels</span></div>",
        f"<div class='chip'><span class='v'>{fint(shapes.get('d_out'))}</span><span class='k'>D_out</span></div>",
        method_chip,
        f"<div class='chip'><span class='v'>{fnum(contention_ratio, 2)}x</span>"
        f"<span class='k'>contention</span></div>",
    ]
    chips_html = "".join(c for c in chips if c)

    # --- gather concurrent workers --------------------------------------
    concurrent = res.get("concurrent") or {}
    workers = concurrent.get("workers") or []
    phases = res.get("phases") or []

    # If phases missing, infer from the first worker's numeric keys.
    if not phases and workers:
        skip = {"total", "test_corr", "train_corr", "model_src"}
        phases = [k for k in workers[0].keys() if k not in skip]

    # --- phase-timing table ---------------------------------------------
    # columns: phase | worker0 .. workerN | max
    def worker_val(w, key):
        return w.get(key)

    def phase_max(key):
        """Max of a phase's timing across the concurrent workers (None if none are numeric)."""
        vals = [worker_val(w, key) for w in workers]
        nums = [float(v) for v in vals if isinstance(v, (int, float))]
        return max(nums) if nums else None

    header_cells = "<th>phase</th>" + "".join(
        f"<th>w{i}</th>" for i in range(len(workers))
    ) + "<th>max</th>"

    body_rows = []
    for ph in phases:
        cells = [f"<td>{esc(ph)}</td>"]
        for w in workers:
            cells.append(f"<td class='num'>{fnum(worker_val(w, ph), 3)}</td>")
        cells.append(f"<td class='num'>{fnum(phase_max(ph), 3)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    # total row (per-worker totals + max-of-totals)
    total_cells = ["<td>total</td>"]
    for w in workers:
        total_cells.append(f"<td class='num'>{fnum(w.get('total'), 3)}</td>")
    total_cells.append(f"<td class='num'>{fnum(phase_max('total'), 3)}</td>")
    body_rows.append("<tr>" + "".join(total_cells) + "</tr>")

    table_html = (
        "<table class='ledger'><thead><tr>"
        + header_cells
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )

    # --- horizontal bar chart (max-across-concurrent per phase) ---------
    phase_maxes = {ph: phase_max(ph) for ph in phases}
    # total for bar widths = sum of per-phase maxes (so bars sum to ~100%)
    total_for_bars = sum(v for v in phase_maxes.values() if isinstance(v, (int, float)))
    bar_rows = []
    for ph in phases:
        v = phase_maxes.get(ph)
        if isinstance(v, (int, float)) and total_for_bars > 0:
            pct = max(0.0, min(100.0, v / total_for_bars * 100.0))
        else:
            pct = 0.0
        bar_rows.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>{esc(ph)}</div>"
            f"<div class='bar-container'><div class='bar-fill' style='width:{pct:.1f}%'></div></div>"
            f"<div class='bar-value'>{fnum(v, 2)}s</div>"
            "</div>"
        )
    bar_html = (
        "<div class='figure'><div class='ftitle'>Phase time "
        "(max across concurrent workers)</div>"
        "<div class='bar-chart'>" + "".join(bar_rows) + "</div></div>"
    )

    # --- contention callout (only when solo present) --------------------
    callout_html = ""
    solo = res.get("solo")
    if solo:
        solo_workers = solo.get("workers") or []
        # slowest concurrent worker total
        conc_totals = [
            float(w.get("total"))
            for w in workers
            if isinstance(w.get("total"), (int, float))
        ]
        solo_totals = [
            float(w.get("total"))
            for w in solo_workers
            if isinstance(w.get("total"), (int, float))
        ]
        if conc_totals and solo_totals:
            slow_conc = max(conc_totals)
            fast_solo = min(solo_totals)
            ratio = slow_conc / fast_solo if fast_solo else None
            ratio_txt = f" ({fnum(ratio, 2)}x)" if ratio is not None else ""
            callout_html = (
                "<div class='callout'><h4>Contention under concurrency</h4>"
                f"<p>Slowest concurrent worker {fnum(slow_conc, 2)}s vs "
                f"{fnum(fast_solo, 2)}s solo{ratio_txt}.</p></div>"
            )

    # --- assemble section ------------------------------------------------
    return (
        "<section class='section'>"
        f"<h2 class='sec'>benchmark result</h2>"
        f"<h3 class='head'>{head_html}</h3>"
        f"<div class='small'>model_src: <code>{esc(model_src)}</code></div>"
        f"<div class='chips'>{chips_html}</div>"
        + table_html
        + bar_html
        + callout_html
        + "</section>"
    )


# ---------------------------------------------------------------------------
# Page assembly + I/O.
# ---------------------------------------------------------------------------
HEADER_HTML = (
    '<header><div class="wrap"><div class="brand">neuro<span class="dot">·</span>'
    'autoresearch <span class="sub">/ readout benchmark</span></div>'
    '<div class="spectrum-rule"></div></div></header>'
)


def render_page(results):
    """Wrap rendered cards in a full HTML document."""
    if results:
        cards = "".join(render_card(r) for r in results)
        body = f'<main><div class="wrap">{cards}</div></main>'
    else:
        body = (
            '<main><div class="wrap"><section class="section">'
            '<h2 class="sec">benchmark result</h2>'
            '<h3 class="head">No results yet</h3>'
            '<p class="small">Run <code>python scripts/profile_readout/profile.py</code> '
            'to produce result JSONs, then re-render.</p>'
            '</section></div></main>'
        )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>neuro·autoresearch · readout benchmark</title>"
        f"<style>{STYLE}</style></head><body>"
        + HEADER_HTML
        + body
        + '<footer><div class="wrap">Generated by render_html.py · '
        f'{esc(len(results))} result(s)</div></footer>'
        "</body></html>"
    )


def load_results(results_dir):
    """Load all result JSONs, sorted most-recent first by timestamp."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    out = []
    for p in paths:
        try:
            with open(p, "r") as f:
                data = json.load(f)
            data["_path"] = p  # keep provenance (unused in render, handy for debug)
            out.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"warning: skipping {p}: {e}")
    # most recent first; missing timestamps sort last
    out.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return out


def main():
    """Load every benchmark results JSON and write the self-contained HTML dashboard."""
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(here, "results")
    ap = argparse.ArgumentParser(description="Render benchmark results to HTML.")
    ap.add_argument("--results-dir", default=default_dir,
                    help="directory of result JSONs (default: scripts/profile_readout/results)")
    ap.add_argument("--out", default=None,
                    help="output HTML path (default: <results-dir>/index.html)")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.results_dir, "index.html")
    results = load_results(args.results_dir)
    page = render_page(results)
    with open(out_path, "w") as f:
        f.write(page)
    print(out_path)


if __name__ == "__main__":
    main()
