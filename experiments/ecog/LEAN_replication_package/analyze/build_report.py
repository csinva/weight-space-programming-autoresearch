# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build an interactive HTML report from runs/*/results/overall_results.csv.

For each task it shows accuracy vs. parameter count (points numbered by the
order tried and connected along the search path) plus an auto-generated writeup
of what was tried and which circuits worked / didn't. A summary plot at the
bottom overlays every task's search path on one chart.
"""
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "analyze" / "report.html"

CATEGORY = {
    "addition-five-digits": "Easy",
    "multiplication-five-digits": "Easy",
    "sort-five-digits": "Easy",
    "digit-counting-10": "Easy",
    "parity-upto10-bits": "Easy",
    "boolean-circuit-5-bits": "Easy",
    "linear-interpolation-two-points": "Easy",
    "word-reversal-3x3": "Harder",
    "gcd-three-digits": "Harder",
    "decimal-to-binary-8bit": "Harder",
    "sentiment-sst2": "Real world",
    "paraphrase-mrpc": "Real world",
    "nli-snli": "Real world",
}
CAT_ORDER = ["Easy", "Harder", "Real world"]


def fmt_params(p: float) -> str:
    if p >= 1e6:
        return f"{p/1e6:.2f}M"
    if p >= 1e3:
        return f"{p/1e3:.1f}K"
    return f"{p:.0f}"


def make_writeup(task: str, methods: list[dict]) -> str:
    """Generate a grounded narrative from the per-method descriptions."""
    accs = [m["acc"] for m in methods]
    best_acc = max(accs)
    best = next(m for m in methods if m["acc"] == best_acc)
    base = methods[0]
    n = len(methods)
    thresh = best_acc - 0.01  # "near-best"
    breakthrough = next((m for m in methods if m["acc"] >= thresh), best)
    good = [m for m in methods if m["acc"] >= thresh]
    compact = min(good, key=lambda m: m["params"])
    failed = [m for m in methods if m["status"] == "failed" or m["acc"] < 0.1]

    def esc(s):
        return html.escape(str(s))

    parts = [
        f"<b>{n}</b> hand-written transformer circuit{'s' if n > 1 else ''} were tried for "
        f"<b>{esc(task)}</b>. Accuracy ranged from <b>{min(accs):.1%}</b> to a best of "
        f"<b>{best_acc:.1%}</b>.",
        f"The first attempt (#1, <b>{esc(base['name'])}</b>, {fmt_params(base['params'])} params) "
        f"reached {base['acc']:.1%} &mdash; {esc(base['desc'])}",
    ]
    if breakthrough["i"] != 1:
        parts.append(
            f"The breakthrough came at attempt #{breakthrough['i']} "
            f"(<b>{esc(breakthrough['name'])}</b>), which hit {breakthrough['acc']:.1%}: "
            f"{esc(breakthrough['desc'])}"
        )
    parts.append(
        f"The best circuit was #{best['i']} <b>{esc(best['name'])}</b> "
        f"({best_acc:.1%}, {fmt_params(best['params'])} params)."
    )
    if compact["i"] != best["i"]:
        parts.append(
            f"The most parameter-efficient circuit reaching near-best accuracy was #{compact['i']} "
            f"<b>{esc(compact['name'])}</b> at only {fmt_params(compact['params'])} params "
            f"({compact['acc']:.1%})."
        )
    if failed:
        names = ", ".join(f"#{m['i']} {esc(m['name'])}" for m in failed[:6])
        more = "" if len(failed) <= 6 else f" (and {len(failed)-6} more)"
        parts.append(
            f"<b>{len(failed)}</b> attempt{'s' if len(failed) > 1 else ''} failed or barely "
            f"beat chance: {names}{more}."
        )
    return "".join(f"<p>{p}</p>" for p in parts)


def collect() -> list[dict]:
    tasks = []
    for run_dir in sorted(RUNS.iterdir()):
        csv_path = run_dir / "results" / "overall_results.csv"
        if not csv_path.exists():
            continue
        methods = []
        with csv_path.open() as fh:
            for i, row in enumerate(csv.DictReader(fh), start=1):
                methods.append({
                    "i": i,
                    "name": row.get("model_shorthand_name", "") or "?",
                    "acc": float(row["accuracy"]) if row.get("accuracy") else 0.0,
                    "params": float(row["n_params"]) if row.get("n_params") else 0.0,
                    "status": "failed" if row.get("status") == "failed" else "success",
                    "desc": row.get("description", "") or "",
                })
        if not methods:
            continue
        task = methods and (next(iter(csv.DictReader(csv_path.open())), {}).get("task") or run_dir.name)
        tasks.append({
            "task": task,
            "category": CATEGORY.get(task, "Other"),
            "methods": methods,
            "writeup": make_writeup(task, methods),
        })
    tasks.sort(key=lambda t: (CAT_ORDER.index(t["category"]) if t["category"] in CAT_ORDER else 99, t["task"]))
    return tasks


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Circuit-Building Autoresearch &mdash; Results</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --ink:#1a1a1a; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; --bg:#ffffff; }
  * { box-sizing: border-box; }
  body {
    margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    line-height:1.55; -webkit-font-smoothing:antialiased;
  }
  .wrap { max-width:1100px; margin:0 auto; padding:48px 28px 100px; }
  header h1 { font-size:30px; font-weight:700; margin:0 0 6px; letter-spacing:-0.02em; }
  header p.sub { color:var(--muted); margin:0 0 28px; font-size:15px; }
  h2 { font-size:21px; font-weight:650; margin:48px 0 14px; letter-spacing:-0.01em; }
  .tabs { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 8px; }
  .tab {
    border:1px solid var(--line); background:#fff; color:var(--ink);
    padding:7px 13px; border-radius:8px; cursor:pointer; font-size:13px;
    transition:all .12s; white-space:nowrap;
  }
  .tab:hover { border-color:#cbd5e1; background:#f8fafc; }
  .tab.active { background:var(--accent); border-color:var(--accent); color:#fff; }
  .tab .cat { font-size:10px; opacity:.6; margin-right:5px; text-transform:uppercase; letter-spacing:.04em; }
  .tab.active .cat { opacity:.85; }
  .panel { display:grid; grid-template-columns: 1.15fr 1fr; gap:24px; align-items:start; margin-top:14px; }
  @media (max-width:880px){ .panel { grid-template-columns:1fr; } }
  .card { border:1px solid var(--line); border-radius:12px; padding:4px; background:#fff; }
  .writeup { border:1px solid var(--line); border-radius:12px; padding:18px 20px; background:#fff; }
  .writeup h3 { margin:0 0 4px; font-size:16px; }
  .writeup .meta { color:var(--muted); font-size:12px; margin-bottom:10px; }
  .writeup p { margin:0 0 11px; font-size:14px; }
  .writeup b { font-weight:650; }
  details.methods { margin-top:6px; }
  details.methods summary { cursor:pointer; font-size:13px; color:var(--accent); font-weight:550; }
  table.mlist { width:100%; border-collapse:collapse; margin-top:12px; font-size:12.5px; }
  table.mlist th { text-align:left; color:var(--muted); font-weight:600; padding:5px 8px; border-bottom:1px solid var(--line); position:sticky; top:0; background:#fff; }
  table.mlist td { padding:6px 8px; border-bottom:1px solid #f1f5f9; vertical-align:top; }
  table.mlist td.num { color:var(--muted); width:26px; }
  table.mlist td.desc { color:#374151; }
  .acc-pill { font-variant-numeric:tabular-nums; font-weight:600; }
  .scroll { max-height:340px; overflow:auto; }
  .note { color:var(--muted); font-size:13px; margin:6px 0 0; }
  footer { color:var(--muted); font-size:12px; margin-top:60px; border-top:1px solid var(--line); padding-top:18px; }
  code { background:#f1f5f9; padding:1px 5px; border-radius:4px; font-size:12.5px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Circuit-Building Autoresearch</h1>
    <p class="sub">Autonomous AI research that hand-writes transformer weights to solve tasks. Each run iteratively
      proposes interpretable circuits; below, every point is one circuit, numbered in the order it was tried.</p>
  </header>

  <h2>Per-task exploration</h2>
  <p class="note">Pick a task. The plot shows <b>accuracy vs. number of parameters</b>; points are numbered by the
    order tried and connected along the path the search took. Hover a point for its method &amp; description.</p>
  <div class="tabs" id="tabs"></div>
  <div class="panel">
    <div class="card"><div id="taskPlot" style="width:100%;height:440px;"></div></div>
    <div class="writeup" id="writeup"></div>
  </div>

  <h2>All tasks together</h2>
  <p class="note">Every task overlaid. Each line traces one task's search path through (parameters, accuracy) space,
    in the order circuits were tried.</p>
  <div class="card"><div id="summaryPlot" style="width:100%;height:560px;"></div></div>

  <footer>
    Generated from <code>runs/*/results/overall_results.csv</code> by <code>analyze/build_report.py</code>.
  </footer>
</div>

<script>
const DATA = __DATA__;
const PALETTE = ['#2563eb','#dc2626','#059669','#d97706','#7c3aed','#0891b2','#db2777',
                 '#65a30d','#ea580c','#4f46e5','#0d9488','#be123c','#a16207'];
const LAYOUT_BASE = {
  paper_bgcolor:'#fff', plot_bgcolor:'#fff',
  font:{family:'-apple-system,Segoe UI,Roboto,sans-serif', color:'#1a1a1a', size:12},
  margin:{l:58,r:18,t:14,b:50},
};

function renderTask(idx){
  const t = DATA[idx];
  const m = t.methods;
  // For dense sweeps we can't legibly stamp a number on every point, so we drop
  // the on-plot labels (order stays in the hover + the ordered table below) and
  // lighten the path so the cloud of points stays readable. "Dense" also covers
  // tasks where most attempts share one param count (e.g. solved immediately then
  // refined in place) -- there the points are coincident and numbers just pile up.
  const counts = {}; let modal = 0;
  m.forEach(d=>{ counts[d.params]=(counts[d.params]||0)+1; modal=Math.max(modal,counts[d.params]); });
  const dense = m.length > 35 || modal/m.length >= 0.6;
  const trace = {
    x: m.map(d=>d.params), y: m.map(d=>d.acc),
    customdata: m.map(d=>String(d.i)),
    text: m.map(d=>
      `#${d.i} <b>${esc(d.name)}</b><br>acc ${(d.acc*100).toFixed(1)}%`+
      `<br>${d.params.toExponential(2)} params`+
      `${d.status==='failed'?'<br><i>failed</i>':''}`+
      `<br>${wrap(d.desc,60)}`),
    mode: dense ? 'lines+markers' : 'lines+markers+text',
    line:{color: dense?'rgba(148,163,184,0.35)':'#cbd5e1', width: dense?1:1.4},
    marker:{
      // Color encodes the ORDER tried: light (first) -> dark (last).
      size: dense?8:15, opacity: dense?0.85:1, color:m.map(d=>d.i),
      colorscale:[[0,'#dbeafe'],[0.5,'#60a5fa'],[1,'#1e3a8a']],
      cmin:1, cmax:m.length, line:{color:'#fff', width: dense?0.6:1.2},
      colorbar:{title:{text:'order tried',side:'right'}, thickness:10, len:0.6, x:1.02},
    },
    textposition:'top center', textfont:{size:9, color:'#475569'},
    texttemplate:'%{customdata}',
    hovertemplate:'%{text}<extra></extra>',
  };
  const layout = Object.assign({}, LAYOUT_BASE, {
    xaxis:{title:{text:'# parameters (log scale)'}, type:'log', gridcolor:'#f1f5f9', zeroline:false},
    yaxis:{title:{text:'accuracy'}, range:[-0.04,1.06], gridcolor:'#f1f5f9', zeroline:false, tickformat:'.0%'},
    showlegend:false,
  });
  Plotly.react('taskPlot', [trace], layout, {displayModeBar:false, responsive:true});

  const rows = m.map(d=>
    `<tr><td class="num">${d.i}</td>`+
    `<td><b>${esc(d.name)}</b></td>`+
    `<td class="acc-pill" style="color:${d.acc>=0.99?'#059669':d.acc<0.5?'#b91c1c':'#374151'}">${(d.acc*100).toFixed(1)}%</td>`+
    `<td style="white-space:nowrap;color:#6b7280">${fmtP(d.params)}</td>`+
    `<td class="desc">${esc(d.desc)}</td></tr>`).join('');
  document.getElementById('writeup').innerHTML =
    `<h3>${esc(t.task)}</h3>`+
    `<div class="meta">${esc(t.category)} &middot; ${m.length} circuits tried</div>`+
    t.writeup+
    `<details class="methods" open><summary>All ${m.length} circuits, in order</summary>`+
    `<div class="scroll"><table class="mlist"><thead><tr>`+
    `<th>#</th><th>method</th><th>acc</th><th>params</th><th>description</th>`+
    `</tr></thead><tbody>${rows}</tbody></table></div></details>`;
}

function renderSummary(){
  const traces = DATA.map((t,i)=>{
    const c = PALETTE[i%PALETTE.length];
    // Connect along increasing parameter count (not tried-order) so each task is
    // a clean left-to-right curve instead of a scribble jumping around the plot.
    const pts = t.methods.slice().sort((a,b)=> a.params-b.params || a.acc-b.acc);
    return {
      name: t.task, legendgroup:t.task,
      x: pts.map(d=>d.params), y: pts.map(d=>d.acc),
      text: pts.map(d=>`#${d.i} ${esc(d.name)}<br>${(d.acc*100).toFixed(1)}% &middot; ${d.params.toExponential(1)}p`),
      mode:'lines+markers',
      line:{color:c, width:1.6},
      marker:{size:5, color:c, opacity:0.75, line:{color:'#fff',width:0.6}},
      hovertemplate:`<b>${esc(t.task)}</b><br>%{text}<extra></extra>`,
    };
  });
  const layout = Object.assign({}, LAYOUT_BASE, {
    xaxis:{title:{text:'# parameters (log scale)'}, type:'log', gridcolor:'#f1f5f9', zeroline:false},
    yaxis:{title:{text:'accuracy'}, range:[-0.04,1.06], gridcolor:'#f1f5f9', zeroline:false, tickformat:'.0%'},
    legend:{font:{size:11}, bgcolor:'rgba(255,255,255,0.6)'},
  });
  Plotly.react('summaryPlot', traces, layout, {displayModeBar:false, responsive:true});
}

function buildTabs(){
  const el = document.getElementById('tabs');
  el.innerHTML = DATA.map((t,i)=>
    `<button class="tab${i===0?' active':''}" data-i="${i}">`+
    `<span class="cat">${esc(t.category)}</span>${esc(t.task)}</button>`).join('');
  el.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{
    el.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    renderTask(+b.dataset.i);
  }));
}

function esc(s){ return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function wrap(s, n){ s=esc(s); let out='',line=0;
  for(const w of s.split(' ')){ if(line+w.length>n){out+='<br>';line=0;} out+=w+' '; line+=w.length+1; } return out; }
function fmtP(p){ if(p>=1e6) return (p/1e6).toFixed(2)+'M'; if(p>=1e3) return (p/1e3).toFixed(1)+'K'; return p.toFixed(0); }

buildTabs();
renderTask(0);
renderSummary();
</script>
</body>
</html>
"""


def main():
    tasks = collect()
    OUT.write_text(TEMPLATE.replace("__DATA__", json.dumps(tasks)))
    n_methods = sum(len(t["methods"]) for t in tasks)
    print(f"Wrote {OUT} with {len(tasks)} tasks, {n_methods} circuits.")


if __name__ == "__main__":
    main()
