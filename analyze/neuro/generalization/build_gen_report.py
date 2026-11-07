"""Build analyze/neuro/report_generalization.html from results.csv.

Run with the project env (it imports config → src.data → joblib): `uv run build_gen_report.py`.

Presents the ORIGINAL runs (recap) separately from the NEW generalization
experiment (new subjects + new stories), all measured under one identical
pipeline so original-vs-new comparisons are apples-to-apples.
"""
import csv
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUNS_DIR = os.path.join(REPO, "experiments", "runs-neuro")
RESULTS = os.path.join(HERE, "results.csv")
OUT = os.path.join(HERE, "..", "report_generalization.html")

import config as C  # noqa: E402

# ---- run metadata (model / effort) ----
RUN_META = {}
for folder in os.listdir(RUNS_DIR):
    mp = os.path.join(RUNS_DIR, folder, "metadata.json")
    if os.path.exists(mp):
        with open(mp) as f:
            RUN_META[folder] = json.load(f)

PALETTE = {
    "fmri-jun3-run1": "#2563eb", "fmri-jun03-run2": "#db2777",
    "fmri-jun03-run3": "#059669", "fmri-jun03-run4": "#d97706",
    "fmri-jun04-run1": "#0e7490", "fmri-may27-run1": "#7c3aed",
}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---- load results.csv → per (run, model) record ----
rows = []
with open(RESULTS, newline="") as f:
    rows = list(csv.DictReader(f))

models = {}  # (run, model) -> dict
order = []
for run, model, reported, note in C.SELECTED:
    key = (run, model)
    models[key] = {"run": run, "model": model, "reported": reported, "note": note,
                   "orig": None, "UTS01": None, "UTS02": None, "newstory": None,
                   "orig_train": None}
    order.append(key)

for r in rows:
    if r.get("status") != "ok":
        continue
    key = (r["run"], r["model"])
    if key not in models:
        continue
    m = models[key]
    corr = fnum(r["test_corr"])
    if r["setting"] == "orig":
        m["orig"] = corr
        m["orig_train"] = fnum(r["train_corr"])
    elif r["setting"] == "newsubj":
        m[r["subject"]] = corr
    elif r["setting"] == "newstory":
        m["newstory"] = corr

# Snapshot files sometimes carry a trailing date/status tag
# (e.g. "FeatBag_v1116_Emo40_0810_CEILING_BREAK"); strip it for a clean display name.
import re as _re  # noqa: E402

def clean_name(model):
    return _re.sub(r"_\d{4}_[A-Za-z0-9_.]+$", "", model)

def short(model):
    model = clean_name(model)
    return model if len(model) <= 26 else model[:24] + "…"

DATA = []
for key in order:
    m = models[key]
    DATA.append({
        "run": m["run"], "model": m["model"], "short": short(m["model"]),
        "label": RUN_META.get(m["run"], {}).get("label", m["run"]),
        "modelLLM": RUN_META.get(m["run"], {}).get("model", ""),
        "effort": RUN_META.get(m["run"], {}).get("effort", ""),
        "color": PALETTE.get(m["run"], "#555"),
        "note": m["note"], "reported": m["reported"],
        "orig": m["orig"], "UTS01": m["UTS01"], "UTS02": m["UTS02"],
        "newstory": m["newstory"], "orig_train": m["orig_train"],
        "untrimmed_orig": m["run"] == "fmri-may27-run1",
    })

DATA_JS = json.dumps(DATA)
NEW_STORIES_JS = json.dumps(C.NEW_STORIES)
ORIG_TEST_JS = json.dumps(C.ORIG_TEST)

# ---- derived statistics for the prose (kept in sync with the data) ----
def _pct(x):
    return f"{round(100 * x)}%"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


N_MODELS = len(DATA)
_subj_ret, _story_ret, _uts02_better = [], [], 0
for d in DATA:
    o = d["orig"]
    if not o:
        continue
    sub = _mean([d["UTS01"], d["UTS02"]])
    if sub is not None:
        _subj_ret.append(sub / o)
    if d["newstory"] is not None:
        _story_ret.append(d["newstory"] / o)
    if d["UTS01"] is not None and d["UTS02"] is not None and d["UTS02"] >= d["UTS01"]:
        _uts02_better += 1
SUBJ_RET_LO, SUBJ_RET_HI = _pct(min(_subj_ret)), _pct(max(_subj_ret))
STORY_RET_LO, STORY_RET_HI = _pct(min(_story_ret)), _pct(max(_story_ret))
STORY_RET_MEAN = _pct(_mean(_story_ret))
UTS02_BETTER_ALL = _uts02_better == sum(
    1 for d in DATA if d["UTS01"] is not None and d["UTS02"] is not None)
# May-27 best model, re-measured under the trimmed pipeline (for finding #4).
_may = next((d for d in DATA if d["model"] == "WordNetMorphLingPerceptual"), None)
MAY_TRIMMED = f"{_may['orig']:.3f}" if _may and _may["orig"] is not None else "—"

# ---- original-runs recap table ----
def recap_rows():
    out = []
    seen = []
    for d in DATA:
        if d["run"] not in seen:
            seen.append(d["run"])
    for run in seen:
        meta = RUN_META.get(run, {})
        ds = [d for d in DATA if d["run"] == run]
        names = ", ".join(html.escape(clean_name(d["model"])) for d in ds)
        out.append(
            f'<tr><td><span class="dot" style="background:{PALETTE.get(run,"#555")}"></span>'
            f'{html.escape(meta.get("label", run))}</td>'
            f'<td>{html.escape(meta.get("model",""))}</td>'
            f'<td>{html.escape(meta.get("effort",""))}</td>'
            f'<td>{"yes" if meta.get("untrimmed") else "no"}</td>'
            f'<td class="mono">{names}</td></tr>'
        )
    return "".join(out)


RECAP = recap_rows()

HTML = f'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>fMRI encoding — generalization report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {{ --ink:#1a1a1a; --dim:#6b7280; --line:#e5e7eb; --panel:#fafafa; --accent:#2563eb;
           --good:#15803d; --goodbg:#dcfce7; --below:#b45309; --belowbg:#fef3c7; }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; background:#fff; color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased; line-height:1.55; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:48px 28px 96px; }}
  h1 {{ font-size:30px; font-weight:700; letter-spacing:-0.02em; margin:0 0 6px; }}
  h2 {{ font-size:21px; font-weight:650; margin:54px 0 6px; padding-bottom:8px;
        border-bottom:1px solid var(--line); }}
  h3 {{ font-size:16px; font-weight:650; margin:26px 0 4px; }}
  p {{ margin:10px 0; }} .lead {{ color:var(--dim); font-size:15px; max-width:780px; }}
  code {{ background:var(--panel); border:1px solid var(--line); border-radius:5px; padding:1px 5px;
    font-size:12.5px; font-family:ui-monospace,Menlo,monospace; }}
  .note {{ background:var(--belowbg); border:1px solid #fcd34d; border-radius:10px;
    padding:14px 18px; font-size:14px; margin:16px 0; color:#78350f; }}
  .panel {{ border:1px solid var(--line); border-radius:14px; padding:18px 20px; margin:18px 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; color:var(--dim); font-weight:600; padding:8px 10px;
        border-bottom:2px solid var(--line); }}
  td {{ padding:7px 10px; border-bottom:1px solid #f1f1f1; vertical-align:top; }}
  .mono {{ font-family:ui-monospace,Menlo,monospace; font-size:11.5px; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px;
          vertical-align:middle; }}
  .plot {{ width:100%; }}
  .tall {{ height:430px; }} .sq {{ height:480px; }}
  .legend-note {{ font-size:12.5px; color:var(--dim); margin-top:8px; }}
  .pill {{ display:inline-block; font-size:11px; border-radius:999px; padding:2px 9px;
           border:1px solid var(--line); background:var(--panel); color:var(--dim); }}
  .ret-good {{ color:var(--good); font-weight:600; }} .ret-bad {{ color:var(--below); font-weight:600; }}
  footer {{ margin-top:60px; color:var(--dim); font-size:12.5px; border-top:1px solid var(--line);
            padding-top:18px; }}
</style></head>
<body><div class="wrap">
  <h1>Generalization of the hand-written fMRI encoders</h1>
  <p class="lead">
    We took a handful of the best, non-redundant hand-written models from each evolutionary run
    and tested how well they transfer (1) to <b>new subjects</b> (UTS01, UTS02) on the same stories,
    and (2) to <b>new stories</b> (held-out stories from the training pool) on the same subject (UTS03).
    Every number on this page — including each model's "original" score — is re-measured under one
    <b>identical pipeline</b> (10-gram features, 30-TR edge trim, <code>ndelays=4</code>, 8 training
    stories, bootstrapped ridge), so original-vs-new comparisons are apples-to-apples.
    See the original report (<code>report.html</code>) for the full evolution curves.
    Only genuinely <b>hand-wired</b> models are tested here. Disallowed iterations flagged in the headline
    report are <b>excluded</b> from selection: Jun-03 run 3's <b>external pretrained encoders</b>
    (Qwen / DistilBERT / RoBERTa / GloVe) and trained model, and Jun-04 run 1's late <b>corpus-statistics</b>
    push (an n-gram surprisal language model + LSA / PPMI co-occurrence word vectors built from the stimulus
    text, which reached ~0.088 but breaks the "no corpus statistics" rule). All six runs are represented.
    <b>Jun-04 run 1</b> (Claude Opus 4.8, xhigh) was given <b>all prior runs' results</b> and resumed run 4's
    <code>FeatBag</code>; its best <i>legitimate</i> model adds a content-free <b>within-story novelty</b>
    block (first-mention / repetition-suppression, computed per story at inference) plus a name gazetteer
    (<code>FeatBagNovelty_NamesDense_xrun</code>, 0.084 — itself just past GPT-2 XL), which we test here as a
    genuinely new mechanism alongside the FeatBag family.
  </p>

  <div class="note">
    <b>Two differences from the headline report.</b>
    (1) The headline <code>report.html</code> scored each run with its own recorded number; here we
    <b>re-run</b> every model through the same fixed pipeline, so the "original" bars can differ slightly
    from the headline values (and substantially for the May-27 run).
    (2) The <b>May-27</b> models were originally scored <b>untrimmed</b> (and with <code>ndelays=3</code>);
    re-measured here with 30-TR trimming + <code>ndelays=4</code>, their "original" score drops well below
    the 0.11 reported there. Their headline (untrimmed) numbers are shown in the recap table for reference only.
  </div>

  <h2>Key findings</h2>
  <div class="panel">
    <p><b>1. Cross-subject transfer is partial but consistent.</b> Every one of the {N_MODELS} models keeps
    roughly <b>half</b> of its correlation when moved to a new subject (subject retention ~{SUBJ_RET_LO}–{SUBJ_RET_HI}),
    {"with <b>UTS02 transferring better than UTS01</b> for every model" if UTS02_BETTER_ALL else "with <b>UTS02 generally transferring better than UTS01</b>"}. So the hand-built circuits
    capture genuine, subject-general language signal — but a substantial part of each model's score is
    subject-specific (the ridge readout is always refit per subject; the feature circuit is what
    transfers).</p>
    <p><b>2. Cross-story transfer is weaker than cross-subject, but holds up.</b> Averaged over
    <b>all {len(C.NEW_STORIES)} held-out shared stories</b>, models keep <b>~{STORY_RET_LO}–{STORY_RET_HI} (mean ~{STORY_RET_MEAN})</b> of
    their original correlation — below the cross-subject ~{SUBJ_RET_LO}–{SUBJ_RET_HI}, but well above what a small sample
    implied. (An earlier 3-story version of this test gave only ~19–32%: those particular stories were
    a noisy, pessimistic draw. Averaging over the full shared set raises and stabilizes the estimate,
    and the new-story score now <b>tracks the original ranking</b> — the strongest original models stay
    strongest — rather than looking flat.) New stories remain the harder axis of generalization, just
    not as severe as the 3-story sample suggested.</p>
    <p><b>3. Model rankings are preserved.</b> The strongest original models (LexFeat / FeatBag /
    WordNet families) stay at or near the top in both new settings, so the relative conclusions from
    the evolutionary runs hold up under transfer — the differences are attenuated, not reordered.</p>
    <p><b>4. The May-27 models are not special once measured consistently.</b> Re-measured under the
    same trimmed pipeline, <code>WordNetMorphLingPerceptual</code> drops from its reported (untrimmed)
    0.115 to <b>{MAY_TRIMMED}</b>, and its transfer is in line with the other runs.</p>
  </div>

  <h2>Original runs (recap)</h2>
  <p class="lead">The six evolutionary runs and the models sampled from each for this experiment.</p>
  <div class="panel"><table>
    <thead><tr><th>Run</th><th>Driver model</th><th>Effort</th><th>Untrimmed</th>
      <th>Models tested here</th></tr></thead>
    <tbody>{RECAP}</tbody>
  </table></div>

  <h2>New subjects — same stories, train+test on UTS01 / UTS02</h2>
  <p class="lead">Each model's feature circuit is fixed; only the ridge readout is refit per subject
  (the encoding model is always subject-specific). Bars compare the original subject (UTS03) with the
  two new subjects on the identical test stories.</p>
  <div class="panel"><div id="subj-bars" class="plot tall"></div>
    <p class="legend-note">Grouped by model (colored by source run). Higher = better transfer.</p></div>
  <h3>Summary — original vs. new-subject correlation</h3>
  <div class="panel"><div id="subj-scatter" class="plot sq"></div>
    <p class="legend-note">Each marker is one model on one new subject; the dashed line is
    <i>y = x</i> (perfect transfer). Points below the line lost correlation on the new subject.</p></div>

  <h2>New stories — same subject (UTS03), held-out stories</h2>
  <p class="lead">Same subject and the same fixed circuits, but evaluated on <b>all {len(C.NEW_STORIES)}
  shared stories</b> that were never used for training (the 8 fit stories) or testing (the 3 original
  test stories) — i.e. the entire remainder of the cross-subject shared story pool, not just a 3-story
  sample. The ridge readout is refit on the original 8 training stories; only the held-out evaluation
  set changes. Averaging over {len(C.NEW_STORIES)} stories makes the new-story estimate far more stable
  than the earlier 3-story version.</p>
  <div class="panel"><div id="story-bars" class="plot tall"></div>
    <p class="legend-note">Original test stories vs. the new held-out stories, per model.</p></div>
  <h3>Summary — original vs. new-story correlation</h3>
  <div class="panel"><div id="story-scatter" class="plot sq"></div>
    <p class="legend-note">Each marker is one model; dashed line is <i>y = x</i>.</p></div>

  <h2>Transfer retention</h2>
  <p class="lead">New-setting correlation as a fraction of the (re-measured) original. ~1.0 means the
  model transferred with little loss.</p>
  <div class="panel"><table id="rettab">
    <thead><tr><th>Run</th><th>Model</th><th class="num">orig</th>
      <th class="num">UTS01</th><th class="num">UTS02</th><th class="num">new stories</th>
      <th class="num">subj retention</th><th class="num">story retention</th></tr></thead>
    <tbody></tbody>
  </table></div>

  <footer>
    Generated from <code>generalization/results.csv</code> (harness:
    <code>generalization/harness.py</code>, models from <code>experiments/runs-neuro/*/interpretable_transformers_lib/</code>).
    Pipeline: 30-TR edge trim, 10-gram, ndelays=4, 8 train stories, bootstrapped ridge over ~95.6k voxels.
  </footer>
</div>

<script>
const DATA = {DATA_JS};
const FONT = {{ family:"-apple-system,Segoe UI,Roboto,sans-serif", color:"#1a1a1a", size:12 }};
const CFG = {{ displayModeBar:false, responsive:true }};
const base = (extra) => Object.assign({{
  paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:FONT,
  margin:{{l:56,r:18,t:14,b:90}},
  xaxis:{{ gridcolor:"#f0f0f0", zeroline:false, showline:true, linecolor:"#d1d5db" }},
  yaxis:{{ gridcolor:"#f0f0f0", zeroline:false, showline:true, linecolor:"#d1d5db",
           title:"test correlation" }},
  hovermode:"closest"
}}, extra||{{}});

const labels = DATA.map(d => d.short);
const colors = DATA.map(d => d.color);

// ---- new-subject grouped bars ----
Plotly.newPlot("subj-bars", [
  {{ x:labels, y:DATA.map(d=>d.orig),  name:"UTS03 (orig)", type:"bar",
     marker:{{color:"#9ca3af"}} }},
  {{ x:labels, y:DATA.map(d=>d.UTS01), name:"UTS01 (new)", type:"bar",
     marker:{{color:"#60a5fa"}} }},
  {{ x:labels, y:DATA.map(d=>d.UTS02), name:"UTS02 (new)", type:"bar",
     marker:{{color:"#2563eb"}} }},
], base({{ barmode:"group", xaxis:{{tickangle:-40, automargin:true}},
   legend:{{orientation:"h", y:1.08, x:1, xanchor:"right"}} }}), CFG);

// ---- scatter (orig on x, new-setting on y) ----
function scatterFig(div, names) {{
  const traces = names.map((nm) => ({{
    x:DATA.map(d=>d.orig), y:DATA.map(d=>d[nm.key]),
    text:DATA.map(d=>d.short + "<br>" + d.modelLLM + " ("+d.effort+")"),
    mode:"markers", type:"scatter", name:nm.label,
    marker:{{size:11, color:nm.color, line:{{color:"#fff", width:1}}, symbol:nm.symbol}},
    hovertemplate:"%{{text}}<br>orig %{{x:.4f}} → new %{{y:.4f}}<extra></extra>"
  }}));
  let all = [];
  DATA.forEach(d => {{ all.push(d.orig); names.forEach(nm => all.push(d[nm.key])); }});
  all = all.filter(v=>v!=null);
  const mx = Math.max.apply(null, all) * 1.08;
  traces.push({{ x:[0,mx], y:[0,mx], mode:"lines", name:"y = x", hoverinfo:"skip",
    line:{{color:"#9ca3af", width:1.4, dash:"dash"}} }});
  Plotly.newPlot(div, traces, base({{
    xaxis:{{title:"original (UTS03) test correlation", gridcolor:"#f0f0f0", showline:true,
            linecolor:"#d1d5db", range:[0,mx]}},
    yaxis:{{title:"new-setting test correlation", gridcolor:"#f0f0f0", showline:true,
            linecolor:"#d1d5db", range:[0,mx]}},
    legend:{{orientation:"h", y:1.07, x:1, xanchor:"right"}}, margin:{{l:56,r:18,t:14,b:56}}
  }}), CFG);
}}
scatterFig("subj-scatter",
  [{{key:"UTS01", label:"UTS01", color:"#60a5fa", symbol:"circle"}},
   {{key:"UTS02", label:"UTS02", color:"#2563eb", symbol:"diamond"}}]);

// ---- new-story bars ----
Plotly.newPlot("story-bars", [
  {{ x:labels, y:DATA.map(d=>d.orig),     name:"orig test stories", type:"bar",
     marker:{{color:"#9ca3af"}} }},
  {{ x:labels, y:DATA.map(d=>d.newstory), name:"new (held-out) stories", type:"bar",
     marker:{{color:"#059669"}} }},
], base({{ barmode:"group", xaxis:{{tickangle:-40, automargin:true}},
   legend:{{orientation:"h", y:1.08, x:1, xanchor:"right"}} }}), CFG);

// ---- new-story scatter ----
scatterFig("story-scatter",
  [{{key:"newstory", label:"new stories (UTS03)", color:"#059669", symbol:"circle"}}]);

// ---- retention table ----
function fmt(v) {{ return v==null ? "—" : v.toFixed(4); }}
function ret(n,o) {{ return (n==null||o==null||o<=0) ? null : n/o; }}
function retCell(r) {{
  if (r==null) return '<td class="num">—</td>';
  const cls = r>=0.85 ? "ret-good" : (r<0.6 ? "ret-bad" : "");
  return `<td class="num ${{cls}}">${{(r*100).toFixed(0)}}%</td>`;
}}
const tb = document.querySelector("#rettab tbody");
DATA.forEach(d => {{
  const subjR = ret((avg2(d.UTS01,d.UTS02)), d.orig);
  const storyR = ret(d.newstory, d.orig);
  const tr = document.createElement("tr");
  tr.innerHTML =
    `<td><span class="dot" style="background:${{d.color}}"></span>${{d.label}}</td>`+
    `<td class="mono">${{d.short}}</td>`+
    `<td class="num">${{fmt(d.orig)}}</td>`+
    `<td class="num">${{fmt(d.UTS01)}}</td>`+
    `<td class="num">${{fmt(d.UTS02)}}</td>`+
    `<td class="num">${{fmt(d.newstory)}}</td>`+
    retCell(subjR)+retCell(storyR);
  tb.appendChild(tr);
}});
function avg2(a,b) {{ const xs=[a,b].filter(v=>v!=null); return xs.length? xs.reduce((s,v)=>s+v,0)/xs.length : null; }}
</script>
</body></html>'''

with open(OUT, "w") as f:
    f.write(HTML)
print(f"wrote {os.path.abspath(OUT)} ({len(HTML)} bytes)")
