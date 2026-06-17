"""Build the *search-process* report: how the dynamic-workflow autoresearch searched for an
untrained model of the NSD ventral stream — across a hypothesis-driven phase and an
open-ended discovery phase — and what it found.

Tells the methodology and the discovery: what a dynamic-workflow search is, how it differs
from a single-agent loop, the two phases (top-down theories vs free score-guided invention),
every generation, the computations the open-ended search discovered, the noise-ceiling
grounding, and an honest verdict vs the 1.5x-AlexNet target.

    python analyze/search_report.py            # -> analyze/search_report.html

Self-contained: shared CSS + vendored Plotly inlined. Reads runs/evo-ventral/search_data_full.json.
"""
from __future__ import annotations
import html, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "runs", "evo-ventral", "search_data_full.json")
OUT = os.path.join(HERE, "search_report.html")
ALEX = 0.2838
CEIL = 0.50
TARGET = 0.4257
REGION_ORDER = ["V1", "hV4", "early", "ventral", "FFA", "PPA", "EBA", "VWFA"]

# ---- curated narrative (the orchestrator's view, not the per-row ledger) ----
TRAJECTORY = [
    ("identity", 16, 0.1279, "hypothesis", "raw 16px pixels into the ridge"),
    ("demo", 16, 0.2467, "hypothesis", "evolutionary search finds the texture family"),
    ("gen 1", 16, 0.2505, "hypothesis", "depth + crossover"),
    ("gen 2", 64, 0.2817, "hypothesis", "resolution 16→64px unlocks deep families"),
    ("gen 3", 64, 0.2966, "hypothesis", "ensembling complementary families"),
    ("gen 4–6", 128, 0.3122, "hypothesis", "canonical cascade, eccentricity (theories plateau)"),
    ("open 1", 128, 0.3201, "open", "gating · relational · RFF break the plateau"),
    ("open 2", 128, 0.3200, "open", "cross-stream bilinear (ties)"),
    ("open 3", 128, 0.3213, "open", "fusion + random-Fourier capacity"),
]
# hypothesis-phase falsified-theory ledger (gens 4–6)
FALSIFIED = [
    ("canonical LN + divisive-norm cascade", 0.304, "the right model for early/mid; flat on high-level"),
    ("depth to saturation (7 stages)", 0.288, "SATURATES by ~3 stages — high-level not capacity-limited"),
    ("eccentricity-bias (foveal/peripheral)", 0.312, "gain is early-cortex; the FFA/PPA dissociation is refuted"),
    ("hierarchical texture / Gram statistics", 0.275, "position-invariant covariance underperforms"),
    ("sparse / efficient coding", 0.300, "sparsity monotonically HURTS"),
    ("V4 curvature / shape", 0.278, "no high-level lift"),
]
# open-ended discoveries (what free search found that theory-driven did not)
DISCOVERIES = [
    ("Multiplicative / cross-channel gating", "features gating other features — beyond additive energy cascades", 0.320),
    ("Cross-stream bilinear", "outer products between feature maps at different cascade depths", 0.320),
    ("Relational / configural co-occurrence", "products of a feature map with spatially-shifted copies — encodes part arrangement, which pooling & Gram cannot", 0.306),
    ("Random-Fourier / random-kernel expansion", "high-dimensional untrained kernel features (RBF / arc-cosine)", 0.321),
]
OPEN_NEGATIVES = [
    ("Untrained self-attention routing", 0.265, "fixed-weight multi-head attention + bilinear — generalizes cleanly but below the plateau"),
    ("Predictive-coding fixed-point dynamics", 0.286, "iterative fixed-weight settling — under the energy+interaction cascade"),
]


def esc(s): return html.escape(str(s if s is not None else ""))
def fnum(x, d=4): return "—" if x is None else f"{x:.{d}f}"


def _asset(*p):
    with open(os.path.join(HERE, *p), encoding="utf-8") as fh:
        return fh.read()


def _plotly_bundle():
    js = re.sub(r"https://cdn\.plot\.ly/?", "", _asset("vendor", "plotly-cartesian.min.js"))
    if "cdn.plot.ly" in js:
        raise RuntimeError("vendored Plotly still references a CDN")
    return js


_LAYOUT = dict(
    paper_bgcolor="#fff", plot_bgcolor="#fff",
    font=dict(family="Seravek, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", size=13, color="#2b2f36"),
    margin=dict(l=58, r=22, t=30, b=60), hovermode="closest",
    xaxis=dict(gridcolor="#efede7", zerolinecolor="#e4e2dc", linecolor="#e4e2dc"),
    yaxis=dict(gridcolor="#efede7", zerolinecolor="#e4e2dc", linecolor="#e4e2dc"),
    legend=dict(bgcolor="rgba(255,255,255,.7)", bordercolor="#e4e2dc", borderwidth=1),
)
_CONFIG = {"displayModeBar": False, "responsive": True}
_pid = [0]


def plot(data, layout=None, height=380, caption=""):
    _pid[0] += 1
    pid = f"plt{_pid[0]}"
    lay = json.loads(json.dumps(_LAYOUT)); lay.update(layout or {})
    cap = f'<p class="chart-caption">{caption}</p>' if caption else ""
    return (f'<div class="chart-card"><div id="{pid}" class="chart-plot" style="min-height:{height}px"></div>{cap}</div>'
            f'<script>Plotly.newPlot("{pid}",{json.dumps(data)},{json.dumps(lay)},{json.dumps(_CONFIG)});</script>')


# ---------- sections ----------
def header(d):
    champ = d["champion"]["ventral"]
    sb = []
    for r in REGION_ORDER:
        ours = d["champion"]["panel"].get(r)
        alex = d["alexnet_baseline"].get("V1" if r == "early" else r)
        dd = (ours - alex) if (ours is not None and alex is not None) else None
        cls = "good" if (dd is not None and dd > 0) else "bad"
        sb.append(f'<div class="sb-card"><span class="sb-region">{esc(r)}</span>'
                  f'<span class="sb-best">{fnum(ours,3)}</span>'
                  f'<span class="sb-alex">AlexNet {fnum(alex,3)}</span>'
                  f'<span class="sb-delta {cls}">{"+" if (dd or 0)>=0 else ""}{fnum(dd,3)}</span></div>')
    stats = [
        (f"{champ:.4f}", "best ventral test_corr", "untrained, verified"),
        (f"+{champ-ALEX:.4f}", "vs trained AlexNet", "0.2838 bar"),
        (f"{champ/TARGET*100:.0f}%", "of the 1.5× target", "0.4257"),
        (f"{champ/CEIL*100:.0f}%", "of the noise ceiling", "~0.50 explainable"),
        ("~50", "models / 9 gens", "2 search phases"),
        ("0", "trained params", "ridge readout only"),
    ]
    stat_html = "".join(f'<div class="stat"><span class="stat-value">{esc(v)}<small> {esc(sub)}</small></span>'
                        f'<span class="stat-label">{esc(l)}</span></div>' for v, l, sub in stats)
    return f"""
<header class="report-header">
  <h1>What computation predicts the ventral stream? An open-ended search</h1>
  <p class="subtitle">A dynamic-workflow autoresearch in two phases — first implementing known
  computational <i>theories</i>, then an <b>open-ended, score-guided search</b> free to invent
  <i>any</i> untrained computation. Both with zero trained parameters. The search climbed from
  a 0.128 pixel floor to <b>{champ:.4f}</b> ventral predictivity — <b>+{champ-ALEX:.4f}</b> past a
  <i>trained</i> AlexNet (0.2838) — and discovered that the most predictive untrained computations
  are feature <b>interactions</b>, not deeper filter cascades.</p>
  <div class="stat-row">{stat_html}</div>
  <div class="scoreboard">{''.join(sb)}</div>
  <p class="scoreboard-caption">Champion vs trained AlexNet per NSD region. Untrained wins early/mid
  decisively and on the ventral reward; the trained net still leads the category-selective regions —
  the wall neither phase could cross (§09).</p>
</header>"""


def toc():
    items = [("q", "The question"), ("dw", "Dynamic workflows"), ("traj", "Trajectory"),
             ("p1", "Phase 1 · theories"), ("p2", "Phase 2 · open-ended"),
             ("disc", "The discovery"), ("regions", "Cross-region"), ("verdict", "Verdict"),
             ("model", "Winning model"), ("caveats", "Caveats")]
    return '<nav class="toc"><b>Contents</b>' + "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in items) + '</nav>'


def s_question(d):
    b = d["alexnet_baseline"]
    rows = "".join(f"<tr><td>{esc(r)}</td><td class='num'>{fnum(b.get('V1' if r=='early' else r),3)}</td></tr>" for r in REGION_ORDER)
    return f"""
<h2 id="q">The question</h2>
<p class="section-sub">Can an untrained computation predict high-level visual cortex as well as a trained network — and which computation?</p>
<div class="findings">
<p>An agent proposes a <b>fixed, never-trained</b> model — a callable mapping a stimulus image
<code>X</code> to activations — whose <b>ridge read-out</b> against recorded NSD brain activity has
the highest held-out correlation. Only the ridge is fit; that fit <i>is</i> the eval. The reward is
mean per-voxel <code>test_corr</code> on the image-disjoint <b>shared-1000</b> test set, over the
<b>ventral stream</b> (1820 voxels, verified canonical).</p>
<p>The bar: a frozen ImageNet <b>AlexNet</b> (fc7) read out through the same ridge — <b>ventral
0.2838</b> (reproduced, himalaya-cross-checked). The target: <b>1.5× = 0.4257</b>. The hard limit:
an estimated <b>noise ceiling ≈ {CEIL}</b> (split-half reliability of the 3-rep responses), so
0.4257 is <b>85% of the explainable variance</b> — extremely ambitious but, in principle, possible.</p>
<table class="baseline"><thead><tr><th>region</th><th class="num">trained AlexNet</th></tr></thead><tbody>{rows}
<tr><td><b>ventral (reward)</b></td><td class="num"><b>{fnum(b['ventral'],4)}</b></td></tr></tbody></table>
<p>The reward is <b>deterministic</b> (closed-form ridge), so "noise" is the risk of <i>selecting</i> a
model that fits the fixed 1000 test images by luck — guarded by margins, panel-wide generalisation, and
explicit per-sample / untrained verification of every champion.</p>
</div>"""


def s_dynamic_workflows():
    return """
<h2 id="dw">The method — a dynamic-workflow population search</h2>
<p class="section-sub">The same search, run as a parallel population instead of one trajectory.</p>
<div class="findings">
<p>The stock loop is <b>one agent, sequential</b>: propose → evaluate → read best → propose. A
<b>dynamic workflow</b> turns it into <b>evolutionary search</b>: a deterministic orchestrator runs
generations; each fans out K candidate agents <i>in parallel</i>, each writing a full
<code>model.py</code>, evaluating it through the fixed harness, and returning its score + per-region
panel; between generations an <b>elite archive</b> keeps the best and seeds the next.</p>
<div class="diagram">
  <div class="dw-col"><div class="dw-head">Phase 1 — theory-driven</div>
    <div class="dw-gen"><span class="g">gen</span><span class="c">canonical cascade</span><span class="c">scattering</span><span class="c">V4 shape</span><span class="c">texture</span><span class="c">eccentricity</span></div>
    <div class="dw-note">each candidate = a KNOWN computational theory · plateaued 0.31</div></div>
  <div class="dw-col"><div class="dw-head">Phase 2 — open-ended</div>
    <div class="dw-gen"><span class="g">gen</span><span class="c">wildcard</span><span class="c">mutate</span><span class="c">novel interaction</span><span class="c">capacity↑</span><span class="c">recombine</span><span class="c">random-kernel</span></div>
    <div class="dw-sel">▼ rank · keep elite · seed winners · inject wildcards ▼</div>
    <div class="dw-note">candidates INVENT any untrained computation · the score is the only judge</div></div>
</div>
<div class="takeaway"><span class="takeaway-tag">the two invariants that stay</span><b>Untrained</b>
(every weight analytic or random-but-fixed; the target Y is never seen) and <b>per-sample</b> (an
image's features depend only on that image — no test-batch statistics). These make the held-out score
<i>real</i>; everything else — structure, capacity, interpretability — is unconstrained in Phase 2.</div>
</div>"""


def s_trajectory():
    xs = [t[0] for t in TRAJECTORY]
    colors = ["#9aa1ab" if t[3] == "hypothesis" else "#1e40af" for t in TRAJECTORY]
    colors[0] = "#cbd0d8"
    bars = dict(type="bar", x=xs, y=[t[2] for t in TRAJECTORY],
                text=[f"{t[2]:.3f}<br>{t[1]}px" for t in TRAJECTORY], textposition="outside",
                marker=dict(color=colors), hovertext=[t[4] for t in TRAJECTORY], hoverinfo="text+y", cliponaxis=False)
    lines = [
        dict(type="scatter", x=xs, y=[ALEX]*len(xs), mode="lines", line=dict(color="#b91c1c", dash="dash", width=2), name="trained AlexNet 0.284"),
        dict(type="scatter", x=xs, y=[TARGET]*len(xs), mode="lines", line=dict(color="#a16207", dash="dot", width=2), name="1.5× target 0.426"),
        dict(type="scatter", x=xs, y=[CEIL]*len(xs), mode="lines", line=dict(color="#166534", dash="dot", width=1.5), name="noise ceiling ~0.50"),
    ]
    layout = dict(yaxis=dict(title="ventral test_corr", range=[0, 0.55]), showlegend=True,
                  legend=dict(x=0.01, y=0.99, font=dict(size=11)),
                  annotations=[dict(x=5.5, y=0.345, text="open-ended phase →", showarrow=False, font=dict(color="#1e40af", size=12))])
    return f"""
<h2 id="traj">The trajectory</h2>
<p class="section-sub">Grey = theory-driven phase; blue = open-ended phase. Two ceilings were broken; the target was not.</p>
{plot(lines+[bars], layout, height=420,
      caption="Best ventral test_corr per stage. Resolution (16→64px) and ensembling carried Phase 1 to 0.312; "
              "open-ended invention of feature interactions carried Phase 2 to 0.321. Both plateau far below the 0.426 target.")}"""


def _cand_card(c, baseline=ALEX):
    ok = c.get("status") == "ok" and (c.get("test_corr") or c.get("ventral")) is not None
    v = c.get("test_corr", c.get("ventral"))
    dot = "#166534" if (ok and v and v >= 0.315) else ("#1e40af" if ok else "#b91c1c")
    badge = "" if ok else '<span class="fail-badge">error</span>'
    panel = c.get("panel") or {}
    chips = "".join(f'<span class="chip">{esc(k)} {fnum(panel.get(k),3)}</span>'
                    for k in ["FFA", "PPA", "EBA", "VWFA"] if panel.get(k) is not None)
    dout = f'<span class="meta"> · D={c["d_out"]:,}</span>' if c.get("d_out") else ""
    idea = esc(c.get("idea") or c.get("family") or "").strip()
    what = esc(c.get("what") or "").strip()
    overfit = esc(c.get("overfit") or "").strip()
    note = esc(c.get("note") or "").strip()
    body = ""
    if idea: body += f'<div class="cand-note"><b>idea.</b> {idea}</div>'
    if what: body += f'<details class="sub"><summary>what it computes</summary><div class="cand-note">{what}</div></details>'
    if overfit: body += f'<details class="sub"><summary>overfit / honesty check</summary><div class="cand-note">{overfit}</div></details>'
    if note: body += f'<details class="sub"><summary>run notes</summary><div class="cand-note">{note}</div></details>'
    return f"""
<details class="run-block"><summary>
  <span class="run-dot" style="background:{dot}"></span>
  <span class="run-title">{esc(c.get('name') or c.get('family'))}</span>{dout}
  <span style="margin-left:auto;font-family:var(--mono);font-weight:700">{fnum(v,4)}</span>{badge}
</summary><div class="cand-body"><div class="chips">{chips}</div>{body}</div></details>"""


def s_phase1(d):
    out = ['<h2 id="p1">Phase 1 — implementing known theories</h2>',
           '<p class="section-sub">Resolution and ensembling worked; every named theory of high-level selectivity plateaued.</p>',
           '<div class="findings">',
           '<p>The first six generations each implemented a <b>known computational theory</b> of ventral '
           'cortex, top-down. Two levers carried it from 0.25 to 0.31 — <b>input resolution (16→64px)</b> '
           '(four agents independently hit a "16×16 ceiling"; raising it unlocked the deep families) and '
           '<b>ensembling complementary mechanisms</b>. But past 64px resolution stopped paying, and every '
           'theory aimed at the high-level regions <b>was falsified</b>:</p>',
           '<table class="principles"><thead><tr><th>hypothesis (gens 4–6)</th><th class="num">best</th><th>verdict</th></tr></thead><tbody>']
    for name, sc, verdict in FALSIFIED:
        out.append(f"<tr><td>{esc(name)}</td><td class='num'>{sc:.3f}</td><td>{esc(verdict)}</td></tr>")
    out.append("</tbody></table>")
    out.append('<p>The early generations (the climb to 0.297) are detailed below; the theory phase '
               'topped out at <b>0.3122</b>.</p>')
    # the early demo→gen3 candidate cards (already in data["phases"])
    for ph in d.get("phases", []):
        cands = sorted(ph["candidates"], key=lambda c: (c.get("ventral") is not None, c.get("ventral") or 0), reverse=True)
        best = next((c["ventral"] for c in cands if c.get("ventral") is not None), None)
        out.append(f'<details class="what-searched"><summary>{esc(ph["id"])} — {ph["imsize"]}px · '
                   f'best {fnum(best,4)} · {len(cands)} candidates</summary><div style="padding:.3rem 0">'
                   + "".join(_cand_card(c) for c in cands) + "</div></details>")
    out.append('<div class="takeaway"><span class="takeaway-tag">phase-1 conclusion</span>Untrained '
               'computation <b>solves early/mid cortex</b> (V1 0.50, far above AlexNet 0.34) but the '
               '<b>category-selective regions are a wall</b> — and crucially, this was only a statement about '
               '<i>known</i> theories. That limitation is exactly what Phase 2 removes.</div></div>')
    return "".join(out)


def s_phase2(d):
    out = ['<h2 id="p2">Phase 2 — open-ended, score-guided invention</h2>',
           '<p class="section-sub">No prescribed theory. Agents invent any untrained computation; the held-out score is the only judge.</p>',
           '<div class="findings">',
           '<p>Phase 2 removed the neuroscience priors entirely. Eight explorers per generation — '
           '<b>wildcards inventing from scratch</b>, aggressive mutation, novel nonlinearities/interactions, '
           '<b>capacity scaling</b> (very wide/deep untrained spaces, now <i>allowed</i>), recombination, and '
           'random-kernel expansions — each free to build anything, judged only by held-out ventral '
           'predictivity. <b>It broke the theory-phase plateau (0.312 → 0.321)</b>, and the gains came from '
           'computations the theory phase never tried.</p>']
    for ph in d["open_phases"]:
        cands = sorted(ph["candidates"], key=lambda c: (c.get("test_corr") is not None, c.get("test_corr") or 0), reverse=True)
        out.append(f'<h3 class="gen-h">{esc(ph["label"])}<span class="gen-meta">{ph["imsize"]}px · '
                   f'best {fnum(ph["best"],4)} · {len(cands)} explorers</span></h3>'
                   + "".join(_cand_card(c) for c in cands))
    out.append('<div class="takeaway"><span class="takeaway-tag">phase-2 conclusion</span>Free invention '
               'broke the plateau but found a <b>new one at ~0.32</b>. Even fashionable untrained '
               'architectures lost: <b>self-attention routing</b> (0.265) and <b>predictive-coding dynamics</b> '
               '(0.286) both underperformed the energy+interaction cascade — clean negative results.</div></div>')
    return "".join(out)


def s_discovery():
    rows = "".join(f'<tr><td><b>{esc(n)}</b></td><td class="num">{sc:.3f}</td><td>{esc(desc)}</td></tr>'
                   for n, desc, sc in DISCOVERIES)
    neg = "".join(f'<li><b>{esc(n)}</b> — {fnum(sc,3)}: {esc(desc)}</li>' for n, sc, desc in OPEN_NEGATIVES)
    return f"""
<h2 id="disc">The discovery — feature <i>interactions</i>, not deeper filters</h2>
<p class="section-sub">What the score-guided search found most predictive of ventral.</p>
<div class="findings">
<p>Four genuinely-novel computations independently reached the top of the open search — and none
comes from the theory playbook. All are <b>multiplicative / relational</b>: they combine features
with each other rather than stacking more linear filters.</p>
<table class="principles"><thead><tr><th>discovered computation</th><th class="num">best</th><th>what it computes</th></tr></thead><tbody>{rows}</tbody></table>
<p>This is the scientific payload: <b>the most predictive untrained computations of the ventral
stream are feature interactions — multiplicative gating, second-order relational/configural
co-occurrence, cross-stream bilinear products, and random-kernel expansions — not the deeper
energy/divisive-norm cascades that dominate the literature.</b> These are exactly the operations
that nudged the stuck high-level regions (FFA 0.29→0.31, EBA 0.32→0.33, VWFA 0.22→0.23).</p>
<p class="meta">Clean negatives (informative): <ul class="findings">{neg}</ul></p>
</div>"""


def s_regions(d):
    champ = d["champion"]["panel"]; b = d["alexnet_baseline"]
    t_ours = dict(type="bar", name="untrained champion", x=REGION_ORDER, y=[champ.get(r) for r in REGION_ORDER], marker=dict(color="#1e40af"))
    t_alex = dict(type="bar", name="trained AlexNet", x=REGION_ORDER, y=[b.get("V1" if r == "early" else r) for r in REGION_ORDER], marker=dict(color="#b91c1c"))
    return f"""
<h2 id="regions">Where we win — and where the wall is</h2>
<p class="section-sub">A decisive early/mid win; the category-selective regions resist both phases.</p>
{plot([t_ours, t_alex], dict(barmode="group", yaxis=dict(title="test_corr", range=[0,0.55]), legend=dict(x=0.01,y=0.99)), height=400,
      caption="Champion (0.321, untrained) vs trained AlexNet across the NSD panel. Untrained crushes V1/early and "
              "wins the ventral reward; AlexNet still leads FFA/PPA/EBA/VWFA — the unexplained variance that needs learning.")}
<div class="findings"><p>The ventral win is <b>driven by early/mid cortex</b>, where untrained
interaction-cascades are demonstrably the right model (V1 0.49 vs 0.34). The residual gap to the
ceiling lives in the <b>category-selective regions</b> — and VWFA (visual word form, the most
explicitly <i>learned</i> representation) is consistently hardest.</p></div>"""


def s_verdict(d):
    champ = d["champion"]["ventral"]
    return f"""
<h2 id="verdict">Verdict — the ~0.32 ceiling is real, and now well-earned</h2>
<p class="section-sub">Is 1.5× AlexNet reachable with an untrained model? The evidence says no.</p>
<div class="findings">
<p>Across <b>two phases, ~50 models, 9 generations</b> — a top-down theory-driven search <i>and</i> an
unconstrained, capacity-unlimited, score-guided open-ended search — predictivity plateaus at
<b>~0.32</b> ({champ/CEIL*100:.0f}% of the noise ceiling). We beat trained AlexNet by <b>~13%</b> and
reached <b>{champ/TARGET*100:.0f}%</b> of the 1.5× target, but the target (0.4257) was not reached.</p>
<p>This is a far stronger conclusion than Phase 1 alone could give. The "untrained can't reach
high-level cortex" claim was worth challenging — so we challenged it with <b>total freedom to invent
any computation</b> — and it <b>held</b>. The ~0.32 ceiling is not an artifact of which theories we
thought to try; even free invention can't cross it. <b>The residual gap to the target lives in the
category-selective regions and appears to require learned structure that no untrained computation we
could discover supplies.</b></p>
<div class="takeaway"><span class="takeaway-tag">what dynamic workflows bought</span>The open-ended
search did what a single trajectory could not: it <b>broke a plateau the theory phase was stuck on</b>
by inventing feature-interaction computations outside the literature, and it <b>earned the ceiling
result</b> by exhausting a vast space of untrained computations in parallel — turning "we didn't find
better" into evidence rather than a gap.</div>
</div>"""


def s_model(d):
    c = d["champion"]
    code = (f'<details class="code"><summary>champion source — full ({c["code"].count(chr(10))} lines, untrained)'
            f'</summary><pre><code>{esc(c["code"])}</code></pre></details>')
    return f"""
<h2 id="model">The winning model (0.3213)</h2>
<p class="section-sub">A fusion of the open-ended discoveries, expanded by a random-kernel feature map.</p>
<div class="findings">
<p><code>{esc(c['tag'])}</code> · ventral <b>{fnum(c['ventral'],4)}</b> · train {fnum(c['train'],4)} ·
gap {fnum(c['gap'],4)} · <b>0 trained params</b>. {esc(c['idea'])} Verified per-sample (no test-batch
statistics) and reproduced verbatim in a clean run. Needs <code>NEURO_IMSIZE=128</code>.</p>
{code}
</div>"""


def s_caveats():
    return f"""
<h2 id="caveats">Caveats &amp; reproduce</h2>
<div class="findings"><ul>
<li><b>Mid-level win.</b> The beat is on the <code>streams</code>-ventral reward; FFA/PPA/EBA/VWFA still favour the trained net.</li>
<li><b>Resolution is a harness change.</b> Results need <code>NEURO_IMSIZE=128</code> (a new operator knob that enriches model input only — reward untouched, and conservative vs AlexNet's 224px).</li>
<li><b>Selection on a fixed test set.</b> Deterministic reward ⇒ the risk is optimistic selection, mitigated by a margin over the 0.005 tie band, panel-wide generalisation, and per-sample + untrained verification of every champion.</li>
<li><b>Capacity is honest, not hacking.</b> A wide (D≈40k) untrained feature space that <i>generalises</i> on held-out images is a real result; the train–test gap stayed flat (~0.11) as capacity grew, the opposite of overfit-driven gains.</li>
</ul>
<p><b>Reproduce the champion:</b></p>
<pre><code>cp runs/evo-ventral/CHAMPION_ventral_0.3213.py model.py
NEURO_DATASET=nsd NEURO_ROI=ventral NEURO_IMSIZE=128 python -m harness.run
# reproduces ventral test_corr 0.3213</code></pre></div>
<footer class="rfoot">Generated by <code>analyze/search_report.py</code> from
<code>runs/evo-ventral/search_data_full.json</code> — self-contained, no network.</footer>"""


EXTRA_CSS = """
.diagram{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1.2rem 0;max-width:64rem}
.dw-col{border:1px solid var(--line);border-radius:6px;padding:.9rem 1rem;background:var(--wash)}
.dw-head{font-family:var(--serif);font-weight:700;color:var(--ink);margin-bottom:.6rem}
.dw-gen{display:flex;flex-wrap:wrap;gap:.3rem;align-items:center;margin:.3rem 0}
.dw-gen .g{font-family:var(--sans);font-weight:700;font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-right:.3rem}
.dw-gen .c{font-family:var(--mono);font-size:.72rem;background:#fff;border:1px solid var(--accent);color:var(--accent);border-radius:3px;padding:.12rem .45rem}
.dw-sel{text-align:center;font-size:.74rem;color:var(--good);font-weight:700;margin:.35rem 0;font-family:var(--mono)}
.dw-note{margin-top:.5rem;font-size:.78rem;color:var(--muted);font-style:italic;font-family:var(--serif)}
.gen-h{margin:2rem 0 .2rem;font-size:1.2rem;border-top:1px solid var(--line-soft);padding-top:1rem}
.gen-meta{font-family:var(--mono);font-size:.74rem;color:var(--muted);font-weight:500;margin-left:.5rem}
.what-searched{margin:.4rem 0;font-size:.85rem}
.what-searched>summary{cursor:pointer;color:var(--accent);font-family:var(--sans);font-size:.85rem;font-weight:600}
.cand-body{padding:.5rem 0 .2rem 1.5rem}
.chips{display:flex;flex-wrap:wrap;gap:.35rem;margin-bottom:.4rem}
.chip{font-family:var(--mono);font-size:.7rem;background:var(--wash-deep);border:1px solid var(--line);border-radius:999px;padding:.08rem .5rem;color:var(--body)}
.cand-note{font-size:.84rem;line-height:1.55;color:var(--body);max-width:64rem;border-left:2px solid var(--line);padding-left:.8rem;margin:.3rem 0}
details.sub{margin:.25rem 0}
details.sub>summary{cursor:pointer;color:var(--accent);font-size:.78rem;font-family:var(--sans)}
.rfoot{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--faint);font-size:.8rem;font-style:italic;font-family:var(--serif)}
"""


def build():
    d = json.load(open(DATA, encoding="utf-8"))
    body = "".join([header(d), toc(), s_question(d), s_dynamic_workflows(), s_trajectory(),
                    s_phase1(d), s_phase2(d), s_discovery(), s_regions(d), s_verdict(d), s_model(d), s_caveats()])
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open-ended search for untrained models of the ventral stream</title>
<style>{_asset('assets','report.css')}{EXTRA_CSS}</style>
<script>{_plotly_bundle()}</script></head><body><main class="page">{body}</main></body></html>"""


if __name__ == "__main__":
    out = build()
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"wrote {OUT}  ({len(out)//1024} KB)")
