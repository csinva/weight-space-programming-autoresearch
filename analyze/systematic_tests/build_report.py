"""Build analyze/systematic_tests/report.html from the metrics written by systematic-tests/analyze.py, analyze_filtered.py,
exp2_analyze.py and exp3.py, plus the predictor audit files. Main text: Opus 5.5, no-weights vs raw-weights. Views and Fable are in expandables."""
import base64, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ST = os.path.join(REPO, "systematic-tests")
FAMS = ["scalar", "neuron", "layer", "token"]
LOWER_BETTER = {"mae"}


def J(name):
    return json.load(open(os.path.join(HERE, name)))


def img(name):
    return "data:image/png;base64," + base64.b64encode(open(os.path.join(HERE, name), "rb").read()).decode()


def cell(t, ref=None, metric="spearman", nd=2):
    v, lo, hi = t[0], t[1], t[2]
    if v != v:
        return "<td>n/a</td>"
    s = f"{v:.{nd}f} <span class='ci'>± {(hi - lo) / 2:.{nd}f}</span>"
    if ref is not None and ref[0] == ref[0] and abs(v - ref[0]) >= 0.5 * 10 ** -nd:
        d = v - ref[0]
        s += f" <span class='{'pos' if ((d < 0) if metric in LOWER_BETTER else (d > 0)) else 'neg'}'>({d:+.{nd}f})</span>"
    return f"<td>{s}</td>"


def pcell(t, metric="spearman"):
    v, lo, hi = t[0], t[1], t[2]
    if v != v:
        return "<td>n/a</td>"
    better = (v < 0) if metric in LOWER_BETTER else (v > 0)
    se = (hi - lo) / 2
    cls = ("pos" if better else "neg") if abs(v) > 1.96 * se else ""
    return f"<td class='{cls}'>{v:+.2f} <span class='ci'>± {se:.2f}</span></td>"


def audits():
    rows = []
    paths = glob.glob(os.path.join(ST, "rounds*", "*", "*", "*", "audit.json")) + glob.glob(os.path.join(ST, "exp2*", "r*", "*", "*", "audit.json")) + \
        glob.glob(os.path.join(ST, "exp3", "*", "*", "audit.json"))
    for p in sorted(paths):
        a = json.load(open(p))
        rel = os.path.relpath(p, ST)
        exp = "exp3" if rel.startswith("exp3") else "exp2" if rel.startswith("exp2") else "pilot" if rel.startswith("rounds_pilot") else "exp1"
        wf = {os.path.basename(x) for x in a["reads"] if "weights/" in x and not os.path.basename(x).startswith("shared")}
        rows.append(dict(exp=exp, model=a["init"]["model"].replace("claude-", ""), cond=rel.split(os.sep)[-2], reads=a["tool_calls"], wfiles=len(wf),
                         outside=len(a["reads_outside_sandbox"]), cost=a["cost_usd"] or 0, dur=a["duration_s"], retry=a.get("attempt", 0)))
    return rows


def audit_table(rows):
    agg = {}
    for r in rows:
        a = agg.setdefault((r["exp"], r["model"], r["cond"]), dict(n=0, wf=0, reads=0, cost=0, dur=0, out=0, retry=0))
        for k, v in [("n", 1), ("wf", r["wfiles"]), ("reads", r["reads"]), ("cost", r["cost"]), ("dur", r["dur"]), ("out", r["outside"]), ("retry", r["retry"])]:
            a[k] += v
    lab = lambda c: {"none": "Baseline", "raw": "Read raw weights", "rawplus": "Read raw weights + needed numbers", "compute": "Read raw weights + recipe"}.get(
        c.split("_")[0], c.split("_")[0]) + (f" {c.split('_')[1]}" if "_" in c else "")
    return "".join(f"<tr><td>{e}</td><td>{m}</td><td>{lab(c)}</td><td>{a['n']}</td><td>{a['reads'] / a['n']:.1f}</td><td>{a['wf'] / a['n']:.1f}</td><td>{a['out']}</td>"
                   f"<td>{a['retry']}</td><td>${a['cost']:.2f}</td><td>{a['dur'] / a['n']:.0f} s</td></tr>" for (e, m, c), a in sorted(agg.items()))


def abbreviate(text, keep=2):
    """Cut every run of number-grid lines ('  16: +0.0123 ...') to its first `keep` lines plus a note; cut long single lines."""
    import re
    out, run = [], []
    def flush():
        if run:
            out.extend(run[:keep])
            if len(run) > keep:
                out.append(f"   ... ({len(run) - keep} more lines of 16 numbers each)")
            run.clear()
    for line in text.split("\n"):
        if re.match(r"^\s*\d+: [+-]", line):
            run.append(line[:150] + (" ..." if len(line) > 150 else ""))
        else:
            flush()
            out.append(line if len(line) <= 400 else line[:400] + " ...")
    flush()
    return "\n".join(out)


def sample_call():
    """Exact files of one raw-weights call from experiment 1 (round 1, scalar), with numbers abbreviated."""
    import html as _h
    b = os.path.join(ST, "rounds", "r0", "scalar", "raw", "bundle")
    parts = [("README.md", open(os.path.join(b, "README.md")).read()), ("test_interventions.md", open(os.path.join(b, "test_interventions.md")).read())]
    for d, iid in [("scalar", "SC00"), ("neuron", "NE00"), ("layer", "LA00")]:
        p = os.path.join(ST, "rounds", "r0", d, "raw", "bundle", "weights", f"{iid}.txt")
        if os.path.exists(p):
            parts.append((f"weights/{iid}.txt ({d} edit)", abbreviate(open(p).read())))
    sh = os.path.join(b, "weights", "shared_embeddings_targets.txt")
    if os.path.exists(sh):
        parts.append(("weights/shared_embeddings_targets.txt (first entries)", abbreviate("\n".join(open(sh).read().split("\n")[:12]))))
    return "".join(f"<p><b><code>{_h.escape(n)}</code></b></p><pre>{_h.escape(t)}</pre>" for n, t in parts)


AUDIT_HEAD = "<tr><th>experiment</th><th>predictor</th><th>condition</th><th>calls</th><th>tool calls / call</th><th>weight files / call</th><th>reads outside sandbox</th><th>retries</th><th>cost (list price)</th><th>time / call</th></tr>"


def main():
    M, FI, X3 = J("metrics.json"), J("filtered.json"), J("exp3_metrics.json")["res"]
    X2o, XRo = J("exp2_opus_metrics.json")["res"], J("exp2_rep_opus_metrics.json")["res"]
    X2f, XRf = J("exp2_metrics.json")["res"], J("exp2_rep_metrics.json")["res"]
    items = J("items.json")
    zmae = {f: sum(abs(r["true"]) for r in items if r["family"] == f) / sum(r["family"] == f for r in items) for f in FAMS}
    O, OP, FS, sc = M["overall"], M["overall_paired"], M["fewshot"], M["paired"]["scalar"]["raw-none"]
    AU = audits()
    opus_all = [r for r in AU if r["model"] == "opus-5-5" and r["exp"] != "pilot"]
    opus = [r for r in opus_all if r["cond"] != "views"]
    opus_views = [r for r in opus_all if r["cond"] == "views"]
    fable = [r for r in AU if r["model"] == "fable-5-1"]
    pilot = [r for r in AU if r["exp"] == "pilot"]
    cost_o, cost_f = sum(r["cost"] for r in opus), sum(r["cost"] for r in fable)

    # Table 4: exp2 (Opus), none vs raw + baselines
    t4 = ""
    for fam in ["neuron", "mlp", "gain", "all"]:
        cc, pp = X2o["cells"][f"all|{fam}"], X2o["paired"][f"all|{fam}"]
        t4 += (f"<tr><td><b>{fam}</b></td><td>{cc['n']}</td>{cell(cc['none']['pct'])}{cell(cc['raw']['pct'], cc['none']['pct'], 'pct')}{pcell(pp['raw-none']['pct'])}"
               f"<td>{cc['likely']['pct'][0]:.2f}</td><td>{cc['dla_oracle']['pct'][0]:.2f}</td><td>{cc['none']['hit50'][0]:.2f}</td><td>{cc['raw']['hit50'][0]:.2f}</td></tr>")
    rco, rpo = XRo["cells"]["all|gain"], XRo["paired"]["all|gain"]
    t4 += (f"<tr><td><b>gain, second set</b></td><td>{rco['n']}</td>{cell(rco['none']['pct'])}{cell(rco['raw']['pct'], rco['none']['pct'], 'pct')}{pcell(rpo['raw-none']['pct'])}"
           f"<td>{rco['likely']['pct'][0]:.2f}</td><td>{rco['dla_oracle']['pct'][0]:.2f}</td><td>{rco['none']['hit50'][0]:.2f}</td><td>{rco['raw']['hit50'][0]:.2f}</td></tr>")
    x2all = X2o["cells"]["all|all"]
    # Table 5: exp3
    k3 = ["none", "raw", "rawplus", "compute", "code_recipe", "oracle_n"]
    lab3 = dict(none="Baseline", raw="Read raw weights", rawplus="Read raw weights + needed numbers", compute="Read raw weights + numbers + calculation recipe", code_recipe="code runs the recipe", oracle_n="code with the true n_d (needs activations)")
    t5 = ""
    for fam in ["emb", "gain", "all"]:
        cc = X3[fam]["cells"]
        for i, k in enumerate(k3):
            t5 += ("<tr>" + (f"<td rowspan={len(k3)}><b>{fam}</b><br><span class='ci'>{X3[fam]['n']} edits; predict-0 MAE {cc['none']['mae_zero']:.2f}</span></td>" if i == 0 else "") +
                   f"<td>{lab3[k]}</td><td>{cc[k]['pct']:.2f}</td><td>{cc[k]['hit50']:.2f}</td><td>{cc[k]['rho']:+.2f}</td><td>{cc[k]['sign_big']:.2f}</td><td>{cc[k]['mae']:.2f}</td></tr>")
    t5p = ""
    for fam in ["emb", "gain", "all"]:
        pp = X3[fam]["paired"]
        t5p += f"<tr><td><b>{fam}</b></td>" + "".join(pcell(pp[f"{a}|{m}"]) for a in ["compute-none", "compute-rawplus", "rawplus-raw", "raw-none"] for m in ["pct", "rho"]) + "</tr>"
    e3, g3 = X3["emb"]["cells"], X3["gain"]["cells"]

    # expandable: views
    tv = ""
    for fam in FAMS:
        tv += (f"<tr><td>{fam}</td>{cell(M['pooled'][fam]['none']['spearman'])}{cell(M['pooled'][fam]['raw']['spearman'])}{cell(M['pooled'][fam]['views']['spearman'])}"
               f"{pcell(M['paired'][fam]['views-none']['spearman'])}{pcell(M['paired'][fam]['views-none']['big_auc'])}</tr>")
    tv += f"<tr><td><b>all</b></td>{cell(O['none']['spearman'])}{cell(O['raw']['spearman'])}{cell(O['views']['spearman'])}{pcell(OP['views-none']['spearman'])}{pcell(OP['views-none']['big_auc'])}</tr>"
    tvf = "".join(f"<tr><td>{fam}</td><td>{X2f['cells'][f'all|{fam}']['none']['pct'][0]:.2f}</td><td>{X2f['cells'][f'all|{fam}']['views']['pct'][0]:.2f}</td>"
                  f"{pcell(X2f['paired'][f'all|{fam}']['views-none']['pct'])}{pcell(X2f['paired'][f'all|{fam}']['views-dla_plus']['pct'])}</tr>" for fam in ["neuron", "mlp", "gain", "all"])
    # expandable: Fable
    MO, MK = M["models"], M["models_k0"]
    tf = "".join(f"<tr><td>{fam}</td><td>{MO[fam]['none.fable']['spearman'][0]:.2f}</td><td>{MO[fam]['raw.fable']['spearman'][0]:.2f}</td>"
                 f"{pcell(MO[fam]['none.fable-none']['spearman'])}{pcell(MO[fam]['raw.fable-raw']['spearman'])}{pcell(MO[fam]['raw.fable-none.fable']['spearman'])}"
                 f"<td>{MK[fam]['none']['spearman'][0]:.2f} / {MK[fam]['none.fable']['spearman'][0]:.2f}</td><td>{MK[fam]['raw']['spearman'][0]:.2f} / {MK[fam]['raw.fable']['spearman'][0]:.2f}</td></tr>"
                 for fam in FAMS + ["all"])
    tf2 = "".join(f"<tr><td>{s}, {fam}</td><td>{X2f['cells'][f'{s}|{fam}']['none']['pct'][0]:.2f}</td><td>{X2f['cells'][f'{s}|{fam}']['raw']['pct'][0]:.2f}</td>"
                  f"{pcell(X2f['paired'][f'{s}|{fam}']['raw-none']['pct'])}</tr>" for s, fam in [("all", "all"), ("KL>0.1", "all"), ("all", "gain"), ("KL>0.1", "gain")])
    tf2 += f"<tr><td>gain, second set (32 edits)</td><td>{XRf['cells']['all|gain']['none']['pct'][0]:.2f}</td><td>{XRf['cells']['all|gain']['raw']['pct'][0]:.2f}</td>{pcell(XRf['paired']['all|gain']['raw-none']['pct'])}</tr>"

    Z = J("zeroshot.json")
    T = Z["test"]
    zs = Z["all_zeroshot"]["scalar"]
    B, R, RA = "Baseline", "Read raw weights", "Read all weights"
    gts = {}
    for g in glob.glob(os.path.join(ST, "rounds", "r*", "ground_truth.json")):
        gts.update(json.load(open(g)))
    fam_of = {it["iid"]: f for f, v in json.load(open(os.path.join(ST, "data", "interventions.json"))).items() for it in v}
    zs_iids = set()
    for d in [os.path.join(ST, "rounds", "r0")] + [os.path.join(ST, "rounds", f"r{r}") for r in range(3, 9)] + [os.path.join(ST, "rounds_fewshot", "r2_k0")]:
        for pp in glob.glob(os.path.join(d, "*", "none", "prediction.json")):
            zs_iids |= {k.split("|")[0] for k in json.load(open(pp))["predictions"]}
    zmae = {}
    for f in ["scalar", "neuron", "layer"]:
        v = [abs(x["dlogp"]) for i in zs_iids if fam_of.get(i) == f for x in gts[i]["per_prompt"].values()]
        zmae[f] = sum(v) / len(v)
    F4 = ["scalar", "neuron", "layer"]
    beaten = [f for f in F4 if zmae[f] < min(Z["all_zeroshot"][f][k]["mae"][0] for k in ["none_v0", "raw_v0"])]
    se = lambda t: (t[0], t[0] - t[1], t[0] + t[1])
    t1 = ""
    for fam in F4 + ["all"]:
        d = Z["all_zeroshot"][fam]
        mlist = [("spearman", "Spearman ρ (items)"), ("edit_rank", "edit-level rank ρ"), ("big_auc", "large-effect AUC"), ("mae", "MAE (nats)")]
        for i, (m, lab) in enumerate(mlist):
            t1 += ("<tr>" + (f"<td rowspan={len(mlist)}><b>{fam}</b><br><span class='ci'>{d['none_v0']['n_edits']} edits" + (f"; predict-0 MAE {zmae[fam]:.3f}" if fam in zmae else "") + "</span></td>" if i == 0 else "") +
                   f"<td>{lab}</td>{cell(se(d['none_v0'][m]))}{cell(se(d['raw_v0'][m]), se(d['none_v0'][m]), m)}{pcell(se(d['raw_v0-none_v0'][m]), m)}</tr>")
    alt_sec = ""
    nav_alt = ""
    alt_summary = ""
    if os.path.exists(os.path.join(HERE, "zeroshot_alt.json")):
        ZA = J("zeroshot_alt.json")
        ALTT = json.load(open(os.path.join(ST, "data", "alt_targets.json")))
        t1a = ""
        for fam in F4 + ["all"]:
            d = ZA["all_zeroshot"][fam]
            mlist = [("spearman", "Spearman ρ (items)"), ("edit_rank", "edit-level rank ρ"), ("big_auc", "large-effect AUC"), ("mae", "MAE (nats)")]
            has_all = "raw_all" in d
            for i, (m, lab) in enumerate(mlist):
                t1a += ("<tr>" + (f"<td rowspan={len(mlist)}><b>{fam}</b><br><span class='ci'>{d['none_v0']['n_edits']} edits</span></td>" if i == 0 else "") +
                        f"<td>{lab}</td>{cell(se(d['none_v0'][m]))}{cell(se(d['raw_v0'][m]), se(d['none_v0'][m]), m)}{pcell(se(d['raw_v0-none_v0'][m]), m)}" +
                        (f"{cell(se(d['raw_all'][m]), se(d['none_v0'][m]), m)}{pcell(se(d['raw_all-none_v0'][m]), m)}{pcell(se(d['raw_all-raw_v0'][m]), m)}" if has_all else "") + "</tr>")
        za, zas = ZA["all_zeroshot"], ZA["all_zeroshot"]["scalar"]
        all_head = f"<th>{RA}</th><th>{RA} − {B}</th><th>{RA} − {R}</th>" if "raw_all" in za["all"] else ""
        fa = ZA["filtered"]
        alt_summary = (f"When the scored token is a plausible lower-ranked one instead of the top-1 (Sec. <a href=\"#sec-2-2\">2.2</a>), the item-level gain disappears "
                       f"(ρ {ZA['all_zeroshot']['all']['raw_v0-none_v0']['spearman'][0]:+.2f} ± {ZA['all_zeroshot']['all']['raw_v0-none_v0']['spearman'][1]:.2f}), although {R} still ranks which edits matter better "
                       f"(edit rank {ZA['all_zeroshot']['all']['raw_v0-none_v0']['edit_rank'][0]:+.2f} ± {ZA['all_zeroshot']['all']['raw_v0-none_v0']['edit_rank'][1]:.2f}; "
                       f"{ZA['all_zeroshot']['scalar']['raw_v0-none_v0']['edit_rank'][0]:+.2f} for scalars, {ZA['all_zeroshot']['neuron']['raw_v0-none_v0']['edit_rank'][0]:+.2f} for neurons; Table <a href=\"#tab-1b\">1b</a>).\n")
        nav_alt = ', <a href="#sec-2-2">2.2 Lower-ranked token</a>'
        alt_sec = rf"""<h3 id="sec-2-2">2.2 Predicting a lower-ranked token</h3>
<p>The same experiment, repeated with each prompt scored on a plausible next token ranked 6 to 10 before the edit instead of the top-1:
{', '.join(f"{k} {v['token']!r} (rank {v['rank']}, p {v['p']:.4f})" for k, v in ALTT.items())}.
The predictor's README names the scored token and its rank for each prompt; everything else is unchanged. All {za['all']['none_v0']['n_edits']} scalar, neuron and layer edits were predicted with no examples,
{B} against {R} with the original prompt.</p>
<p id="tab-1b"><b>Table 1b.</b> Lower-ranked target. Values ± 1 standard error; in parentheses {R} − {B}; last column the paired difference, shaded if beyond 1.96 standard errors.</p>
<table><tr><th>family</th><th>metric</th><th>{B}</th><th>{R}</th><th>{R} − {B} (paired)</th>{all_head}</tr>{t1a}</table>
<img id="fig-1b" src="{img('fig1_zeroshot_alt.png')}"><p class="cap"><b>Fig. 1b.</b> Spearman ρ and MAE per family for the lower-ranked target, {B} against {R} (n = edits). Error bars are ±1 standard error of the mean (bootstrap over edits).</p>
<img id="fig-2b" src="{img('fig2_zeroshot_scatter_alt.png')}"><p class="cap"><b>Fig. 2b.</b> Predicted against measured Δlogp of the lower-ranked token, {B} (top) and {R} (bottom), on symmetric log axes (linear within ±0.01 nats). No error bars (one point per item).</p>
<p>{R} − {B}: scalar {zas['raw_v0-none_v0']['spearman'][0]:+.2f} ± {zas['raw_v0-none_v0']['spearman'][1]:.2f}, neuron {za['neuron']['raw_v0-none_v0']['spearman'][0]:+.2f} ± {za['neuron']['raw_v0-none_v0']['spearman'][1]:.2f},
layer {za['layer']['raw_v0-none_v0']['spearman'][0]:+.2f} ± {za['layer']['raw_v0-none_v0']['spearman'][1]:.2f}, all {za['all']['raw_v0-none_v0']['spearman'][0]:+.2f} ± {za['all']['raw_v0-none_v0']['spearman'][1]:.2f}.
Sign accuracy on items with |Δlogp| &gt; 0.1: {B} {fa['|Δ|>0.1|all']['sign']['none_v0']:.2f}, {R} {fa['|Δ|>0.1|all']['sign']['raw_v0']:.2f} ({fa['|Δ|>0.1|all']['n']} items).</p>

"""
    explain_sec = ""
    if os.path.exists(os.path.join(HERE, "explain.json")) and os.path.exists(os.path.join(HERE, "explain_alt.json")):
        EX, EXA = J("explain.json")["res"], J("explain_alt.json")["res"]
        kinds = dict(add_value="add ±4 to one entry", max_weight="zero one entry", norm_gain="zero or ×10 one norm gain", neuron="zero or ×5 one neuron", layer="skip a sublayer or block")
        tex = "".join(f"<tr><td>{fam}</td>" + "".join(f"<td>{X[fam]['mean_benefit'][0]:+.3f} <span class='ci'>± {X[fam]['mean_benefit'][1]:.3f}</span></td><td>{X[fam]['rho_rel']:+.2f}</td><td>{X[fam]['rho_effect']:+.2f}</td>" for X in [EX, EXA]) + "</tr>"
                      for fam in ["scalar", "neuron", "layer", "all"])
        tek = "".join(f"<tr><td>{lab}</td>" + "".join(f"<td>{X['kind|' + k][1]:+.3f} <span class='ci'>± {X['kind|' + k][2]:.3f}</span> (n={X['kind|' + k][0]})</td>" for X in [EX, EXA]) + "</tr>" for k, lab in kinds.items())
        explain_sec = f"""<h3 id="sec-2-3">2.3 What predicts when reading helps?</h3>
<p>For each edit, the benefit of reading the weights is how much closer {R} gets to the true effect sizes than {B}: the mean over the edit's prompts of
|pr(|{B} prediction|) − pr(|true Δlogp|)| − |pr(|{R} prediction|) − pr(|true Δlogp|)|, where pr is the percentile rank within the family (positive = reading helps; this ignores calibration).
It is compared with the edit's size relative to its matrix (|ΔW| over the matrix std for one entry, relative norm for a neuron, norm for a layer) and with the size of the true effect.</p>
<p id="tab-2-3"><b>Table 2.3.</b> Mean benefit ± 1 standard error, and its Spearman correlation with relative size and with true effect size, for the top-1 target (Table <a href="#tab-1">1</a>) and the lower-ranked target (Table <a href="#tab-1b">1b</a>).</p>
<table><tr><th rowspan=2>family</th><th colspan=3>top-1 target</th><th colspan=3>lower-ranked target</th></tr><tr><th>benefit</th><th>ρ with size</th><th>ρ with effect</th><th>benefit</th><th>ρ with size</th><th>ρ with effect</th></tr>{tex}</table>
<p>By edit type:</p><table><tr><th>edit</th><th>benefit, top-1 target</th><th>benefit, lower-ranked target</th></tr>{tek}</table>
<p>No simple size measure explains it: the correlations are at most 0.23 with size and near 0 with the true effect size. The clearest pattern is the edit type: zeroing one entry (half of them the largest-magnitude weight in their matrix) benefits most for both targets,
and single scalars with a change 10 to 200 times the matrix std benefit more than smaller or much larger ones.</p>
"""
    VL = dict(none_v0=f"{B}", none_v1=f"{B} v1", raw_v0=f"{R}", raw_v1=f"{R} v1", raw_v2=f"{R} v2")
    DK = ["none_v0", "none_v1", "raw_v0", "raw_v1", "raw_v2"]
    TK = ["none_v0", "none_v1", "raw_v0", "raw_v2"]
    tdev = "".join(f"<tr><td><b>{fam}</b></td>" + "".join(f"<td>{Z['dev'][fam][k]['spearman'][0]:+.2f} <span class='ci'>± {Z['dev'][fam][k]['spearman'][1]:.2f}</span><br><span class='ci'>MAE {Z['dev'][fam][k]['mae'][0]:.3f}</span></td>" for k in DK) + "</tr>" for fam in F4 + ["all"])
    ttest = "".join(f"<tr><td><b>{fam}</b></td>" + "".join(f"<td>{T[fam][k]['spearman'][0]:+.2f} <span class='ci'>± {T[fam][k]['spearman'][1]:.2f}</span><br><span class='ci'>MAE {T[fam][k]['mae'][0]:.3f}</span></td>" for k in TK) +
                    pcell(se(T[fam]["raw_v0-none_v0"]["spearman"])) + pcell(se(T[fam]["raw_v2-none_v1"]["spearman"])) + "</tr>" for fam in F4 + ["all"])
    FZ = Z["fewshot"]
    tfs = ""
    for c, lab_c in [("none", B), ("raw", R)]:
        for i, (m, lab) in enumerate([("spearman", "Spearman ρ"), ("edit_rank", "edit-level rank ρ"), ("mae", "MAE (nats)")]):
            ref = se(FZ[f"{c}@k0"][m])
            tfs += ("<tr>" + (f"<td rowspan=3><b>{lab_c}</b></td>" if i == 0 else "") + f"<td>{lab}</td>" + cell(ref) +
                    "".join(cell(se(FZ[f"{c}@{k}"][m]), ref, m) for k in ["k4", "k24"]) + "</tr>")
    tfi = ""
    for key in ["|Δ|>0.1|all", "|Δ|>0.1|scalar", "|Δ|>0.1|layer", "|Δ|>1|all", "|Δ|>1|scalar", "edit KL>0.1|all", "edit KL>0.1|scalar", "edit KL>0.1|layer"]:
        v = Z["filtered"][key]
        b = v["boot"]
        tfi += (f"<tr><td>{key.replace('|', ', ').replace('>', ' &gt; ')}</td><td>{v['n']}</td><td>{b['none_v0']['spearman'][0]:+.2f}</td><td>{b['raw_v0']['spearman'][0]:+.2f}</td>"
                f"<td>{v['sign']['none_v0']:.2f}</td><td>{v['sign']['raw_v0']:.2f}</td>{pcell(se(b['raw_v0-none_v0']['spearman']))}</tr>")
    import html as _html
    INSTR = ("Read README.md in the current directory and follow it exactly. Read test_interventions.md and any other files there "
             "that you find useful (they are the only material you have). End with the JSON block described in README.md.")
    tprompts = "".join(
        f"<tr><td>{p['pid']}</td><td><code style='white-space:pre'>{_html.escape(repr(p['text']))}</code></td><td><code style='white-space:pre-wrap'>{_html.escape(' | '.join(repr(t) for t in p['input_tokens']))}</code></td>"
        f"<td><code>{_html.escape(repr(p['target']))}</code> (id {p['target_id']})</td><td>{p['top10'][0]['p']:.3f}</td>"
        f"<td><code>{_html.escape(', '.join(f'{x[chr(116) + chr(111) + chr(107) + chr(101) + chr(110)]!r} {x[chr(112)]:.3f}' for x in p['top10']))}</code></td></tr>"
        for p in json.load(open(os.path.join(ST, "data", "prompts.json"))))
    H = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reading Qwen3.5 weights</title>
<style>
:root {{ --ink:#1a1a1a; --dim:#6b7280; --line:#e5e7eb; --bg:#fff; --panel:#fafafa; --accent:#2563eb; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; line-height:1.55; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:40px 24px 90px; }}
h1 {{ font-size:28px; margin:0 0 6px; }} h2 {{ font-size:20px; margin:44px 0 8px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
a {{ color:var(--accent); text-decoration:none; }} .top {{ font-size:13px; color:var(--dim); margin-bottom:18px; }}
.summary {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 20px; }}
table {{ border-collapse:collapse; font-size:13px; margin:10px 0; font-variant-numeric:tabular-nums; }}
th, td {{ border-bottom:1px solid var(--line); padding:4px 10px; text-align:left; vertical-align:top; }} th {{ background:var(--panel); }}
.ci {{ color:var(--dim); font-size:11px; }} .pos {{ color:#1d4ed8; }} .neg {{ color:#dc2626; }} td.pos {{ background:#eff6ff; }} td.neg {{ background:#fef2f2; }}
img {{ max-width:100%; border:1px solid var(--line); border-radius:6px; }} .cap {{ color:var(--dim); font-size:13px; }}
code {{ background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:12px; }}
details {{ margin:14px 0; border:1px solid var(--line); border-radius:8px; padding:8px 14px; }} summary {{ cursor:pointer; font-weight:600; }}
pre {{ white-space:pre-wrap; font-size:12px; background:var(--panel); padding:10px; border-radius:6px; }}
</style></head><body><div class="wrap">
<div class="top"><a href="../index.html">&larr; Analysis Reports index</a></div>
<h1>Can Claude Opus 5.5 predict weight edits by reading Qwen3.5-0.8B's weights?</h1>

<div class="summary">
<p><b>Bottom line: this is a fairly negative result.</b> Reading the weights helps Claude only a little in predicting what a weight edit does, mostly for single-scalar edits,
and the gain disappears when the predicted token is not the top-1. It can often tell which edits are large, but the direction of an effect depends on activations that the weights
alone do not reveal. Reading all of a model's weights at once, without running the network, is not yet a plausible way to predict what an edit does: given every weight of the model
and asked to trace each edit through the downstream weights, it read a handful of rows and declined the arithmetic, noting that 1024-dimensional dot products by hand are not practical.</p>
<p><b>What was done.</b> As a controlled follow-up to recent weight-space programming posts (<a href="#sec-0">Context</a>), Claude Opus 5.5, run headless with only the file Read tool (no code execution), predicted how weight edits change the
next-token distribution of Qwen3.5-0.8B, either from the edit description alone (<i>{B}</i>) or after reading the raw weight numbers of the edited component (<i>{R}</i>).
There were three experiments ({len(opus)} predictor calls in total).
<b>(1)</b> {Z['all_zeroshot']['all']['none_v0']['n_edits']} distinct ablation and scaling edits at 3 granularities (single scalar, MLP neuron, layer or sublayer skip), each scored on 8 prompts by the change in log-probability of the top-1 token.
<b>(2)</b> 124 amplification edits that visibly shift the output, where the model names the 10 tokens that gain the most probability.
<b>(3)</b> 48 single-dimension edits whose effect is an exact small formula, with the model asked to do the arithmetic by hand.</p>
<p><b>Key takeaway.</b> Reading raw weights helps for single-scalar edits and a little for layer skips, and not for neurons.
For scalars, ρ between predicted and measured Δlogp rises by {zs['raw_v0-none_v0']['spearman'][0]:+.2f} ± {zs['raw_v0-none_v0']['spearman'][1]:.2f} (± 1 standard error), the large-effect AUC from {zs['none_v0']['big_auc'][0]:.2f} to {zs['raw_v0']['big_auc'][0]:.2f},
and the sign of large effects (|Δlogp| &gt; 1) is right {Z['filtered']['|Δ|>1|scalar']['sign']['raw_v0']:.0%} of the time against {Z['filtered']['|Δ|>1|scalar']['sign']['none_v0']:.0%} (Table <a href="#tab-1">1</a>, Table <a href="#tab-5">5</a>).
For layer skips ρ rises by {Z['all_zeroshot']['layer']['raw_v0-none_v0']['spearman'][0]:+.2f} ± {Z['all_zeroshot']['layer']['raw_v0-none_v0']['spearman'][1]:.2f}. Across all families the gain is {Z['all_zeroshot']['all']['raw_v0-none_v0']['spearman'][0]:+.2f} ± {Z['all_zeroshot']['all']['raw_v0-none_v0']['spearman'][1]:.2f}, and absolute accuracy stays low
(ρ {Z['all_zeroshot']['all']['raw_v0']['spearman'][0]:.2f}); for {', '.join(beaten)} edits, always predicting "no change" has lower error than either condition.
{alt_summary}Prompts with calibration guidance or a calculation procedure lower the error of both conditions but do not widen the gap (Table <a href="#tab-3">3</a>).
In (2), naming the tokens that rise works no better than listing the tokens that were already likely ({x2all['raw']['pct'][0]:.2f} and {x2all['none']['pct'][0]:.2f} against {x2all['likely']['pct'][0]:.2f}, Table <a href="#tab-6">6</a>).
In (3), asking for the explicit calculation helps on embedding-entry edits (score {e3['compute']['pct']:.2f} against {e3['none']['pct']:.2f}, Table <a href="#tab-7">7</a>) but hurts on final-norm gain edits ({g3['compute']['pct']:.2f} against {g3['none']['pct']:.2f}):
the arithmetic is exact once the final hidden state at the edited dimension is known (oracle {g3['oracle_n']['pct']:.2f}), and the bottleneck is estimating that hidden state, which the weights alone do not determine.</p>
</div>

<h2 id="sec-0">Context</h2>
<p>Two recent posts argue that language models can work directly in weight space.
<a href="https://rikiyat.github.io/blog/weight-space-programming/">Sparks of Weight-Space Programming</a> (Takehi, Teoh, Zhang and Khattab) reports a frontier model that names the game behind
four PPO policies from their raw weights, repairs a stuck policy by editing 3.5% of its weights, and edits 0.016% of Qwen3.5-0.8B's weights to raise MATH-500 from 7.9% to 48.1%.
<a href="../toy_examples/report.html">Weight-Space Programming from scratch on simple tasks</a> had Claude set every weight of small transformers by hand for 13 tasks with known algorithms.
Both show striking individual successes, mostly from few trials and without controls that separate reading the weights from what the model already knows.</p>
<p>This report is a simpler but more rigorous follow-up. It asks a narrower question about a real model, the same Qwen3.5-0.8B: if Claude reads the weights, can it predict what an edit will do to the output?
The task is easy to score, and the design is built to tell real reading of weights apart from other explanations:</p>
<ul><li><b>Many trials:</b> {Z['all_zeroshot']['all']['none_v0']['n_edits'] + 124 + 48} edits, from a single scalar to a whole layer, each measured on 8 prompts.</li>
<li><b>A no-weights control:</b> the same prompt and the same edit descriptions without the numbers. Any gain must come from the weights, not from priors about transformers.</li>
<li><b>No code:</b> the predictor's only tool is Read, and every transcript is checked, so any calculation is the model's own.</li>
<li><b>Uncertainty and baselines:</b> paired comparisons with standard errors over edits, and code baselines (always predict 0, list the already likely tokens, copy the vocabulary projection).</li></ul>

<p>Sections: <a href="#sec-0">Context</a> · <a href="#sec-1">1 Setup</a> · <a href="#sec-2">2 Ablation and scaling edits</a> (<a href="#sec-2-1">2.1 Prompt variants</a>{nav_alt}, <a href="#sec-2-3">2.3 What predicts when reading helps</a>) · <a href="#sec-3">3 Learning from examples</a> ·
<a href="#sec-4">4 Only cases where the output changed</a> · <a href="#sec-5">5 Which tokens rise?</a> · <a href="#sec-6">6 Simulating the calculation</a> ·
<a href="#sec-7">7 Did it read the weights?</a> · <a href="#sec-8">8 Caveats</a></p>

<h2 id="sec-1">1 Setup</h2>
<p><b>Model.</b> <code>Qwen/Qwen3.5-0.8B</code> (revision 2fc0636), text part, fp32 on CPU. 24 layers: 18 Gated DeltaNet (linear attention, 16 heads)
and 6 full softmax attention layers (3, 7, 11, 15, 19, 23; 8 heads with an output gate), hidden size 1024, 3,584 MLP neurons, tied embeddings.</p>
<p><b>Prompts.</b> 8 fixed prompts where the model is fairly confident (top-1 probability 0.27 to 0.99): <i>Paris</i> after "Germany: Berlin / Italy: Rome / France:",
<i>five</i>, <i>+</i> in <code>return a</code>, <i>Thursday</i>, <i>Paris</i> (Eiffel Tower), <i>oxygen</i>, <i>Shakespeare</i>, <i>are</i> ("The keys to the cabinet").</p>
<details><summary>All prompts and completions, exactly</summary>
<p>Each prompt is fed as raw text (no chat template, no BOS token added). The target is the unedited model's top-1 next token; scores use its log-probability.
Strings are shown in Python notation, so <code>\\n</code> is a newline and leading spaces are part of the token.</p>
<table><tr><th>id</th><th>prompt text</th><th>input tokens</th><th>target (top-1)</th><th>p</th><th>unedited top-10 next tokens</th></tr>{tprompts}</table>
</details>
<p><b>Predictor.</b> <code>claude -p --model claude-opus-5-5 --restricted --tools Read --strict-mcp-config</code> in a fresh sandbox directory outside any repository
(no CLAUDE.md, no memory). Its only tool is Read, so it cannot run code, and any calculation is its own. Every transcript is checked (Sec. <a href="#sec-7">7</a>).
With <i>{R}</i> it must read every edit's weight file; calls that skip one are rejected and rerun. <i>{B}</i> gets the same prompt without the weight files.
Both conditions see an architecture summary, the prompts, and the unedited model's top-10 next tokens. No activations are given.</p>
<details><summary>Exactly what the predictor sees (one {R} call, numbers abbreviated)</summary>
<p>The predictor is started in a directory holding these files and told: <code>{_html.escape(INSTR)}</code>
It can only open files with Read. The {B} call gets the same README without the weight files and without the "Required" paragraph. Weight grids are printed with 4 decimals, 16 values per line, each line starting with the index of its first value; edited entries are marked with [ ]. Below, each grid is cut to its first two lines.</p>
{sample_call()}
</details>
<p>Edits are sampled with a fixed seed. Magnitudes such as ±4, ×5 or ×30 were chosen from pilots on separate components. Uncertainties are ±1 standard error, estimated by bootstrap over edits (for means this is the standard error of the mean over edits). Paired differences are shaded when they exceed 1.96 standard errors.</p>

<h2 id="sec-2">2 Experiment 1: ablation and scaling edits</h2>
<p>Edits at 4 granularities, each predicted with no examples: <b>scalar</b> (zero or ×10 one RMSNorm gain; zero the largest-|w| or a random entry of a matrix; add ±4 to one embedding or down_proj entry),
<b>neuron</b> (zero or ×5 one MLP neuron), <b>layer</b> (skip the mixer, the MLP or the whole block of one layer; all 72 such edits are used).
{Z['all_zeroshot']['scalar']['none_v0']['n_edits']} scalar and {Z['all_zeroshot']['neuron']['none_v0']['n_edits']} neuron edits and {Z['all_zeroshot']['layer']['none_v0']['n_edits']} layer edits
({Z['all_zeroshot']['all']['none_v0']['n_items']} scored items), predicted 12 to 16 per call, each scored on the 8 prompts by the change in log-probability of each prompt's top-1 token (Δlogp, nats).
Metrics: Spearman ρ over items; edit-level rank ρ (does it know which edits matter); large-effect AUC (|pred| as a detector of |Δlogp| &gt; 0.1); MAE.</p>
<p id="tab-1"><b>Table 1.</b> {B} against {R}, original prompt. Values ± 1 standard error. In parentheses: {R} − {B} (blue = better, red = worse; lower MAE is better). Last column: paired difference, shaded if beyond 1.96 standard errors.</p>
<table><tr><th>family</th><th>metric</th><th>{B}</th><th>{R}</th><th>{R} − {B} (paired)</th></tr>{t1}</table>
<img id="fig-1" src="{img('fig1_zeroshot.png')}"><p class="cap"><b>Fig. 1.</b> Spearman ρ and MAE per family, {B} against {R} (n = edits). Error bars are ±1 standard error of the mean (bootstrap over edits).</p>
<img id="fig-2" src="{img('fig2_zeroshot_scatter.png')}"><p class="cap"><b>Fig. 2.</b> Predicted against measured Δlogp, {B} (top) and {R} (bottom), on symmetric log axes (logarithmic in both directions, linear within ±0.01 nats); the dashed line is perfect prediction.
Both conditions almost always predict a drop, while measured effects go both ways. No error bars (one point per item).</p></table>
<ul>
<li><b>Single scalars: reading the weights helps.</b> ρ {zs['raw_v0-none_v0']['spearman'][0]:+.2f} ± {zs['raw_v0-none_v0']['spearman'][1]:.2f}, edit rank {zs['raw_v0-none_v0']['edit_rank'][0]:+.2f} ± {zs['raw_v0-none_v0']['edit_rank'][1]:.2f},
large-effect AUC {zs['raw_v0']['big_auc'][0]:.2f} against {zs['none_v0']['big_auc'][0]:.2f}, and the sign of large effects is right far more often:
{Z['filtered']['|Δ|>0.1|scalar']['sign']['raw_v0']:.2f} against {Z['filtered']['|Δ|>0.1|scalar']['sign']['none_v0']:.2f} for |Δlogp| &gt; 0.1 ({Z['filtered']['|Δ|>0.1|scalar']['n']} items) and
{Z['filtered']['|Δ|>1|scalar']['sign']['raw_v0']:.2f} against {Z['filtered']['|Δ|>1|scalar']['sign']['none_v0']:.2f} for |Δlogp| &gt; 1 ({Z['filtered']['|Δ|>1|scalar']['n']} items).
Most large effects come from adding ±4 to an embedding entry of a target token, and the difference is almost all on prompts whose target is a <i>different</i> token:
pushing up one token's logit on every prompt takes probability from the other prompts' targets, so the true effect is a drop. On these 30 items {B} predicted no change or a rise 26 times, while {R}
(seeing, for example, a large final-norm gain on the edited dimension) predicted a drop 29 times.
When the edit hits the prompt's own target and the effect is a rise, both conditions get the sign wrong (8 items), because the sign depends on the hidden state at that dimension.
The size of the effects stays hard: MAE is slightly worse than {B} because {R} predicts larger changes.
On the 80 held-out edits of rounds 5 to 9 alone the gain is {T['scalar']['raw_v0-none_v0']['spearman'][0]:+.2f} ± {T['scalar']['raw_v0-none_v0']['spearman'][1]:.2f}.</li>
<li><b>Layer skips:</b> a smaller gain, ρ {Z['all_zeroshot']['layer']['raw_v0-none_v0']['spearman'][0]:+.2f} ± {Z['all_zeroshot']['layer']['raw_v0-none_v0']['spearman'][1]:.2f} and edit rank {Z['all_zeroshot']['layer']['raw_v0-none_v0']['edit_rank'][0]:+.2f} ± {Z['all_zeroshot']['layer']['raw_v0-none_v0']['edit_rank'][1]:.2f}, from the norm tables of each layer.</li>
<li><b>Neurons:</b> every edit moves the target log-prob by very little (predict-0 MAE {zmae['neuron']:.3f}); {R} ranks edits better (edit rank {Z['all_zeroshot']['neuron']['raw_v0-none_v0']['edit_rank'][0]:+.2f}) but item ρ stays near 0.</li>
</ul>
<p><b>Example.</b> Adding −4 to entry 880 of the <i>Paris</i> embedding (final-norm gain at dim 880 = 4.70) changes the Paris prompts by +0.01 and +0.70 and the <b>Shakespeare</b> prompt by −8.74, because <i>Paris</i> overtakes it.
With raw weights the model predicted −2.5 / −2.2 / −0.02; without, −0.4 / −0.5 / 0.0. It read the large gain, but it could not know the sign of the hidden state at dim 880, which decides where the effect lands.</p>

<h3 id="sec-2-1">2.1 Prompt variants</h3>
<p>Two changes to the prompt were tried, chosen on a dev set (round 4: 16 edits per family, 12 for layer) and then tested on held-out edits (rounds 5 to 9: {T['all']['none_v0']['n_edits']} edits).
<b>v1</b> adds calibration guidance (most single edits move a log-probability by under 0.05 nats; predict near 0 unless there is a concrete reason). It was given to both conditions.
<b>v2</b>, for {R} only, adds a step-by-step procedure: judge the edit's size from the norm tables, its path to the output, and its direction from the dot product with the target embedding rows, then convert a logit change into a log-probability change by hand.
On the dev set, {R} v2 against {B} v1 had the largest family-averaged ρ gap, so that pair went to the test set together with the original prompts.</p>
<p id="tab-2"><b>Table 2.</b> Dev set (round 4), Spearman ρ and MAE ± 1 standard error.</p>
<table><tr><th>family</th>{''.join(f'<th>{VL[k]}</th>' for k in DK)}</tr>{tdev}</table>
<p id="tab-3"><b>Table 3.</b> Held-out test set (rounds 5 to 9): ρ, MAE, and paired ρ differences ± 1 standard error.</p>
<table><tr><th>family</th>{''.join(f'<th>{VL[k]}</th>' for k in TK)}<th>{R} − {B}, original</th><th>{R} v2 − {B} v1</th></tr>{ttest}</table>
<p>Calibration guidance lowers MAE for both conditions (all families: {T['all']['none_v0']['mae'][0]:.3f} to {T['all']['none_v1']['mae'][0]:.3f} for {B}).
It does not widen the gap between the conditions: on the test set {R} v2 − {B} v1 is {T['all']['raw_v2-none_v1']['spearman'][0]:+.2f} ± {T['all']['raw_v2-none_v1']['spearman'][1]:.2f}, against {T['all']['raw_v0-none_v0']['spearman'][0]:+.2f} ± {T['all']['raw_v0-none_v0']['spearman'][1]:.2f} with the original prompt,
and on scalars the procedure removes most of the original prompt's gain ({T['scalar']['raw_v2-none_v1']['spearman'][0]:+.2f} against {T['scalar']['raw_v0-none_v0']['spearman'][0]:+.2f}). Pushing predictions toward 0 appears to flatten the large scalar predictions that carried the signal.</p>

{alt_sec}{explain_sec}<h2 id="sec-3">3 Learning from examples</h2>
<p>Earlier rounds of experiment 1 showed the model the measured effects (and weight files) of previous edits. A fixed test set (round 3's {Z['fewshot_n_edits']} non-token edits) was predicted with 0, 4 or 24 solved examples.</p>
<p id="tab-4"><b>Table 4.</b> ρ, edit-level rank and MAE ± 1 standard error; in parentheses the change from 0 examples.</p>
<table><tr><th>condition</th><th>metric</th><th>0 examples</th><th>4 examples</th><th>24 examples</th></tr>{tfs}</table>
<p>Examples improve the edit ranking with raw weights ({R}: {FZ['raw@k24-raw@k0']['edit_rank'][0]:+.2f} ± {FZ['raw@k24-raw@k0']['edit_rank'][1]:.2f} with 24 examples) and lower MAE for {B}. The item ρ changes stay within about 2 standard errors.</p>

<h2 id="sec-4">4 Only cases where the output changed</h2>
<p>The zero-shot predictions of Table <a href="#tab-1">1</a>, scored only on items or edits whose output changed noticeably (the same subset for both conditions).</p>
<p id="tab-5"><b>Table 5.</b> n items, Spearman ρ, sign accuracy (|Δlogp| &gt; 0.1), and paired {R} − {B} ρ ± 1 standard error.</p>
<table><tr><th>subset</th><th>n</th><th>ρ {B}</th><th>ρ {R}</th><th>sign {B}</th><th>sign {R}</th><th>{R} − {B}</th></tr>{tfi}</table>
<p>On real changes the clearest difference is the sign of large scalar effects. Within these cuts ρ is noisy and often negative for both conditions, because the ranking among items that all changed a lot depends on sizes that neither condition predicts well.</p>

<h2 id="sec-5">5 Experiment 2: which tokens rise under an amplification?</h2>
<p>Edits large enough to change the output: one MLP neuron ×100, a whole MLP ×3, or one final-norm gain dimension ×30 (median mean-KL 0.2 to 0.7 in a pilot).
For each edit the model named the 10 tokens whose log-probability rises most on average over the 8 prompts, among tokens that reach p ≥ 0.001.
<b>Score</b> = mean percentile of the named tokens in the measured ranking (0.5 = a random plausible token, 0 = not plausible). <b>Hit</b> = share of named tokens in the measured top 50.
Baselines from code: <i>already likely</i> (the 10 most likely tokens before the edit, excluding targets) and <i>copy</i> (top 10 of the edited component's vocabulary projection, with the better of the two signs).
The final-gain edits were repeated on a second set of 32 dimensions.</p>
<p id="tab-6"><b>Table 6.</b> Score ± 1 standard error of the mean over edits, paired {R} − {B}, baselines, hit rate.</p>
<table><tr><th>family</th><th>edits</th><th>{B}</th><th>{R}</th><th>{R} − {B}</th><th>already likely</th><th>copy</th><th>hit {B}</th><th>hit {R}</th></tr>{t4}</table>
<p>The tokens that gain probability are mostly ones already plausible in context, so {B} is about as good as "already likely". Copying the vocabulary projection does badly ({x2all['dla_oracle']['pct'][0]:.2f}),
because the tokens it names are rare tokens that never become plausible.</p>

<h2 id="sec-6">6 Experiment 3: simulating the calculation</h2>
<p>These edits have an effect that is an exact formula given one unknown per prompt, n<sub>d</sub> = the RMS-normalised final hidden state at dimension d.</p>
<ul><li><b>gain</b> (24 edits): final-norm gain ×30 at dim d, giving Δlogit<sub>t</sub> = 29 (1+w<sub>d</sub>) n<sub>d</sub> E[t,d] for every token.</li>
<li><b>emb</b> (24 edits): add ±4 to E[t,d] of a target or runner-up token, giving Δlogit<sub>t</sub> = ±4 (1+w<sub>d</sub>) n<sub>d</sub> for that token.</li></ul>
<p>A check with the true n<sub>d</sub> reproduces the edited model to within 2e-6 nats. Conditions:
<i>{R}</i> (gain and the targets' embedding entries); <i>{R} + needed numbers</i> (also each prompt's top-50 unedited tokens with probabilities and E[t,d], and the column's extremes);
<i>+ calculation recipe</i> (the same numbers plus step-by-step instructions: estimate n<sub>d</sub> from the probability-weighted embedding entries, compute the new logits, renormalise; do the arithmetic yourself).
The recipe was also run exactly by code, and once with the true n<sub>d</sub> (which needs activations) as a ceiling. Same score as Table <a href="#tab-6">6</a>, plus ρ and sign accuracy (|Δ| &gt; 0.1) for target Δlogp.</p>
<p id="tab-7"><b>Table 7.</b> Experiment 3 results.</p>
<table><tr><th>family</th><th>condition</th><th>score</th><th>hit</th><th>Δlogp ρ</th><th>sign acc.</th><th>MAE</th></tr>{t5}</table>
<p>Paired differences ± 1 standard error (bootstrap over edits); shaded if beyond 1.96 standard errors:</p>
<table><tr><th rowspan=2>family</th><th colspan=2>recipe − {B}</th><th colspan=2>recipe − needed numbers</th><th colspan=2>needed numbers − {R}</th><th colspan=2>{R} − {B}</th></tr>
<tr><th>score</th><th>ρ</th><th>score</th><th>ρ</th><th>score</th><th>ρ</th><th>score</th><th>ρ</th></tr>{t5p}</table>
<ul><li><b>Embedding-entry edits:</b> the recipe helps (score {e3['compute']['pct']:.2f} against {e3['none']['pct']:.2f} for {B} and {e3['rawplus']['pct']:.2f} with the numbers alone; hit {e3['compute']['hit50']:.2f} against {e3['none']['hit50']:.2f}).
Giving the numbers without the recipe does not, so the gain comes from doing the calculation. It stays below the same recipe run by code ({e3['code_recipe']['pct']:.2f}).</li>
<li><b>Final-gain edits:</b> the recipe hurts (score {g3['compute']['pct']:.2f} against {g3['none']['pct']:.2f}). Even run exactly by code the recipe reaches only {g3['code_recipe']['pct']:.2f}, while the true n<sub>d</sub> gives {g3['oracle_n']['pct']:.2f}.
The arithmetic is not the problem. The estimate of n<sub>d</sub> from the output distribution is too weak, and a 30× gain turns a wrong sign into large confident errors.</li>
<li>For target Δlogp no condition beats predicting 0 on MAE.</li></ul>

<h2 id="sec-7">7 Did it read the weights?</h2>
<p id="tab-6"><b>Table 6.</b> Transcript audit of the Opus calls. "weight files" = distinct per-edit weight files read.</p>
<table>{AUDIT_HEAD}{audit_table(opus)}</table>
<p>The only tool used was Read and every read stayed inside the sandbox. In a first pass of experiment 1 with optional reading, some calls read 1 of 12 weight files, so reading was made mandatory
and that set was rerun. Notes cite specific values (for example "W[11751,880]=-0.00519 shifted by -4 (~200 std); ... final gain 1+3.7031=4.70"), so the numbers were read.
In most cases they did not change the predictions in a useful way.</p>

<h2 id="sec-8">8 Caveats</h2>
<ul><li>Each (set, family, condition) is one predictor sample, so predictor run-to-run variance is not separated from condition effects. The standard errors resample edits only.</li>
<li>For layers, the raw files cannot hold all numbers, so they give norm tables, norm gains and the largest neurons.</li>
<li>Most large scalar effects come from one edit type (±4 on a target-token embedding entry). One model and 8 prompts; other prompts or models may differ.</li>
<li>Experiment 2 was repeated for the main text with Opus on the same edits after a first run with Fable (expandable below), so its measured effects existed before the Opus predictions. The predictor could not read them: reads outside the sandbox cause a call to be rejected, and there were none.</li></ul>

<details><summary>Views condition (raw weights plus derived views): not better than raw</summary>
<p>A third condition gave raw weights plus views computed from weights only: vocabulary projections through the tied unembedding, input-token read-ins, SVDs, OV maps of the prompt tokens, and norm ranks.</p>
<p>Experiment 1 (first 180 edits, token edits included), Spearman ρ and paired views − {B}:</p>
<table><tr><th>family</th><th>{B}</th><th>{R}</th><th>views</th><th>views − {B} ρ</th><th>views − {B} large-effect AUC</th></tr>{tv}</table>
<p>Experiment 2 (run with Fable 5.1 as predictor), score: views led the model to name vocabulary-projection tokens, which rarely become plausible, so views did worse than {B}. They still beat plain copying of the projection.</p>
<table><tr><th>family</th><th>{B}</th><th>views</th><th>views − {B}</th><th>views − copy</th></tr>{tvf}</table>
<p>Audit of the Opus views calls:</p><table>{AUDIT_HEAD}{audit_table(opus_views)}</table>
</details>

<details><summary>Fable 5.1 as the predictor: not better than Opus</summary>
<p>The same bundles were given to <code>claude-fable-5-1</code> with the same Read-only setup and checks (0 reads outside the sandbox). Experiment 1 zero-shot, Spearman ρ:</p>
<table><tr><th>family</th><th>Fable {B}</th><th>Fable {R}</th><th>Fable − Opus, {B}</th><th>Fable − Opus, {R}</th><th>Fable {R} − {B}</th><th>round-3 edits, {B} O / F</th><th>round-3 edits, {R} O / F</th></tr>{tf}</table>
<p>Experiment 2 with Fable, score for {B} and {R}. For final-gain edits with KL &gt; 0.1, raw looked better on the first 16 edits, but not on a second set of 32:</p>
<table><tr><th>subset</th><th>{B}</th><th>{R}</th><th>{R} − {B}</th></tr>{tf2}</table>
<p>Fable audit:</p><table>{AUDIT_HEAD}{audit_table(fable)}</table>
</details>

<details><summary>Computational cost</summary>
<ul><li>Opus predictor: {len(opus)} none/raw calls ({cost_o:.2f} USD at list price) and {len(opus_views)} views calls (${sum(r['cost'] for r in opus_views):.2f}), plus {len(pilot)} first-pass calls (${sum(r['cost'] for r in pilot):.2f}). Fable predictor: {len(fable)} calls, {cost_f:.2f} USD at list price.
All calls ran through the user's Claude Max subscription (OAuth, no API key), so these are list-price equivalents, not charges. 2 to 5 calls ran at a time; a call takes 1 to 15 minutes.</li>
<li>Measuring effects: about {len(zs_iids) + 36 + 124 + 48} edit runs × 8 prompts, fp32 on CPU (one 176-core node), about 0.4 s per forward pass, under 30 minutes in total. No GPU (0 H100 node hours).</li>
<li>Building the predictor files takes about 5 minutes per set on CPU, and the bootstrap analysis about 6 minutes.</li></ul></details>

<details><summary>Reproducibility</summary>
<pre>Repository https://github.com/csinva/weight-space-programming-autoresearch, branch systematic-tests (local). Use the repo-root uv env (uv sync).
Question: can Claude Opus 5.5, reading only weight files (Read tool, no code), predict how weight edits of Qwen/Qwen3.5-0.8B
(revision 2fc06364715b967f1860aea9cf38778875588b17) change the next-token distribution on 8 fixed prompts?
Code in systematic-tests/: common.py (model, edit context manager), 00_make_prompts.py (data/prompts.json), 00_pilot.py (SCALAR_ADD=4),
interventions.py + bundles.py + run_round.py (experiment 1), analyze.py, analyze_filtered.py, exp2_interventions.py + exp2_pilot.py + exp2_run.py
+ exp2_analyze.py (experiment 2), exp3.py + exp3_check.py (experiment 3). The model downloads to systematic-tests/.hf_cache.
Predictor command, run with cwd = a copy of one bundle directory outside any repository:
  claude -p "Read README.md ... End with the JSON block" --model claude-opus-5-5 --restricted --tools Read --strict-mcp-config --output-format stream-json --verbose
Experiment 1: uv run python systematic-tests/00_make_prompts.py; for R in 0 1 2: run_round.py build/predict --round R, git commit, run_round.py measure --round R
  ; few-shot: run_round.py build/predict --round 2 --variant k0|k4 before measuring set 3.
  Then analyze.main(), analyze_filtered.py.
Experiment 2: WSP_X2_DIR=exp2_opus WSP_X2_CONDS=none,raw exp2_run.py predict --round 0 (bundles and measured effects as in exp2/; K: neuron x100,
  MLP x3, final gain x30); second gain set: WSP_X2_DIR=exp2_rep_opus WSP_X2_FAMS=gain. Score with exp2_analyze.py under the same env.
Experiment 3: exp3.py build, exp3.py predict, git commit, exp3.py measure, exp3.py analyze.
Report: uv run python analyze/systematic_tests/build_report.py
Fable runs: the same scripts with WSP_MODEL=claude-fable-5-1 (experiment 1 with WSP_TAG=fable writes rounds_fable/; experiment 2 in exp2/, exp2_rep/).</pre></details>
</div></body></html>"""
    open(os.path.join(HERE, "report.html"), "w").write(H)
    print("wrote", os.path.join(HERE, "report.html"))


if __name__ == "__main__":
    main()
