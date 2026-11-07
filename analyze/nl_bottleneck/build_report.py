"""Build analyze/nl_bottleneck/report.html (and its two PNG figures) from experiments/nl-bottleneck/results.json and the saved summaries / metadata.
  ~/weight-space-programming-autoresearch/.venv/bin/python analyze/nl_bottleneck/build_report.py"""
import glob, html, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(os.path.dirname(os.path.dirname(HERE)), "experiments", "nl-bottleneck")
R = json.load(open(f"{NB}/results.json"))
SEL = json.load(open(f"{NB}/selected.json"))
NLP = ["sentiment-sst2", "paraphrase-mrpc", "nli-snli", "trec-coarse"]
GEN = [t for t in R if t not in NLP]
P = lambda x: f"{100 * x:.1f}"


def d(x, ref):  # diff to the original in percentage points, blue positive / red negative
    v = 100 * (x - ref)
    return "" if abs(v) < 0.05 else f" <span class='{'pos' if v > 0 else 'neg'}'>({v:+.1f})</span>"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


# ---------- figures
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(7.2, 3.2))
w = 0.16
for i, t in enumerate(NLP):
    e = R[t]
    vals = [("original", e["orig"]["hidden"], "#111827"), ("reconstruction 1", e["recon"][f"{t}-r1"]["acc"]["hidden"], "#2563eb"),
            ("reconstruction 2", e["recon"][f"{t}-r2"]["acc"]["hidden"], "#60a5fa"), ("no description", e["control"][f"{t}-r1"]["acc"]["hidden"], "#f59e0b"),
            ("majority class", e["majority"], "#d1d5db")]
    for j, (lab, v, c) in enumerate(vals):
        ax.bar(i + (j - 2) * w, 100 * v, w, color=c, label=lab if i == 0 else None)
ax.set_xticks(range(4)); ax.set_xticklabels(NLP); ax.set_ylabel("hidden accuracy (%)"); ax.set_ylim(0, 100)
ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
fig.tight_layout(); fig.savefig(f"{HERE}/fig1_hidden_acc.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.6))
labels, only_o, both, only_r, exp = [], [], [], [], []
for t in NLP:
    for kind, tag in (("recon", f"{t}-r1"), ("recon", f"{t}-r2"), ("control", f"{t}-r1")):
        h = R[t][kind][tag]["vs_orig"]["hidden"]; n = h["n"]
        labels.append(f"{t.split('-')[-1] if t != 'trec-coarse' else 'trec'} {'r' + tag[-1] if kind == 'recon' else 'ctrl'}")
        only_o.append(100 * (h["err_a"] - h["both_wrong"]) / n); both.append(100 * h["both_wrong"] / n); only_r.append(100 * (h["err_b"] - h["both_wrong"]) / n); exp.append(100 * h["expected_both_if_independent"] / n)
x = range(len(labels))
ax.bar(x, both, color="#7c3aed", label="both wrong"); ax.bar(x, only_o, bottom=both, color="#111827", label="only original wrong")
ax.bar(x, only_r, bottom=[a + b for a, b in zip(both, only_o)], color="#60a5fa", label="only reconstruction wrong")
ax.scatter(x, exp, marker="_", s=200, color="#dc2626", zorder=5, label="both wrong if independent")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=60, ha="right"); ax.set_ylabel("% of hidden examples")
ax.legend(fontsize=7, ncol=2, frameon=False); fig.tight_layout(); fig.savefig(f"{HERE}/fig2_error_overlap.png", dpi=150); plt.close(fig)

# ---------- numbers
words = {t: len(open(f"{NB}/summaries/{t}.txt").read().split()) for t in R}
rec_ok = {k: [e["acc"] for tag, e in R[t]["recon"].items() for k in ["hidden"]] for t in GEN for k in [0]}
gen_perfect = sum(all(v == 1.0 for v in e["acc"].values()) for t in GEN for e in R[t]["recon"].values())
gen_n = sum(len(R[t]["recon"]) for t in GEN)
ctl_perfect = sum(all(v == 1.0 for v in R[t]["control"][f"{t}-r1"]["acc"].values()) for t in GEN)
nlp_rows = [(t, tag, e) for t in NLP for tag, e in R[t]["recon"].items()]
agree_r = mean([e["vs_orig"]["hidden"]["agree"] for _, _, e in nlp_rows]); agree_c = mean([R[t]["control"][f"{t}-r1"]["vs_orig"]["hidden"]["agree"] for t in NLP])
agree_pair = mean([R[t]["recon_pair"]["hidden"]["agree"] for t in NLP])
gap_r = mean([100 * (e["acc"]["hidden"] - R[t]["orig"]["hidden"]) for t, _, e in nlp_rows])
meta = [json.load(open(f)) for f in glob.glob(f"{NB}/recon/*/meta.json")]; cmeta = [json.load(open(f)) for f in glob.glob(f"{NB}/recon_control/*/meta.json")]
slog = [json.load(open(f)) for f in glob.glob(f"{NB}/summaries/*.log.json")]
cost = dict(sum=sum(x["cost"] for x in slog), rec=sum(x["cost_usd"] for x in meta), ctl=sum(x["cost_usd"] for x in cmeta))
hrs = dict(sum=sum(x["seconds"] for x in slog) / 3600, rec=sum(x["seconds"] for x in meta) / 3600, ctl=sum(x["seconds"] for x in cmeta) / 3600)
import math
def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))
rr = [[v["n_params"] / R[t]["orig_params"] for v in R[t]["recon"].values()] for t in GEN]
cr = [R[t]["control"][f"{t}-r1"]["n_params"] / R[t]["orig_params"] for t in GEN]
rec_within = sum(all(0.5 <= x <= 2 for x in r) for r in rr)
rec_gm, ctl_gm, ctl_min = gmean([x for r in rr for x in r]), gmean(cr), min(cr)
one_shot = sum(x["experiments"] == 1 for x in meta)
TOK = json.load(open(f"{NB}/tokens.json"))
def tsum(prefix, k):
    return sum(v[k] or 0 for n, v in TOK.items() if n.startswith(prefix))
tok_rows = "".join(f"<tr><td>{lab}</td><td>{sum(1 for n in TOK if n.startswith(p))}</td><td>{tsum(p, 'output'):,}</td><td>{tsum(p, 'thinking'):,}</td><td>{tsum(p, 'cache_write'):,}</td><td>{tsum(p, 'cache_read'):,}</td><td>{tsum(p, 'input'):,}</td><td>${tsum(p, 'cost'):.2f}</td></tr>" for lab, p in [("reconstruct (with description)", "recon/"), ("control (no description)", "recon_control/")])
size_rows = ""
for t in sorted(R):
    fb_ = os.path.getsize(f"{NB}/originals/{t}.py"); db_ = len(open(f"{NB}/summaries/{t}.txt").read().encode())
    rb = [os.path.getsize(f"{NB}/recon_final/{t}-r{k}.py") for k in (1, 2)]
    size_rows += f"<tr><td>{t}</td><td>{fb_ / 1024:.1f}</td><td>{db_ / 1024:.2f}</td><td>{fb_ / db_:.0f}</td><td>{rb[0] / 1024:.1f} / {rb[1] / 1024:.1f}</td></tr>"
aud = json.load(open(f"{NB}/audit_recon_final.json")); aud_c = json.load(open(f"{NB}/audit_recon_control_final.json"))
fb = [k for k, v in json.load(open(f"{NB}/recon_selected.json")).items() if v["fallback_to_last_file"]]

# ---------- tables
def row_all(t):
    e = R[t]; o = e["orig"]; cells = [f"<td>{t}</td><td>{words[t]}</td>"]
    cells.append(f"<td>{P(o['visible'])} / {P(o['hidden'])}" + (f" / {P(o['extra'])}" if "extra" in o else "") + f"</td><td>{int(e['orig_params']):,}</td>")
    for tag, v in list(e["recon"].items()) + list(e["control"].items()):
        a = v["acc"]
        s = f"{P(a['visible'])}{d(a['visible'], o['visible'])} / {P(a['hidden'])}{d(a['hidden'], o['hidden'])}" + (f" / {P(a['extra'])}{d(a['extra'], o['extra'])}" if "extra" in a else "")
        cells.append(f"<td>{s}</td><td>{int(v['n_params']):,}</td>")
    return "<tr>" + "".join(cells) + "</tr>"

tab1 = "<table><thead><tr><th rowspan=2>task</th><th rowspan=2>words</th><th colspan=2>original</th><th colspan=2>reconstruction 1</th><th colspan=2>reconstruction 2</th><th colspan=2>no description</th></tr>" \
       "<tr>" + "<th>visible / hidden / extra</th><th>params</th>" * 4 + "</tr></thead><tbody>" + "".join(row_all(t) for t in sorted(R, key=lambda t: (t in NLP, t))) + "</tbody></table>"

def f2(x):
    return "n/a" if x is None else f"{x:.2f}"

rows = []
for t in NLP:
    for kind, tag in [("recon", f"{t}-r1"), ("recon", f"{t}-r2"), ("control", f"{t}-r1")]:
        e = R[t][kind][tag]; h = e["vs_orig"]["hidden"]; v = e["vs_orig"]["visible"]
        rows.append(f"<tr><td>{t}</td><td>{'reconstruction ' + tag[-1] if kind == 'recon' else 'no description'}</td><td>{P(R[t]['orig']['hidden'])}</td><td>{P(e['acc']['hidden'])}{d(e['acc']['hidden'], R[t]['orig']['hidden'])}</td>"
                    f"<td>{100 * h['agree']:.1f}</td><td>{f2(h['kappa'])}</td><td>{f2(h['phi'])}</td><td>{100 * h['p_b_given_a_wrong']:.0f} / {100 * h['p_b_given_a_right']:.0f}</td>"
                    f"<td>{h['both_wrong']} / {h['expected_both_if_independent']:.0f} (of {h['err_a']})</td><td>{100 * v['agree']:.1f}</td></tr>")
    h = R[t]["recon_pair"]["hidden"]
    rows.append(f"<tr class='sub'><td>{t}</td><td>reconstruction 1 vs 2</td><td></td><td></td><td>{100 * h['agree']:.1f}</td><td>{f2(h['kappa'])}</td><td>{f2(h['phi'])}</td><td>{100 * h['p_b_given_a_wrong']:.0f} / {100 * h['p_b_given_a_right']:.0f}</td><td>{h['both_wrong']} / {h['expected_both_if_independent']:.0f} (of {h['err_a']})</td><td></td></tr>")
tab2 = "<table><thead><tr><th>task</th><th>compared model</th><th>original hidden acc.</th><th>its hidden acc.</th><th>identical output (%)</th><th>kappa</th><th>phi of errors</th><th>P(wrong | original wrong / right) (%)</th><th>both wrong: observed / if independent</th><th>identical output, visible (%)</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

cls_rows = []
for t in NLP:
    e = R[t]
    for name, rec, dist in [("original", e["orig_recall"], e["orig_pred_dist"])] + [(tag.split("-")[-1] if kind == "recon" else "no description", v["recall"], v["pred_dist"]) for kind in ("recon", "control") for tag, v in e[kind].items()]:
        cls_rows.append(f"<tr><td>{t}</td><td>{'reconstruction ' + name if name.startswith('r') else name}</td><td>" + " ".join(f"{100 * rec[c]:.0f}" for c in rec) + "</td><td>" + " ".join(f"{100 * dist[c]:.0f}" for c in dist) + "</td></tr>")
tab3 = "<table><thead><tr><th>task</th><th>model</th><th>recall per class (%)</th><th>predicted-class share (%)</th></tr></thead><tbody>" + "".join(cls_rows) + "</tbody></table>"

sums = "".join(f"<details><summary>{t} ({words[t]} words; original {SEL[t]['program']}, {P(SEL[t]['visible'])}% visible, {int(SEL[t]['n_params']):,} params)</summary><pre>{html.escape(open(f'{NB}/summaries/{t}.txt').read())}</pre></details>" for t in sorted(R))

css = open(f"{HERE}/../systematic_tests/report.html").read().split("<style>")[1].split("</style>")[0]
page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Can a 200-word description recreate a weight program?</title><style>{css}
tr.sub td {{ color:var(--dim); font-style:italic; }} .scroll {{ overflow-x:auto; }}</style></head><body><div class="wrap">
<div class="top"><a href="../index.html">&larr; Analysis Reports index</a> &middot; <a href="report_raw.html">Raw data</a> (prompts, every description, example inputs, per-run logs)</div>
<h1>Can a 200-word description recreate a hand-written weight program?</h1>

<div class="summary">
<p><b>Bottom line.</b> A description of at most 200 words, written by Opus 5.5 from the program file, was enough for a fresh Opus 5.5 session to rebuild a program with
exact accuracy and nearly the original's parameter count on all 13 generated tasks, and hidden accuracy within 5 points of the original on the 4 dataset tasks, with most of the original's errors repeated. The no-description control is also exact on the 13 generated tasks, but with {ctl_gm:.0f} times the parameters on a geometric mean, so on those tasks the description transmits how compact the design is, not whether it works.</p>
<ul>
<li><b>Setup.</b> Per task, the best program from the Opus 5.5 runs ({len(R)} tasks, 2 runs each; highest visible accuracy among programs whose weights do not depend on task data, ties to fewest parameters). Opus 5.5 read it and wrote a description (mean {mean(words.values()):.0f} words). Two fresh Opus 5.5 sessions per task then rebuilt a model from the description plus the task statement ({len(meta)} reconstructions), with up to 6 scored experiments on the visible 200 examples; {one_shot} of {len(meta)} used only one. A control session per task got the task but no description.</li>
<li><b>Generated tasks (13).</b> {gen_perfect} of {gen_n} reconstructions are 100% on visible, hidden (2,000 fresh) and extra (20,000 fresh) examples, like all originals. The no-description control is also 100% on {ctl_perfect} of 13. With no errors on either side there is no error profile to compare; [Sec. 3](#sec-3) compares program sizes instead: {rec_within} of 13 tasks have reconstructions within a factor 2 of the original's parameter count (geometric mean ratio {rec_gm:.2f}), against {ctl_gm:.0f} for the control.</li>
<li><b>Dataset tasks (4).</b> Hidden accuracy of the reconstructions is {gap_r:+.1f} points on average relative to the original ([Table 1](#tab-1), [Fig. 1](#fig-1)). They give the identical output as the original on {100 * agree_r:.0f}% of hidden examples on average, against {100 * agree_c:.0f}% for the control and {100 * agree_pair:.0f}% between the two reconstructions ([Table 2](#tab-2)).
When the original is wrong, a reconstruction is wrong with probability 77 to 90% (SST-2, SNLI, TREC), against 7 to 15% when the original is right ([Fig. 2](#fig-2)).</li>
<li><b>Caveat.</b> The original MRPC program answers 1 for 90% of pairs and scores the same as always answering 1 (68.9% against 68.8%), so matching it says little about transmitting a useful rule ([Sec. 4](#sec-4)). On SNLI the two reconstructions agree with each other on 98% of examples, so the description fixes the program closely; the originals and reconstructions share a feature recipe, not weights.</li>
</ul></div>

<h2 id="sec-1">1. Setup</h2>
<p><b>Sets.</b> <i>Visible</i>: the 200 seed-0 examples that every attempt in the original runs was scored on (the reconstructors also see this score). <i>Hidden</i>: generated tasks use 2,000 fresh examples from an unseen seed (987654321, as in <code>experiments/evolve/heldout_fresh_seed.py</code>); dataset tasks use up to 2,000 training rows outside the 2,000-example pool the agents saw (1,668 for MRPC, which has no more). <i>Extra</i>: generated tasks only, 20,000 more fresh examples, to find rare errors. One standard error on 2,000 examples is about 1 point, so differences under 2 points on hidden accuracy are noise.</p>
<p><b>Summarizer.</b> Headless <code>claude -p --model claude-opus-5-5 --restricted --tools Read</code> in a folder containing only the program file and asked for at most 200 words on the algorithm, practical hints and known imperfections, with no code and no tensor values. Over-length outputs were sent back to shorten.</p>
<p><b>Reconstructor.</b> A fresh headless Opus 5.5 session in a sandbox outside any repo with the template <code>interpretable_transformer.py</code>, <code>src/</code> and the description in the prompt, plus the task statement, the original's visible accuracy and parameter count, and the repo's rules (no training, no external computation, no fitting to dataset examples). Only the scoring command could be run; the final program is the saved snapshot with the highest visible accuracy (ties to fewest parameters), the rule used for the originals. For {len(fb)} reconstructions ({", ".join(fb)}) the agent's saved copies were truncated, so the agent's last working file was used.
<b>Audits.</b> All {sum(v == 'data_independent' for v in aud.values())} of {len(aud)} final reconstructions and all {sum(v == 'data_independent' for v in aud_c.values())} of {len(aud_c)} controls are data-independent under the repo's counterfactual audit (<code>experiments/evolve/audit_data_use.py</code>); no transcript read outside its sandbox. Agents tried to write scratch files in /tmp, which were refused (47 refused calls over the reconstructions).</p>

<h2 id="sec-2">2. Accuracy of original, reconstructions and control</h2>
<p id="tab-1"><b>Table 1.</b> Accuracy (%) as visible / hidden / extra (extra only for generated tasks); differences to the original in parentheses. Words = length of the description.</p>
<div class="scroll">{tab1}</div>
<p id="fig-1"><b>Fig. 1.</b> Hidden accuracy on the dataset tasks.</p><img src="fig1_hidden_acc.png" width="720">
<p>Visible accuracy overstates the original on TREC (92.0 against 82.3 hidden), because the original was tuned on those 200 examples. The reconstructions, tuned with at most a few runs on the same 200, show a smaller visible-to-hidden gap (84.0 / 84.5 against 77.5 / 78.2), so the hidden difference to the original (&minus;4.8 and &minus;4.1) is smaller than the visible one (&minus;8.0 and &minus;7.5). The description cannot carry the 200-example tuning; the reconstruction lands near what the algorithm generalizes to. On SNLI the reconstructions are 3.8 and 4.0 points below the original on hidden examples and the control (59.7) is above all of them, so on SNLI and TREC the description did not help accuracy relative to starting from scratch. On SST-2 it did (hidden 78.1 and 77.2 against 74.8 for the control); on MRPC the control scores the same as the original by always answering 1.</p>

<h2 id="sec-3">3. Generated tasks: size of the programs</h2>
<p>All originals and reconstructions are exact on every example tested (22,200 per program), so accuracy and errors cannot tell them apart. Program size can. The reconstructions reproduce the original's parameter count closely: {", ".join(f"{t} {int(R[t]['orig_params']):,} against {int(list(R[t]['recon'].values())[0]['n_params']):,} and {int(list(R[t]['recon'].values())[1]['n_params']):,}" for t in ["sort-five-digits", "parity-upto10-bits", "word-reversal-3x3", "edit-distance-8"])} (original against reconstructions 1 and 2; full list in [Table 1](#tab-1)). {rec_within} of 13 tasks are within a factor 2 for both reconstructions, and the largest ratio is {max(max(r) for r in rr):.1f}. The control, given only the task, reaches 100% with {ctl_gm:.0f} times the original's parameters on a geometric mean (smallest ratio {ctl_min:.0f}), mostly in the range of the template's 152,000-parameter default or larger.
So on these tasks the 200 words carry the structure of the compressed design (which architecture, which scalar tricks), which the control runs did not reproduce. They do not show that the description is needed for exact accuracy, since the control is exact too.</p>

<h2 id="sec-4">4. Dataset tasks: error profiles</h2>
<p id="tab-2"><b>Table 2.</b> Comparison on the hidden set (last column: visible set). Kappa: agreement of the two models' outputs beyond chance given their output frequencies. Phi: correlation of the two error indicators. P(wrong | original wrong / right): error rate of the compared model on examples the original gets wrong / right; equal values mean independent errors. Both wrong: observed count against the count expected if the errors were independent.</p>
<div class="scroll">{tab2}</div>
<p id="fig-2"><b>Fig. 2.</b> Error overlap on the hidden set. The red mark is the both-wrong share expected for independent errors with the same error rates.</p><img src="fig2_error_overlap.png" width="720">
<ul>
<li><b>Errors are strongly shared.</b> On SST-2, SNLI and TREC the reconstructions repeat 77 to 90% of the original's errors, 2 to 4 times the count expected by chance. For SNLI and TREC the shared wrong answer is the same class in 90 to 93% of the shared errors (for the binary tasks this is 100% by construction).</li>
<li><b>The control shares fewer errors but still many.</b> Phi drops from 0.62 to 0.75 (SST-2, SNLI, TREC) to 0.44 to 0.53 without the description, and identical outputs on SNLI drop from 84 to 85% to 67%. Part of the shared errors comes from the task itself: hard examples are hard for any character-level rule system.</li>
<li><b>Reconstructions are closer to each other than to the original.</b> Agreement between reconstructions 1 and 2 is 91 to 100%, above their agreement with the original (84 to 100%). The description pins down most of the program, and what it leaves out (exact word lists, constants) differs from the original.</li>
<li><b>Class-level behaviour is kept.</b> [Table 3](#tab-3) shows the predicted-class shares match the original closely, for example TREC predicts ABBR for 1% of questions in all models (recall 61% for it in the original and both reconstructions), and SNLI keeps its profile of entailment recall above neutral above contradiction. SST-2 reconstructions lean more toward positive (65 and 66% against 61%) and have lower negative recall (65 and 63% against 69%).</li>
<li><b>MRPC is a degenerate case.</b> The original answers "paraphrase" when an 8-character window near the start of sentence 2 appears in sentence 1 under a coarse 2-bit character hash. Collisions make this true for 90% of pairs (recall 16% for the negative class, 93% for the positive), and its accuracy equals the majority baseline. Both reconstructions rebuild the same lookup and give identical outputs on 99.9 and 100% of hidden examples. The control predicts class 1 always (kappa 0.00, so agreement is 90% only because the original is itself 90% class 1) and its phi of 0.77 comes from the same overlap. The description transmits this rule exactly, which is plausible because the rule is short and fully specified.</li>
</ul>
<p id="tab-3"><b>Table 3.</b> Per-class behaviour on the hidden set (classes in label order; SST-2 negative/positive, MRPC not paraphrase/paraphrase, SNLI entailment/neutral/contradiction, TREC ABBR/ENTY/DESC/HUM/LOC/NUM).</p>
<div class="scroll">{tab3}</div>

<h2 id="sec-5">5. Limitations</h2>
<ul>
<li>Token counts for the summarizer calls were not recorded.</li>
<li>One original program per task, two reconstructions and one control. Differences of 1 to 2 points on 2,000 examples are within sampling noise.</li>
<li>Selecting the original by visible accuracy picked programs tuned to the visible 200 examples. The hidden numbers are the fair comparison.</li>
<li>The reconstructor is told the original's visible accuracy and parameter count, and may score on the visible 200 examples. These are task context, not weights, but they let it calibrate; most runs ({one_shot} of {len(meta)}) used a single experiment.</li>
<li>Same model (Opus 5.5) writes and reads the description and rebuilds the program, so shared habits in program layout may raise agreement. A different reader model was not tried.</li>
<li>The generated tasks are saturated at 100%, so they give no information on how faithfully errors are transmitted. Harder or noisier versions of these tasks (for example by adding noise to the weights, as the original runs did) would test this.</li>
</ul>

<h2 id="sec-6">6. Descriptions</h2>
<p>The 200-word descriptions that the reconstructors received. Prompts, example inputs, attempts and per-example disagreements are in the <a href="report_raw.html">raw data page</a>.</p>{sums}

<h2 id="sec-7">7. Token and dollar cost</h2>
<p id="tab-4"><b>Table 4.</b> Opus 5.5 usage of the reconstruction and control sessions (list-price dollars, as reported by Claude Code). Cache read tokens are the context re-read on each turn and dominate the totals; output tokens are what the model generated. The 17 summarizer calls cost ${cost['sum']:.2f} in total ({mean([x['cost'] for x in slog]):.2f} each, 2 to 3 turns) but did not record token counts. Per-run numbers are in the <a href="report_raw.html#sec-5">raw data</a>.</p>
<table><thead><tr><th>step</th><th>sessions</th><th>output tokens</th><th>of which thinking</th><th>cache write</th><th>cache read</th><th>uncached input</th><th>cost</th></tr></thead><tbody>{tok_rows}</tbody></table>
<p>All three steps together cost ${cost['sum'] + cost['rec'] + cost['ctl']:.2f}. A reconstruction costs about ${cost['rec'] / len(meta):.2f} on average and a control ${cost['ctl'] / len(cmeta):.2f}, so the description does not make rebuilding cheaper; its value is in the program it produces. The most expensive reconstructions are TREC ($3.01 and $2.15), edit distance ($2.34) and date normalization ($2.11); the most expensive control is SNLI ($4.46). Twenty-three of the 34 reconstructions cost under $1.</p>
<p id="tab-5"><b>Table 5.</b> Size of the description against the program it describes (file sizes in KB, the original file includes its comments and helper code, so the ratio is a rough measure of compression). The last column gives the sizes of the two final reconstructions.</p>
<div class="scroll"><table><thead><tr><th>task</th><th>original file KB</th><th>description KB</th><th>file / description</th><th>reconstructions KB</th></tr></thead><tbody>{size_rows}</tbody></table></div>

<details><summary>Computational cost</summary>
<p>No GPU was used (0 H100 node hours). Models are tiny and were scored on CPU (about 0.3 to 190 s per program for 2,200 to 22,200 examples, 8 to 16 processes at a time). The cost is Opus 5.5 usage through headless Claude Code, at list price:</p>
<table><thead><tr><th>step</th><th>runs</th><th>API cost</th><th>agent time (sum over runs)</th></tr></thead><tbody>
<tr><td>summarize</td><td>{len(slog)}</td><td>${cost['sum']:.2f}</td><td>{hrs['sum']:.1f} h</td></tr>
<tr><td>reconstruct</td><td>{len(meta)}</td><td>${cost['rec']:.2f}</td><td>{hrs['rec']:.1f} h</td></tr>
<tr><td>control (no description)</td><td>{len(cmeta)}</td><td>${cost['ctl']:.2f}</td><td>{hrs['ctl']:.1f} h</td></tr></tbody></table>
<p>For comparison, the Opus 5.5 original runs on the easy tasks cost about $8 to $12 each in the toy-examples report; the reconstructions cost about ${cost['rec'] / len(meta):.2f} each. Nothing here is measured as throughput or MFU because no model was trained.</p></details>

<details><summary>Reproducibility</summary>
<p>Prompt for a coding agent:</p>
<pre>In the repo https://github.com/csinva/weight-space-programming-autoresearch (use the repo-root uv environment; run on CPU; if /data is full work on /data18), reproduce analyze/nl_bottleneck/report.html.
Question: can a description of at most 200 words, written by Claude Opus 5.5 from a hand-written transformer program, let a fresh Opus 5.5 session rebuild a program with similar accuracy and similar errors?
Inputs: the Opus 5.5 runs in runs-opus55/ (results/overall_results.csv, snapshots in reconstructed/ or interpretable_transformers_lib/, audit_data_use.json and audit_data_use_reconstructed.json); the 17 tasks in experiments/evolve/src/task.py; offline HF cache of sst2, glue/mrpc, snli, CogComp/trec.
Code, all in nl-bottleneck/ (run in this order from that folder):
 01_select.py       best program per task (highest visible accuracy among data_independent programs, ties to fewest params) -> originals/, selected.json
 eval_program.py    scores one program on visible (200, seed 0), hidden (generated: 2,000 at seed 987654321; dataset tasks: up to 2,000 training rows outside the 2,000-example pool, sampled with seed 1) and extra (generated: 20,000 at seed 555)
 02_eval_originals.sh   scores the originals -> preds/orig/
 03_summarize.py    claude -p --model claude-opus-5-5 --restricted --tools Read in a sandbox holding only model.py; at most 200 words -> summaries/
 04_reconstruct.py --reps 1 2 --workers 12   fresh sessions with the description (tools Read,Edit,Write,Glob,Grep,Bash; only the scoring command allowed; max 6 experiments) -> recon/
 04_reconstruct.py --control --reps 1        same without the description -> recon_control/
 05_eval_recon.py recon; 05_eval_recon.py recon_control    final program = best visible snapshot (fallback: the sandbox's last file if the snapshot is truncated), scored with eval_program.py
 07_audit.py recon_final; 07_audit.py recon_control_final; 08_audit_transcripts.py recon   audits
 06_analyze.py      agreement, kappa, phi, conditional error rates -> results.json
 09_tokens.py        token usage and final messages from the transcripts -> tokens.json
 analyze/nl_bottleneck/build_report.py   renders this report (needs HF_HOME offline cache, as above)
 analyze/nl_bottleneck/build_report_raw.py   renders report_raw.html (prompts, descriptions, attempts, example inputs and disagreements; needs experiments/nl-bottleneck/preds/)
Sandboxes live outside the repo (/data18/chandan.singh/wsp-nl/sandbox); set HF_HOME to an offline copy of the HF cache and HF_HUB_OFFLINE=1.
Expected: all 13 generated tasks 100% for all models on all sets; hidden accuracies on SST-2/MRPC/SNLI/TREC about 78.7/68.9/57.9/82.3 for the originals. Reconstruction numbers vary with the agent's sampling.</pre></details>
</div></body></html>"""
page = re.sub(r"\[(Table|Fig\.|Sec\.) (\d+)\]\((#[^)]+)\)", r'\1 <a href="\3">\2</a>', page)
open(f"{HERE}/report.html", "w").write(page)
print("ok", len(page))
