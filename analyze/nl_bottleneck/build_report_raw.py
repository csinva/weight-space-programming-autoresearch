"""Build analyze/nl_bottleneck/report_raw.html: verbatim prompts, example task inputs, every description, every attempt, token usage per run,
and per-example disagreements on the dataset tasks. Needs experiments/nl-bottleneck/preds/ (regenerate with 02_eval_originals.sh and 05_eval_recon.py)."""
import csv, glob, html, importlib.util, json, os, random, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NB = f"{REPO}/experiments/nl-bottleneck"
sys.path.insert(0, f"{REPO}/experiments/evolve")
import src.task as T
R = json.load(open(f"{NB}/results.json")); SEL = json.load(open(f"{NB}/selected.json")); TOK = json.load(open(f"{NB}/tokens.json"))
RS = json.load(open(f"{NB}/recon_selected.json")); CS = json.load(open(f"{NB}/recon_control_selected.json"))
NLP = ["sentiment-sst2", "paraphrase-mrpc", "nli-snli", "trec-coarse"]
LAB = {"sentiment-sst2": ["negative", "positive"], "paraphrase-mrpc": ["not paraphrase", "paraphrase"], "nli-snli": ["entailment", "neutral", "contradiction"],
       "trec-coarse": ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]}
E_ = html.escape


def mod(name):
    sp = importlib.util.spec_from_file_location("m" + name[:2], f"{NB}/{name}")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
S3, R4 = mod("03_summarize.py"), mod("04_reconstruct.py")


def clean(p):
    return re.sub(r"_{2,}", " ", p)


def lab(t, x):
    return LAB[t][int(x)] if t in LAB else x


def pre(s):
    return f"<pre>{E_(s)}</pre>"


# ---- prompts
ex_doc = "Example task statement (sort-five-digits), as inserted into the prompts:\n" + re.sub(r"\n +", "\n", T.get_task("sort-five-digits").__doc__.strip())
fmt = dict(task="sort-five-digits", doc="<task docstring>", vis=1.0, npar=53, nw=185, summary="<the description>", maxexp=R4.MAX_EXP, py="<python>", name="sort-five-digits")
p_sum = S3.PROMPT.replace("{name}", "<task name>").replace("{doc}", "<task docstring from src/task.py>")
p_rec = (R4.HEAD_RECON + R4.TAIL.format(task="<task>", py="<venv python>", maxexp=R4.MAX_EXP, guide=", guided by the description", end=R4.END_RECON)).replace("{task}", "<task>").replace("{doc}", "<task docstring>").replace("{vis:.1%}", "<original visible accuracy>").replace("{npar:,}", "<original params>").replace("{nw}", "<n>").replace("{summary}", "<description>")
p_ctl = (R4.HEAD_CONTROL + R4.TAIL.format(task="<task>", py="<venv python>", maxexp=R4.MAX_EXP, guide="", end=R4.END_CONTROL)).replace("{task}", "<task>").replace("{doc}", "<task docstring>").replace("{maxexp}", str(R4.MAX_EXP))
p_rec = p_rec.replace("{maxexp}", str(R4.MAX_EXP))

# ---- example inputs per task
def examples(t, n=3):
    task = T.get_task(t)
    return "\n".join(f"{clean(e.prompt)!r}  ->  {lab(t, e.answer)!r}" if t in NLP else f"{e.prompt!r}  ->  {e.answer!r}" for e in task.generate_examples(n, seed=0))
ex_tab = "".join(f"<details><summary>{t}</summary>{pre(examples(t, 5))}</details>" for t in sorted(R))

# ---- per task blocks
def attempts_tab(info, tag):
    rows = ""
    for a in info["attempts"]:
        acc = "" if a["visible"] is None else "%.1f" % (100 * a["visible"])
        npar = "" if a["n_params"] is None else "{:,}".format(int(a["n_params"]))
        fin = " (final)" if a["name"] == info["final"] else ""
        snap = "yes" if a["snapshot"] else "no (truncated or missing)"
        rows += "<tr><td>%s%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (E_(a["name"]), fin, acc, npar, snap, E_(a["description"]))
    return "<table><thead><tr><th>attempt</th><th>visible acc. (%)</th><th>params</th><th>snapshot saved</th><th>agent's description of the attempt</th></tr></thead><tbody>" + rows + "</tbody></table>"


def run_block(kind, tag):
    sel = (RS if kind == "recon" else CS)[tag]; e = R[sel["task"]][kind][tag]; tk = TOK[f"{'recon' if kind == 'recon' else 'recon_control'}/{tag}"]
    acc = " / ".join(f"{k} {100 * v:.2f}" for k, v in e["acc"].items())
    d = "recon_final" if kind == "recon" else "recon_control_final"
    return (f"<details><summary>{'Reconstruction' if kind == 'recon' else 'Control (no description)'} {tag[-2:]}: {acc}; {int(e['n_params']):,} params; {tk['turns']} turns, ${tk['cost']:.2f}</summary>"
            f"<p>Final program: <code>experiments/nl-bottleneck/{d}/{tag}.py</code> (snapshot <code>{E_(str(sel['final']))}</code>{', agent last file because its copy was truncated' if sel['fallback_to_last_file'] else ''}). "
            f"Tokens: {tk['output']:,} output ({tk['thinking'] or 0:,} thinking), {tk['cache_write']:,} cache write, {tk['cache_read']:,} cache read, {tk['input']:,} uncached input.</p>"
            f"{attempts_tab(sel, tag)}<p>Agent's final message:</p>{pre(tk['final_text'])}</details>")


blocks = []
for t in sorted(R):
    s = SEL[t]; e = R[t]
    blocks.append(f"<h3 id='task-{t}'>{t}</h3><p>Original: run <code>{s['run']}</code>, program <code>{E_(s['program'])}</code> (<code>experiments/nl-bottleneck/originals/{t}.py</code>), "
                  f"visible {100 * s['visible']:.1f}%, hidden {100 * e['orig']['hidden']:.2f}%, {int(s['n_params']):,} params.</p>"
                  f"<p>The original agent's own description of this program: <i>{E_(s['description'])}</i></p>"
                  f"<p><b>Description given to the reconstructors</b> ({len(open(f'{NB}/summaries/{t}.txt').read().split())} words):</p>{pre(open(f'{NB}/summaries/{t}.txt').read())}"
                  + run_block("recon", f"{t}-r1") + run_block("recon", f"{t}-r2") + run_block("control", f"{t}-r1"))

# ---- disagreement examples on dataset tasks
def load(p):
    return json.load(open(p))
dis = []
for t in NLP:
    o = load(f"{NB}/preds/orig/{t}.json")["hidden"]; r1 = load(f"{NB}/preds/recon/{t}-r1.json")["hidden"]; r2 = load(f"{NB}/preds/recon/{t}-r2.json")["hidden"]; c = load(f"{NB}/preds/recon_control/{t}-r1.json")["hidden"]
    n = o["n"]; rng = random.Random(0)
    idx = {"original wrong, reconstruction 1 right": [i for i in range(n) if not o["correct"][i] and r1["correct"][i]],
           "original right, reconstruction 1 wrong": [i for i in range(n) if o["correct"][i] and not r1["correct"][i]],
           "both wrong, different answers": [i for i in range(n) if not o["correct"][i] and not r1["correct"][i] and o["preds"][i] != r1["preds"][i]],
           "both wrong, same wrong answer": [i for i in range(n) if not o["correct"][i] and not r1["correct"][i] and o["preds"][i] == r1["preds"][i]],
           "both right (random)": [i for i in range(n) if o["correct"][i] and r1["correct"][i]]}
    cnt = {k: len(v) for k, v in idx.items()}
    out = f"<h3 id='dis-{t}'>{t}</h3><p>Hidden set, {n} examples. Counts: " + ", ".join(f"{k} {v}" for k, v in cnt.items()) + ". Up to 8 random examples per group (seed 0); padding underscores removed; pair tasks show the two sentences separated by <code>|</code>. Columns: true label, original, reconstruction 1, reconstruction 2, control.</p>"
    for k, v in idx.items():
        pick = sorted(rng.sample(v, min(8, len(v))))
        rows = "".join(f"<tr><td>{E_(clean(o['prompts'][i])[:-1])}</td><td>{lab(t, o['targets'][i])}</td><td>{lab(t, o['preds'][i])}</td><td>{lab(t, r1['preds'][i])}</td><td>{lab(t, r2['preds'][i])}</td><td>{lab(t, c['preds'][i])}</td></tr>" for i in pick)
        out += f"<details><summary>{k} ({len(v)})</summary><table><thead><tr><th>input</th><th>label</th><th>original</th><th>recon 1</th><th>recon 2</th><th>control</th></tr></thead><tbody>{rows}</tbody></table></details>"
    dis.append(out)

# ---- token table + full metrics
trs = "".join(f"<tr><td>{k}</td><td>{v['turns']}</td><td>{v['output']:,}</td><td>{v['thinking'] or 0:,}</td><td>{v['cache_write']:,}</td><td>{v['cache_read']:,}</td><td>{v['input']:,}</td><td>${v['cost']:.2f}</td><td>{v['seconds']}</td></tr>" for k, v in sorted(TOK.items(), key=lambda kv: kv[0].split("/")[0] + kv[0].split("/")[1]))
tok_tab = f"<table><thead><tr><th>run</th><th>turns</th><th>output tok.</th><th>of which thinking</th><th>cache write</th><th>cache read</th><th>uncached input</th><th>cost</th><th>seconds</th></tr></thead><tbody>{trs}</tbody></table>"
mrows = []
for t in sorted(R):
    for kind in ("recon", "control"):
        for tag, v in R[t][kind].items():
            for st, c in v["vs_orig"].items():
                f = lambda x: "" if x is None else f"{x:.3f}"
                mrows.append(f"<tr><td>{tag}</td><td>{kind}</td><td>{st}</td><td>{c['n']}</td><td>{c['err_a']}</td><td>{c['err_b']}</td><td>{c['both_wrong']}</td><td>{c['expected_both_if_independent']:.1f}</td><td>{f(c['jaccard'])}</td><td>{f(c['phi'])}</td><td>{f(c['agree'])}</td><td>{f(c['kappa'])}</td><td>{f(c['same_wrong'])}</td></tr>")
met_tab = "<table><thead><tr><th>run</th><th>kind</th><th>set</th><th>n</th><th>orig errors</th><th>compared errors</th><th>both wrong</th><th>expected both wrong</th><th>Jaccard</th><th>phi</th><th>identical output</th><th>kappa</th><th>same wrong answer</th></tr></thead><tbody>" + "".join(mrows) + "</tbody></table>"

css = open(f"{HERE}/../systematic_tests/report.html").read().split("<style>")[1].split("</style>")[0]
page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Raw data: description bottleneck</title><style>{css} .scroll {{ overflow-x:auto; }} td {{ max-width:520px; }}</style></head><body><div class="wrap">
<div class="top"><a href="../index.html">&larr; Analysis Reports index</a> &middot; <a href="report.html">Main report</a></div>
<h1>Raw data: the 200-word description experiment</h1>
<div class="summary"><p>Everything the main report ([Sec. 1](report.html#sec-1) to [Sec. 6](report.html#sec-6)) is computed from: the exact prompts, example task inputs, every description, every attempt and final message of each reconstruction and control session, token usage per run, per-example disagreements on the four dataset tasks, and the full metric table.
Programs themselves are files in the repo: <code>experiments/nl-bottleneck/originals/</code>, <code>recon_final/</code>, <code>recon_control_final/</code>; compressed agent transcripts in <code>recon/*/agent_log.jsonl.gz</code> and <code>recon_control/*/</code>. Contents: [Sec. 1](#sec-1) prompts, [Sec. 2](#sec-2) example inputs, [Sec. 3](#sec-3) per-task descriptions and runs, [Sec. 4](#sec-4) per-example disagreements, [Sec. 5](#sec-5) tokens, [Sec. 6](#sec-6) full metrics.</p></div>

<h2 id="sec-1">1. Prompts, verbatim</h2>
<p>Placeholders in angle brackets are filled per task. All calls used <code>claude -p --model claude-opus-5-5 --restricted --strict-mcp-config</code>; the summarizer had <code>--tools Read</code> only; reconstructors and controls had Read, Edit, Write, Glob, Grep and Bash, with Bash allowed only for the scoring command.</p>
<details open><summary>Summarizer prompt</summary>{pre(p_sum)}<p>If over 200 words, a follow-up in the same session: "That is N words. Rewrite it in at most 190 words, keeping the most important information. Output only the description."</p></details>
<details><summary>Reconstructor prompt (with description)</summary>{pre(p_rec)}</details>
<details><summary>Control prompt (no description)</summary>{pre(p_ctl)}</details>
<details><summary>Task statement as inserted (sort-five-digits example)</summary>{pre(ex_doc)}</details>

<h2 id="sec-2">2. Example task inputs</h2>
<p>Five visible examples per task (seed 0), as the model sees them (prompt, then the target answer). For the dataset tasks the sentences are lowercased and padding underscores are collapsed; labels are shown by name here, but the model sees the digit.</p>{ex_tab}

<h2 id="sec-3">3. Per task: description, reconstructions and control</h2>
<p>For every task: the selected original, the exact description the reconstructors received, and for each of two reconstructions and the control the attempts they logged, their final message and usage.</p>{"".join(blocks)}

<h2 id="sec-4">4. Dataset tasks: example disagreements</h2>{"".join(dis)}

<h2 id="sec-5">5. Token usage per run</h2>
<p>From the result record of each session. Cache read tokens are the repeated context re-read on each turn and dominate the count. Summarizer calls (17 tasks, $5.06 total) did not record tokens.</p><div class="scroll">{tok_tab}</div>

<h2 id="sec-6">6. Full comparison metrics</h2>
<p>Definitions as in the main report. Each row compares a reconstruction or control with the original on one example set.</p><div class="scroll">{met_tab}</div>
</div></body></html>"""
page = re.sub(r"\[(Sec\.) (\d+)\]\(((?:report\.html)?#[^)]+)\)", r'\1 <a href="\3">\2</a>', page)
open(f"{HERE}/report_raw.html", "w").write(page)
print("ok", len(page))
