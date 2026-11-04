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

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "report_raw.html"

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
    "edit-distance-8": "Harder",
    "smallest-prime-factor": "Harder",
    "sentiment-sst2": "Real world",
    "paraphrase-mrpc": "Real world",
    "nli-snli": "Real world",
    "trec-coarse": "Real world",
    "date-normalization": "Real world",
}
CAT_ORDER = ["Easy", "Harder", "Real world"]

# The agent model and thinking effort for these runs are not recorded in the repository
# (the Copilot CLI logs from May 2026 are gone). Fill these in before publishing.
AGENT_MODEL = None
AGENT_EFFORT = None
RUN_DATE = "May 15, 2026"


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


def collect(runs_root: Path = RUNS) -> list[dict]:
    """One entry per run folder: task, seed (from metadata.json if present), attempts and an auto-generated writeup."""
    out = []
    if not runs_root.exists():
        return out
    for run_dir in sorted(runs_root.iterdir()):
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
                    "status": "failed" if row.get("status") in ("failed", "crash") else "success",
                    "desc": row.get("description", "") or "",
                })
        if not methods:
            continue
        task = next(iter(csv.DictReader(csv_path.open())), {}).get("task") or run_dir.name
        meta = json.loads((run_dir / "metadata.json").read_text()) if (run_dir / "metadata.json").exists() else {}
        audit = {}
        ap = runs_root / "audit_data_use.json"
        if ap.exists():
            per = json.loads(ap.read_text())["per_snapshot"].get(run_dir.name, {})
            for x in methods:
                x["data"] = per.get(x["name"], "missing")
        out.append({
            "task": task, "run": run_dir.name, "seed": meta.get("seed", 1),
            "category": CATEGORY.get(task, "Other"), "methods": methods, "writeup": make_writeup(task, methods),
            "cost": sum((x.get("cost_usd") or 0) for x in meta.get("sessions", [])), "sessions": len(meta.get("sessions", [])),
            "target": meta.get("target"),
        })
    out.sort(key=lambda t: (CAT_ORDER.index(t["category"]) if t["category"] in CAT_ORDER else 99, t["task"], t["seed"]))
    return out


def group_by_task(runs: list[dict]) -> list[dict]:
    tasks = {}
    for r in runs:
        tasks.setdefault(r["task"], dict(task=r["task"], category=r["category"], seeds=[]))["seeds"].append(r)
    return list(tasks.values())


def summarize(r: dict) -> dict:
    m = r["methods"]
    best = max(x["acc"] for x in m)
    first_best = next(x["i"] for x in m if x["acc"] >= best - 1e-9)
    near = [x for x in m if x["acc"] >= best - 0.01]
    smallest = min(near, key=lambda x: x["params"])
    return dict(best=best, first_best=first_best, smallest=smallest, n=len(m))


def trec_table() -> str:
    p = ROOT / "runs-opus55" / "heldout_trec.json"
    if not p.exists():
        return ""
    rows = "".join(
        f"<tr><td>{i}</td><td>{html.escape(r['program'])}</td><td>{r['eval_200']:.1%}</td><td>{r['official_test'][0]:.1%}</td><td>{r['rest_of_train'][0]:.1%}</td></tr>"
        for i, r in enumerate(json.loads(p.read_text()).values(), 1))
    fp = ROOT / "runs-opus55" / "heldout_fresh_seed.json"
    fresh = json.loads(fp.read_text()) if fp.exists() else {}
    n_ok = sum(1 for v in fresh.values() if v.get("fresh_seed") == 1.0)
    note = (f"""<p class="note" style="margin-top:14px"><b>Fresh-seed check on generated tasks.</b> In {n_ok} of the {len(fresh)} runs on tasks with a generator (the 12 algorithmic tasks and date
    normalization), the first saved program to reach the run's best accuracy also scores 100% on 2,000 new examples from an unseen seed, so these scores do not depend on the fixed 200 evaluation examples
    (<code>evolve/heldout_fresh_seed.py</code>).</p>""" if fresh else "")
    return (note + """<h3 style="margin-top:24px">Held-out check on TREC</h3>
  <p class="note">The best program of each TREC run, scored on data outside its 2,000-example pool, which the agent never saw: the official test split (500 questions)
    and the remaining 3,452 training questions. The best-of-50 score on the fixed 200 examples is inflated by up to about 4 points relative to the test split.</p>
  <div class="scroll" style="max-height:none"><table class="mlist"><thead><tr><th>run</th><th>program</th><th>fixed 200 examples</th><th>official test</th><th>rest of training split</th></tr></thead><tbody>"""
            + rows + "</tbody></table></div>")


def n_dep(r: dict) -> str:
    m = r["methods"]
    if not any("data" in x for x in m):
        return "n/a"
    d = sum(1 for x in m if x.get("data") == "data_dependent")
    return f"{d} of {len(m)}"


def best_indep(r: dict) -> str:
    ok = [x for x in r["methods"] if x.get("data") == "data_independent"]
    if not any("data" in x for x in r["methods"]):
        return "n/a"
    return f"{max(x['acc'] for x in ok):.1%}" if ok else "none"


def results_table(new_tasks: list[dict]) -> str:
    rows = []
    for t in new_tasks:
        for i, r in enumerate(t["seeds"]):
            sm = summarize(r)
            rows.append(
                (f"<tr><td rowspan={len(t['seeds'])}><code>{html.escape(t['task'])}</code></td><td rowspan={len(t['seeds'])}>{t['category']}</td>" if i == 0 else "<tr>") +
                f"<td>{r['seed']}</td><td>{sm['n']}</td><td class='acc-pill' style='color:{'#059669' if sm['best'] >= 0.99 else '#374151'}'>{sm['best']:.1%}</td>"
                f"<td>{sm['first_best']}</td><td>{fmt_params(sm['smallest']['params'])} <span style='color:#6b7280'>({html.escape(sm['smallest']['name'])}, {sm['smallest']['acc']:.1%})</span></td>"
                f"<td>{n_dep(r)}</td><td>{best_indep(r)}</td>"
                f"<td>{r['sessions']}</td><td>${r['cost']:.0f}</td></tr>")
    return "".join(rows)


def task_rows(tasks: list[dict]) -> str:
    """One table row per task, from the task definitions in evolve/src/task.py."""
    import inspect
    import sys
    sys.path.insert(0, str(ROOT / "evolve"))
    from src.task import TASK_REGISTRY
    n_attempts = {}
    for t in tasks:
        n_attempts[t["task"]] = n_attempts.get(t["task"], 0) + len(t["methods"])
    rows = []
    for t in sorted(TASK_REGISTRY, key=lambda n: (CAT_ORDER.index(CATEGORY.get(n, "Real world")), n)):
        cls = TASK_REGISTRY[t]
        doc = [l.strip() for l in (inspect.getdoc(cls) or "").split("\n")]
        fmt = {k: next((l.split(":", 1)[1].strip() for l in doc if l.lower().startswith(k + ":")), "") for k in ["prompt", "answer", "example"]}
        rows.append(
            f"<tr><td><code>{html.escape(t)}</code></td><td>{CATEGORY.get(t, '')}</td><td>{html.escape(doc[0])}</td>"
            f"<td><code>{html.escape(fmt['prompt'])}</code></td><td><code>{html.escape(fmt['answer'])}</code></td>"
            f"<td>{html.escape(fmt['example'])}</td><td>{len(cls.vocab)}</td><td>{n_attempts.get(t, 0)}</td></tr>")
    return "".join(rows)


SHARED_DEFS = """
    <p><b>Network.</b> A decoder-only (GPT-style) transformer defined in <code>interpretable_transformer.py</code>: learned token and position
      embeddings, a stack of pre-LayerNorm blocks (causal multi-head self-attention and a ReLU MLP, each with a residual connection), a final LayerNorm and a
      linear output head over a character-level vocabulary specific to each task (4 to 43 symbols). The starting configuration is 3 layers, width 64,
      4 heads and MLP width 256; the agent may change the architecture, depth, width, number of heads and MLP width. All parameters are set inside
      <code>write_weights(model, task)</code>. Parameter count is <code>sum(p.numel() for p in model.parameters())</code>.</p>
    <p><b>Rules given to the agent</b> (<code>evolve/program.md</code>). Allowed: editing anything in its copy of
      <code>interpretable_transformer.py</code> except the evaluation code in the main block, including the architecture and hard-coded constants,
      arrays, hand-built attention circuits and lookup tables. Not allowed: any training (gradient descent, optimizer steps, fitting loops,
      backpropagation), computing the task output with Python functions or libraries outside the transformer, using pretrained weights, and
      modifying the shared task and evaluation code. Goal: maximize accuracy; all else equal, prefer more interpretable solutions, with
      parameter count as a suggested proxy.</p>
    <p><b>Evaluation.</b> Running <code>uv run interpretable_transformer.py --task &lt;task&gt;</code> builds the network, generates the answer greedily
      (argmax at each step) for a fixed number of answer tokens, and scores <b>accuracy</b>: the fraction of examples whose generated answer string
      exactly matches the target (no partial credit or numerical tolerance). Each attempt is scored on the same 200 examples, produced by the task's
      generator with seed 0, so there is no separate held-out set, and the agent could read the generator code. For the four dataset tasks the
      examples are sampled from a fixed subset of 2,000 examples of the training split (SST-2, GLUE/MRPC, SNLI, TREC), lowercased, with characters
      outside the task vocabulary replaced by spaces and each sentence truncated or padded to a fixed length. The real-world group also includes
      date normalization, whose examples are generated from nine common date formats rather than drawn from a dataset.</p>
"""


def definitions(new_runs: list[dict]) -> str:
    """Precise description of the Opus 5.5 runs, shown at the top of the report."""
    n_att = sum(len(r["methods"]) for r in new_runs)
    tasks = sorted({r["task"] for r in new_runs})
    return f"""
  <section class="defs" id="setup">
    <h2>Setup</h2>
    <p><b>What was done.</b> For each of {len(tasks)} tasks with a known algorithm or a known dataset, an autonomous coding agent was asked to set
      every weight of a small transformer by hand, with no training, so that the transformer solves the task by generating the answer token by token.
      Each task was run twice, independently (seeds 1 and 2: two separate agent sessions from the same starting files; the agent's own sampling makes
      them differ). In each run the agent repeatedly edited the weights, evaluated them and logged the result, until it reached at least 30 attempts for the
      easy tasks, 40 for the harder ones and 50 for the real-world ones. This page shows all {n_att} logged attempts of the {len(new_runs)} runs.
      An attempt is one version of the weight-writing code (a <i>weight-space program</i>) and its evaluation.</p>
    <p><b>Agent.</b> Claude Opus 5.5 (<code>claude-opus-5-5</code>, default thinking effort) running headless in Claude Code (<code>claude -p</code>),
      driven by <code>evolve/launch_claude.py</code>. The prompt tells it to read and follow <code>program.md</code> (the same instructions as the earlier runs,
      with its task name filled in), notes that its run folder already exists, and asks for at least the target number of attempts. If a session ends early,
      the driver resumes the same session with a short "continue the loop" message, until the target is reached or two sessions in a row add no attempts.
      There was no human intervention.</p>
    <p><b>What the harness enforces.</b> Each run works in its own folder <code>runs-opus55/&lt;task&gt;-opus55-s&lt;seed&gt;</code>, created by
      <code>evolve/setup_run.py</code>, with a private copy of <code>interpretable_transformer.py</code> and <code>program.md</code>; the task and
      evaluation code (<code>src/</code>) is a symlink to the shared <code>evolve/src</code>, so scoring is identical across runs. Claude Code runs in
      restricted mode with only the Read, Edit, Write, Glob, Grep and Bash tools: file reads are confined to the run folder (so the earlier runs'
      solutions cannot be opened), there is no web access, and the only shell command it may run, apart from read-only listings of its own folder, is
      <code>uv run interpretable_transformer.py ...</code>; every other command is refused. Snapshots are saved with the Write tool. The rules on training,
      outside computation and pretrained weights remain instructions: the agent's own Python could in principle break them, which the snapshots are checked for.</p>
    <p><b>Audits after the runs.</b> Four checks, all in <code>evolve/</code>.
      <i>Transcripts</i> (<code>audit_transcripts.py</code>): every tool call of every session. Runs used only the six allowed tools, and no call read or
      changed another run's files; the few paths outside a run folder were failed calls on nonexistent files and Claude Code's own saved command output.
      <i>Static scan</i> (<code>audit_runs.py</code>): no saved program contains training or fitting code (optimizers, <code>.backward()</code>,
      <code>.fit(</code>, least squares, pseudo-inverse).
      <i>Counterfactual data check</i> (<code>audit_data_use.py</code>): each saved program is built normally, built again (to confirm determinism), and built
      with the task data altered (a different sample with answers shuffled; for the dataset tasks the other half of the training split with labels
      shuffled). A program is <b>data-dependent</b> if its weights change, which means they were computed from task examples or chosen by checking accuracy
      on them. No saved program is data-dependent; a positive control (programs that set weights from label counts or generated answers) is flagged.
      <i>Indirect fitting</i>: no session printed data statistics (word frequencies, log-odds and so on) or ran a fitting procedure (logistic regression,
      perceptron) whose numbers could have been copied into a later program.
      Across all runs, 1,277 saved programs are data-independent, none are data-dependent, 7 are not deterministic (so the check cannot tell) and 9 fail to build.</p>
    <p><b>Data-guided hand tuning.</b> A caveat for the dataset tasks: in 7 of the 8 runs the agent added a measurement-only "dev" check that prints accuracy
      (and sometimes a confusion matrix) on examples from the same 2,000-example pool, usually excluding the 200 evaluation prompts, and adjusted its hand-set
      word lists, thresholds and biases across attempts. The two TREC runs went further: nearly every attempt printed misclassified dev questions with their true and predicted
      labels, and the agent wrote rules in response. This is manual feature engineering guided by data, as a person would do it, rather than training, but those rules
      are informed by labeled examples. Few evaluation prompts appeared in tool outputs (0 to 4 of 200 in 7 of the 8 runs; up to 26 in NLI run 2, an upper bound because
      SNLI reuses premises across examples). In every run the agent also sees the accuracy on the fixed evaluation set after each attempt.</p>
{SHARED_DEFS}
    <p><b>Plots and tables.</b> Each point is one attempt: x is its parameter count (log scale), y its accuracy; attempts are numbered in the order
      they were logged in <code>results/overall_results.csv</code>. "First attempt at best" is the attempt number at which a run first reached its best
      accuracy; "smallest near-best" is the attempt with the fewest parameters within 1 point of the run's best accuracy.</p>
    <details><summary><b>Tasks</b></summary>
    <div class="scroll" style="max-height:none"><table class="mlist">
      <thead><tr><th>task</th><th>group</th><th>definition</th><th>prompt</th><th>answer</th><th>example</th><th>vocab</th><th>attempts (both runs)</th></tr></thead>
      <tbody>{task_rows(new_runs)}</tbody></table></div></details>
  </section>
"""


def old_definitions(old_runs: list[dict]) -> str:
    todo = "<span class='todo'>[not recorded in the repository; fill in]</span>"
    model = html.escape(AGENT_MODEL) if AGENT_MODEL else todo
    effort = html.escape(AGENT_EFFORT) if AGENT_EFFORT else todo
    return f"""
    <p class="note" style="font-size:14px">An earlier, smaller study on {RUN_DATE}: one run per task ({sum(len(r['methods']) for r in old_runs)} attempts in total) with Claude,
      model version {model}, thinking effort {effort}, driven by the GitHub Copilot CLI (<code>copilot --autopilot --yolo</code>), prompted with
      "Read and follow the instructions in <code>evolve/program.md</code>" and told to make at least 20 attempts. Same network, rules and evaluation as above;
      the run folders are <code>runs/&lt;task&gt;-may15-run&lt;k&gt;</code>. In these runs the agent had the CLI's default tools and the rules were not
      enforced by the harness.</p>
"""


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weight-Space Programming Autoresearch &mdash; Results</title>
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
  .defs p { font-size:14.5px; margin:0 0 12px; max-width:900px; }
  .defs details summary { cursor:pointer; margin:6px 0; }
  .todo { background:#fef3c7; color:#92400e; padding:1px 6px; border-radius:4px; font-weight:600; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Weight-Space Programming Autoresearch</h1>
    <p class="sub">An autonomous agent writes transformer weights directly, with no training, to solve tasks with known solutions.
      Every point below is one weight-space program, numbered in the order it was tried.</p>
  </header>
__DEFS__
  <h2>Results: Claude Opus 5.5, two runs per task</h2>
  <p class="note">Best accuracy per run, the attempt at which it was first reached, and the smallest program within 1 point of the best.</p>
  <div class="scroll" style="max-height:none"><table class="mlist">
    <thead><tr><th>task</th><th>group</th><th>run (seed)</th><th>attempts</th><th>best accuracy</th><th>first attempt at best</th><th>smallest near-best (params)</th><th>attempts whose weights use task data</th><th>best without task data</th><th>sessions</th><th>cost (list price)</th></tr></thead>
    <tbody>__RESULTS__</tbody></table></div>

__TREC__
  <h2>Per-task exploration</h2>
  <p class="note">Pick a task. The plot shows accuracy against number of parameters for both runs (colour = run); points are connected in the
    order tried. Hover a point for its name and description.</p>
  <div class="tabs" id="tabsNew"></div>
  <div class="panel">
    <div class="card"><div id="taskPlotNew" style="width:100%;height:440px;"></div></div>
    <div class="writeup" id="writeupNew"></div>
  </div>

  <h2>All tasks together</h2>
  <p class="note">Each line is one run, connected in order of increasing parameter count.</p>
  <div class="card"><div id="summaryPlotNew" style="width:100%;height:560px;"></div></div>

  <h2 id="earlier">Earlier runs (May 2026, GitHub Copilot CLI)</h2>
__OLDDEFS__
  <div class="tabs" id="tabsOld"></div>
  <div class="panel">
    <div class="card"><div id="taskPlotOld" style="width:100%;height:440px;"></div></div>
    <div class="writeup" id="writeupOld"></div>
  </div>
  <h3 style="margin-top:28px">All tasks together (earlier runs)</h3>
  <div class="card"><div id="summaryPlotOld" style="width:100%;height:520px;"></div></div>

  <footer>
    Generated from <code>runs-opus55/*/results/overall_results.csv</code> and <code>runs/*/results/overall_results.csv</code> by <code>analyze/toy_examples/build_report.py</code>.
  </footer>
</div>

<script>
const NEW = __NEW__;
const OLD = __OLD__;
const PALETTE = ['#2563eb','#dc2626','#059669','#d97706','#7c3aed','#0891b2','#db2777',
                 '#65a30d','#ea580c','#4f46e5','#0d9488','#be123c','#a16207'];
const SEEDCOL = ['#2563eb','#ea580c','#059669'];
const LAYOUT_BASE = {
  paper_bgcolor:'#fff', plot_bgcolor:'#fff',
  font:{family:'-apple-system,Segoe UI,Roboto,sans-serif', color:'#1a1a1a', size:12},
  margin:{l:58,r:18,t:14,b:50},
};
const AXES = {
  xaxis:{title:{text:'# parameters (log scale)'}, type:'log', gridcolor:'#f1f5f9', zeroline:false},
  yaxis:{title:{text:'accuracy'}, range:[-0.04,1.06], gridcolor:'#f1f5f9', zeroline:false, tickformat:'.0%'},
};

function hoverText(d){
  return `#${d.i} <b>${esc(d.name)}</b><br>acc ${(d.acc*100).toFixed(1)}%<br>${d.params.toExponential(2)} params`+
         `${d.status==='failed'?'<br><i>failed</i>':''}<br>${wrap(d.desc,60)}`;
}
function methodTable(m){
  const rows = m.map(d=>
    `<tr><td class="num">${d.i}</td><td><b>${esc(d.name)}</b></td>`+
    `<td class="acc-pill" style="color:${d.acc>=0.99?'#059669':d.acc<0.5?'#b91c1c':'#374151'}">${(d.acc*100).toFixed(1)}%</td>`+
    `<td style="white-space:nowrap;color:#6b7280">${fmtP(d.params)}</td><td class="desc">${esc(d.desc)}</td></tr>`).join('');
  return `<div class="scroll"><table class="mlist"><thead><tr><th>#</th><th>method</th><th>acc</th><th>params</th><th>description</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
function renderTask(task, plotId, writeupId){
  const runs = task.seeds;
  const traces = runs.map((r,k)=>({
    name:`run ${r.seed}`, x:r.methods.map(d=>d.params), y:r.methods.map(d=>d.acc),
    text:r.methods.map(hoverText), mode:'lines+markers',
    line:{color:SEEDCOL[k%3], width:1, dash:k?'dot':'solid'}, opacity:0.9,
    marker:{size:7, color:SEEDCOL[k%3], line:{color:'#fff', width:0.6}},
    hovertemplate:'%{text}<extra>run '+r.seed+'</extra>',
  }));
  Plotly.react(plotId, traces, Object.assign({}, LAYOUT_BASE, AXES, {showlegend:runs.length>1, legend:{x:0.01,y:0.99}}), {displayModeBar:false, responsive:true});
  document.getElementById(writeupId).innerHTML =
    `<h3>${esc(task.task)}</h3><div class="meta">${esc(task.category)} &middot; ${runs.map(r=>`run ${r.seed}: ${r.methods.length} attempts`).join(' &middot; ')}</div>`+
    runs.map(r=>(runs.length>1?`<p style="margin-top:10px"><b>Run ${r.seed}.</b></p>`:'')+r.writeup+
      `<details class="methods"><summary>All ${r.methods.length} attempts of run ${r.seed}, in order</summary>${methodTable(r.methods)}</details>`).join('');
}
function renderSummary(tasks, plotId){
  const traces = [];
  tasks.forEach((t,i)=>{
    const c = PALETTE[i%PALETTE.length];
    t.seeds.forEach((r,k)=>{
      const pts = r.methods.slice().sort((a,b)=> a.params-b.params || a.acc-b.acc);
      traces.push({
        name: t.seeds.length>1?`${t.task} (run ${r.seed})`:t.task, legendgroup:t.task,
        x: pts.map(d=>d.params), y: pts.map(d=>d.acc),
        text: pts.map(d=>`#${d.i} ${esc(d.name)}<br>${(d.acc*100).toFixed(1)}% &middot; ${d.params.toExponential(1)}p`),
        mode:'lines+markers', line:{color:c, width:1.4, dash:k?'dot':'solid'},
        marker:{size:4, color:c, opacity:0.75}, hovertemplate:`<b>${esc(t.task)}</b> run ${r.seed}<br>%{text}<extra></extra>`,
      });
    });
  });
  Plotly.react(plotId, traces, Object.assign({}, LAYOUT_BASE, AXES, {legend:{font:{size:10}, bgcolor:'rgba(255,255,255,0.6)'}}), {displayModeBar:false, responsive:true});
}
function buildTabs(tasks, tabsId, plotId, writeupId){
  const el = document.getElementById(tabsId);
  el.innerHTML = tasks.map((t,i)=>`<button class="tab${i===0?' active':''}" data-i="${i}"><span class="cat">${esc(t.category)}</span>${esc(t.task)}</button>`).join('');
  el.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{
    el.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); renderTask(tasks[+b.dataset.i], plotId, writeupId);
  }));
  if (tasks.length) renderTask(tasks[0], plotId, writeupId);
}
function esc(s){ return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function wrap(s, n){ s=esc(s); let out='',line=0;
  for(const w of s.split(' ')){ if(line+w.length>n){out+='<br>';line=0;} out+=w+' '; line+=w.length+1; } return out; }
function fmtP(p){ if(p>=1e6) return (p/1e6).toFixed(2)+'M'; if(p>=1e3) return (p/1e3).toFixed(1)+'K'; return p.toFixed(0); }

buildTabs(NEW, 'tabsNew', 'taskPlotNew', 'writeupNew'); renderSummary(NEW, 'summaryPlotNew');
buildTabs(OLD, 'tabsOld', 'taskPlotOld', 'writeupOld'); renderSummary(OLD, 'summaryPlotOld');
</script>
</body>
</html>
"""


def main():
    new_runs = collect(ROOT / "runs-opus55")
    old_runs = collect(RUNS)
    new_tasks, old_tasks = group_by_task(new_runs), group_by_task(old_runs)
    out = (TEMPLATE.replace("__DEFS__", definitions(new_runs)).replace("__OLDDEFS__", old_definitions(old_runs))
           .replace("__RESULTS__", results_table(new_tasks)).replace("__TREC__", trec_table())
           .replace("__NEW__", json.dumps(new_tasks)).replace("__OLD__", json.dumps(old_tasks)))
    OUT.write_text(out)
    print(f"Wrote {OUT}: {len(new_runs)} Opus 5.5 runs ({sum(len(r['methods']) for r in new_runs)} attempts), "
          f"{len(old_runs)} earlier runs ({sum(len(r['methods']) for r in old_runs)} attempts).")


if __name__ == "__main__":
    main()
