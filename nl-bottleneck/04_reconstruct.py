"""Step 2: a fresh Opus 5.5 session (no access to the original program) rebuilds a model from the <=200-word description.
Sandbox: template interpretable_transformer.py + src/ + results/, outside any repo. The agent may score on the visible 200 examples (up to MAX_EXP experiments).
Outputs: recon/<task>-r<k>/ (agent log, results csv, snapshots)

  python 04_reconstruct.py --reps 1 2 --workers 12
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
EVOLVE = os.path.join(os.path.dirname(HERE), "evolve")
sys.path.insert(0, EVOLVE)
import src.task as T
SB = "/data18/chandan.singh/wsp-nl/sandbox"
OUT = f"{HERE}/recon"  # --control writes to recon_control (same agent and budget, but no description of the original)
PY = os.path.expanduser("~/weight-space-programming-autoresearch/.venv/bin/python")
MAX_EXP = 6
CONTROL = False
TOOLS = "Read,Edit,Write,Glob,Grep,Bash"
ALLOWED = [f"Bash({PY} interpretable_transformer.py:*)", "Edit", "Write"]
HEAD_RECON = """You are rebuilding a hand-written transformer. An earlier agent wrote, with no training, the weights of a small transformer for the task `{task}`. You will not see that model. All you get is the description below, written by another engineer who did read it. Your job is to hand-write a transformer that implements the same algorithm, so that its accuracy AND its mistakes resemble the original's.

TASK:
{doc}
Prompts are fed token by token (character level); the model generates the answer autoregressively with greedy decoding; score = exact-match accuracy on 200 fixed examples (seed 0). The original model scored {vis:.1%} on these 200 examples and has {npar:,} parameters.

DESCRIPTION OF THE ORIGINAL MODEL ({nw} words):
\"\"\"
{summary}
\"\"\"
"""
HEAD_CONTROL = """You are hand-writing the weights of a small transformer, with no training, for the task `{task}`.

TASK:
{doc}
Prompts are fed token by token (character level); the model generates the answer autoregressively with greedy decoding; score = exact-match accuracy on 200 fixed examples (seed 0). Goal: as high an accuracy as you can get within {maxexp} experiments.
"""
TAIL = """
Your working folder has interpretable_transformer.py (a template transformer; edit the architecture and `write_weights(model, task)` so every parameter is set directly in closed form), src/task.py and src/eval.py (read-only task and scoring code), and results/.
Rules:
- NO training of any kind: no gradient steps, optimizers, fitting loops, least squares or searches over weights. Parameters are set in closed form from the algorithm.
- Do not compute the answer with Python outside the transformer; the network must produce it autoregressively. Do not use pretrained weights.
- Do not fit weights, lexicons or thresholds to examples or labels from the dataset (for dataset tasks write any word lists from your own knowledge{guide}). Do not read anything outside this folder.
- Only edit interpretable_transformer.py and write snapshots; do not modify src/.
Procedure: implement your approach, then score it with exactly `{py} interpretable_transformer.py --task {task}` (set `model_shorthand_name` and `model_description` in the file before each run; each run appends a row to results/overall_results.csv). Run at most {maxexp} experiments in total. After each run save a copy of the file to interpretable_transformers_lib/<model_shorthand_name>.py with the Write tool. Use the experiments to fix bugs and improve the model.{end}"""
END_RECON = " Do not tune it to beat the original: the goal is faithfulness, so do not 'fix' limitations the description says the original has. Finish with a one-line statement of which snapshot is your final model."
END_CONTROL = " Finish with a one-line statement of which snapshot is your final model."


def run(job):
    task, k = job
    rd = f"{OUT}/{task}-r{k}"
    if os.path.exists(f"{rd}/done"):
        return f"{task}-r{k} cached"
    shutil.rmtree(rd, ignore_errors=True); os.makedirs(rd)
    sel = json.load(open(f"{HERE}/selected.json"))[task]
    summary = open(f"{HERE}/summaries/{task}.txt").read().strip()
    sb = f"{SB}/rec-{task}-r{k}"
    shutil.rmtree(sb, ignore_errors=True); os.makedirs(sb)
    shutil.copytree(f"{EVOLVE}/src", f"{sb}/src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(f"{EVOLVE}/interpretable_transformer.py", sb)
    os.makedirs(f"{sb}/results"); os.makedirs(f"{sb}/interpretable_transformers_lib")
    doc = re.sub(r"\n +", "\n", T.get_task(task).__doc__.strip())
    head = (HEAD_CONTROL if CONTROL else HEAD_RECON).format(task=task, doc=doc, vis=sel["visible"], npar=int(sel["n_params"]), nw=len(summary.split()), summary=summary, maxexp=MAX_EXP)
    prompt = head + TAIL.format(task=task, py=PY, maxexp=MAX_EXP, guide="" if CONTROL else ", guided by the description", end=END_CONTROL if CONTROL else END_RECON)
    env = dict(os.environ, HF_HOME="/data18/chandan.singh/wsp-nl/hf_home", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", DATASETS_VERBOSITY="error", OMP_NUM_THREADS="4")
    cmd = ["claude", "-p", prompt, "--model", "claude-opus-5-5", "--restricted", "--tools", TOOLS, "--strict-mcp-config",
           "--output-format", "stream-json", "--verbose", "--allowedTools", *ALLOWED]
    t0 = time.time()
    with open(f"{rd}/agent_log.jsonl", "w") as fh:
        subprocess.run(cmd, cwd=sb, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env, timeout=3 * 3600)
    res, tools = {}, {}
    for line in open(f"{rd}/agent_log.jsonl"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "result":
            res = d
        if d.get("type") == "assistant":
            for c in d["message"]["content"]:
                if c.get("type") == "tool_use":
                    tools[c["name"]] = tools.get(c["name"], 0) + 1
    for sub in ("results", "interpretable_transformers_lib"):
        shutil.copytree(f"{sb}/{sub}", f"{rd}/{sub}", ignore=shutil.ignore_patterns("*.png"))
    n = sum(1 for _ in csv.DictReader(open(f"{sb}/results/overall_results.csv"))) if os.path.exists(f"{sb}/results/overall_results.csv") else 0
    json.dump(dict(task=task, rep=k, experiments=n, cost_usd=res.get("total_cost_usd"), turns=res.get("num_turns"), seconds=round(time.time() - t0), tools=tools, final_text=res.get("result", "")[-500:]), open(f"{rd}/meta.json", "w"), indent=1)
    open(f"{rd}/done", "w").write("1")
    return f"{task}-r{k}: {n} experiments, ${res.get('total_cost_usd')}, {round(time.time() - t0)}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--control", action="store_true", help="no description of the original model")
    a = ap.parse_args()
    if a.control:
        CONTROL, OUT = True, f"{HERE}/recon_control"
        SB = SB + "/control"
    tasks = sorted(json.load(open(f"{HERE}/selected.json"))) if a.tasks == ["all"] else a.tasks
    jobs = [(t, k) for k in a.reps for t in tasks]
    with ThreadPoolExecutor(a.workers) as ex:
        for m in ex.map(run, jobs):
            print(m, flush=True)
