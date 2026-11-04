"""Run the autoresearch loop with Claude Code in headless mode (`claude -p`), several independent runs per task.

  uv run evolve/launch_claude.py --tasks all --seeds 1 2 --workers 8

Each run gets a fresh folder runs-opus55/<task>-opus55-s<seed>/ made by setup_run.py (its own copy of program.md with the
task name filled in). The agent works only there: tools are restricted to Read/Edit/Write/Glob/Grep/Bash, file reads outside
the folder are refused (restricted mode), and the only shell command it may run is `uv run interpretable_transformer.py ...`.
The driver resumes the same session until results/overall_results.csv has the target number of attempts
(easy 30, harder 40, real world 50) or two sessions in a row add nothing. Transcripts go to <run>/agent_logs/."""
import argparse, csv, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from setup_run import setup_run

MODEL = "claude-opus-5-5"
RUNS_SUBDIR = "runs-opus55"
TASKS = {
    "addition-five-digits": "easy", "multiplication-five-digits": "easy", "sort-five-digits": "easy", "digit-counting-10": "easy",
    "parity-upto10-bits": "easy", "boolean-circuit-5-bits": "easy", "linear-interpolation-two-points": "easy",
    "word-reversal-3x3": "harder", "gcd-three-digits": "harder", "decimal-to-binary-8bit": "harder",
    "edit-distance-8": "harder", "smallest-prime-factor": "harder",
    "sentiment-sst2": "real", "paraphrase-mrpc": "real", "nli-snli": "real", "trec-coarse": "real", "date-normalization": "real",
}
TARGET = dict(easy=30, harder=40, real=50)
MAX_SESSIONS = 25
TOOLS = "Read,Edit,Write,Glob,Grep,Bash"
ALLOWED = ["Bash(uv run interpretable_transformer.py:*)", "Edit", "Write"]

FIRST = ("Read and follow the instructions in program.md. Your task name is `{task}`. Notes on this harness: the run folder has already been created "
         "and you are in it, so skip the setup steps that create it; run experiments on CPU (the default device); the only shell command available "
         "is `uv run interpretable_transformer.py --task {task}` (redirecting output, e.g. `> run.log 2>&1`, is fine), so use the Read and Grep "
         "tools to inspect run.log and results/overall_results.csv, and save each snapshot by writing a copy of interpretable_transformer.py to "
         "interpretable_transformers_lib/<model_shorthand_name>.py with the Write tool. Keep looping: do at least {target} experiments "
         "(each one logged as a row of results/overall_results.csv); there are {n} so far.")
CONT = ("Continue the autoresearch loop from program.md for task `{task}`. results/overall_results.csv now has {n} experiments; keep going "
        "until there are at least {target}, then keep simplifying if time allows. Same rules and tools as before.")


def n_attempts(run_dir):
    p = os.path.join(run_dir, "results", "overall_results.csv")
    with open(p) as f:
        return sum(1 for _ in csv.DictReader(f))


def run_one(task, seed):
    tag = f"{task}-opus55-s{seed}"
    run_dir = os.path.join(ROOT, RUNS_SUBDIR, tag)
    if not os.path.exists(run_dir):
        setup_run(tag, runs_subdir=RUNS_SUBDIR)
        pm = os.path.join(run_dir, "program.md")
        s = open(pm).read()
        s = re.sub(r"your task name is `[^`]+`", f"your task name is `{task}`", s, count=1)
        open(pm, "w").write(s)
    os.makedirs(os.path.join(run_dir, "agent_logs"), exist_ok=True)
    meta_path = os.path.join(run_dir, "metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else dict(
        run=tag, task=task, seed=seed, model=MODEL, effort="default", driver="claude -p (headless Claude Code)", tools=TOOLS,
        allowed=ALLOWED, target=TARGET[TASKS[task]], sessions=[], started=datetime.now(timezone.utc).isoformat())
    target, stalls, sid = meta["target"], 0, (meta["sessions"][-1]["session_id"] if meta["sessions"] else None)
    env = dict(os.environ, HF_HOME=os.path.join(ROOT, ".hf_home"), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    while len(meta["sessions"]) < MAX_SESSIONS:
        n0 = n_attempts(run_dir)
        if n0 >= target:
            break
        prompt = (FIRST if sid is None else CONT).format(task=task, target=target, n=n0)
        cmd = ["claude", "-p", prompt, "--model", MODEL, "--restricted", "--tools", TOOLS, "--strict-mcp-config",
               "--output-format", "stream-json", "--verbose", "--allowedTools", *ALLOWED] + (["--resume", sid] if sid else [])
        k = len(meta["sessions"])
        log = os.path.join(run_dir, "agent_logs", f"session_{k:02d}.jsonl")
        t0 = time.time()
        with open(log, "w") as fh:
            p = subprocess.run(cmd, cwd=run_dir, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env, timeout=6 * 3600)
        res, tools_used = {}, set()
        for line in open(log):  # transcripts are gzipped after the runs (gzip -9 runs-opus55/*/agent_logs/*.jsonl)
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "result":
                res = d
            if d.get("type") == "assistant":
                tools_used |= {c["name"] for c in d["message"]["content"] if c.get("type") == "tool_use"}
        n1 = n_attempts(run_dir)
        sid = res.get("session_id", sid)
        meta["sessions"].append(dict(k=k, session_id=sid, exit=p.returncode, is_error=res.get("is_error"), cost_usd=res.get("total_cost_usd"),
                                     turns=res.get("num_turns"), seconds=round(time.time() - t0), attempts_before=n0, attempts_after=n1,
                                     tools_used=sorted(tools_used), log=os.path.relpath(log, run_dir)))
        json.dump(meta, open(meta_path, "w"), indent=1)
        print(f"{tag} session {k}: {n0} -> {n1} attempts, ${res.get('total_cost_usd')}, {round(time.time() - t0)}s", flush=True)
        if "limit" in json.dumps(res).lower() and n1 == n0:
            time.sleep(1800)  # usage limit: wait and resume
            continue
        stalls = stalls + 1 if n1 == n0 else 0
        if stalls >= 2:
            break
    meta.update(finished=datetime.now(timezone.utc).isoformat(), attempts=n_attempts(run_dir))
    json.dump(meta, open(meta_path, "w"), indent=1)
    return f"{tag}: {meta['attempts']} attempts in {len(meta['sessions'])} sessions"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tasks = list(TASKS) if a.tasks == ["all"] else a.tasks
    # hardest first so the long runs start early
    order = sorted([(t, s) for t in tasks for s in a.seeds], key=lambda x: {"real": 0, "harder": 1, "easy": 2}[TASKS[x[0]]])
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda x: run_one(*x), order):
            print(msg, flush=True)
