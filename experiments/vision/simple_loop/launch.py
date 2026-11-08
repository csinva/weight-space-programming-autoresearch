#!/usr/bin/env python3
"""Simple iWSP loop for the vision encoding models: headless Claude Code (Opus 5.5) resumed until a run has the target number of scored
experiments. Adapted from experiments/evolve/launch_claude.py; no orchestrator, coordinator or proposer pool (search/ is not used).

  python simple_loop/launch.py --dataset nsd --region visual --seeds 1 --target 60      (from experiments/vision)

Run folder: experiments/runs/vision_opus55/<dataset>-<region>-s<seed>/ with model.py (the repository's starting model), program.md,
eval.py (the only allowed shell command; scores the VALIDATION split on a free local GPU), .run_config.json, scratch/notes.md, results/,
models_lib/ and agent_logs/. Tools are Read/Edit/Write/Glob/Grep/Bash in restricted mode (no reads outside the folder).
"""
import argparse
import csv
import json
import os
import shutil
import stat
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
VISION = os.path.dirname(HERE)
RUNS = os.path.join(os.path.dirname(VISION), "runs", "vision_opus55")
PY = "/data19/chandan/wsp_vision/venv/bin/python"
MODEL = "claude-opus-5-5"
TOOLS = "Read,Edit,Write,Glob,Grep,Bash"
ALLOWED = ["Bash(./eval.py:*)", "Edit", "Write"]
MAX_SESSIONS = 25
COHORT = {
    "nsd": "NSD human 7T fMRI; search subjects 1-4, hidden subjects 5-8 and a hidden half of the shared images",
    "laion": "LAION-fMRI human fMRI; search subjects sub-01, 03, 05, hidden sub-06, 07 and a hidden half of the shared images",
    "triplen": "Triple-N macaque spiking; search monkeys share all images, hidden images held out",
    "tvsd": "TVSD macaque spiking on THINGS images; search monkey F, hidden monkey N and hidden images",
}
FIRST = ("Read and follow the instructions in program.md. The run folder is ready and you are in it. The only shell command available is "
         "./eval.py (use --check for a quick CPU test and --idea '<label>' to score). Use the Read and Grep tools to inspect "
         "results/overall_results.csv and your notes. Do at least {target} scored experiments; there are {n} so far.")
CONT = ("Continue the loop from program.md. results/overall_results.csv has {n} scored experiments; keep going until there are at least "
        "{target}, building new interpretable motifs and recombining them. Same rules and tools as before.")


def n_scored(run_dir):
    p = os.path.join(run_dir, "results", "overall_results.csv")
    return sum(1 for _ in csv.DictReader(open(p))) if os.path.exists(p) else 0


def setup(run_dir, dataset, region, metric, target, data_env, score_key):
    os.makedirs(os.path.join(run_dir, "scratch"))
    for d in ("results", "models_lib", "agent_logs"):
        os.makedirs(os.path.join(run_dir, d))
    shutil.copy(os.path.join(VISION, "model.py"), os.path.join(run_dir, "model.py"))
    prog = open(os.path.join(HERE, "program.md")).read().format(dataset=dataset, region=region, metric=metric, cohort=COHORT[dataset],
                                                                 imsize=128, target=target)
    open(os.path.join(run_dir, "program.md"), "w").write(prog)
    ev = os.path.join(run_dir, "eval.py")
    shutil.copy(os.path.join(HERE, "eval_template.py"), ev)
    os.chmod(ev, os.stat(ev).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    json.dump(dict(dataset=dataset, region=region, metric=metric, vision_dir=VISION, python=PY, score_key=score_key, imsize=128,
                   env=data_env), open(os.path.join(run_dir, ".run_config.json"), "w"), indent=1)
    open(os.path.join(run_dir, "scratch", "notes.md"), "w").write("# Notes\n\n(no experiments yet)\n")


def run_one(dataset, region, metric, seed, target, data_env, score_key):
    tag = f"{dataset}-{region}-s{seed}"
    run_dir = os.path.join(RUNS, tag)
    if not os.path.exists(run_dir):
        setup(run_dir, dataset, region, metric, target, data_env, score_key)
    meta_path = os.path.join(run_dir, "metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else dict(
        run=tag, dataset=dataset, region=region, metric=metric, seed=seed, model=MODEL, driver="claude -p (headless Claude Code)",
        tools=TOOLS, allowed=ALLOWED, target=target, sessions=[], started=datetime.now(timezone.utc).isoformat())
    stalls, sid = 0, (meta["sessions"][-1]["session_id"] if meta["sessions"] else None)
    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    while len(meta["sessions"]) < MAX_SESSIONS:
        n0 = n_scored(run_dir)
        if n0 >= target:
            break
        prompt = (FIRST if sid is None else CONT).format(target=target, n=n0)
        cmd = ["claude", "-p", prompt, "--model", MODEL, "--restricted", "--tools", TOOLS, "--strict-mcp-config",
               "--output-format", "stream-json", "--verbose", "--allowedTools", *ALLOWED] + (["--resume", sid] if sid else [])
        k = len(meta["sessions"])
        log = os.path.join(run_dir, "agent_logs", f"session_{k:02d}.jsonl")
        t0 = time.time()
        with open(log, "w") as fh:
            p = subprocess.run(cmd, cwd=run_dir, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env, timeout=8 * 3600)
        res, tools_used = {}, set()
        for line in open(log):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "result":
                res = d
            if d.get("type") == "assistant":
                tools_used |= {c["name"] for c in d["message"]["content"] if c.get("type") == "tool_use"}
        n1 = n_scored(run_dir)
        sid = res.get("session_id", sid)
        meta["sessions"].append(dict(k=k, session_id=sid, exit=p.returncode, is_error=res.get("is_error"), cost_usd=res.get("total_cost_usd"),
                                     turns=res.get("num_turns"), seconds=round(time.time() - t0), scored_before=n0, scored_after=n1,
                                     tools_used=sorted(tools_used), log=os.path.relpath(log, run_dir)))
        json.dump(meta, open(meta_path, "w"), indent=1)
        print(f"{tag} session {k}: {n0} -> {n1} scored, ${res.get('total_cost_usd')}, {round(time.time() - t0)}s", flush=True)
        if "limit" in json.dumps(res).lower() and n1 == n0:
            time.sleep(1800)
            continue
        stalls = stalls + 1 if n1 == n0 else 0
        if stalls >= 2:
            break
    meta.update(finished=datetime.now(timezone.utc).isoformat(), scored=n_scored(run_dir))
    json.dump(meta, open(meta_path, "w"), indent=1)
    return f"{tag}: {meta['scored']} scored in {len(meta['sessions'])} sessions"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(COHORT))
    ap.add_argument("--region", default="visual")
    ap.add_argument("--metric", default="encoding")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--score-key", default="test_corr")
    a = ap.parse_args()
    data_env = json.load(open(os.path.join(HERE, "data_env.json")))
    with ThreadPoolExecutor(len(a.seeds)) as ex:
        for msg in ex.map(lambda s: run_one(a.dataset, a.region, a.metric, s, a.target, data_env, a.score_key), a.seeds):
            print(msg, flush=True)
