"""Step 3: pick each reconstruction's final program (best visible accuracy among its logged experiments that have a saved snapshot, ties -> fewest
parameters, the same rule used for the originals), copy it to <dir>_final/, and score it with eval_program.py (visible / hidden / extra).

  python 05_eval_recon.py [recon|recon_control]
"""
import csv, glob, json, os, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.expanduser("~/weight-space-programming-autoresearch/.venv/bin/python")
d = sys.argv[1] if len(sys.argv) > 1 else "recon"
os.makedirs(f"{HERE}/{d}_final", exist_ok=True); os.makedirs(f"{HERE}/preds/{d}", exist_ok=True)
env = dict(os.environ, HF_HOME="/data18/chandan.singh/wsp-nl/hf_home", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", DATASETS_VERBOSITY="error", OMP_NUM_THREADS="4")
sel, jobs = {}, []
for rd in sorted(glob.glob(f"{HERE}/{d}/*-r[0-9]")):
    tag = os.path.basename(rd); task = re.sub(r"-r\d+$", "", tag)
    best, attempts = None, []
    p = f"{rd}/results/overall_results.csv"
    for r in (csv.DictReader(open(p)) if os.path.exists(p) else []):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", r["model_shorthand_name"])
        path = next((q for q in [f"{rd}/interpretable_transformers_lib/{safe}.py", f"{rd}/interpretable_transformers_lib/{r['model_shorthand_name']}.py"] if os.path.exists(q)), None)
        acc = float(r["accuracy"]) if r["accuracy"] else None
        attempts.append(dict(name=r["model_shorthand_name"], visible=acc, n_params=float(r["n_params"]) if r["n_params"] else None, snapshot=bool(path), description=r["description"]))
        if acc is not None and path and (best is None or (acc, -float(r["n_params"])) > best[0]):
            best = ((acc, -float(r["n_params"])), path, r["model_shorthand_name"])
    fallback = False
    if best and "def build_model" not in open(best[1]).read():  # the agent's Write copy was truncated: use the agent's last working file instead
        sb = f"/data18/chandan.singh/wsp-nl/sandbox/{'control/' if d.endswith('control') else ''}rec-{tag}/interpretable_transformer.py"
        best, fallback = (best[0], sb, re.search(r'^model_shorthand_name = "([^"]+)"', open(sb).read(), re.M).group(1)), True
    sel[tag] = dict(task=task, attempts=attempts, final=best[2] if best else None, fallback_to_last_file=fallback)
    if best:
        shutil.copy(best[1], f"{HERE}/{d}_final/{tag}.py")
        jobs.append((tag, task))
json.dump(sel, open(f"{HERE}/{d}_selected.json", "w"), indent=1)


def go(j):
    tag, task = j
    r = subprocess.run([PY, f"{HERE}/eval_program.py", f"{HERE}/{d}_final/{tag}.py", task, f"{HERE}/preds/{d}/{tag}.json"], capture_output=True, text=True, env=env, timeout=3600)
    return tag + " " + (r.stdout.strip().splitlines() or [""])[-1] + ("" if r.returncode == 0 else " FAILED " + r.stderr.strip()[-300:])


with ThreadPoolExecutor(8) as ex:
    for m in ex.map(go, jobs):
        print(m, flush=True)
