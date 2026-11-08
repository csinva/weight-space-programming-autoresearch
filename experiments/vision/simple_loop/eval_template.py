#!/data19/chandan/wsp_vision/venv/bin/python
"""The only command the agent may run in a simple-loop vision run folder (copied in as ./eval.py).

  ./eval.py --idea "<plain-language label>"     score model.py on the VALIDATION split and log it
  ./eval.py --check                             run model.py on synthetic images on CPU (shape, finiteness, time); nothing is logged

Scoring calls the repository's trusted scorer (search.score: static reward-hack scan, image-only bwrap sandbox, ridge readout fit on
each search subject's own images, Pearson r on the validation half of the shared images, averaged over the search subjects).
The hidden half and the held-out subjects are never touched here. Each scored model is copied to models_lib/<id>.py and logged to
results/overall_results.csv. The GPU is chosen at call time: the local card with the most free memory and the lowest utilisation.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = json.load(open(os.path.join(HERE, ".run_config.json")))
VISION = CONFIG["vision_dir"]
PY = CONFIG["python"]
FIELDS = ["id", "time", "idea", "status", "score", "best_so_far", "seconds", "gpu", "details"]


def pick_gpu(min_free_mib=25000):
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    cards = []
    for line in q:
        i, tot, used, util = (int(x) for x in line.split(","))
        cards.append((tot - used >= min_free_mib, util, used, i))
    cards.sort(key=lambda c: (not c[0], c[1], c[2]))
    return str(cards[0][3])


def rows():
    p = os.path.join(HERE, "results", "overall_results.csv")
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p)))


def check():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    code = ("import time, torch, importlib.util as u; s=u.spec_from_file_location('m','model.py'); m=u.module_from_spec(s); s.loader.exec_module(m);"
            f"X=torch.rand(8, 3*{CONFIG.get('imsize', 128)}**2); t=time.time(); f=m.MODEL(X); f=torch.as_tensor(f);"
            "print('output', tuple(f.shape), 'finite', bool(torch.isfinite(f).all()), 'seconds/8 images', round(time.time()-t,2),"
            "'n_params', getattr(m.MODEL,'n_params',None))")
    r = subprocess.run([PY, "-c", code], cwd=HERE, env=env, capture_output=True, text=True, timeout=600)
    print((r.stdout + r.stderr)[-3000:])


def score(idea):
    prev = rows()
    run_id = f"m{len(prev):03d}"
    gpu = pick_gpu()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, NEURO_STAGE="search")
    env.update(CONFIG.get("env", {}))
    t0 = time.time()
    cmd = [PY, "-m", "search.score", "--dataset", CONFIG["dataset"], "--region", CONFIG["region"], "--metric", CONFIG["metric"],
           "--model", os.path.join(HERE, "model.py")]
    r = subprocess.run(cmd, cwd=VISION, env=env, capture_output=True, text=True, timeout=1800)
    secs = round(time.time() - t0)
    try:
        res = json.loads(r.stdout[r.stdout.index("{"):])
    except ValueError:
        res = {"status": "error: scorer output unreadable", "log": (r.stdout + r.stderr)[-1500:]}
    status = str(res.get("status", ""))
    val = res.get(CONFIG["score_key"]) if status.startswith("ok") else None
    best = max([float(x["score"]) for x in prev if x["score"]] + ([val] if val is not None else []), default=None)
    os.makedirs(os.path.join(HERE, "models_lib"), exist_ok=True)
    shutil.copy(os.path.join(HERE, "model.py"), os.path.join(HERE, "models_lib", f"{run_id}.py"))
    details = {k: v for k, v in res.items() if k not in ("log", "repair_log", "candidate_error") and not isinstance(v, (list, dict))}
    for k in ("regions", "subjects", "roi_scores", "subject_scores"):
        if k in res:
            details[k] = res[k]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "overall_results.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(dict(id=run_id, time=time.strftime("%Y-%m-%d %H:%M:%S"), idea=idea, status=status[:200],
                        score="" if val is None else f"{val:.5f}", best_so_far="" if best is None else f"{best:.5f}",
                        seconds=secs, gpu=gpu, details=json.dumps(details)[:4000]))
    print(f"{run_id}: status {status[:200]}")
    if val is not None:
        prev_best = max([float(x["score"]) for x in prev if x["score"]], default=None)
        delta = "" if prev_best is None else f" (best before {prev_best:.4f}, delta {val - prev_best:+.4f})"
        print(f"validation score ({CONFIG['score_key']}): {val:.4f}{delta}")
    print("details:", json.dumps(details)[:2500])
    if not status.startswith("ok"):
        print("error log:", (res.get("candidate_error") or res.get("log") or "")[-1500:])
    print(f"{secs}s on GPU {gpu}; snapshot models_lib/{run_id}.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    elif a.idea:
        score(a.idea)
    else:
        sys.exit("usage: ./eval.py --idea '<label>'  or  ./eval.py --check")
