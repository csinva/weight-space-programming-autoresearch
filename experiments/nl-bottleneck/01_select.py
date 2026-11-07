"""Pick the best Opus 5.5 program per task: highest visible accuracy (200 seed-0 examples), among programs the data-use audit
marks data_independent, ties broken by fewest parameters. Copies each to originals/<task>.py and writes selected.json.
Reads the source runs from SRC (read-only)."""
import csv, glob, json, os, re, shutil, sys
SRC = os.path.expanduser("~/weight-space-programming-autoresearch/runs-opus55")
HERE = os.path.dirname(os.path.abspath(__file__))
aud = [json.load(open(os.path.join(SRC, f)))["per_snapshot"] for f in ("audit_data_use.json", "audit_data_use_reconstructed.json")]
out = {}
for rd in sorted(glob.glob(f"{SRC}/*-opus55-s[0-9]")):
    run = os.path.basename(rd); task = re.sub(r"-opus55-s\d+$", "", run)
    for r in csv.DictReader(open(f"{rd}/results/overall_results.csv")):
        name = r["model_shorthand_name"]; safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        path = next((p for p in [f"{rd}/reconstructed/{safe}.py", f"{rd}/interpretable_transformers_lib/{safe}.py", f"{rd}/interpretable_transformers_lib/{name}.py"] if os.path.exists(p)), None)
        labs = {a.get(run, {}).get(name) or a.get(run, {}).get(safe) for a in aud} - {None}
        if path is None or "data_independent" not in labs or not r["accuracy"]:
            continue
        key = (float(r["accuracy"]), -float(r["n_params"]))
        if task not in out or key > out[task]["key"]:
            out[task] = dict(key=key, run=run, program=name, path=path, visible=float(r["accuracy"]), n_params=float(r["n_params"]), description=r["description"])
os.makedirs(f"{HERE}/originals", exist_ok=True)
for t, v in out.items():
    shutil.copy(v.pop("path"), f"{HERE}/originals/{t}.py"); v.pop("key")
    print(t, v["run"], v["program"], v["visible"], v["n_params"])
json.dump(out, open(f"{HERE}/selected.json", "w"), indent=1)
