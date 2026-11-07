"""Run the repo's data-use audit (evolve/audit_data_use.py) on the final reconstructed programs: label_dependent / input_dependent programs
would break the no-fitting rule. Builds a temporary audit root outside the repo with one pseudo run folder per program.

  python 07_audit.py recon_final   -> audit_recon_final.json
"""
import glob, json, os, re, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(os.path.dirname(HERE), "evolve")
sys.path.insert(0, EV)
import audit_data_use as A
d = sys.argv[1]
root = f"/data18/chandan.singh/wsp-nl/audit_root_{d}"
shutil.rmtree(root, ignore_errors=True)
jobs = []
for p in sorted(glob.glob(f"{HERE}/{d}/*.py")):
    tag = os.path.basename(p)[:-3]; task = re.sub(r"-r\d+$", "", tag)
    rd = f"{root}/{task}-opus55-s{tag.rsplit('-r', 1)[1]}"
    os.makedirs(f"{rd}/lib"); os.symlink(f"{EV}/src", f"{rd}/src"); shutil.copy(p, f"{rd}/lib/{tag}.py")
    jobs.append((rd, f"{rd}/lib/{tag}.py"))
if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    with ProcessPoolExecutor(16) as ex:
        for path, res in ex.map(A.check, jobs):
            out[os.path.basename(path)[:-3]] = res
    json.dump(out, open(f"{HERE}/audit_{d}.json", "w"), indent=1)
    from collections import Counter
    print(Counter(out.values()))
    print({k: v for k, v in out.items() if v != "data_independent"})
