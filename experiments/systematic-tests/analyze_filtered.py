"""Post-hoc: score each condition only on items / edits where the output really changed (same subset for every condition).
Writes analyze/systematic_tests/filtered.json."""
import json, os
import numpy as np
from scipy.stats import spearmanr
from analyze import OUT, CONDS

rows = json.load(open(os.path.join(OUT, "items.json")))
rng = np.random.default_rng(0)


def score(sub, key):
    y = np.array([r["true"] for r in sub]); p = np.array([r[key] for r in sub], float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    return dict(rho=spearmanr(p, y)[0] if len(y) > 3 else np.nan, sign=float(np.mean(np.sign(p) == np.sign(y))),
                mae=float(np.mean(np.abs(p - y))), mae_zero=float(np.mean(np.abs(y))),
                log_mag_rho=spearmanr(np.abs(p), np.abs(y))[0] if len(y) > 3 else np.nan)


def paired(sub, a, b, metric, n=1000):
    groups = {}
    for r in sub:
        groups.setdefault((r["round"], r["iid"]), []).append(r)
    g = list(groups.values())
    d = []
    for _ in range(n):
        rs = [x for i in rng.integers(0, len(g), len(g)) for x in g[i]]
        d.append(score(rs, a)[metric] - score(rs, b)[metric])
    d = np.array(d); d = d[~np.isnan(d)]
    e = float(score(sub, a)[metric] - score(sub, b)[metric])
    return e, e - float(np.std(d)), e + float(np.std(d))  # +-1 bootstrap SE


out = {}
# main + few-shot keys present on round-2 rows; use the main-round predictions (k24 on round 2) for all rounds
for thr in [0.1, 0.5, 1.0]:
    for fam in ["all", "scalar", "layer", "token"]:
        sub = [r for r in rows if abs(r["true"]) > thr and (fam == "all" or r["family"] == fam)]
        if len(sub) < 8:
            continue
        out[f"item>{thr}|{fam}"] = dict(n=len(sub), n_edits=len({(r['round'], r['iid']) for r in sub}),
                                        **{c: score(sub, c) for c in CONDS},
                                        **{f"{c}-none": {m: paired(sub, c, "none", m) for m in ["rho", "sign", "log_mag_rho"]} for c in ["raw", "views"]})
# edit-level significance: edits whose mean KL over the 8 prompts exceeds a threshold (all their prompts kept)
kl = {}
for r in rows:
    kl.setdefault((r["round"], r["iid"]), []).append(r["kl"])
for thr in [0.01, 0.1, 0.5]:
    keep = {k for k, v in kl.items() if np.mean(v) > thr}
    sub = [r for r in rows if (r["round"], r["iid"]) in keep]
    for fam in ["all", "scalar", "layer", "token"]:
        s = [r for r in sub if fam == "all" or r["family"] == fam]
        if len({(r['round'], r['iid']) for r in s}) < 4:
            continue
        out[f"editKL>{thr}|{fam}"] = dict(n=len(s), n_edits=len({(r['round'], r['iid']) for r in s}), **{c: score(s, c) for c in CONDS},
                                          **{f"{c}-none": {m: paired(s, c, "none", m) for m in ["rho", "sign"]} for c in ["raw", "views"]})
json.dump(out, open(os.path.join(OUT, "filtered.json"), "w"), indent=1)
for k, v in out.items():
    print(k.ljust(20), f"n={v['n']:4d} edits={v['n_edits']:3d}", " | ".join(f"{c}: rho {v[c]['rho']:+.2f} sign {v[c]['sign']:.2f}" for c in CONDS),
          " || raw-none rho {:+.2f} [{:+.2f},{:+.2f}] views-none rho {:+.2f} [{:+.2f},{:+.2f}]".format(*v["raw-none"]["rho"], *v["views-none"]["rho"]))
