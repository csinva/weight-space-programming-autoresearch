"""What explains when reading raw weights helps? Per edit: benefit b = mean over its prompts of
|pr(|Baseline pred|) - pr(|true|)| - |pr(|Read raw pred|) - pr(|true|)|, where pr = percentile rank within the family
(calibration-free closeness in effect size; > 0 = Read raw weights ranks this edit's effect sizes closer to the truth).
Compared against weights-only size features of the edit (norm of the weight change, relative size, layer) and the true effect size."""
import json, os
import numpy as np
import torch
from scipy.stats import spearmanr
from common import ALT, load
from interventions import P, mixer_out
import analyze_zeroshot as A
from analyze import OUT

L = lambda x: np.log10(np.abs(np.asarray(x)) + 1e-3)


def features(it):
    k = it["kind"]
    if k in ("max_weight", "add_value"):
        W = P(it["param"]); r, c = it["index"]
        d = abs(float(W[r, c])) if k == "max_weight" else abs(it["value"])
        return dict(delta=d, rel=d / float(W.std()), layer=int(it["param"].split(".")[2]) if ".layers." in it["param"] else (-1 if "embed" in it["param"] else 24))
    if k == "norm_gain":
        g = 1 + float(P(it["param"])[it["index"][0]])
        d = abs(g) if it["op"] == "zero_gain" else 9 * abs(g)
        gv = 1 + P(it["param"]).float()
        return dict(delta=d, rel=d / float(gv.abs().mean()), layer=int(it["param"].split(".")[2]) if ".layers." in it["param"] else 24)
    if k == "neuron":
        Wd = P(f"model.layers.{it['layer']}.mlp.down_proj.weight")
        n = float(Wd[:, it["neuron"]].norm()); f = 1.0 if it["op"] == "zero" else 4.0
        return dict(delta=n * f, rel=n / float(Wd.norm(dim=0).median()) * f, layer=it["layer"])
    if k == "layer":
        d2 = 0.0
        if it["op"] in ("skip_mixer", "skip_block"):
            d2 += float(P(mixer_out(it["layer"])).norm()) ** 2
        if it["op"] in ("skip_mlp", "skip_block"):
            d2 += float(P(f"model.layers.{it['layer']}.mlp.down_proj.weight").norm()) ** 2
        return dict(delta=d2 ** 0.5, rel=d2 ** 0.5, layer=it["layer"])
    raise ValueError(k)


def main():
    load()
    rows = [r for r in A.load_rows() if "none_v0" in r and "raw_v0" in r]
    iv = {it["iid"]: it for v in json.load(open(os.path.join(A.ROOT, "data", "interventions.json"))).values() for it in v}
    # calibration-free closeness: within each family, percentile rank of |prediction| vs percentile rank of |true effect|
    from scipy.stats import rankdata
    for fam in {r["family"] for r in rows}:
        rs = [r for r in rows if r["family"] == fam]
        for k in ["true", "none_v0", "raw_v0"]:
            pr = rankdata([abs(r[k]) for r in rs]) / len(rs)
            for r, v in zip(rs, pr):
                r["pr_" + k] = v
    by = {}
    for r in rows:
        by.setdefault(r["iid"], []).append(r)
    E = []
    for iid, rs in by.items():
        y = np.array([r["true"] for r in rs])
        b = np.mean([abs(r["pr_none_v0"] - r["pr_true"]) - abs(r["pr_raw_v0"] - r["pr_true"]) for r in rs])
        E.append(dict(iid=iid, family=rs[0]["family"], kind=iv[iid]["kind"], benefit=float(b), effect=float(np.mean(np.abs(y))), **features(iv[iid])))
    res = {}
    rng = np.random.default_rng(0)
    for fam in ["scalar", "neuron", "layer", "all"]:
        sub = [e for e in E if fam == "all" or e["family"] == fam]
        b = np.array([e["benefit"] for e in sub])
        out = dict(n=len(sub), mean_benefit=(float(b.mean()), float(b.std() / np.sqrt(len(b)))))
        for f in ["delta", "rel", "layer", "effect"]:
            x = np.array([e[f] for e in sub])
            out[f"rho_{f}"] = float(spearmanr(x, b)[0])
            qs = np.quantile(x, [0, .25, .5, .75, 1])
            bins = []
            for i in range(4):
                m = (x >= qs[i]) & (x <= qs[i + 1]) if i == 3 else (x >= qs[i]) & (x < qs[i + 1])
                bins.append(dict(lo=float(qs[i]), hi=float(qs[i + 1]), n=int(m.sum()), benefit=float(b[m].mean()) if m.any() else np.nan,
                                 se=float(b[m].std() / np.sqrt(max(m.sum(), 1))) if m.any() else np.nan))
            out[f"bins_{f}"] = bins
        res[fam] = out
        print(fam, f"n={len(sub)} mean benefit {out['mean_benefit'][0]:+.3f}±{out['mean_benefit'][1]:.3f}",
              " ".join(f"ρ({f})={out[f'rho_{f}']:+.2f}" for f in ["delta", "rel", "layer", "effect"]))
        for f in ["rel", "effect"]:
            print("   by", f, " | ".join(f"[{q['lo']:.3g},{q['hi']:.3g}] n={q['n']} {q['benefit']:+.3f}±{q['se']:.3f}" for q in out[f"bins_{f}"]))
    # by edit kind
    for k in sorted({e["kind"] for e in E}):
        b = np.array([e["benefit"] for e in E if e["kind"] == k])
        print("kind", k, len(b), f"{b.mean():+.3f}±{b.std() / np.sqrt(len(b)):.3f}")
        res[f"kind|{k}"] = (len(b), float(b.mean()), float(b.std() / np.sqrt(len(b))))
    json.dump(dict(res=res, edits=E), open(os.path.join(OUT, f"explain{'_alt' if ALT else ''}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
