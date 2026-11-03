"""Zero-shot experiment-1 predictions only (no feedback): round 0, round 2 at k=0, and rounds 3+ (fresh edits).
Token edits are excluded. Prompt variants: v0 (original), v1 (calibration guidance, both conditions), v2 (v1 + calculation procedure, raw only).
Round 3 is the dev set used to pick a variant; rounds 4+ are the held-out test set. Writes analyze/systematic_tests/zeroshot.json and fig1_zeroshot.png."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import OUT, ROOT, metrics
from common import load_prompts, ALT
SUF = "_alt" if ALT else ""  # lower-ranked-token targets: rounds_alt/, outputs *_alt

FAMS = ["scalar", "neuron", "layer"]
NBOOT = int(os.environ.get("NBOOT", 500))
LABEL = {"none": "Baseline", "raw": "Read raw weights"}
METRICS = ["spearman", "edit_rank", "big_auc", "mae"]


def load_rows():
    prm = [p["pid"] for p in load_prompts()]
    base = os.path.join(ROOT, "rounds_alt" if ALT else "rounds")
    gts = {int(os.path.basename(os.path.dirname(g))[1:]): json.load(open(g)) for g in glob.glob(os.path.join(base, "r*", "ground_truth.json"))}
    if ALT:  # every round was predicted with no examples
        sources = [(r, os.path.join(base, f"r{r}")) for r in sorted(gts)]
    else:
        sources = [(0, os.path.join(base, "r0")), (2, os.path.join(ROOT, "rounds_fewshot", "r2_k0"))] + [(r, os.path.join(base, f"r{r}")) for r in sorted(gts) if r >= 3]
    rows = {}
    for r, d in sources:
        for fam in FAMS:
            for pp in glob.glob(os.path.join(d, fam, "*", "prediction.json")):
                cond = pp.split(os.sep)[-2]
                if cond.startswith("views"):
                    continue
                key = cond if "_" in cond else cond + "_v0"
                for k, v in json.load(open(pp))["predictions"].items():
                    iid, pid = k.split("|")
                    t = gts[r][iid]["per_prompt"][pid]
                    row = rows.setdefault((iid, pid), dict(round=r, family=fam, iid=iid, pid=pid, true=t["dlogp"], flip=bool(t["flip"]), kl=t["kl"]))
                    row[key], row[key + "_pf"] = v["dlogp"], v["p_flip"]
    return list(rows.values())


def boot_all(rows, keys, pairs, rng):
    by = {}
    for r in rows:
        by.setdefault(r["iid"], []).append(r)
    g = list(by.values())
    def stats(rs):
        out = {k: metrics([x for x in rs if k in x], k) for k in keys}
        for a, b in pairs:
            sub = [x for x in rs if a in x and b in x]
            ma, mb = metrics(sub, a), metrics(sub, b)
            out[f"{a}-{b}"] = {m: ma.get(m, np.nan) - mb.get(m, np.nan) for m in METRICS}
        return out
    est = stats(rows)
    draws = [stats([x for i in rng.integers(0, len(g), len(g)) for x in g[i]]) for _ in range(NBOOT)]
    res = {}
    for k in est:
        res[k] = {m: (float(est[k].get(m, np.nan)), float(np.nanstd([d[k].get(m, np.nan) for d in draws]))) for m in METRICS}
        res[k]["n_edits"] = len({x["iid"] for x in rows if (k in x if "-" not in k else all(z in x for z in k.split("-")))})
        res[k]["n_items"] = sum(1 for x in rows if (k in x if "-" not in k else all(z in x for z in k.split("-"))))
    return res


def fig_scatter(rows, lin=0.01):
    """Predicted against measured dlogp on symmetric log axes (linear for |x| < lin nats), so small and large effects are both visible."""
    fig, axs = plt.subplots(2, len(FAMS), figsize=(13, 6.4))
    for ri, k in enumerate(["none_v0", "raw_v0"]):
        for fi, fam in enumerate(FAMS):
            sub = [r for r in rows if r["family"] == fam and k in r]
            a_ = axs[ri, fi]
            a_.scatter([r["true"] for r in sub], [r[k] for r in sub], s=6, alpha=0.4, color="#9aa0a6" if ri == 0 else "#e8710a", lw=0)
            vals = [abs(r["true"]) for r in sub] + [abs(r[k]) for r in sub]
            m = max(vals + [1.0]) * 1.5
            a_.plot([-m, m], [-m, m], "k--", lw=0.5)
            for ax_ in (a_.set_xscale, a_.set_yscale):
                ax_("symlog", linthresh=lin)
            a_.set_xlim(-m, m); a_.set_ylim(-m, m)
            ticks = [t for t in [-10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10] if abs(t) <= m]
            lab = [("0" if t == 0 else f"{t:g}") for t in ticks]
            a_.set_xticks(ticks); a_.set_xticklabels(lab, fontsize=7, rotation=45); a_.set_yticks(ticks); a_.set_yticklabels(lab, fontsize=7)
            a_.minorticks_off()
            a_.axhline(0, color="k", lw=0.3); a_.axvline(0, color="k", lw=0.3)
            a_.set_title(f"{fam} / {LABEL[k[:-3]]}", fontsize=9)
            if fi == 0: a_.set_ylabel("predicted Δlogp (nats)")
            if ri == 1: a_.set_xlabel("measured Δlogp (nats)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig2_zeroshot_scatter{SUF}.png"), dpi=130); plt.close()


def main():
    rows = load_rows()
    rng = np.random.default_rng(0)
    keys_all = sorted({k for r in rows for k in r if k.endswith(("_v0", "_v1", "_v2"))})
    res = dict(all_zeroshot={}, dev={}, test={})
    for fam in FAMS + ["all"]:
        sub = [r for r in rows if fam == "all" or r["family"] == fam]
        extra = ["raw_all"] if any("raw_all" in r for r in sub) else []  # "Read all weights" condition (lower-ranked target)
        res["all_zeroshot"][fam] = boot_all(sub, ["none_v0", "raw_v0"] + extra, [("raw_v0", "none_v0")] + [(e, b) for e in extra for b in ["none_v0", "raw_v0"]], rng)
        dev = [r for r in sub if r["round"] == 3 and not ALT]
        if dev:
            ks = [k for k in keys_all if any(k in r for r in dev)]
            res["dev"][fam] = boot_all(dev, ks, [(a, "none_v0") for a in ks if a != "none_v0"] + [("raw_v1", "none_v1"), ("raw_v2", "none_v1")], rng)
        test = [r for r in sub if r["round"] >= 4 and not ALT]
        if test:
            ks = [k for k in keys_all if any(k in r for r in test)]
            pr = [(a, "none_v0") for a in ks if a != "none_v0"] + [(a, "none_v1") for a in ks if a.startswith("raw") and any("none_v1" in r for r in test) and a != "raw_v0"]
            res["test"][fam] = boot_all(test, ks, pr, rng)
    # significant changes only (same subset for both conditions), original prompt
    kl = {}
    for r in rows:
        kl.setdefault(r["iid"], []).append(r["kl"])
    res["filtered"] = {}
    for name, keep in [("|Δ|>0.1", lambda r: abs(r["true"]) > 0.1), ("|Δ|>1", lambda r: abs(r["true"]) > 1.0), ("edit KL>0.01", lambda r: np.mean(kl[r["iid"]]) > 0.01),
                       ("edit KL>0.1", lambda r: np.mean(kl[r["iid"]]) > 0.1)]:
        for fam in FAMS + ["all"]:
            sub = [r for r in rows if (fam == "all" or r["family"] == fam) and keep(r) and "none_v0" in r and "raw_v0" in r]
            if len({r["iid"] for r in sub}) >= 5:
                b = boot_all(sub, ["none_v0", "raw_v0"], [("raw_v0", "none_v0")], rng)
                sign = {k: float(np.mean([np.sign(r[k]) == np.sign(r["true"]) for r in sub if abs(r["true"]) > 0.1])) if any(abs(r["true"]) > 0.1 for r in sub) else np.nan for k in ["none_v0", "raw_v0"]}
                res["filtered"][f"{name}|{fam}"] = dict(n=len(sub), boot=b, sign=sign)
    if not ALT:
        # few-shot on set 3's edits (round 2) without token edits: k0, k4 (rounds_fewshot) and k24 (the round-2 run)
        prm = [p["pid"] for p in load_prompts()]
        gt2 = json.load(open(os.path.join(ROOT, "rounds", "r2", "ground_truth.json")))
        fsrows = {}
        for k, d in [("k0", os.path.join(ROOT, "rounds_fewshot", "r2_k0")), ("k4", os.path.join(ROOT, "rounds_fewshot", "r2_k4")), ("k24", os.path.join(ROOT, "rounds", "r2"))]:
            for fam in FAMS:
                for cond in ["none", "raw"]:
                    for kk, v in json.load(open(os.path.join(d, fam, cond, "prediction.json")))["predictions"].items():
                        iid, pid = kk.split("|")
                        t = gt2[iid]["per_prompt"][pid]
                        row = fsrows.setdefault((iid, pid), dict(round=2, family=fam, iid=iid, pid=pid, true=t["dlogp"], flip=bool(t["flip"])))
                        row[f"{cond}@{k}"], row[f"{cond}@{k}_pf"] = v["dlogp"], v["p_flip"]
        fsr = list(fsrows.values())
        res["fewshot"] = boot_all(fsr, [f"{c}@{k}" for c in ["none", "raw"] for k in ["k0", "k4", "k24"]],
                                  [(f"{c}@{k}", f"{c}@k0") for c in ["none", "raw"] for k in ["k4", "k24"]] + [(f"raw@{k}", f"none@{k}") for k in ["k0", "k4", "k24"]], rng)
        res["fewshot_n_edits"] = len({r["iid"] for r in fsr})
    json.dump(res, open(os.path.join(OUT, f"zeroshot{SUF}.json"), "w"), indent=1)
    # Fig 1: Baseline vs Read raw weights, all zero-shot items, original prompt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for mi, m in enumerate(["spearman", "mae"]):
        for ci, (k, col) in enumerate([("none_v0", "#9aa0a6"), ("raw_v0", "#e8710a")]):
            est = [res["all_zeroshot"][f][k][m][0] for f in FAMS]; se = [res["all_zeroshot"][f][k][m][1] for f in FAMS]
            ax[mi].bar(np.arange(len(FAMS)) + (ci - 0.5) * 0.36, est, 0.36, yerr=se, color=col, label=LABEL[k[:-3]], capsize=2)
        ax[mi].set_xticks(range(len(FAMS))); ax[mi].set_xticklabels([f"{f}\n(n={res['all_zeroshot'][f]['none_v0']['n_edits']})" for f in FAMS])
        ax[mi].set_ylabel(dict(spearman="Spearman ρ (pred vs true Δlogp)", mae="MAE of Δlogp (nats)")[m]); ax[mi].axhline(0, color="k", lw=0.6)
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig1_zeroshot{SUF}.png"), dpi=150); plt.close()
    fig_scatter(rows)
    for part in ["all_zeroshot", "dev", "test"]:
        for fam, d in res[part].items():
            print(part, fam, " | ".join(f"{k}: ρ {v['spearman'][0]:+.2f}±{v['spearman'][1]:.2f} mae {v['mae'][0]:.3f} n={v['n_edits']}" for k, v in d.items()))
    return res


if __name__ == "__main__":
    import sys
    fig_scatter(load_rows()) if sys.argv[1:] == ["scatter"] else main()
