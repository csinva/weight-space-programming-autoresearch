"""Score predictions against measured effects; write metrics + figures to analyze/systematic_tests/.
Cluster bootstrap over interventions (an intervention's 8 prompt items move together)."""
import glob, json, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, load_prompts
from interventions import FAMILIES

OUT = os.path.join(os.path.dirname(ROOT), "analyze", "systematic_tests")
CONDS = ["none", "raw", "views"]
BASELINES = ["zero", "feedback_mean"]
TAGS = ["fable"]  # other predictor models, stored in rounds_<tag>/ (same bundles and items)
PLOT_CONDS = ["none", "raw"]  # the report shows views only in an expandable
COLORS = dict(none="#9aa0a6", raw="#e8710a", views="#1a73e8", zero="#cccccc", feedback_mean="#34a853")
RNG = np.random.default_rng(0)
NBOOT = int(os.environ.get("NBOOT", 2000))


def op_key(it):
    return it.get("op", "") + "|" + it.get("kind", "")


def load_all():
    with open(os.path.join(ROOT, "data", "interventions.json")) as f:
        allv = json.load(f)
    prm = [p["pid"] for p in load_prompts()]
    rows = []  # one row per (round, family, iid, pid) with truth and each predictor
    gts = {}
    for gpath in sorted(glob.glob(os.path.join(ROOT, "rounds", "r*", "ground_truth.json"))):
        r = int(os.path.basename(os.path.dirname(gpath))[1:])
        if r > 2:  # this analysis covers the first three rounds (views, few-shot, Fable); later rounds are in analyze_zeroshot.py
            continue
        with open(gpath) as f:
            gts[r] = json.load(f)
    for r, gt in gts.items():
        for fam in FAMILIES:
            items = [it for it in allv[fam] if it["round"] == r]
            train = [it for it in allv[fam] if it["round"] < r]
            preds = {}
            for c in CONDS:
                pth = os.path.join(ROOT, "rounds", f"r{r}", fam, c, "prediction.json")
                if os.path.exists(pth):
                    with open(pth) as f:
                        preds[c] = json.load(f)["predictions"]
                for tag in TAGS:
                    tp = os.path.join(ROOT, f"rounds_{tag}", f"r{r}", fam, c, "prediction.json")
                    if os.path.exists(tp):
                        with open(tp) as f:
                            preds[f"{c}.{tag}"] = json.load(f)["predictions"]
                    for vp in glob.glob(os.path.join(ROOT, f"rounds_{tag}_fewshot", f"r{r}_k*", fam, c, "prediction.json")):
                        k = vp.split(os.sep)[-4].split("_")[1]
                        with open(vp) as f:
                            preds[f"{c}.{tag}@{k}"] = json.load(f)["predictions"]
                for vp in glob.glob(os.path.join(ROOT, "rounds_fewshot", f"r{r}_k*", fam, c, "prediction.json")):
                    k = vp.split(os.sep)[-4].split("_")[1]
                    with open(vp) as f:
                        preds[f"{c}@{k}"] = json.load(f)["predictions"]
            for it in items:
                for pid in prm:
                    t = gt[it["iid"]]["per_prompt"][pid]
                    row = dict(round=r, family=fam, iid=it["iid"], pid=pid, true=t["dlogp"], flip=bool(t["flip"]), kl=t["kl"])
                    for c, P in preds.items():
                        x = P.get(f"{it['iid']}|{pid}")
                        row[c] = x["dlogp"] if x else np.nan
                        row[c + "_pf"] = x["p_flip"] if x else np.nan
                    row["zero"], row["zero_pf"] = 0.0, 0.0
                    same = [gts[q.get("round")][q["iid"]]["per_prompt"][pid] for q in train if op_key(q) == op_key(it)]
                    alltr = [gts[q.get("round")][q["iid"]]["per_prompt"][pid] for q in train]
                    src = same or alltr
                    row["feedback_mean"] = float(np.mean([s["dlogp"] for s in src])) if src else np.nan
                    row["feedback_mean_pf"] = float(np.mean([s["flip"] for s in src])) if src else np.nan
                    rows.append(row)
    return rows, allv


def metrics(rows, key):
    y = np.array([r["true"] for r in rows]); p = np.array([r[key] for r in rows], float)
    f = np.array([r["flip"] for r in rows]); pf = np.array([r[key + "_pf"] for r in rows], float)
    ok = ~np.isnan(p)
    if ok.sum() < 3:
        return {}
    y, p, f, pf = y[ok], p[ok], f[ok], pf[ok]
    big = np.abs(y) > 0.1
    out = dict(n=int(ok.sum()), mae=float(np.mean(np.abs(p - y))),
               spearman=float(spearmanr(p, y)[0]) if np.std(p) > 0 else np.nan,
               sign_acc=float(np.mean(np.sign(p[big]) == np.sign(y[big]))) if big.any() else np.nan, n_big=int(big.sum()),
               brier=float(np.mean((pf - f) ** 2)),
               auc=float(roc_auc_score(f, pf)) if 0 < f.sum() < len(f) and np.std(pf) > 0 else np.nan, flip_rate=float(f.mean()),
               big_auc=float(roc_auc_score(big, np.abs(p))) if 0 < big.sum() < len(big) and np.std(p) > 0 else np.nan)
    # edit-level: rank edits by mean |dlogp| over their prompts (does it know which edits matter?)
    keys = [(r["round"], r["iid"]) for r, k in zip(rows, ok) if k]
    g = {}
    for kk, a, b in zip(keys, np.abs(p), np.abs(y)):
        g.setdefault(kk, []).append((a, b))
    if len(g) >= 4:
        pa = np.array([np.mean([x[0] for x in v]) for v in g.values()]); ya = np.array([np.mean([x[1] for x in v]) for v in g.values()])
        out["edit_rank"] = float(spearmanr(pa, ya)[0]) if np.std(pa) > 0 else np.nan
    return out


_BOOT_CACHE = {}


def boot(rows, key, metric, ref=None):
    """Point estimate and +-1 standard error (bootstrap over interventions); with ref, of metric(key) - metric(ref) on the same draws.
    All metrics are computed once per draw and cached per (rows, key, ref)."""
    ck = (id(rows), len(rows), key, ref)
    if ck not in _BOOT_CACHE:
        by = {}
        for r in rows:
            by.setdefault((r["round"], r["iid"]), []).append(r)
        groups = list(by.values())
        def val(rs):
            a = metrics(rs, key)
            if ref:
                b = metrics(rs, ref)
                return {m: a.get(m, np.nan) - b.get(m, np.nan) for m in a}
            return a
        est = val(rows)
        rng = np.random.default_rng(0)
        draws = [val([x for i in rng.integers(0, len(groups), len(groups)) for x in groups[i]]) for _ in range(NBOOT)]
        _BOOT_CACHE[ck] = (est, draws)
    est, draws = _BOOT_CACHE[ck]
    bs = np.array([d.get(metric, np.nan) for d in draws], float)
    bs = bs[~np.isnan(bs)]
    e = float(est.get(metric, np.nan))
    se = float(np.std(bs)) if len(bs) else np.nan  # bootstrap standard error over edits (= SEM over edits for means)
    return e, e - se, e + se, (float(np.mean(bs > 0)) if ref else None)


def main():
    rows, allv = load_all()
    os.makedirs(OUT, exist_ok=True)
    rounds = sorted({r["round"] for r in rows})
    res = dict(rounds=rounds, cells={}, pooled={}, paired={}, curve={})
    for fam in FAMILIES:
        for r in rounds:
            sub = [x for x in rows if x["family"] == fam and x["round"] == r]
            res["cells"][f"{fam}|{r}"] = {k: metrics(sub, k) for k in CONDS + BASELINES}
        sub = [x for x in rows if x["family"] == fam]
        res["pooled"][fam] = {k: {m: boot(sub, k, m) for m in ["spearman", "mae", "sign_acc", "auc", "big_auc", "edit_rank"]} for k in CONDS}
        res["paired"][fam] = {f"{k}-none": {m: boot(sub, k, m, ref="none") for m in ["spearman", "mae", "big_auc", "edit_rank"]} for k in ["raw", "views"]}
    for r in rounds:
        sub = [x for x in rows if x["round"] == r]
        res["curve"][r] = {k: {m: boot(sub, k, m) for m in ["spearman", "mae"]} for k in CONDS + (["feedback_mean"] if r > 0 else [])}
    # few-shot on a fixed test set (round-2 items): k0, k4 from rounds_fewshot, k24 = the round-2 run itself
    fs_rows = [x for x in rows if x["round"] == 2 and any("@" in k for k in x)]
    if fs_rows:
        for x in fs_rows:
            for c in CONDS:
                x[f"{c}@k24"], x[f"{c}@k24_pf"] = x[c], x[c + "_pf"]
        ks = sorted({k.split("@")[1] for k in fs_rows[0] if "@" in k and not k.endswith("_pf")}, key=lambda z: int(z[1:]))
        res["fewshot"] = dict(ks=ks, pooled={}, by_family={}, paired={})
        for c in CONDS:
            res["fewshot"]["pooled"][c] = {k: {m: boot(fs_rows, f"{c}@{k}", m) for m in ["spearman", "mae", "big_auc", "edit_rank"]} for k in ks}
        for fam in FAMILIES:
            sub = [x for x in fs_rows if x["family"] == fam]
            res["fewshot"]["by_family"][fam] = {c: {k: metrics(sub, f"{c}@{k}") for k in ks} for c in CONDS}
        for c in CONDS:
            res["fewshot"]["paired"][c] = {f"{k}-k0": {m: boot(fs_rows, f"{c}@{k}", m, ref=f"{c}@k0") for m in ["spearman", "mae"]} for k in ks if k != "k0"}
        for k in ks:
            res["fewshot"]["paired"][f"views-none@{k}"] = {m: boot(fs_rows, f"views@{k}", m, ref=f"none@{k}") for m in ["spearman", "mae"]}
            res["fewshot"]["paired"][f"raw-none@{k}"] = {m: boot(fs_rows, f"raw@{k}", m, ref=f"none@{k}") for m in ["spearman", "mae"]}
    if any("none.fable" in x for x in rows):
        res["models"] = {}
        for fam in FAMILIES + ["all"]:
            sub = [x for x in rows if (fam == "all" or x["family"] == fam) and "views.fable" in x and x["round"] == 0]  # zero-shot only
            res["models"][fam] = {
                **{f"{c}.{t}": {m: boot(sub, f"{c}.{t}", m) for m in ["spearman", "mae", "big_auc", "edit_rank"]} for c in CONDS for t in TAGS},
                **{f"{c}.{t}-{c}": {m: boot(sub, f"{c}.{t}", m, ref=c) for m in ["spearman", "mae", "big_auc", "edit_rank"]} for c in CONDS for t in TAGS},
                **{f"{c}.{t}-none.{t}": {m: boot(sub, f"{c}.{t}", m, ref=f"none.{t}") for m in ["spearman", "mae", "big_auc", "edit_rank"]} for c in ["raw", "views"] for t in TAGS}}
        fs2 = [x for x in rows if x["round"] == 2 and "views.fable@k0" in x]
        if fs2:
            for x in fs2:
                for c in CONDS:
                    x[f"{c}.fable@k24"], x[f"{c}.fable@k24_pf"] = x.get(f"{c}.fable", np.nan), x.get(f"{c}.fable_pf", np.nan)
            res["models_k0"] = {fam: {**{f"{c}{t}": {m: boot([x for x in fs2 if fam == "all" or x["family"] == fam], f"{c}{t}@k0", m) for m in ["spearman", "mae", "edit_rank", "big_auc"]}
                                         for c in CONDS for t in ["", ".fable"]},
                                      **{f"{c}.fable-{c}": {m: boot([x for x in fs2 if fam == "all" or x["family"] == fam], f"{c}.fable@k0", m, ref=f"{c}@k0") for m in ["spearman", "mae"]} for c in CONDS}}
                                for fam in FAMILIES + ["all"]}
    allrows = rows
    res["overall"] = {k: {m: boot(allrows, k, m) for m in ["spearman", "mae", "sign_acc", "auc", "big_auc", "edit_rank"]} for k in CONDS}
    res["overall_paired"] = {f"{k}-none": {m: boot(allrows, k, m, ref="none") for m in ["spearman", "mae", "big_auc", "edit_rank"]} for k in ["raw", "views"]}
    if len(rounds) > 1:
        late = [x for x in rows if x["round"] > 0]
        res["late_vs_feedback_mean"] = {k: {m: boot(late, k, m, ref="feedback_mean") for m in ["spearman", "mae"]} for k in CONDS}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT, "items.json"), "w") as f:
        json.dump(rows, f)
    figures(rows, res)
    return res


def fig_fewshot(res):
    fs = res["fewshot"]
    ks = fs["ks"]; x = [int(k[1:]) for k in ks]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    for c in PLOT_CONDS:
        for mi, m in enumerate(["spearman", "mae", "edit_rank"]):
            est = [fs["pooled"][c][k][m][0] for k in ks]
            lo = [fs["pooled"][c][k][m][1] for k in ks]; hi = [fs["pooled"][c][k][m][2] for k in ks]
            ax[mi].errorbar(x, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], marker="o", color=COLORS[c], label=c, capsize=2)
            ax[mi].set_xscale("symlog", linthresh=4); ax[mi].set_xticks(x); ax[mi].set_xticklabels(x)
            ax[mi].set_xlabel("solved examples shown (same 60 test edits)")
            ax[mi].set_ylabel(dict(spearman="Spearman ρ (items)", mae="MAE (nats)", edit_rank="edit-level rank ρ")[m])
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_fewshot.png"), dpi=150); plt.close()


def figures(rows, res):
    if "fewshot" in res:
        fig_fewshot(res)
    # Fig 1: pooled Spearman per family x condition
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for mi, m in enumerate(["spearman", "mae"]):
        for ci, c in enumerate(PLOT_CONDS):
            xs = np.arange(len(FAMILIES)) + (ci - 0.5) * 0.36
            est = [res["pooled"][f][c][m][0] for f in FAMILIES]
            err = [[e - res["pooled"][f][c][m][1] for f, e in zip(FAMILIES, est)], [res["pooled"][f][c][m][2] - e for f, e in zip(FAMILIES, est)]]
            ax[mi].bar(xs, est, 0.36, yerr=err, color=COLORS[c], label=c, capsize=2)
        ax[mi].set_xticks(range(len(FAMILIES))); ax[mi].set_xticklabels(FAMILIES)
        ax[mi].set_ylabel(dict(spearman="Spearman ρ (pred vs true Δlogp)", mae="MAE of Δlogp (nats)")[m]); ax[mi].axhline(0, color="k", lw=0.6)
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_by_family.png"), dpi=150); plt.close()
    # Fig 2: learning curve
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for mi, m in enumerate(["spearman", "mae"]):
        for c in PLOT_CONDS + ["feedback_mean"]:
            rs = [r for r in res["curve"] if c in res["curve"][r]]
            if not rs:
                continue
            est = [res["curve"][r][c][m][0] for r in rs]
            lo = [res["curve"][r][c][m][1] for r in rs]; hi = [res["curve"][r][c][m][2] for r in rs]
            x = [12 * int(r) for r in rs]
            ax[mi].errorbar(x, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], marker="o", color=COLORS[c], label=c, capsize=2)
        ax[mi].set_xlabel("measured examples revealed per family"); ax[mi].set_ylabel(dict(spearman="Spearman ρ", mae="MAE (nats)")[m])
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_learning.png"), dpi=150); plt.close()
    # Fig 3: scatter pred vs true, views and none
    fig, axs = plt.subplots(2, len(FAMILIES), figsize=(15, 5.6), sharex=False)
    for ri, c in enumerate(["none", "raw"]):
        for fi, fam in enumerate(FAMILIES):
            sub = [x for x in rows if x["family"] == fam and not np.isnan(x.get(c, np.nan))]
            a = axs[ri, fi]
            a.scatter([x["true"] for x in sub], [x[c] for x in sub], s=8, c=[["#1a73e8", "#e8710a", "#34a853"][x["round"]] for x in sub], alpha=0.7)
            lim = [min([x["true"] for x in sub] + [x[c] for x in sub] + [-0.5]), max([x["true"] for x in sub] + [x[c] for x in sub] + [0.5])]
            a.plot(lim, lim, "k--", lw=0.5); a.set_title(f"{fam} / {c}", fontsize=9)
            if fi == 0: a.set_ylabel("predicted Δlogp")
            if ri == 1: a.set_xlabel("measured Δlogp")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_scatter.png"), dpi=130); plt.close()


def figures_only():
    """Redraw figures from saved items.json / metrics.json (no bootstrap rerun)."""
    figures(json.load(open(os.path.join(OUT, "items.json"))), json.load(open(os.path.join(OUT, "metrics.json"))))


if __name__ == "__main__":
    r = main()
    print(json.dumps(r["overall"], indent=0)[:1500])
