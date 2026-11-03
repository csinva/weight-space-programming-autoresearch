"""Score experiment 2: predicted 'up' token lists against measured promoted tokens, plus target dlogp.
Code baselines use weights only: copy the top-10 of the direct-logit vocabulary projection (both signs, and the better sign as an
oracle), and 'already likely' tokens (highest mean unedited probability, excluding targets)."""
import glob, json, os
import numpy as np
import torch
from scipy.stats import spearmanr
from common import ROOT, load, load_prompts, next_token_logprobs
import exp2_interventions as X
from interventions import FULL, mixer_out, P
from analyze import OUT

BASE = os.path.join(ROOT, os.environ.get("WSP_X2_DIR", "exp2"))
CONDS = os.environ.get("WSP_X2_CONDS", "none,raw,views").split(",")
rng = np.random.default_rng(0)


def write_direction(it):
    if it["kind"] == "neuron":
        return P(f"model.layers.{it['layer']}.mlp.down_proj.weight")[:, it["neuron"]].float()
    if it["kind"] == "gain":
        v = torch.zeros(1024); v[it["dim"]] = 1.0
        return v
    W = P(f"model.layers.{it['layer']}.mlp.down_proj.weight").float()
    return torch.linalg.svd(W, full_matrices=False)[0][:, 0]


def main():
    m, tok = load()
    prm = load_prompts()
    E, gf = m.model.embed_tokens.weight.data, 1 + m.model.norm.weight.data
    U = E * gf
    s2id = {}
    for i in range(E.shape[0]):
        s2id.setdefault(tok.decode([i]), i)
    base = next_token_logprobs([p["text"] for p in prm]).exp().mean(0)
    targets = {p["target_id"] for p in prm}
    likely = [int(i) for i in torch.argsort(base, descending=True) if int(i) not in targets][:10]
    allv = json.load(open(os.path.join(BASE, "interventions.json")))
    rows = []
    for gpath in sorted(glob.glob(os.path.join(BASE, "r*", "ground_truth.json"))):
        rd = os.path.dirname(gpath); r = int(os.path.basename(rd)[1:])
        gt = json.load(open(gpath)); dense = np.load(os.path.join(rd, "dense_effects.npz"))
        for fam in [f for f in X.FAMILIES if f in allv]:
            preds = {c: json.load(open(os.path.join(rd, fam, c, "prediction.json")))["predictions"] for c in CONDS
                     if os.path.exists(os.path.join(rd, fam, c, "prediction.json"))}
            for it in [x for x in allv[fam] if x["round"] == r]:
                d, elig = dense[it["iid"]]
                elig = elig > 0.5
                order = np.argsort(-np.where(elig, d, -np.inf))
                ne = int(elig.sum())
                pct = np.zeros(len(d)); pct[order[:ne]] = 1 - np.arange(ne) / max(ne - 1, 1)
                top50, top10 = set(order[:50].tolist()), set(order[:10].tolist())

                def sc(ids):
                    ids = [i for i in ids if i is not None][:10] + [None] * (10 - len([i for i in ids if i is not None][:10]))
                    return dict(hit50=sum(i in top50 for i in ids if i is not None) / 10, hit10=sum(i in top10 for i in ids if i is not None) / 10,
                                pct=float(np.mean([pct[i] if i is not None else 0.0 for i in ids])))
                row = dict(round=r, family=fam, iid=it["iid"], kl=gt[it["iid"]]["mean_kl"], n_elig=ne)
                v = write_direction(it)
                dla = U.float() @ v
                row["dla_plus"], row["dla_minus"] = sc(torch.topk(dla, 10).indices.tolist()), sc(torch.topk(-dla, 10).indices.tolist())
                row["dla_oracle"] = max(row["dla_plus"], row["dla_minus"], key=lambda z: z["pct"])
                row["likely"] = sc(likely)
                for c, Pd in preds.items():
                    e = Pd.get(it["iid"])
                    if not e:
                        continue
                    ids, mapped = [], 0
                    for u in e["up"]:
                        i = u.get("id") if isinstance(u, dict) else None
                        t = u.get("tok") if isinstance(u, dict) else str(u)
                        if i is not None and 0 <= int(i) < len(d) and tok.decode([int(i)]) == t:
                            ids.append(int(i)); mapped += 1
                        elif t in s2id:
                            ids.append(s2id[t]); mapped += 1
                        else:
                            ids.append(None)
                    row[c] = dict(**sc(ids), mapped=mapped / max(len(e["up"]), 1))
                    yt = [gt[it["iid"]]["per_prompt"][p["pid"]]["dlogp"] for p in prm]
                    yp = [e["dlogp"].get(p["pid"], np.nan) for p in prm]
                    row[c + "_dlogp"] = list(zip(yp, yt))
                rows.append(row)
    keys = CONDS + ["dla_plus", "dla_minus", "dla_oracle", "likely"]

    def agg(sub, k, metric):
        vals = np.array([x[k][metric] for x in sub if k in x])
        if not len(vals):
            return (np.nan,) * 3
        bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(2000)]
        return float(vals.mean()), float(vals.mean() - np.std(bs)), float(vals.mean() + np.std(bs))  # +-1 SEM over edits

    def paired(sub, a, b, metric):
        s = [x for x in sub if a in x and b in x]
        dv = np.array([x[a][metric] - x[b][metric] for x in s])
        bs = [dv[rng.integers(0, len(dv), len(dv))].mean() for _ in range(2000)]
        return float(dv.mean()), float(dv.mean() - np.std(bs)), float(dv.mean() + np.std(bs))  # +-1 SEM over edits

    res = dict(n=len(rows), cells={}, paired={}, dlogp={})
    for subset in ["all", "KL>0.1"]:
        for fam in X.FAMILIES + ["all"]:
            sub = [x for x in rows if (fam == "all" or x["family"] == fam) and (subset == "all" or x["kl"] > 0.1)]
            key = f"{subset}|{fam}"
            res["cells"][key] = dict(n=len(sub), **{k: {mt: agg(sub, k, mt) for mt in ["pct", "hit50", "hit10"]} for k in keys})
            res["paired"][key] = {f"{a}-{b}": {mt: paired(sub, a, b, mt) for mt in ["pct", "hit50"]}
                                  for a, b in [("raw", "none"), ("views", "none"), ("views", "raw"), ("views", "dla_oracle"), ("views", "dla_plus"), ("raw", "likely"), ("none", "likely")] if a in keys and b in keys}
            for c in CONDS:
                pairs = [pq for x in sub if c + "_dlogp" in x for pq in x[c + "_dlogp"] if pq[0] == pq[0]]
                if len(pairs) > 5:
                    yp, yt = np.array(pairs).T
                    res["dlogp"][f"{key}|{c}"] = dict(rho=float(spearmanr(yp, yt)[0]), mae=float(np.mean(np.abs(yp - yt))), mae_zero=float(np.mean(np.abs(yt))))
    json.dump(dict(res=res, rows=rows), open(os.path.join(OUT, f"{os.path.basename(BASE)}_metrics.json"), "w"), indent=1, default=float)
    for key, c in res["cells"].items():
        print(key.ljust(14), f"n={c['n']:3d}", " ".join(f"{k}:{c[k]['pct'][0]:.2f}/{c[k]['hit50'][0]:.2f}" for k in keys))
    for key, c in res["paired"].items():
        print("PAIR", key.ljust(14), " ".join(f"{k}: pct {v['pct'][0]:+.2f} [{v['pct'][1]:+.2f},{v['pct'][2]:+.2f}]" for k, v in c.items()))
    for k, v in res["dlogp"].items():
        print("DLOGP", k, {a: round(b, 3) for a, b in v.items()})
    return res


if __name__ == "__main__":
    main()
