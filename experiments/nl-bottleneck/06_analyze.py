"""Step 4: compare original vs reconstructed models (accuracy on visible / hidden / extra sets, error overlap, agreement). Writes results.json.

Error comparison on a set of n examples, with e_o, e_r the 0/1 error indicators of the original and a reconstruction:
  phi = Pearson correlation of e_o and e_r (nan if either has no errors); jaccard = |both wrong| / |either wrong|
  p_r|o_wrong, p_r|o_right = error rate of the reconstruction on examples the original gets wrong / right (equal if errors are independent)
  agree = fraction of identical outputs; same_wrong = fraction of both-wrong examples with the identical wrong output
  kappa = Cohen's kappa of the two models' outputs (agreement beyond what their output marginals predict by chance)
"""
import glob, json, math, os, re
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = ("sentiment-sst2", "paraphrase-mrpc", "nli-snli", "trec-coarse")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def compare(a, b):
    """a, b: per-set dicts from eval_program (same prompts)."""
    assert a["prompts"] == b["prompts"]
    n = a["n"]
    ea = [not c for c in a["correct"]]; eb = [not c for c in b["correct"]]
    both = sum(x and y for x, y in zip(ea, eb)); oa, ob = sum(ea), sum(eb)
    either = oa + ob - both
    agree = sum(x == y for x, y in zip(a["preds"], b["preds"])) / n
    sw = [x == y for x, y, p, q in zip(a["preds"], b["preds"], ea, eb) if p and q]
    ca, cb = Counter(a["preds"]), Counter(b["preds"])
    pe = sum(ca[k] * cb[k] for k in ca) / n ** 2
    d = math.sqrt(oa * (n - oa) * ob * (n - ob))
    return dict(n=n, err_a=oa, err_b=ob, both_wrong=both, jaccard=both / either if either else None,
                phi=(n * both - oa * ob) / d if d else None,
                p_b_given_a_wrong=both / oa if oa else None, p_b_given_a_right=(ob - both) / (n - oa) if n > oa else None,
                agree=agree, same_wrong=sum(sw) / len(sw) if sw else None, kappa=(agree - pe) / (1 - pe) if pe < 1 else None,
                expected_both_if_independent=oa * ob / n)


out = {}
for op in sorted(glob.glob(f"{HERE}/preds/orig/*.json")):
    task = os.path.basename(op)[:-5]
    o = load(op)
    ent = dict(orig={k: o[k]["acc"] for k in ("visible", "hidden", "extra") if k in o}, orig_params=o["n_params"], recon={}, control={}, recon_pair={})
    for kind, d in (("recon", "recon"), ("control", "recon_control")):
        for p in sorted(glob.glob(f"{HERE}/preds/{d}/{task}-r*.json")):
            tag = os.path.basename(p)[:-5]; r = load(p)
            e = dict(acc={k: r[k]["acc"] for k in ("visible", "hidden", "extra") if k in r}, n_params=r["n_params"],
                     vs_orig={k: compare(o[k], r[k]) for k in ("visible", "hidden", "extra") if k in o})
            ent[kind][tag] = e
            if task in DATASET:  # per-class view on the hidden set
                cls = sorted(set(o["hidden"]["targets"]))
                def rec(m):
                    return {c: sum(p == t == c for p, t in zip(m["preds"], m["targets"])) / max(1, sum(t == c for t in m["targets"])) for c in cls}
                e["pred_dist"] = {c: sum(p == c for p in r["hidden"]["preds"]) / r["hidden"]["n"] for c in cls}
                e["recall"] = rec(r["hidden"])
    r1, r2 = (load(f"{HERE}/preds/recon/{task}-r{k}.json") for k in (1, 2))
    if r1 and r2:
        ent["recon_pair"] = {k: compare(r1[k], r2[k]) for k in ("visible", "hidden", "extra") if k in r1}
    if task in DATASET:
        cls = sorted(set(o["hidden"]["targets"]))
        ent["orig_pred_dist"] = {c: sum(p == c for p in o["hidden"]["preds"]) / o["hidden"]["n"] for c in cls}
        ent["orig_recall"] = {c: sum(p == t == c for p, t in zip(o["hidden"]["preds"], o["hidden"]["targets"])) / sum(t == c for t in o["hidden"]["targets"]) for c in cls}
        ent["majority"] = max(Counter(o["hidden"]["targets"]).values()) / o["hidden"]["n"]
    out[task] = ent
json.dump(out, open(f"{HERE}/results.json", "w"), indent=1)
print(len(out), "tasks")
