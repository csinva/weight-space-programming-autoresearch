"""Experiment 3: can Opus 5.5 *simulate the calculation* of an edit's effect when the effect is an exact low-dimensional formula?

Edits (fresh, disjoint from experiment 2):
  gain : final RMSNorm gain of dim d multiplied by 30      -> dlogit_t = 29 (1+g_d) n_d E[t,d]   for every token t
  emb  : add +-4 to E[t,d] of a likely token t (tied)      -> dlogit_t = +-4 (1+g_d) n_d          (output side; input side if t is in the prompt)
where n = RMS-normalised final hidden state (one unknown number n_d per prompt). The predictor must estimate n_d from the unedited
next-token distribution and the embedding column, then compute new logits / softmax by hand (Read tool only, no code).
Conditions: none (no numbers), raw (exp1/exp2-style weight file), rawplus (raw + every number the calculation needs), compute (rawplus +
step-by-step calculation instructions). Code baselines: the same procedure run exactly; oracle that uses the true n_d (needs activations).

  uv run python experiments/systematic-tests/exp3.py build|predict|measure|analyze"""
import json, os, re, shutil, subprocess, sys, tempfile, time, glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from common import ROOT, load, load_prompts, next_token_logprobs, edited
import exp2_interventions as X
from run_round import audit, MODEL_ID, INSTRUCTION

BASE = os.path.join(ROOT, "exp3")
CONDS = ["none", "raw", "rawplus", "compute"]
FAMS = ["gain", "emb"]
KG, DE, NUP, TOPN = 30.0, 4.0, 10, 50


def interventions():
    path = os.path.join(BASE, "interventions.json")
    if os.path.exists(path):
        return json.load(open(path))
    os.makedirs(BASE, exist_ok=True)
    used = {it["dim"] for d in ["exp2", "exp2_rep"] for it in json.load(open(os.path.join(ROOT, d, "interventions.json")))["gain"]}
    g = 1 + X.P("model.norm.weight")
    rng = np.random.default_rng(31)
    big = [int(i) for i in torch.argsort(g.abs(), descending=True) if int(i) not in used][:48]
    rest = [int(i) for i in rng.permutation(1024) if int(i) not in used and int(i) not in big]
    dims = [(int(d), "large final gain") for d in rng.choice(big, 12, replace=False)] + [(d, "random") for d in rest[:12]]
    gain = [dict(kind="gain", family="gain", dim=d, salient=s, iid=f"G{j:02d}") for j, (d, s) in enumerate([dims[i] for i in rng.permutation(24)])]
    prm = load_prompts()
    emb = []
    for j in range(24):
        p = prm[j % 8]
        t = p["target_id"] if j < 8 else p["top10"][1 + (j // 8)]["id"]  # targets, then runner-ups
        d = int(rng.choice(big[:24])) if j % 2 else int(rng.integers(0, 1024))
        emb.append(dict(kind="emb", family="emb", token=int(t), dim=d, value=float(DE * rng.choice([-1, 1])), salient="large final gain" if j % 2 else "random", iid=f"E{j:02d}"))
    allv = dict(gain=gain, emb=emb)
    json.dump(allv, open(path, "w"), indent=1)
    return allv


def to_edits(it):
    if it["kind"] == "gain":
        return [("model.norm.weight", (it["dim"],), lambda w: KG * (1 + w) - 1)]
    return [("model.embed_tokens.weight", (it["token"], it["dim"]), lambda w: w + it["value"])]


def describe(it, tok):
    if it["kind"] == "gain":
        return f"final RMSNorm `model.norm.weight` dimension {it['dim']}: multiply its gain (1+w) by {KG:g}"
    return f"embedding entry E[{it['token']} ({tok.decode([it['token']])!r}), {it['dim']}]: add {it['value']:+g} (tied: changes that token's output logit and its input embedding)"


def fmt_tokens(tok, ids):
    return ", ".join(f"{tok.decode([int(i)])!r}({int(i)})" for i in ids)


def edit_file(it, cond, ctx):
    m, tok, E, gf, prm, top = ctx
    d = it["dim"]
    col = E[:, d].float()
    L = [f"# {it['iid']}: {describe(it, tok)}",
         f"final-norm stored w[{d}] = {float(gf[d] - 1):+.4f}, so gain 1+w = {float(gf[d]):+.4f}",
         f"embedding column {d} over the whole vocabulary: mean {float(col.mean()):+.5f}, std {float(col.std()):.5f}"]
    if it["kind"] == "emb":
        L.append(f"E[{it['token']}, {d}] = {float(E[it['token'], d]):+.5f} (row std {float(E[it['token']].std()):.5f}); "
                 f"token {tok.decode([it['token']])!r} appears in the input of: " + (", ".join(p["pid"] for p in prm if it["token"] in p["input_ids"]) or "no prompt"))
    if cond == "raw":
        L.append(f"## raw embedding column {d} for prompt targets: " + ", ".join(f"{p['target']!r} {float(E[p['target_id'], d]):+.4f}" for p in prm))
        return "\n".join(L)
    hi, lo = torch.topk(col, 30).indices, torch.topk(-col, 30).indices
    L.append(f"## the 30 most positive entries of column {d}: " + ", ".join(f"{tok.decode([int(i)])!r}({int(i)}) {float(col[i]):+.4f}" for i in hi))
    L.append(f"## the 30 most negative entries of column {d}: " + ", ".join(f"{tok.decode([int(i)])!r}({int(i)}) {float(col[i]):+.4f}" for i in lo))
    L.append(f"## for each prompt, its {TOPN} most likely unedited next tokens: token(id) probability, E[token,{d}]")
    for j, p in enumerate(prm):
        ids, ps = top[j]
        L.append(f"{p['pid']}: " + "; ".join(f"{tok.decode([int(i)])!r}({int(i)}) p={float(q):.4f} E={float(E[i, d]):+.4f}" for i, q in zip(ids, ps)))
    return "\n".join(L)


COMPUTE = """## How to compute (do this arithmetic yourself, step by step, in your reasoning; you cannot run code)
The final layer is logits_t = sum_d E[t,d] * (1+w_d) * n_d, where n = h / rms(h) is the RMS-normalised final hidden state of that prompt
(one vector per prompt; across its 1024 dims the mean of n_d^2 is 1, and a few dims can be much larger).
1. **The edit's exact effect on logits.** Gain x30 at dim d: dlogit_t = 29 * (1+w_d) * n_d * E[t,d] for every token t.
   Embedding entry E[a,d] += c: dlogit_a = c * (1+w_d) * n_d for token a only (plus an input-side effect if a is in the prompt, usually small).
   So the only unknown per prompt is n_d.
2. **Estimate n_d for each prompt from the unedited distribution.** High-probability tokens have embeddings aligned with (1+w) * n. A practical estimate:
   s = sum over the listed top tokens of p_t * (E[t,d] - mean_d) / std_d. The sign of s estimates the sign of n_d. Its size (relative to how peaked the
   distribution is) suggests |n_d|; typical |n_d| is around 0.3 to 3. Write the number you pick for each prompt.
3. **Compute new logits.** For each prompt take the listed top tokens and the tokens at the extremes of column d. New logit = old logit + dlogit, where
   old logit = log p (up to a constant). Tokens outside the lists keep their old logits, which are below the listed ones.
4. **Renormalise.** New log p_t = new logit_t - logsumexp(new logits). The listed top-50 hold most of the old mass; the rest of the vocabulary keeps
   its old total mass (1 - listed mass) unless the column extremes were pushed up. Compute dlogp of the target and which tokens gained the most.
5. **Aggregate `up`:** average the dlogp of each candidate over the 8 prompts, keep tokens that reach p >= 0.001 on some prompt, and list the 10 largest.
Show the main numbers in your reasoning; the final answer only needs the JSON.
"""

TASK = """# Task: predict how a single-dimension edit changes Qwen3.5-0.8B's next-token probabilities

You are given {n} edits of one kind ("{fam}"). Each edit is applied on its own to the unedited model, which then reads each of 8 fixed prompts.
For every edit predict:
1. `up`: the {nup} tokens whose log-probability increases the most on average over the 8 prompts, counting only tokens that reach p >= 0.001 on
   some prompt after the edit. Give each as {{"tok": "<exact token string>", "id": <token id or null>}}.
2. `dlogp` for each prompt: change in natural-log probability of that prompt's unedited top-1 token (target), edited minus unedited.

{cond_text}
You may only read files (no code execution is available).{required}
Files: `README.md`, `test_interventions.md`{wl}.

{arch}

## Prompts and the unedited model's next-token distribution
{prompts}
{compute}
## Output format (your final message)
A single fenced ```json block, nothing after it:
{{"edits": [{{"iid": "G00", "up": [{{"tok": " Paris", "id": 11751}}, ... {nup} entries], "dlogp": {{"P0": -0.3, ..., "P7": 0.1}}}}, ... one per edit],
 "notes": {{"G00": "one sentence: the n_d values you used per prompt and the evidence", ...}}}}
"""
CT = dict(none="**Condition: no weights.** Predict from your knowledge of transformers, the edit descriptions, the prompts and the unedited distribution.",
          raw="**Condition: raw weights.** `weights/` has the relevant raw numbers for each edit (the gain, and the target tokens' embedding entries at that dimension).",
          rawplus="**Condition: raw weights and the numbers needed.** `weights/` has, for each edit, the gain, the embedding column's extremes, and each prompt's top-50 unedited tokens with their probabilities and their embedding entry at the edited dimension.",
          compute="**Condition: raw weights and the numbers needed, with a calculation recipe.** `weights/` has, for each edit, the gain, the embedding column's extremes, and each prompt's top-50 unedited tokens with their probabilities and embedding entries. Follow the recipe below and do the arithmetic yourself.")


def ctx():
    import bundles as B
    m, tok = load()
    prm = load_prompts()
    E, gf = m.model.embed_tokens.weight.data, 1 + m.model.norm.weight.data
    lp = next_token_logprobs([p["text"] for p in prm])
    top = [(lp[j].topk(TOPN).indices, lp[j].topk(TOPN).values.exp()) for j in range(len(prm))]
    return (m, tok, E, gf, prm, top), B


def build():
    allv = interventions()
    c, B = ctx()
    tok, prm = c[1], c[4]
    for fam in FAMS:
        for cond in CONDS:
            out = os.path.join(BASE, fam, cond, "bundle")
            os.makedirs(out, exist_ok=True)
            readme = TASK.format(n=len(allv[fam]), fam=fam, nup=NUP, cond_text=CT[cond], required=("\n" + B.REQUIRED) if cond != "none" else "",
                                 wl=", `weights/` (one file per edit)" if cond != "none" else "", arch=B.ARCH.format(revision="revision 2fc0636471"),
                                 prompts=B.prompts_md(prm), compute=COMPUTE if cond == "compute" else "")
            open(os.path.join(out, "README.md"), "w").write(readme)
            open(os.path.join(out, "test_interventions.md"), "w").write("# Edits to predict (each applied alone)\n" + "\n".join(
                f"- **{it['iid']}**: {describe(it, tok)}" + (f" [file weights/{it['iid']}.txt]" if cond != "none" else "") for it in allv[fam]))
            if cond != "none":
                os.makedirs(os.path.join(out, "weights"), exist_ok=True)
                for it in allv[fam]:
                    open(os.path.join(out, "weights", f"{it['iid']}.txt"), "w").write(edit_file(it, cond, c))
            print("built", fam, cond, flush=True)


def predict_one(fam, cond, items, max_tries=6):
    import exp2_run as R2
    out = os.path.join(BASE, fam, cond)
    if os.path.exists(os.path.join(out, "prediction.json")):
        return f"skip {fam} {cond}"
    prm = load_prompts()
    for attempt in range(max_tries):
        sandbox = tempfile.mkdtemp(prefix=f"x3_{fam}_{cond}_", dir=os.environ.get("WSP_SANDBOX", tempfile.gettempdir()))
        shutil.copytree(os.path.join(out, "bundle"), sandbox, dirs_exist_ok=True)
        p = subprocess.run(["claude", "-p", INSTRUCTION, "--model", MODEL_ID, "--restricted", "--tools", "Read", "--strict-mcp-config",
                            "--output-format", "stream-json", "--verbose"], cwd=sandbox, capture_output=True, text=True, timeout=7200)
        try:
            aud, text = audit(p.stdout.splitlines(), sandbox)
            preds, notes, missing = R2.parse(text, items, prm)
        except Exception as e:
            print(f"x3 {fam} {cond} attempt {attempt} failed: {type(e).__name__} {str(e)[:150]}", flush=True)
            shutil.rmtree(sandbox, ignore_errors=True)
            time.sleep(600 if "limit" in (p.stdout + p.stderr).lower() else 30)
            continue
        unread = sorted({f"weights/{it['iid']}.txt" for it in items} - set(aud["reads"])) if cond != "none" else []
        bad = unread or missing > len(items) or aud["reads_outside_sandbox"] or aud["init"]["model"] != MODEL_ID or aud["tools_used"] not in ([], ["Read"])
        if bad:
            print(f"x3 {fam} {cond} attempt {attempt}: unread {unread} missing {missing} outside {aud['reads_outside_sandbox']} tools {aud['tools_used']}, retrying", flush=True)
            shutil.rmtree(sandbox, ignore_errors=True)
            continue
        aud.update(attempt=attempt, missing=missing, model_flag=MODEL_ID)
        json.dump(dict(predictions=preds, notes=notes), open(os.path.join(out, "prediction.json"), "w"), indent=1)
        json.dump(aud, open(os.path.join(out, "audit.json"), "w"), indent=1)
        open(os.path.join(out, "final_message.md"), "w").write(text)
        shutil.rmtree(sandbox, ignore_errors=True)
        return f"done x3 {fam} {cond}: {aud['tool_calls']} reads, ${aud['cost_usd']:.2f}, {aud['duration_s']:.0f}s, out tokens {(aud['usage'] or {}).get('output_tokens')}"
    return f"FAILED x3 {fam} {cond}"


def predict(workers=4):
    allv = interventions()
    jobs = [(fam, cond, allv[fam]) for fam in FAMS for cond in CONDS]
    with ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(lambda j: predict_one(*j), jobs):
            print(msg, flush=True)


def final_normed_states(texts):
    """Oracle only (activations): RMS-normalised final hidden state at the last position, per prompt."""
    m, tok = load()
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        h = m.model(input_ids=ids).last_hidden_state  # already includes the final norm: E-space input is h * (1+w) / rms
        out.append(h[0, -1].float() / (1 + m.model.norm.weight.data.float()))
    return torch.stack(out)


def measure():
    preds = glob.glob(os.path.join(BASE, "*", "*", "prediction.json"))
    assert len(preds) == len(FAMS) * len(CONDS), f"only {len(preds)} prediction files"
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *preds], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(["git", "ls-files", "--", *preds], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert not dirty and len(tracked) == len(preds), "commit the predictions before measuring (pre-registration)"
    _, tok = load()
    allv, prm = interventions(), load_prompts()
    texts = [p["text"] for p in prm]
    base = next_token_logprobs(texts)
    gt, dense = {}, {}
    for fam in FAMS:
        for it in allv[fam]:
            with edited(to_edits(it)):
                lp = next_token_logprobs(texts)
            d, elig, kl = X.effects(lp, base)
            per = {p["pid"]: dict(dlogp=float(lp[j, p["target_id"]] - base[j, p["target_id"]]), flip=int(lp[j].argmax()) != p["target_id"],
                                  new_top1=tok.decode([int(lp[j].argmax())]), kl=float(kl[j])) for j, p in enumerate(prm)}
            gt[it["iid"]] = dict(family=fam, mean_kl=float(kl.mean()), per_prompt=per)
            dense[it["iid"]] = np.stack([d.numpy().astype(np.float32), elig.numpy().astype(np.float32)])
            print(it["iid"], f"KL {float(kl.mean()):.3f}", flush=True)
    json.dump(gt, open(os.path.join(BASE, "ground_truth.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(BASE, "dense_effects.npz"), **dense)


_BASE = {}


def base_logprobs():
    if "lp" not in _BASE:
        _BASE["lp"] = next_token_logprobs([p["text"] for p in load_prompts()])
    return _BASE["lp"]


def simulate(it, n_est, ctxv):
    """Code version of the recipe: given n_d per prompt, exact new logits on the full vocabulary (no other approximation)."""
    m, tok, E, gf, prm, top = ctxv
    base = base_logprobs()
    d = it["dim"]
    new = base.clone()
    for j in range(len(prm)):
        if it["kind"] == "gain":
            new[j] = base[j] + (KG - 1) * gf[d] * n_est[j] * E[:, d]
        else:
            new[j, it["token"]] += it["value"] * gf[d] * n_est[j]
    new = torch.log_softmax(new, -1)
    return X.effects(new, base)


def analyze():
    from scipy.stats import spearmanr
    c, _ = ctx()
    m, tok, E, gf, prm, top = c
    allv = interventions()
    gt = json.load(open(os.path.join(BASE, "ground_truth.json")))
    dense = np.load(os.path.join(BASE, "dense_effects.npz"))
    s2id = {}
    for i in range(E.shape[0]):
        s2id.setdefault(tok.decode([i]), i)
    n_true = final_normed_states([p["text"] for p in prm])
    rng = np.random.default_rng(0)
    rows = []
    for fam in FAMS:
        P = {cd: json.load(open(os.path.join(BASE, fam, cd, "prediction.json")))["predictions"] for cd in CONDS}
        for it in allv[fam]:
            dd, el = dense[it["iid"]]; el = el > 0.5
            order = np.argsort(-np.where(el, dd, -np.inf)); ne = int(el.sum())
            pct = np.zeros(len(dd)); pct[order[:ne]] = 1 - np.arange(ne) / max(ne - 1, 1)
            top50 = set(order[:50].tolist())
            sc = lambda ids: dict(pct=float(np.mean([pct[i] if i is not None else 0.0 for i in (ids + [None] * 10)[:10]])),
                                  hit50=sum(i in top50 for i in ids[:10] if i is not None) / 10)
            yt = np.array([gt[it["iid"]]["per_prompt"][p["pid"]]["dlogp"] for p in prm])
            row = dict(family=fam, iid=it["iid"], kl=gt[it["iid"]]["mean_kl"], true=yt.tolist())
            col = E[:, it["dim"]].float(); mu, sd = col.mean(), col.std()
            n_proc = []
            for j in range(len(prm)):
                ids, ps = top[j]
                s = float((ps * (E[ids, it["dim"]].float() - mu) / sd).sum())
                n_proc.append(float(np.sign(s)) * 1.0)
            for name, n_est in [("code_recipe", n_proc), ("oracle_n", n_true[:, it["dim"]].tolist())]:
                d2, e2, _ = simulate(it, n_est, c)
                ids = [int(i) for i in torch.topk(torch.where(e2, d2, torch.tensor(-1e9)), 10).indices]
                bl = base_logprobs()
                yp = []
                for j, p in enumerate(prm):
                    nl = bl[j].clone()
                    if it["kind"] == "gain":
                        nl = nl + (KG - 1) * gf[it["dim"]] * n_est[j] * E[:, it["dim"]]
                    else:
                        nl[it["token"]] += it["value"] * gf[it["dim"]] * n_est[j]
                    yp.append(float(torch.log_softmax(nl, -1)[p["target_id"]] - bl[j, p["target_id"]]))
                row[name] = dict(**sc(ids), dlogp=yp)
            for cd in CONDS:
                e = P[cd].get(it["iid"])
                ids = []
                for u in e["up"]:
                    i, t = (u.get("id"), u.get("tok")) if isinstance(u, dict) else (None, str(u))
                    ids.append(int(i) if i is not None and 0 <= int(i) < len(dd) and tok.decode([int(i)]) == t else s2id.get(t))
                row[cd] = dict(**sc(ids), dlogp=[e["dlogp"].get(p["pid"], np.nan) for p in prm])
            rows.append(row)
    keys = CONDS + ["code_recipe", "oracle_n"]
    res = {}
    for fam in FAMS + ["all"]:
        sub = [r for r in rows if fam == "all" or r["family"] == fam]
        cell = {}
        for k in keys:
            yp = np.concatenate([r[k]["dlogp"] for r in sub]); yt = np.concatenate([r["true"] for r in sub]); ok = ~np.isnan(yp)
            big = np.abs(yt[ok]) > 0.1
            cell[k] = dict(pct=float(np.mean([r[k]["pct"] for r in sub])), hit50=float(np.mean([r[k]["hit50"] for r in sub])),
                           rho=float(spearmanr(yp[ok], yt[ok])[0]), mae=float(np.mean(np.abs(yp[ok] - yt[ok]))), mae_zero=float(np.mean(np.abs(yt))),
                           sign_big=float(np.mean(np.sign(yp[ok][big]) == np.sign(yt[ok][big]))) if big.any() else np.nan)
        pairs = {}
        for a, b in [("compute", "none"), ("compute", "rawplus"), ("rawplus", "raw"), ("raw", "none"), ("compute", "code_recipe")]:
            for metric in ["pct", "rho"]:
                bs = []
                for _ in range(2000):
                    s = [sub[i] for i in rng.integers(0, len(sub), len(sub))]
                    if metric == "pct":
                        bs.append(np.mean([r[a]["pct"] - r[b]["pct"] for r in s]))
                    else:
                        f = lambda k: spearmanr(np.concatenate([r[k]["dlogp"] for r in s]), np.concatenate([r["true"] for r in s]), nan_policy="omit")[0]
                        bs.append(f(a) - f(b))
                est = cell[a][metric] - cell[b][metric]
                pairs[f"{a}-{b}|{metric}"] = (float(est), float(est - np.nanstd(bs)), float(est + np.nanstd(bs)))  # +-1 bootstrap SE
        res[fam] = dict(n=len(sub), cells=cell, paired=pairs)
        print(fam, len(sub), " ".join(f"{k}: pct {cell[k]['pct']:.2f} hit {cell[k]['hit50']:.2f} rho {cell[k]['rho']:+.2f} sign {cell[k]['sign_big']:.2f} mae {cell[k]['mae']:.2f}" for k in keys))
        print("   zero-MAE", round(cell['none']['mae_zero'], 2), " ".join(f"{k} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]" for k, v in pairs.items()))
    from analyze import OUT
    json.dump(dict(res=res, rows=rows), open(os.path.join(OUT, "exp3_metrics.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    dict(build=build, predict=predict, measure=measure, analyze=analyze)[sys.argv[1]]()
