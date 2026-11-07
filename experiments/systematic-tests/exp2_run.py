"""Experiment 2 (pre-registered like run_round.py): amplification edits; predict the 10 tokens that gain the most probability
and the target-token dlogp on each prompt.

  uv run python systematic-tests/exp2_run.py build|predict|measure --round R      (commit predictions before measure)"""
import argparse, glob, json, os, re, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from common import ROOT, load_prompts
import exp2_interventions as X
from run_round import audit, CONDS, MODEL_ID, INSTRUCTION

BASE = os.path.join(ROOT, os.environ.get("WSP_X2_DIR", "exp2"))
FAMS = os.environ.get("WSP_X2_FAMS", ",".join(X.FAMILIES)).split(",")
CONDS = os.environ.get("WSP_X2_CONDS", ",".join(CONDS)).split(",")
NUP = 10


def rdir(r):
    return os.path.join(BASE, f"r{r}")


def interventions():
    path = os.path.join(BASE, "interventions.json")
    os.makedirs(BASE, exist_ok=True)
    if not os.path.exists(path):
        if os.path.basename(BASE) == "exp2_rep":  # replication: 32 fresh final-gain dims, disjoint from exp2, all zero-shot (round 0)
            used = {it["dim"] for it in json.load(open(os.path.join(ROOT, "exp2", "interventions.json")))["gain"]}
            g = 1 + X.P("model.norm.weight")
            import torch
            big = [int(i) for i in torch.argsort(g.abs(), descending=True) if int(i) not in used][:64]
            rng = np.random.default_rng(7)
            rand = [int(i) for i in rng.permutation(1024) if int(i) not in used and int(i) not in big[:64]]
            dims = [(d, "large final gain") for d in rng.choice(big, 16, replace=False)] + [(d, "random") for d in rand[:16]]
            items = [dict(kind="gain", dim=int(d), salient=s, round=0, iid=f"RGA{j:02d}", family="gain", op=f"x{X.K['gain']:g}")
                     for j, (d, s) in enumerate([dims[i] for i in rng.permutation(32)])]
            json.dump(dict(gain=items), open(path, "w"), indent=1)
        else:
            json.dump(X.sample_all(seed=1), open(path, "w"), indent=1)
    return json.load(open(path))


TASK = """# Task: predict which tokens an amplification edit makes more likely in Qwen3.5-0.8B

You are given {n} weight edits of one kind ("{family}"). Each edit is applied **on its own** to the unedited model, then the model reads
each of 8 fixed prompts. Edits are amplifications strong enough to change the output noticeably (average KL to the unedited
next-token distribution is typically 0.1 to 2 nats).

Predict, for every edit:
1. `up`: the {nup} tokens whose next-token log-probability **increases the most on average over the 8 prompts**, counting only tokens that
   reach probability >= 0.001 on at least one prompt after the edit. Give each as {{"tok": "<exact token string, with its leading space if any>", "id": <token id or null>}}.
   Order does not matter. Tokens are scored by where they fall in the measured ranking.
2. `dlogp` for each prompt: change in natural-log probability of that prompt's unedited top-1 token (target), edited minus unedited.

{condition_text}
{required}
You may only read files (no code execution is available).
Files: `README.md`, `test_interventions.md`{feedback_line}{weights_line}.

{arch}

## Prompts and the unedited model's next-token distribution
{prompts}

## Output format (your final message)
A single fenced ```json block, nothing after it:
{{"edits": [{{"iid": "XX00", "up": [{{"tok": " Paris", "id": 11751}}, ... {nup} entries], "dlogp": {{"P0": -0.3, ..., "P7": 0.1}}}}, ... one per edit],
 "notes": {{"XX00": "one short sentence on the evidence you used", ...}}}}
"""


def build(r):
    import bundles as B
    allv = interventions()
    prm = load_prompts()
    fb = {}
    for q in range(r):
        fb.update(json.load(open(os.path.join(rdir(q), "ground_truth.json"))))

    class Ctx2(B.Ctx):
        def t(self, i):
            return f"{self.tok.decode([int(i)])!r}({int(i)})"
    c = Ctx2()
    for fam in FAMS:
        test = [it for it in allv[fam] if it["round"] == r]
        train = [it for it in allv[fam] if it["round"] < r]
        for cond in CONDS:
            out = os.path.join(rdir(r), fam, cond, "bundle")
            os.makedirs(out, exist_ok=True)
            cf = ", and the measured effects of earlier edits of the same kind in `feedback.md`" if train else ""
            if cond != "none" and train:
                cf += " (their weight files are in `weights/` too)"
            readme = TASK.format(n=len(test), family=fam, nup=NUP, condition_text=B.COND[cond].format(fb=cf),
                                 required=B.REQUIRED if cond != "none" else "",
                                 feedback_line=", `feedback.md`" if train else "", weights_line=", `weights/` (one file per edit, plus shared_*.txt)" if cond != "none" else "",
                                 arch=B.ARCH.format(revision="revision 2fc0636471"), prompts=B.prompts_md(prm))
            if cond == "views":
                readme = readme.replace("Token lists in views show token strings with their ids", "")
                readme += "\nIn the views, tokens are written as 'string'(id).\n"
            open(os.path.join(out, "README.md"), "w").write(readme)
            open(os.path.join(out, "test_interventions.md"), "w").write("# Edits to predict (each applied alone)\n" + "\n".join(
                f"- **{it['iid']}**: {X.describe(it)}" + (f" [file weights/{it['iid']}.txt]" if cond != "none" else "") for it in test))
            if train:
                L = ["# Measured effects of earlier edits (top tokens by mean dlogp over the 8 prompts, among tokens reaching p >= 0.001; then target dlogp per prompt)"]
                for it in train:
                    g = fb[it["iid"]]
                    L.append(f"\n## {it['iid']}: {X.describe(it)}" + (f" [file weights/{it['iid']}.txt]" if cond != "none" else ""))
                    L.append(f"mean KL {g['mean_kl']:.3f}; top promoted: " + ", ".join(f"{x['tok']!r}({x['id']}) {x['d']:+.2f}" for x in g["top_up"][:15]))
                    L.append("target dlogp: " + " | ".join(f"{pid} {v['dlogp']:+.3f}{' new top-1 ' + repr(v['new_top1']) if v['flip'] else ''}" for pid, v in g["per_prompt"].items()))
                open(os.path.join(out, "feedback.md"), "w").write("\n".join(L))
            if cond != "none":
                wd = os.path.join(out, "weights"); os.makedirs(wd, exist_ok=True)
                for name, txt in B.shared_files(c).items():
                    open(os.path.join(wd, name), "w").write(txt)
                for it in test + train:
                    e1 = dict(it)
                    if it["kind"] == "mlp":
                        e1["kind"] = "layer"
                    if it["kind"] == "gain":
                        e1.update(kind="norm_gain", param="model.norm.weight", index=[it["dim"]])
                    open(os.path.join(wd, f"{it['iid']}.txt"), "w").write(f"# {it['iid']}: {X.describe(it)}\n\n" + B.component_files(c, e1, views=(cond == "views")))
        print("built", r, fam)


def parse(text, items, prm):
    obj = json.loads(re.findall(r"```json\s*(.*?)```", text, re.S)[-1])
    want = {it["iid"] for it in items}
    got = {e["iid"]: dict(up=e["up"][:NUP], dlogp={k: max(-20.0, min(3.0, float(v))) for k, v in e["dlogp"].items()}) for e in obj["edits"] if e["iid"] in want}
    missing = len(want - set(got)) + sum(len(prm) - len(v["dlogp"]) for v in got.values())
    return got, obj.get("notes", {}), missing


def predict_one(r, fam, cond, items, prm, max_tries=6):
    out = os.path.join(rdir(r), fam, cond)
    if os.path.exists(os.path.join(out, "prediction.json")):
        return f"skip {fam} {cond}"
    for attempt in range(max_tries):
        sandbox = tempfile.mkdtemp(prefix=f"x2r{r}_{fam}_{cond}_", dir=os.environ.get("WSP_SANDBOX", tempfile.gettempdir()))
        shutil.copytree(os.path.join(out, "bundle"), sandbox, dirs_exist_ok=True)
        t0 = time.time()
        p = subprocess.run(["claude", "-p", INSTRUCTION, "--model", MODEL_ID, "--restricted", "--tools", "Read", "--strict-mcp-config",
                            "--output-format", "stream-json", "--verbose"], cwd=sandbox, capture_output=True, text=True, timeout=5400)
        try:
            aud, text = audit(p.stdout.splitlines(), sandbox)
            preds, notes, missing = parse(text, items, prm)
        except Exception as e:
            print(f"x2 r{r} {fam} {cond} attempt {attempt} failed: {type(e).__name__} {str(e)[:150]}", flush=True)
            shutil.rmtree(sandbox, ignore_errors=True)
            time.sleep(600 if "limit" in (p.stdout + p.stderr).lower() else 30)
            continue
        unread = sorted({f"weights/{it['iid']}.txt" for it in items} - set(aud["reads"])) if cond != "none" else []
        if unread or missing > len(items):
            print(f"x2 r{r} {fam} {cond} attempt {attempt}: unread {unread} missing {missing}, retrying", flush=True)
            shutil.rmtree(sandbox, ignore_errors=True)
            continue
        if aud["reads_outside_sandbox"] or aud["init"]["model"] != MODEL_ID:
            print(f"{fam} {cond} attempt {attempt}: outside reads {aud['reads_outside_sandbox']} model {aud['init']['model']}, retrying", flush=True)
            shutil.rmtree(sandbox, ignore_errors=True)
            continue
        assert aud["tools_used"] in ([], ["Read"]), aud["tools_used"]
        aud.update(attempt=attempt, missing=missing, wall_s=time.time() - t0, model_flag=MODEL_ID)
        json.dump(dict(predictions=preds, notes=notes), open(os.path.join(out, "prediction.json"), "w"), indent=1)
        json.dump(aud, open(os.path.join(out, "audit.json"), "w"), indent=1)
        open(os.path.join(out, "final_message.md"), "w").write(text)
        shutil.rmtree(sandbox, ignore_errors=True)
        return f"done x2 r{r} {fam} {cond}: {aud['tool_calls']} reads, outside={len(aud['reads_outside_sandbox'])}, ${aud['cost_usd']:.2f}, {aud['duration_s']:.0f}s"
    return f"FAILED x2 r{r} {fam} {cond}"


def predict(r, workers=5):
    allv, prm = interventions(), load_prompts()
    jobs = [(fam, cond, [it for it in allv[fam] if it["round"] == r]) for fam in FAMS for cond in CONDS]
    with ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(lambda j: predict_one(r, *j, prm), jobs):
            print(msg, flush=True)


def measure(r):
    from common import next_token_logprobs, edited, load
    preds = glob.glob(os.path.join(rdir(r), "*", "*", "prediction.json"))
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
        for it in [x for x in allv[fam] if x["round"] == r]:
            with edited(X.to_edits(it)):
                lp = next_token_logprobs(texts)
            d, elig, kl = X.effects(lp, base)
            de = d.clone(); de[~elig] = -1e9
            top = de.topk(50).indices
            per = {}
            for j, p in enumerate(prm):
                t, t1 = p["target_id"], int(lp[j].argmax())
                per[p["pid"]] = dict(dlogp=float(lp[j, t] - base[j, t]), flip=t1 != t, new_top1=tok.decode([t1]), kl=float(kl[j]))
            gt[it["iid"]] = dict(family=fam, mean_kl=float(kl.mean()), n_eligible_up=int(((d > 0) & elig).sum()),
                                 top_up=[dict(id=int(i), tok=tok.decode([int(i)]), d=float(d[i])) for i in top if de[i] > -1e8], per_prompt=per)
            dense[it["iid"]] = np.stack([d.numpy().astype(np.float32), elig.numpy().astype(np.float32)])
            print(it["iid"], f"KL {float(kl.mean()):.3f}", [x["tok"] for x in gt[it["iid"]]["top_up"][:6]], flush=True)
    json.dump(gt, open(os.path.join(rdir(r), "ground_truth.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(rdir(r), "dense_effects.npz"), **dense)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "predict", "measure"])
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    dict(build=build, predict=predict, measure=measure)[a.stage](a.round)
