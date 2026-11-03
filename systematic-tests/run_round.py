"""Pre-registered predict-then-measure loop for one round.

  uv run python systematic-tests/run_round.py build   --round R   # sample edits (seeded), write predictor bundles
  uv run python systematic-tests/run_round.py predict --round R   # headless Claude (Opus 5.5), Read tool only, one call per family x condition
  git add/commit the predictions                                   # measure refuses to run on uncommitted predictions
  uv run python systematic-tests/run_round.py measure --round R   # apply each edit, record the true next-token changes

Bundles for round R include measured effects of rounds < R (feedback), so rounds 0, 1, 2 test learning from 0, 12, 24 examples."""
import argparse, glob, json, os, re, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, load_prompts

CONDS = os.environ.get("WSP_CONDS", "none,raw,views").split(",")
ZEROSHOT = os.environ.get("WSP_ZEROSHOT") == "1"  # build without feedback from earlier rounds
MODEL_ID = os.environ.get("WSP_MODEL", "claude-opus-5-5")
TAG = os.environ.get("WSP_TAG", "")  # e.g. "fable": predictions go to rounds_fable/ (same bundles as rounds/)
INSTRUCTION = ("Read README.md in the current directory and follow it exactly. Read test_interventions.md and any other files there "
               "that you find useful (they are the only material you have). End with the JSON block described in README.md.")


VARIANT = None  # few-shot variant name (e.g. "k4"): same test items as round R, only K revealed examples


def rdir(r, variant=None):
    v = variant if variant is not None else VARIANT
    from common import ALT
    return os.path.join(ROOT, "rounds_fewshot", f"r{r}_{v}") if v else os.path.join(ROOT, "rounds_alt" if ALT else "rounds", f"r{r}")


def pdir(r):
    """Where predictions go: next to the bundle, or in a parallel tree for another predictor model (WSP_TAG)."""
    d = rdir(r)
    return d if not TAG else d.replace(os.path.join(ROOT, "rounds"), os.path.join(ROOT, f"rounds_{TAG}"), 1)


def fewshot_ids(allv, fam, r, k, seed=0):
    """k earlier examples per family, covering distinct edit types first (seeded)."""
    import numpy as np
    pool = [it for it in allv[fam] if it["round"] < r]
    rng = np.random.default_rng(seed)
    pool = [pool[i] for i in rng.permutation(len(pool))]
    picked, seen = [], set()
    for it in pool:
        key = (it["kind"], it["op"])
        if key not in seen and len(picked) < k:
            picked.append(it["iid"]); seen.add(key)
    for it in pool:
        if it["iid"] not in picked and len(picked) < k:
            picked.append(it["iid"])
    return picked


def interventions():
    path = os.path.join(ROOT, "data", "interventions.json")
    if not os.path.exists(path):
        from interventions import sample_all
        with open(path, "w") as f:
            json.dump(sample_all(seed=0), f, indent=1)
    with open(path) as f:
        return json.load(f)


def feedback_upto(r):
    fb = {}
    for q in range(r):
        with open(os.path.join(rdir(q, variant=""), "ground_truth.json")) as f:
            for iid, res in json.load(f).items():
                fb[iid] = res["per_prompt"]
    return fb


def build(r, families):
    from bundles import build as build_one
    allv = interventions()
    fb = {} if ZEROSHOT else feedback_upto(r)
    for fam in families:
        shots = [] if ZEROSHOT else (fewshot_ids(allv, fam, r, int(VARIANT[1:])) if VARIANT else None)
        for cond in CONDS:
            build_one(r, fam, cond, allv[fam], fb, os.path.join(rdir(r), fam, cond, "bundle"), shots=shots)
            print("built", r, fam, cond)


def audit(stream_lines, sandbox):
    tools, reads, outside, result, pending, failed = [], [], [], None, {}, []
    for line in stream_lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "system" and d.get("subtype") == "init":
            init = dict(tools=d.get("tools"), model=d.get("model"))
        if d.get("type") == "assistant":
            for c in d["message"]["content"]:
                if c.get("type") == "tool_use":
                    tools.append(c["name"])
                    p = c["input"].get("file_path", "")
                    pending[c.get("id")] = p
                    rp, sb = os.path.realpath(p), os.path.realpath(sandbox)
                    inside = rp == sb or rp.startswith(sb + os.sep)  # the sandbox directory itself counts as inside
                    shared = os.environ.get("WSP_SHARED_READ")  # read-only directory with all weights ("Read all weights" condition)
                    if shared and (rp == os.path.realpath(shared) or rp.startswith(os.path.realpath(shared) + os.sep)):
                        inside = True
                    reads.append(os.path.relpath(os.path.realpath(p), os.path.realpath(sandbox)) if inside else p)
                    if not inside:
                        outside.append(p)
        if d.get("type") == "user":  # tool results: a refused or failed read does not count as a read
            for c in d.get("message", {}).get("content", []) or []:
                if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("is_error"):
                    failed.append(pending.get(c.get("tool_use_id"), ""))
        if d.get("type") == "result":
            result = d
    shared = os.environ.get("WSP_SHARED_READ", "")
    ok_shared = [p for p in pending.values() if shared and os.path.realpath(p).startswith(os.path.realpath(shared) + os.sep) and p not in failed]
    return dict(init=init, tool_calls=len(tools), tools_used=sorted(set(tools)), reads=reads, reads_outside_sandbox=outside, failed_reads=failed,
                shared_reads=[os.path.basename(p) for p in ok_shared],
                cost_usd=result.get("total_cost_usd"), duration_s=(result.get("duration_ms") or 0) / 1000, num_turns=result.get("num_turns"),
                usage=result.get("usage"), is_error=result.get("is_error")), (result or {}).get("result", "")


def parse(text, items, prm):
    blocks = re.findall(r"```json\s*(.*?)```", text, re.S)
    obj = json.loads(blocks[-1])
    want = {(it["iid"], p["pid"]) for it in items for p in prm}
    got = {}
    for x in obj["predictions"]:
        key = (x["iid"], x["pid"])
        if key in want:
            got[f"{key[0]}|{key[1]}"] = dict(dlogp=max(-20.0, min(3.0, float(x["dlogp"]))), p_flip=min(1.0, max(0.0, float(x["p_flip"]))))
    return got, obj.get("notes", {}), len(want) - len(got)


def predict_one(r, fam, cond, items, prm, max_tries=8):
    out, src = os.path.join(pdir(r), fam, cond), os.path.join(rdir(r), fam, cond, "bundle")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "prediction.json")):
        return f"skip {fam} {cond}"
    sandbox_root = os.environ.get("WSP_SANDBOX", tempfile.gettempdir())
    for attempt in range(max_tries):
        sandbox = tempfile.mkdtemp(prefix=f"r{r}_{fam}_{cond}_", dir=sandbox_root)
        shutil.copytree(src, sandbox, dirs_exist_ok=True)
        t0 = time.time()
        extra = ["--add-dir", os.environ["WSP_SHARED_READ"]] if os.environ.get("WSP_SHARED_READ") else []  # read-only access to all weights
        p = subprocess.run(["claude", "-p", INSTRUCTION, "--model", MODEL_ID, "--restricted", "--tools", "Read", "--strict-mcp-config",
                            "--output-format", "stream-json", "--verbose"] + extra, cwd=sandbox, capture_output=True, text=True, timeout=5400)
        lines = p.stdout.splitlines()
        try:
            aud, text = audit(lines, sandbox)
            preds, notes, missing = parse(text, items, prm)
        except Exception as e:  # rate limit, truncated output, malformed JSON: wait and retry from scratch
            print(f"r{r} {fam} {cond} attempt {attempt} failed: {type(e).__name__} {str(e)[:200]} | {p.stdout[-300:]!r} {p.stderr[-300:]!r}", flush=True)
            shutil.rmtree(sandbox, ignore_errors=True)
            time.sleep(600 if "limit" in (p.stdout + p.stderr).lower() else 30)
            continue
        unread = sorted({f"weights/{it['iid']}.txt" for it in items} - set(aud["reads"])) if not cond.startswith("none") else []
        if cond.endswith("_all"):
            if "INDEX.md" not in aud["shared_reads"]:
                unread.append("INDEX.md")
            n_tensor = sum(1 for x in aud["shared_reads"] if x != "INDEX.md")
            if n_tensor < len(items):
                unread.append(f"only {n_tensor} tensor reads in the full-weights directory (need {len(items)})")
        if unread:
            print(f"r{r} {fam} {cond} attempt {attempt}: did not read {unread}, retrying", flush=True)
            shutil.rmtree(sandbox, ignore_errors=True)
            continue
        if missing > len(items) * len(prm) // 10:
            print(f"r{r} {fam} {cond} attempt {attempt}: {missing} missing predictions, retrying", flush=True)
            shutil.rmtree(sandbox, ignore_errors=True)
            continue
        aud.update(attempt=attempt, missing=missing, wall_s=time.time() - t0, model_flag=MODEL_ID, instruction=INSTRUCTION)
        if aud["reads_outside_sandbox"] or aud["init"]["model"] != MODEL_ID:
            print(f"{fam} {cond} attempt {attempt}: outside reads {aud['reads_outside_sandbox']} model {aud['init']['model']}, retrying", flush=True)
            shutil.rmtree(sandbox, ignore_errors=True)
            continue
        assert aud["tools_used"] in ([], ["Read"]), aud["tools_used"]
        with open(os.path.join(out, "prediction.json"), "w") as f:
            json.dump(dict(predictions=preds, notes=notes), f, indent=1)
        with open(os.path.join(out, "audit.json"), "w") as f:
            json.dump(aud, f, indent=1)
        with open(os.path.join(out, "final_message.md"), "w") as f:
            f.write(text)
        shutil.rmtree(sandbox, ignore_errors=True)
        return f"done r{r} {fam} {cond}: {aud['tool_calls']} reads, outside={len(aud['reads_outside_sandbox'])}, ${aud['cost_usd']}, {aud['duration_s']:.0f}s, missing={missing}"
    return f"FAILED r{r} {fam} {cond}"


def predict(r, families, workers):
    allv, prm = interventions(), load_prompts()
    jobs = [(fam, cond, [it for it in allv[fam] if it["round"] == r]) for fam in families for cond in CONDS]
    with ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(lambda j: predict_one(r, *j, prm), jobs):
            print(msg, flush=True)


def measure(r, families):
    import torch
    from common import next_token_logprobs, edited, load
    from interventions import to_edits
    assert VARIANT is None, "measure the base round; few-shot variants share its test items"
    preds = glob.glob(os.path.join(rdir(r), "*", "*", "prediction.json"))
    assert len(preds) == len(families) * len(CONDS), f"only {len(preds)} prediction files"
    preds += glob.glob(os.path.join(ROOT, "rounds_fewshot", f"r{r}_*", "*", "*", "prediction.json"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *preds], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(["git", "ls-files", "--", *preds], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert not dirty and len(tracked) == len(preds), "commit the predictions before measuring (pre-registration)"
    _, tok = load()
    allv, prm = interventions(), load_prompts()
    texts = [p["text"] for p in prm]
    base = next_token_logprobs(texts)
    gt = {}
    for fam in families:
        for it in [x for x in allv[fam] if x["round"] == r]:
            with edited(to_edits(it)):
                lp = next_token_logprobs(texts)
            per = {}
            for j, p in enumerate(prm):
                t, top = p["target_id"], int(lp[j].argmax())
                kl = float((base[j].exp() * (base[j] - lp[j])).sum())
                per[p["pid"]] = dict(dlogp=float(lp[j, t] - base[j, t]), flip=top != t, new_top1=tok.decode([top]), kl=kl)
            gt[it["iid"]] = dict(family=fam, per_prompt=per)
            print(it["iid"], " ".join(f"{v['dlogp']:+.2f}" for v in per.values()), flush=True)
    with open(os.path.join(rdir(r), "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=1)


if __name__ == "__main__":
    from interventions import FAMILIES
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "predict", "measure"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--variant", default=None, help="few-shot variant kK: round R's test items with K revealed examples")
    a = ap.parse_args()
    VARIANT = a.variant
    dict(build=build, predict=predict, measure=measure)[a.stage](a.round, a.families, *([a.workers] if a.stage == "predict" else []))
