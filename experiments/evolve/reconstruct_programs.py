"""Rebuild every evaluated program of a run from its transcript, so audits do not depend on the snapshots the agent chose to save.

Starting from the template interpretable_transformer.py, the successful Write and Edit calls on the run's interpretable_transformer.py are
replayed in order; each successful `uv run interpretable_transformer.py ...` call saves the current file as reconstructed/<model_shorthand_name>.py
(the last evaluation of a name wins, matching the row in results/overall_results.csv). The replayed final file is checked against the real one.

  uv run evolve/reconstruct_programs.py runs-sonnet55      -> <run>/reconstructed/*.py, runs-sonnet55/reconstruct.json"""
import glob, gzip, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = re.compile(r"""^model_shorthand_name\s*=\s*["'](.+?)["']""", re.M)


def replay(run_dir):
    target = os.path.realpath(os.path.join(run_dir, "interpretable_transformer.py"))
    content = open(os.path.join(HERE, "interpretable_transformer.py")).read()
    out_dir = os.path.join(run_dir, "reconstructed")
    os.makedirs(out_dir, exist_ok=True)
    saved, n_edits, failed_apply = {}, 0, 0
    for f in sorted(glob.glob(os.path.join(run_dir, "agent_logs", "session_*.jsonl*"))):
        pend = {}
        for line in (gzip.open(f, "rt") if f.endswith(".gz") else open(f)):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "assistant":
                for c in d["message"]["content"]:
                    if c.get("type") != "tool_use":
                        continue
                    inp = c["input"]
                    p = inp.get("file_path", "")
                    if c["name"] in ("Write", "Edit") and p and os.path.realpath(p if os.path.isabs(p) else os.path.join(run_dir, p)) == target:
                        pend[c["id"]] = (c["name"], inp)
                    elif c["name"] == "Bash" and inp.get("command", "").strip().startswith("uv run interpretable_transformer.py"):
                        pend[c["id"]] = ("Run", inp)
            if d.get("type") == "user":
                for c in d["message"].get("content", []) or []:
                    if not (isinstance(c, dict) and c.get("type") == "tool_result" and c.get("tool_use_id") in pend):
                        continue
                    kind, inp = pend.pop(c["tool_use_id"])
                    if c.get("is_error"):
                        continue
                    if kind == "Write":
                        content = inp["content"]; n_edits += 1
                    elif kind == "Edit":
                        old, new = inp["old_string"], inp["new_string"]
                        if old not in content:
                            failed_apply += 1
                            continue
                        content = content.replace(old, new) if inp.get("replace_all") else content.replace(old, new, 1)
                        n_edits += 1
                    else:
                        m = NAME.search(content)
                        if m:
                            saved[m.group(1)] = content
    for name, src in saved.items():
        open(os.path.join(out_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".py"), "w").write(src)
    real = open(target).read()
    return dict(programs=len(saved), edits=n_edits, edits_not_applied=failed_apply, final_matches=content == real)


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(HERE), sys.argv[1] if len(sys.argv) > 1 else "runs-opus55")
    res = {}
    for rd in sorted(glob.glob(os.path.join(root, "*-s[0-9]"))):
        res[os.path.basename(rd)] = r = replay(rd)
        print(os.path.basename(rd), r)
    json.dump(res, open(os.path.join(root, "reconstruct.json"), "w"), indent=1)
    print("final file matches in", sum(r["final_matches"] for r in res.values()), "of", len(res), "runs")
