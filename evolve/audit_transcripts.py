"""Audit the agent transcripts of each run: tools used, shell commands, refused calls, and file paths outside the run folder.

  uv run evolve/audit_transcripts.py runs-opus55      -> runs-opus55/audit_transcripts.json"""
import collections, glob, gzip, json, os, sys


def main(root):
    out = {}
    for rd in sorted(glob.glob(os.path.join(root, "*", ""))):
        run = os.path.basename(rd.rstrip("/"))
        real_rd = os.path.realpath(rd)
        shared_src = os.path.realpath(os.path.join(rd, "src"))
        tools, cmds, refused, outside, pend = collections.Counter(), collections.Counter(), 0, [], {}
        for f in sorted(glob.glob(os.path.join(rd, "agent_logs", "*.jsonl*"))):
            for line in (gzip.open(f, "rt") if f.endswith(".gz") else open(f)):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("type") == "assistant":
                    for c in d["message"]["content"]:
                        if c.get("type") != "tool_use":
                            continue
                        tools[c["name"]] += 1
                        pend[c["id"]] = c
                        if c["name"] == "Bash":
                            cmds[" ".join(c["input"].get("command", "").split()[:2])] += 1
                        p = c["input"].get("file_path") or c["input"].get("path") or ""
                        if p:
                            rp = os.path.realpath(p if os.path.isabs(p) else os.path.join(rd, p))
                            if not (rp.startswith(real_rd) or rp.startswith(shared_src)):
                                outside.append(p)
                if d.get("type") == "user":
                    for c in d["message"].get("content", []) or []:
                        if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("is_error"):
                            msg = str(c.get("content"))
                            if "requires approval" in msg or "was blocked" in msg or "outside" in msg.lower():
                                refused += 1
        out[run] = dict(tools=dict(tools), bash_commands=dict(cmds), refused_calls=refused, paths_outside_run=sorted(set(outside)))
        print(run, dict(tools), "refused:", refused, "outside:", sorted(set(outside))[:3])
    json.dump(out, open(os.path.join(root, "audit_transcripts.json"), "w"), indent=1)


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), sys.argv[1] if len(sys.argv) > 1 else "runs-opus55"))
