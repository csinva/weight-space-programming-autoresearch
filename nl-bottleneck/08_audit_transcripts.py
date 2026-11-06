"""Audit reconstruction transcripts: tool counts, paths touched outside the agent's sandbox, shell commands, refused calls.
  python 08_audit_transcripts.py recon -> audit_transcripts_recon.json"""
import glob, gzip, json, os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
d = sys.argv[1]
SB = "/data18/chandan.singh/wsp-nl/sandbox" + ("/control" if d.endswith("control") else "")
out = {}
for rd in sorted(glob.glob(f"{HERE}/{d}/*-r[0-9]")):
    tag = os.path.basename(rd); sb = f"{SB}/rec-{tag}"
    tools, cmds, refused, outside, pend = collections.Counter(), collections.Counter(), 0, set(), {}
    for l in (gzip.open(f"{rd}/agent_log.jsonl.gz", "rt") if os.path.exists(f"{rd}/agent_log.jsonl.gz") else open(f"{rd}/agent_log.jsonl")):  # logs are gzipped after the runs
        try: x = json.loads(l)
        except json.JSONDecodeError: continue
        if x.get("type") == "assistant":
            for c in x["message"]["content"]:
                if c.get("type") == "tool_use":
                    tools[c["name"]] += 1; pend[c["id"]] = c
                    i = c["input"]
                    p = i.get("file_path") or i.get("path") or ""
                    if p and not os.path.abspath(p if os.path.isabs(p) else f"{sb}/{p}").startswith(sb):
                        outside.add(p)
                    if c["name"] == "Bash":
                        cm = i.get("command", ""); cmds[" ".join(cm.split()[:2])[:60]] += 1
                        outside |= {m for m in re.findall(r"(?<![\w.])/(?:home|data|data18|tmp|usr)[^\s;'\"]*", cm) if not m.startswith(sb) and "/.venv/bin/python" not in m}
        if x.get("type") == "user":
            for c in x["message"]["content"]:
                if isinstance(c, dict) and c.get("type") == "tool_result" and "requires approval" in str(c.get("content")) + "" or (isinstance(x["message"]["content"], list) and any(isinstance(c, dict) and c.get("type") == "tool_result" and "Blocked" in str(c.get("content"))[:80] for c in x["message"]["content"])):
                    refused += 1
    out[tag] = dict(tools=dict(tools), refused=refused, outside=sorted(outside), top_cmds=cmds.most_common(4))
json.dump(out, open(f"{HERE}/audit_transcripts_{d}.json", "w"), indent=1)
print("tools:", sum((collections.Counter(v["tools"]) for v in out.values()), collections.Counter()))
print("runs with outside paths:", {k: v["outside"] for k, v in out.items() if v["outside"]})
print("refused calls total:", sum(v["refused"] for v in out.values()))
