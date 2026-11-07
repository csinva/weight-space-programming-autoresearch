"""Per-run token usage, cost, turns and the agent's full final message, from the stream-json logs of the reconstruction and control sessions
-> tokens.json. (The summarizer calls only saved cost and turns, not tokens.)"""
import glob, gzip, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
for d in ("recon", "recon_control"):
    for rd in sorted(glob.glob(f"{HERE}/{d}/*-r[0-9]")):
        for l in gzip.open(f"{rd}/agent_log.jsonl.gz", "rt"):
            try:
                x = json.loads(l)
            except json.JSONDecodeError:
                continue
            if x.get("type") == "result":
                u = x["usage"]
                out[f"{d}/{os.path.basename(rd)}"] = dict(input=u["input_tokens"], cache_write=u["cache_creation_input_tokens"], cache_read=u["cache_read_input_tokens"],
                                                         output=u["output_tokens"], thinking=u.get("output_tokens_details", {}).get("thinking_tokens"),
                                                         cost=x["total_cost_usd"], turns=x["num_turns"], seconds=round(x["duration_ms"] / 1000), final_text=x.get("result", ""))
json.dump(out, open(f"{HERE}/tokens.json", "w"), indent=1)
tot = {k: sum(v[k] or 0 for v in out.values()) for k in ("input", "cache_write", "cache_read", "output", "thinking", "cost")}
print(len(out), tot)
