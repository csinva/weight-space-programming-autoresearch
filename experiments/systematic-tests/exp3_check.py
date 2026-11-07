"""Harness check (not shown to the predictor): with the true n_d, the formula must reproduce the edited model exactly. Uses a dim outside the test set."""
import json, torch
from common import load_prompts, next_token_logprobs, edited
import exp3
allv = exp3.interventions()
used = {it["dim"] for f in allv.values() for it in f}
d = next(i for i in range(1024) if i not in used)
c, _ = exp3.ctx()
prm = c[4]
n = exp3.final_normed_states([p["text"] for p in prm])
for it in [dict(kind="gain", dim=d), dict(kind="emb", dim=d, token=prm[0]["target_id"], value=4.0)]:
    with edited(exp3.to_edits(it)):
        real = next_token_logprobs([p["text"] for p in prm])
    sim, _, _ = exp3.simulate(it, n[:, d].tolist(), c)
    base = next_token_logprobs([p["text"] for p in prm])
    print(it["kind"], "dim", d, "max |real mean-dlogp - simulated| =", float(((real - base).mean(0) - sim).abs().max()))
