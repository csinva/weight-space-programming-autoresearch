"""Fix the 8 evaluation prompts and record the unedited model's next-token distribution (data/prompts.json)."""
import json, os
from common import ROOT, load, next_token_logprobs

TEXTS = [
    "Germany: Berlin\nItaly: Rome\nFrance:",
    "one, two, three, four,",
    "def add(a, b):\n    return a",
    "Monday, Tuesday, Wednesday,",
    "The Eiffel Tower is located in the city of",
    "Water is made of hydrogen and",
    "Romeo and Juliet was written by William",
    "The keys to the cabinet",
]

m, tok = load()
lp = next_token_logprobs(TEXTS)
prompts = []
for i, (t, l) in enumerate(zip(TEXTS, lp)):
    v, idx = l.topk(10)
    ids = tok(t).input_ids
    prompts.append(dict(
        pid=f"P{i}", text=t, input_ids=ids, input_tokens=[tok.decode([j]) for j in ids],
        target_id=int(idx[0]), target=tok.decode([int(idx[0])]), target_logp=float(v[0]),
        top10=[dict(id=int(j), token=tok.decode([int(j)]), p=round(float(x.exp()), 4)) for x, j in zip(v, idx)],
    ))
    print(prompts[-1]["pid"], repr(t), prompts[-1]["target"], round(float(v[0].exp()), 3))
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
with open(os.path.join(ROOT, "data", "prompts.json"), "w") as f:
    json.dump(prompts, f, indent=1)
