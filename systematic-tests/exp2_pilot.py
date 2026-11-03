"""Pick K per family: median mean-KL over prompts and how many tokens become promoted, on components not used in the test set."""
import numpy as np, torch
from common import load_prompts, next_token_logprobs, edited
import exp2_interventions as X

prm = load_prompts(); texts = [p["text"] for p in prm]
base = next_token_logprobs(texts)
rng = np.random.default_rng(999)
grid = dict(neuron=[10, 30, 100], head=[4, 8, 16], mlp=[2, 3, 5], gain=[10, 30, 100])
for fam, ks in grid.items():
    its = [dict(family=fam, **x) for x in X.sample(fam, rng, 6)]
    for f in ks:
        kls, nprom = [], []
        for it in its:
            with edited(X.to_edits(it, f)):
                lp = next_token_logprobs(texts)
            d, elig, kl = X.effects(lp, base)
            kls.append(float(kl.mean())); nprom.append(int(((d > 1.0) & elig).sum()))
        print(f"{fam} x{f}: mean-KL median {np.median(kls):.3f} range [{min(kls):.3f}, {max(kls):.3f}]  tokens with +1 nat & eligible: median {np.median(nprom):.0f}")
