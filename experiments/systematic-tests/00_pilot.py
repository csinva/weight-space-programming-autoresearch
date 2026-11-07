"""Harness checks and magnitude pilot (never shown to the predictor):
1) edits restore the model bit-exactly, 2) lm_head is tied to embed_tokens, 3) effect size of adding a value
to a single random down_proj / embedding entry, to pick SCALAR_ADD so single-weight effects are not all ~0."""
import numpy as np, torch
from common import load, load_prompts, next_token_logprobs, edited

m, tok = load()
prm = load_prompts()
texts, tgt = [p["text"] for p in prm], torch.tensor([p["target_id"] for p in prm])
base = next_token_logprobs(texts)
print("tied:", m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr())
rng = np.random.default_rng(12345)
for val in [0.5, 2.0, 4.0, 10.0]:
    effs = []
    for _ in range(6):
        l, r, c = int(rng.integers(0, 24)), int(rng.integers(0, 1024)), int(rng.integers(0, 3584))
        with edited([(f"model.layers.{l}.mlp.down_proj.weight", (r, c), lambda w: w + val * rng.choice([-1, 1]))]):
            lp = next_token_logprobs(texts)
        effs.append((lp - base)[torch.arange(8), tgt].abs().numpy())
    effs = np.concatenate(effs)
    print(f"down_proj add {val}: median |dlogp| {np.median(effs):.4f}  90% {np.quantile(effs, .9):.3f}  max {effs.max():.3f}")
for val in [0.5, 2.0, 4.0]:
    effs = []
    for p in prm[:4]:
        d = int(rng.integers(0, 1024))
        with edited([("model.embed_tokens.weight", (p["target_id"], d), lambda w: w + val * rng.choice([-1, 1]))]):
            lp = next_token_logprobs(texts)
        effs.append((lp - base)[torch.arange(8), tgt].abs().numpy())
    effs = np.concatenate(effs)
    print(f"embed add {val}: median |dlogp| {np.median(effs):.4f}  90% {np.quantile(effs, .9):.3f}  max {effs.max():.3f}")
print("restored exactly:", torch.equal(next_token_logprobs(texts), base))
print("embed row std", float(m.model.embed_tokens.weight.std()), "down_proj std L5", float(m.model.layers[5].mlp.down_proj.weight.std()))
