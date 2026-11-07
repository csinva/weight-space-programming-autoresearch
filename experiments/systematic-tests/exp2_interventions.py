"""Experiment 2: amplification edits that visibly change the output, and the question "which tokens gain probability?".
Families: neuron (one MLP neuron's down_proj column x K), mlp (a whole MLP's down_proj x K),
gain (one dimension of the final RMSNorm gain x K). K per family chosen by exp2_pilot.py on components disjoint from the test set."""
import numpy as np
import torch
from interventions import FULL, mixer_out, P, N_LAYERS, D_MLP, D

FAMILIES = ["neuron", "mlp", "gain"]
SEED_INDEX = dict(neuron=0, mlp=2, gain=3)  # per-family seed offsets, unchanged from the original sampling
K = dict(neuron=100.0, mlp=3.0, gain=30.0)  # from exp2_pilot.py: median mean-KL 0.2 to 0.7
N_PER_ROUND, N_ROUNDS = 16, 2
ELIGIBLE_P = 1e-3  # a token counts as "promoted" only if it reaches this probability on some prompt after the edit


def sample(fam, rng, n):
    out = []
    for k in range(n):
        if fam == "neuron":
            l = int(rng.integers(12, N_LAYERS))
            Wd = P(f"model.layers.{l}.mlp.down_proj.weight")
            i = int(torch.argsort(Wd.norm(dim=0), descending=True)[int(rng.integers(0, 50))]) if k % 2 else int(rng.integers(0, D_MLP))
            out.append(dict(kind="neuron", layer=l, neuron=i, salient="large down_proj norm" if k % 2 else "random"))
        elif fam == "mlp":
            out.append(dict(kind="mlp", layer=int(k * N_LAYERS // n + rng.integers(0, max(1, N_LAYERS // n)))))
        else:
            g = 1 + P("model.norm.weight")
            d = int(torch.argsort(g.abs(), descending=True)[int(rng.integers(0, 32))]) if k % 2 else int(rng.integers(0, D))
            out.append(dict(kind="gain", dim=d, salient="large final gain" if k % 2 else "random"))
    return out


def sample_all(seed=1):
    """16 edits per family per round (MLP: all 24 layers, 12 per round), unique, deterministic given the seed."""
    allv = {}
    for fam in FAMILIES:
        rng = np.random.default_rng(seed * 100 + SEED_INDEX[fam])
        n = N_LAYERS if fam == "mlp" else N_PER_ROUND * N_ROUNDS
        if fam == "mlp":
            items = [dict(kind="mlp", layer=int(l)) for l in rng.permutation(N_LAYERS)]
        else:
            items, seen = [], set()
            while len(items) < n:
                for it in sample(fam, rng, 2 * n):
                    key = tuple(sorted((k, v) for k, v in it.items() if k != "salient"))
                    if key not in seen and len(items) < n:
                        seen.add(key); items.append(it)
        per = n // N_ROUNDS
        for j, pi in enumerate(rng.permutation(len(items))):
            it = items[pi]
            it.update(round=j // per, iid=f"X{fam[:2].upper()}{j:02d}", family=fam, op=f"x{K[fam]:g}")
        allv[fam] = sorted(items, key=lambda x: x["iid"])
    return allv


def describe(it):
    k, f = it["kind"], K[it["family"]]
    if k == "neuron":
        return f"MLP neuron {it['neuron']} in layer {it['layer']}: multiply its down_proj column by {f:g}"
    if k == "mlp":
        return f"layer {it['layer']}: multiply the whole MLP output (down_proj) by {f:g}"
    return f"final RMSNorm `model.norm.weight` dimension {it['dim']}: multiply its gain (1+w) by {f:g}"


def to_edits(it, f=None):
    f = K[it["family"]] if f is None else f
    k = it["kind"]
    if k == "neuron":
        return [(f"model.layers.{it['layer']}.mlp.down_proj.weight", (slice(None), it["neuron"]), lambda w: w * f)]
    if k == "mlp":
        return [(f"model.layers.{it['layer']}.mlp.down_proj.weight", None, lambda w: w * f)]
    return [("model.norm.weight", (it["dim"],), lambda w: f * (1 + w) - 1)]


def effects(lp_edit, lp_base):
    """Per-vocab mean change in log-prob over prompts, restricted to tokens that become plausible (p >= ELIGIBLE_P on some prompt)."""
    d = (lp_edit - lp_base).mean(0)
    elig = (lp_edit.exp() >= ELIGIBLE_P).any(0)
    kl = (lp_base.exp() * (lp_base - lp_edit)).sum(-1)
    return d, elig, kl
