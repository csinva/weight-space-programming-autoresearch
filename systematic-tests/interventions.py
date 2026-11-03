"""Intervention families at 5 granularities, their seeded sampling, and their translation into weight edits.

Every intervention is a pure weight edit (no hooks on activations). Skipping a sublayer = zeroing its output
projection, which removes its residual-stream contribution exactly."""
import numpy as np
import torch
from common import load, load_prompts

N_LAYERS, D, D_MLP = 24, 1024, 3584
FULL = [l for l in range(N_LAYERS) if l % 4 == 3]  # full (softmax) attention layers; the rest are Gated DeltaNet
N_PER_ROUND, N_ROUNDS = 12, 3
FAMILIES = ["scalar", "neuron", "layer", "token"]
SEED_INDEX = dict(scalar=0, neuron=1, layer=3, token=4)  # per-family seed offsets, unchanged from the original sampling
SCALAR_ADD = 4.0  # chosen by 00_pilot.py (pilot entries are disjoint from the test items)


def mixer_out(l):
    return f"model.layers.{l}.self_attn.o_proj.weight" if l in FULL else f"model.layers.{l}.linear_attn.out_proj.weight"


def P(name):
    m, _ = load()
    return dict(m.named_parameters())[name].data


def _sample_scalar(rng, n):
    out = []
    kinds = ["norm_gain", "max_weight", "add_value"] * (n // 3 + 1)
    for k in range(n):
        kind = kinds[k]
        if kind == "norm_gain":
            l = int(rng.integers(-1, N_LAYERS))
            name = "model.norm.weight" if l < 0 else f"model.layers.{l}.{rng.choice(['input_layernorm', 'post_attention_layernorm'])}.weight"
            g = 1 + P(name)
            d = int(torch.argsort(g.abs(), descending=True)[int(rng.integers(0, 8))]) if rng.random() < 0.5 else int(rng.integers(0, D))
            op = str(rng.choice(["zero_gain", "gain_x10"]))
            out.append(dict(kind=kind, param=name, index=[d], op=op, salient=None))
        elif kind == "max_weight":
            l = int(rng.integers(0, N_LAYERS))
            name = str(rng.choice([f"model.layers.{l}.mlp.down_proj.weight", mixer_out(l), f"model.layers.{l}.mlp.gate_proj.weight"]))
            W = P(name)
            if rng.random() < 0.5:
                flat = int(W.abs().argmax())
                r, c = divmod(flat, W.shape[1]); how = "largest |w| in matrix"
            else:
                r, c = int(rng.integers(0, W.shape[0])), int(rng.integers(0, W.shape[1])); how = "random entry"
            out.append(dict(kind=kind, param=name, index=[r, c], op="set_zero", salient=how))
        else:
            if rng.random() < 0.5:
                prm = load_prompts()
                t = int(prm[int(rng.integers(0, len(prm)))]["target_id"])
                name, idx, how = "model.embed_tokens.weight", [t, int(rng.integers(0, D))], "target-token embedding entry"
            else:
                l = int(rng.integers(0, N_LAYERS))
                name = f"model.layers.{l}.mlp.down_proj.weight"
                idx, how = [int(rng.integers(0, D)), int(rng.integers(0, D_MLP))], "random down_proj entry"
            out.append(dict(kind=kind, param=name, index=idx, op="add", value=float(SCALAR_ADD * rng.choice([-1, 1])), salient=how))
    return out


def _sample_neuron(rng, n):
    prm = load_prompts()
    m, _ = load()
    E, g = m.model.embed_tokens.weight.data, 1 + m.model.norm.weight.data
    out = []
    for k in range(n):
        l = int(rng.integers(0, N_LAYERS))
        Wd = P(f"model.layers.{l}.mlp.down_proj.weight")
        how = ["random", "largest down_proj column norm", "largest direct effect on a target logit"][k % 3]
        if how == "random":
            i = int(rng.integers(0, D_MLP))
        elif how.startswith("largest down"):
            i = int(torch.argsort(Wd.norm(dim=0), descending=True)[int(rng.integers(0, 10))])
        else:
            t = prm[int(rng.integers(0, len(prm)))]["target_id"]
            dla = E[t] @ (g[:, None] * Wd)
            i = int(torch.argsort(dla.abs(), descending=True)[int(rng.integers(0, 10))])
        out.append(dict(kind="neuron", layer=l, neuron=i, op=str(rng.choice(["zero", "scale_x5"])), salient=how))
    return out


def _sample_layer(rng, n):
    combos = [(l, k) for l in range(N_LAYERS) for k in ["skip_mixer", "skip_mlp", "skip_block"]]
    idx = rng.permutation(len(combos))[:n]
    return [dict(kind="layer", layer=int(combos[i][0]), op=combos[i][1], salient=None) for i in idx]


def _sample_token(rng, n):
    prm = load_prompts()
    m, tok = load()
    out = []
    for k in range(n):
        p = prm[int(rng.integers(0, len(prm)))]
        op = ["swap", "zero", "copy"][k % 3]
        if op == "zero":
            a = p["target_id"] if rng.random() < 0.5 else int(rng.choice(p["input_ids"]))
            out.append(dict(kind="token", op="zero", a=int(a), a_tok=tok.decode([int(a)]), salient=None))
            continue
        a = p["target_id"]
        if rng.random() < 0.5:
            b, how = p["top10"][int(rng.integers(1, 4))]["id"], "baseline runner-up of a prompt"
        else:
            q = prm[int(rng.integers(0, len(prm)))]
            b, how = (q["target_id"] if q["target_id"] != a else q["top10"][1]["id"]), "another prompt's target"
        out.append(dict(kind="token", op=op, a=int(a), b=int(b), a_tok=tok.decode([int(a)]), b_tok=tok.decode([int(b)]), salient=how))
    return out


SAMPLERS = dict(scalar=_sample_scalar, neuron=_sample_neuron, layer=_sample_layer, token=_sample_token)


def sample_all(seed=0):
    """36 interventions per family, split into 3 rounds of 12. Deterministic given the seed."""
    allv = {}
    for fam in FAMILIES:
        rng = np.random.default_rng(seed * 100 + SEED_INDEX[fam])
        items = SAMPLERS[fam](rng, N_PER_ROUND * N_ROUNDS)
        perm = rng.permutation(len(items))
        for j, pi in enumerate(perm):
            it = items[pi]
            it["round"] = j // N_PER_ROUND
            it["iid"] = f"{fam[:2].upper()}{j:02d}"
            it["family"] = fam
        allv[fam] = sorted(items, key=lambda x: x["iid"])
    return allv


def describe(it):
    """Plain-language description shown to the predictor."""
    from common import ALT
    if ALT and it.get("salient"):  # salience was defined on the top-1 tokens; say so when another token is scored
        it = dict(it, salient=it["salient"].replace("target-token embedding entry", "embedding entry of a prompt's top-1 token")
                  .replace("largest direct effect on a target logit", "largest direct effect on a top-1 token's logit"))
    k = it["kind"]
    if k == "norm_gain":
        op = "set its gain (1+w) to 0 (w := -1)" if it["op"] == "zero_gain" else "multiply its gain (1+w) by 10"
        return f"RMSNorm `{it['param']}` dimension {it['index'][0]}: {op}"
    if k == "max_weight":
        return f"`{it['param']}`[{it['index'][0]}, {it['index'][1]}] ({it['salient']}): set to 0"
    if k == "add_value":
        return f"`{it['param']}`[{it['index'][0]}, {it['index'][1]}] ({it['salient']}): add {it['value']:+g}"
    if k == "neuron":
        op = "zero the neuron (its down_proj column)" if it["op"] == "zero" else "multiply its down_proj column by 5"
        return f"MLP neuron {it['neuron']} in layer {it['layer']}: {op}"
    if k == "layer":
        what = dict(skip_mixer="skip the token mixer (attention / DeltaNet) sublayer", skip_mlp="skip the MLP sublayer",
                    skip_block="skip the whole block (mixer and MLP)")[it["op"]]
        return f"layer {it['layer']}: {what}"
    if k == "token":
        if it["op"] == "zero":
            return f"embedding row of token {it['a']} ({it['a_tok']!r}) set to 0 (tied: input and output)"
        if it["op"] == "swap":
            return f"swap embedding rows of token {it['a']} ({it['a_tok']!r}) and {it['b']} ({it['b_tok']!r}) (tied: input and output)"
        return f"overwrite embedding row of token {it['a']} ({it['a_tok']!r}) with a copy of token {it['b']} ({it['b_tok']!r})'s row (tied)"
    raise ValueError(k)


def to_edits(it):
    k = it["kind"]
    if k == "norm_gain":
        d = it["index"][0]
        fn = (lambda w: torch.full_like(w, -1.0)) if it["op"] == "zero_gain" else (lambda w: 10 * (1 + w) - 1)
        return [(it["param"], (d,), fn)]
    if k == "max_weight":
        return [(it["param"], tuple(it["index"]), lambda w: torch.zeros_like(w))]
    if k == "add_value":
        return [(it["param"], tuple(it["index"]), lambda w: w + it["value"])]
    if k == "neuron":
        f = 0.0 if it["op"] == "zero" else 5.0
        return [(f"model.layers.{it['layer']}.mlp.down_proj.weight", (slice(None), it["neuron"]), lambda w: w * f)]
    if k == "layer":
        e = []
        if it["op"] in ("skip_mixer", "skip_block"):
            e.append((mixer_out(it["layer"]), None, torch.zeros_like))
        if it["op"] in ("skip_mlp", "skip_block"):
            e.append((f"model.layers.{it['layer']}.mlp.down_proj.weight", None, torch.zeros_like))
        return e
    if k == "token":
        E = "model.embed_tokens.weight"
        a = it["a"]
        if it["op"] == "zero":
            return [(E, (a,), torch.zeros_like)]
        Eb = P(E)[it["b"]].clone()
        if it["op"] == "copy":
            return [(E, (a,), lambda w: Eb.clone())]
        Ea = P(E)[a].clone()
        return [(E, (a,), lambda w: Eb.clone()), (E, (it["b"],), lambda w: Ea.clone())]
    raise ValueError(k)


def extend(allv, seed=1, n_new=None, first_round=3, per_round=None, families=("scalar", "neuron", "layer")):
    """Add n_new fresh edits per family (no duplicates of existing ones), split into rounds of per_round starting at first_round.
    Used for the larger zero-shot sample (no feedback)."""
    def key(it):
        return tuple(sorted((k, str(v)) for k, v in it.items() if k in ("kind", "param", "index", "op", "value", "layer", "neuron")))
    n_new = n_new or dict(scalar=96, neuron=96, layer=36)  # layer: only 72 (layer, skip) combinations exist
    per_round = per_round or dict(scalar=16, neuron=16, layer=12)
    for fam in families:
        seen = {key(it) for it in allv[fam]}
        rng = np.random.default_rng(seed * 1000 + SEED_INDEX[fam])
        new = []
        while len(new) < n_new[fam]:
            for it in SAMPLERS[fam](rng, 96):
                if key(it) not in seen and len(new) < n_new[fam]:
                    seen.add(key(it)); new.append(it)
        start = len(allv[fam])
        for j, pi in enumerate(rng.permutation(len(new))):
            it = new[pi]
            it.update(round=first_round + j // per_round[fam], iid=f"{fam[:2].upper()}{start + j:03d}", family=fam)
            allv[fam].append(it)
    return allv

