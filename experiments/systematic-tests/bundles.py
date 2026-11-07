"""Build what each predictor sees. One bundle per (round, family, condition):
  none  : README (task, architecture, prompts, baseline next-token distribution) + test list + past feedback
  raw   : + weights/raw/*.txt  raw weight numbers of the edited component (+ simple stats: norms, std, max)
  views : + weights/views/*.txt  raw files plus weights-only derived views (vocab projections, OV maps, SVD, ranks)
No activations of any (edited or unedited) forward pass are ever given, only the unedited next-token distribution."""
import json, math, os
import torch
from common import ROOT, ALT, load, load_prompts
from interventions import FULL, describe, mixer_out, P

TOPK = 12


def fmt_vec(v, prec=4, per_line=16, mark=None):
    v = v.float().tolist()
    lines = []
    for s in range(0, len(v), per_line):
        cells = []
        for j, x in enumerate(v[s:s + per_line]):
            c = f"{x:+.{prec}f}"
            cells.append(f"[{c}]" if mark is not None and s + j == mark else c)
        lines.append(f"{s:5d}: " + " ".join(cells))
    return "\n".join(lines)


def stats(v):
    v = v.float()
    return f"n={v.numel()} mean={v.mean():+.5f} std={v.std():.5f} min={v.min():+.4f} max={v.max():+.4f} L2={v.norm():.4f}"


class Ctx:
    def __init__(self):
        self.m, self.tok = load()
        self.prm = load_prompts()
        self.E = self.m.model.embed_tokens.weight.data
        self.gf = 1 + self.m.model.norm.weight.data
        self.U = self.E * self.gf  # direct logit of a residual direction v is U @ v (up to the final RMS scale)
        self.targets = sorted({p["target_id"] for p in self.prm})

    def t(self, i):
        return repr(self.tok.decode([int(i)]))

    def vocab_proj(self, v, title, k=TOPK):
        s = self.U @ v.float()
        z = (s - s.mean()) / s.std()
        top, bot = torch.topk(s, k).indices, torch.topk(-s, k).indices
        out = [f"## {title}: direct logit effect of this residual-stream direction, logits = E diag(1+g_final) v",
               "top +: " + ", ".join(f"{self.t(i)} {float(z[i]):+.1f}σ" for i in top),
               "top -: " + ", ".join(f"{self.t(i)} {float(z[i]):+.1f}σ" for i in bot),
               "prompt targets: " + ", ".join(f"{p['pid']} {p['target']!r} {float(z[p['target_id']]):+.2f}σ" for p in self.prm)]
        return "\n".join(out)

    def input_proj(self, w_row, norm_gain, title, k=TOPK):
        """Which token embeddings (fed straight through the norm) most excite a read-in row (direct path only)."""
        s = (self.E * (1 + norm_gain)) @ w_row.float()
        z = (s - s.mean()) / s.std()
        top, bot = torch.topk(s, k).indices, torch.topk(-s, k).indices
        return (f"## {title}: tokens whose (normed) embedding gives the largest read-in, E diag(1+g) w\n"
                "top +: " + ", ".join(f"{self.t(i)} {float(z[i]):+.1f}σ" for i in top) + "\n"
                "top -: " + ", ".join(f"{self.t(i)} {float(z[i]):+.1f}σ" for i in bot))

    def emb_col(self, d, k=TOPK):
        col = self.E[:, d]
        z = (col - col.mean()) / col.std()
        top, bot = torch.topk(col, k).indices, torch.topk(-col, k).indices
        return (f"## embedding column {d} (tied; also the unembedding weight of dim {d}); final-norm gain 1+g[{d}]={float(self.gf[d]):+.4f}\n"
                "top +: " + ", ".join(f"{self.t(i)} {float(z[i]):+.1f}σ" for i in top) + "\n"
                "top -: " + ", ".join(f"{self.t(i)} {float(z[i]):+.1f}σ" for i in bot) + "\n"
                "prompt targets E[t,d]: " + ", ".join(f"{p['pid']} {p['target']!r} {float(col[p['target_id']]):+.4f} ({float(z[p['target_id']]):+.1f}σ)" for p in self.prm))

    def prompt_tokens_ov(self, M, norm_gain, title):
        """For a linear read-in -> write-out map M (1024x1024), show what each prompt token pushes if it is the source."""
        lines = [f"## {title}: if a prompt token x is read (embedding through the layer's input norm) the head writes M x; its direct logit effect:"]
        for p in self.prm:
            cells = []
            for x in sorted(set(p["input_ids"])):
                s = self.U @ (M @ (self.E[x] * (1 + norm_gain)))
                z = (s - s.mean()) / s.std()
                top = torch.topk(s, 3).indices
                cells.append(f"{self.t(x)}->[" + ", ".join(self.t(i) for i in top) + f"] target {float(z[p['target_id']]):+.1f}σ")
            lines.append(f"{p['pid']}: " + "; ".join(cells))
        return "\n".join(lines)


def shared_files(c):
    """Raw files every weights condition gets: target-token embedding rows and the final norm gain."""
    rows = [f"# raw embedding rows (tied with unembedding) of each prompt's {'scored token and its top-4 tokens' if ALT else 'target and its top-3 runner-ups'}. Shape (248320, 1024), {stats(c.E)}"]
    seen = set()
    for p in c.prm:
        for t in [p["target_id"]] + [x["id"] for x in p["top10"][:4]]:
            if t in seen:
                continue
            seen.add(t)
            rows.append(f"\n## token {t} {c.t(t)} ({p['pid']}) {stats(c.E[t])}\n" + fmt_vec(c.E[t]))
    norms = [f"# final RMSNorm `model.norm.weight` w (applied as x/rms(x) * (1+w)). {stats(c.m.model.norm.weight.data)}\n" + fmt_vec(c.m.model.norm.weight.data)]
    for l in range(24):
        for nm in ["input_layernorm", "post_attention_layernorm"]:
            w = P(f"model.layers.{l}.{nm}.weight")
            norms.append(f"\n# `model.layers.{l}.{nm}.weight` w (gain 1+w). {stats(w)}\n" + fmt_vec(w, prec=3))
    return {"shared_embeddings_targets.txt": "\n".join(rows), "shared_norm_gains.txt": "\n".join(norms)}


def layer_norm_table():
    lines = ["# Frobenius norms of every matrix, per layer (F = full attention, L = Gated DeltaNet)",
             "layer type  mixer_in(q|qkv)  mixer_out(o|out)  gate_proj  up_proj  down_proj  in_norm_gain_mean  post_norm_gain_mean"]
    for l in range(24):
        inn = P(f"model.layers.{l}.self_attn.q_proj.weight") if l in FULL else P(f"model.layers.{l}.linear_attn.in_proj_qkv.weight")
        lines.append(f"{l:5d} {'F' if l in FULL else 'L'}    {inn.norm():10.2f}  {P(mixer_out(l)).norm():10.2f}  "
                     f"{P(f'model.layers.{l}.mlp.gate_proj.weight').norm():9.2f} {P(f'model.layers.{l}.mlp.up_proj.weight').norm():8.2f} "
                     f"{P(f'model.layers.{l}.mlp.down_proj.weight').norm():9.2f}   {float((1 + P(f'model.layers.{l}.input_layernorm.weight')).mean()):+.3f}   "
                     f"{float((1 + P(f'model.layers.{l}.post_attention_layernorm.weight')).mean()):+.3f}")
    return "\n".join(lines)


def component_files(c, it, views):
    k, out = it["kind"], []
    if k == "norm_gain":
        d, w = it["index"][0], P(it["param"])
        out.append(f"# {it['param']} (gain = 1+w), edited dim {d} marked [ ]; w[{d}]={float(w[d]):+.4f}. {stats(w)}\n" + fmt_vec(w, mark=d))
        out.append(f"# raw embedding column {d} for prompt targets: " + ", ".join(f"{p['target']!r} {float(c.E[p['target_id'], d]):+.4f}" for p in c.prm))
        if views:
            g = 1 + w
            out.append(f"## views: |gain| rank of dim {d} in this vector: {int((g.abs() > g[d].abs()).sum()) + 1} of 1024")
            out.append(c.emb_col(d))
    elif k in ("max_weight", "add_value"):
        r, col = it["index"]
        W = P(it["param"])
        out.append(f"# {it['param']} shape {tuple(W.shape)}; W[{r},{col}]={float(W[r, col]):+.5f}. matrix: {stats(W)}; "
                   f"99.9th pct |w|={float(W.abs().flatten()[::97].float().quantile(0.999)):.4f}")
        out.append(f"## raw row {r} (edited column {col} marked [ ])\n{fmt_vec(W[r], mark=col)}")
        if W.shape[0] <= 6144 and "embed" not in it["param"]:
            out.append(f"## raw column {col} (edited row {r} marked [ ])\n{fmt_vec(W[:, col], mark=r)}")
        if views:
            if "embed" in it["param"]:
                out.append(c.emb_col(col))
            elif "down_proj" in it["param"] or "o_proj" in it["param"] or "out_proj" in it["param"]:
                out.append(f"## views: residual dim {r} is the written dimension\n" + c.emb_col(r))
                if "down_proj" in it["param"]:
                    out.append(c.vocab_proj(W[:, col], f"neuron {col} write direction (down_proj column)"))
            elif "gate_proj" in it["param"]:
                l = int(it["param"].split(".")[2])
                out.append(c.vocab_proj(P(f"model.layers.{l}.mlp.down_proj.weight")[:, r], f"neuron {r} write direction"))
                out.append(c.emb_col(col))
    elif k == "neuron":
        l, i = it["layer"], it["neuron"]
        g, u, dn = P(f"model.layers.{l}.mlp.gate_proj.weight")[i], P(f"model.layers.{l}.mlp.up_proj.weight")[i], P(f"model.layers.{l}.mlp.down_proj.weight")[:, i]
        norms = P(f"model.layers.{l}.mlp.down_proj.weight").norm(dim=0)
        out.append(f"# layer {l} MLP neuron {i}: out = down[:,i] * silu(gate[i]·x) * (up[i]·x), x = post_attention_layernorm(h)")
        out.append(f"down column L2={float(dn.norm()):.4f} (layer median {float(norms.median()):.4f}, max {float(norms.max()):.4f})")
        out.append(f"## gate_proj row {i} {stats(g)}\n{fmt_vec(g)}\n## up_proj row {i} {stats(u)}\n{fmt_vec(u)}\n## down_proj column {i} {stats(dn)}\n{fmt_vec(dn)}")
        if views:
            pn = P(f"model.layers.{l}.post_attention_layernorm.weight")
            out.append(f"## views: down column norm rank {int((norms > norms[i]).sum()) + 1} of 3584 in layer {l}")
            out.append(c.vocab_proj(dn, f"neuron {i} write direction"))
            out.append(c.input_proj(g, pn, "gate row") + "\n" + c.input_proj(u, pn, "up row"))
            out.append("cosine(down col, target embedding): " + ", ".join(f"{p['pid']} {float(torch.cosine_similarity(dn, c.E[p['target_id']], 0)):+.3f}" for p in c.prm))
    elif k == "layer":
        l = it["layer"]
        out.append(layer_norm_table())
        out.append(f"# layer {l} type {'full attention' if l in FULL else 'Gated DeltaNet'}; norm gains:\n## input_layernorm w\n{fmt_vec(P(f'model.layers.{l}.input_layernorm.weight'), prec=3)}\n"
                   f"## post_attention_layernorm w\n{fmt_vec(P(f'model.layers.{l}.post_attention_layernorm.weight'), prec=3)}")
        if l not in FULL:
            out.append(f"A_log: {fmt_vec(P(f'model.layers.{l}.linear_attn.A_log'))}\ndt_bias: {fmt_vec(P(f'model.layers.{l}.linear_attn.dt_bias'))}")
        Wd = P(f"model.layers.{l}.mlp.down_proj.weight")
        out.append(f"## down_proj column L2 norms, top 20 neurons: " + ", ".join(f"{int(i)}:{float(Wd[:, i].norm()):.3f}" for i in torch.argsort(Wd.norm(dim=0), descending=True)[:20]))
        if views:
            for nm, W in [("mixer out-projection", P(mixer_out(l))), ("down_proj", Wd)]:
                Uo, S, _ = torch.linalg.svd(W.float(), full_matrices=False)
                out.append(f"## views: {nm} top singular values: " + " ".join(f"{float(s):.2f}" for s in S[:8]))
                for j in range(2):
                    out.append(c.vocab_proj(Uo[:, j], f"{nm} left singular vector {j} (sign arbitrary)"))
            for i in torch.argsort(Wd.norm(dim=0), descending=True)[:3]:
                out.append(c.vocab_proj(Wd[:, i], f"largest neuron {int(i)} write direction"))
    elif k == "token":
        toks = [it["a"]] + ([it["b"]] if "b" in it else [])
        for t in toks:
            out.append(f"# raw embedding row of token {t} {c.t(t)} {stats(c.E[t])}\n{fmt_vec(c.E[t])}")
        if views:
            En = torch.nn.functional.normalize(c.E.float(), dim=1)
            for t in toks:
                nn_ = torch.topk(En @ En[t], 11).indices[1:]
                out.append(f"## views: nearest tokens to {c.t(t)} by cosine: " + ", ".join(c.t(i) for i in nn_))
                out.append(f"cosine with prompt targets: " + ", ".join(f"{p['pid']} {p['target']!r} {float(En[t] @ En[p['target_id']]):+.3f}" for p in c.prm))
                occ = [f"{p['pid']}:{'target' if p['target_id'] == t else ''}{'input' if t in p['input_ids'] else ''}{'top10' if any(x['id'] == t for x in p['top10'][1:]) else ''}" for p in c.prm]
                out.append(f"where {c.t(t)} occurs: " + ", ".join(o for o in occ if not o.endswith(":")))
    return "\n\n".join(out)


ARCH = """## Model: Qwen3.5-0.8B (text part), {revision}
- 24 decoder layers, hidden size 1024, vocab 248,320, **tied embeddings** (`model.embed_tokens.weight` is also the output unembedding).
- Pre-norm residual: h = h + Mixer(RMSNorm_in(h)); h = h + MLP(RMSNorm_post(h)); logits = E · RMSNorm_final(h).
- RMSNorm(x) = x / rms(x) * (1 + w)  (note the **1 + w**: stored w near 0 means gain near 1).
- Token mixer: layers 3, 7, 11, 15, 19, 23 are full softmax attention (8 query heads of dim 256, 2 KV heads, per-head q/k RMSNorm,
  partial RoPE, and a per-head sigmoid output gate computed from the input; o_proj columns [256h, 256h+256) belong to head h).
  All other layers are Gated DeltaNet linear attention (16 heads, head dim 128, short causal conv, learned decay A_log/dt_bias,
  output = gated RMSNorm(read-out) * silu(z); out_proj columns [128h, 128h+128) belong to head h).
- MLP: down_proj( silu(gate_proj x) * up_proj x ), 3,584 neurons; neuron i writes down_proj[:, i].
- Weight edits are applied, then the 8 prompts are run with no other change. "Skip a sublayer" = zero its output projection
  (o_proj/out_proj or down_proj), which removes its residual contribution exactly."""

TASK = """# Task: predict how weight edits change Qwen3.5-0.8B's next-token probabilities

You are given {n} weight edits (interventions) of one kind ("{family}" granularity). Each edit is applied **on its own** to the
unedited model, and then the model reads each of 8 fixed prompts. For every (edit, prompt) pair predict:
- `dlogp`: change in the natural-log probability of the prompt's {target_def}, edited minus unedited, in nats.
  0 = no change, negative = target becomes less likely. Range clipped to [-20, +3].
- `p_flip`: your probability that the model's top-1 next token is different after the edit.

{condition_text}
{required}
You may only read files (no code execution is available). Work by reasoning about the model and the provided material.
Files in this directory: `README.md` (this file), `test_interventions.md` (what to predict){feedback_line}{weights_line}.

{arch}

## Prompts and the unedited model's next-token distribution
{prompts}

## Output format (your final message)
A single fenced ```json block, nothing else after it:
{{"predictions": [{{"iid": "XX00", "pid": "P0", "dlogp": -0.3, "p_flip": 0.05}}, ...  one entry for each of the {nitems} (edit, prompt) pairs],
 "notes": {{"XX00": "one short sentence on the evidence you used", ...}}}}
Scoring: Spearman correlation and mean absolute error of dlogp against the measured values, and Brier score of p_flip.
"""

COND = dict(
    none="**Condition: no weights.** You do not get any weight values. Predict from your general knowledge of transformers, the edit descriptions, "
         "the prompts, the unedited distribution{fb}.",
    raw="**Condition: raw weights.** `weights/` has the raw numbers (4 decimals, 16 per line, index of the first value at line start) of each "
        "edited component, plus simple statistics (norms, std, max). Shared raw files: target-token embedding rows and all norm gains. Read the weights and use them{fb}.",
    views="**Condition: weights + weights-only views.** `weights/` has the raw numbers of each edited component plus views computed from weights "
          "only (never from running the model): vocabulary projections through the tied unembedding (direct logit effect), input-token "
          "read-in projections, SVDs, OV maps of the prompt tokens, norm ranks. Read the weights and views and use them{fb}.",
)


REQUIRED = ("\n**Required:** before predicting, Read every `weights/<iid>.txt` file of the edits listed in `test_interventions.md` "
            "(the shared_*.txt files are optional). In `notes`, cite at least one specific number from each edit's file. "
            "Calls that skip a test edit's file are rejected.\n")


CALIB = ("**Calibration guidance.** In this model most single edits change a target's log-probability very little: a typical edit moves it "
         "by less than 0.05 nats and many by less than 0.01. Large changes (over 1 nat) need a perturbation that is large relative to the residual "
         "stream, one aimed at the target token's (un)embedding, or the removal of a component the prediction depends on. Predict values near 0 "
         "unless you have a concrete reason for a larger effect, and make the size of each prediction depend on how large the edit is.\n")
PROC = ("**Procedure (do the arithmetic yourself; you cannot run code).** For each edit, before predicting:\n"
        "1. *Size.* From the weight file, judge how large the change is: the norm of the removed or scaled weights against the other rows, columns or "
        "heads of the same matrix (norm tables are given), or a changed scalar against the matrix std. An edit to a small part of a large layer is small.\n"
        "2. *Path to the output.* Late-layer edits act mostly through the residual stream straight to the logits; early edits pass through many later "
        "layers and are usually damped. Final-norm gains and embedding entries act directly on logits.\n"
        "3. *Direction.* For direct effects, compare the changed write direction with the target's embedding row in `weights/shared_embeddings_targets.txt` "
        "and the final-norm gains: estimate the dot product from the largest entries. Positive alignment raises that target's logit when the component's activation is positive.\n"
        "4. *Number.* Estimate the logit change of each prompt's target and convert it: a logit change d on a token with probability p changes its "
        "log-probability by about d(1 - p) when d is small.\n"
        "Show the key numbers briefly in your reasoning.\n")


ALLW = ("**All weights are available.** Besides `weights/`, every tensor of the unedited model is in `{full}` (read-only). Start with `{full}/INDEX.md`, "
        "which lists each tensor, its shape and the line where any row starts, so you can open specific rows with Read offset/limit.\n"
        "Reason about the weights as a whole, not only the edited component. For each edit, trace how the change reaches the scored token: read the "
        "scored token's and the top-1 token's rows of `model.embed_tokens.weight` (it is also the output unembedding; row = token id) and the final-norm gain; "
        "for an MLP neuron or a layer, compare its write direction with those rows and with the read-in rows of later layers; for an embedding entry, "
        "consider both its input and output roles. When the top-1 token loses probability, the freed mass goes to the other tokens, so a lower-ranked "
        "token can rise. Do all arithmetic yourself (no code is available) and cite the specific numbers you used in `notes`.\n")


def prompts_md(prm):
    lines = []
    for p in prm:
        lines.append(f"- **{p['pid']}** text {p['text']!r}; input tokens {p['input_tokens']} (ids {p['input_ids']})\n  " +
                     (f"scored token {p['target']!r} (id {p['target_id']}, rank {p['target_rank']}, p {math.exp(p['target_logp']):.4f}; top-1 is {p['top1']!r}), unedited top-10: " if ALT else
                      f"target {p['target']!r} (id {p['target_id']}), unedited top-10: ") +
                     ", ".join(f"{x['token']!r} {x['p']:.3f}" for x in p["top10"]))
    return "\n".join(lines)


def build(round_, family, cond, items_all, feedback, out_dir, shots=None):
    """items_all: list of this family's interventions; feedback: dict iid -> measured results for earlier rounds.
    shots: if given, only these earlier iids are revealed and they are shown inline (weights included) as worked examples."""
    from common import REVISION
    os.makedirs(out_dir, exist_ok=True)
    prm = load_prompts()
    test = [it for it in items_all if it["round"] == round_]
    train = [it for it in items_all if it["round"] < round_ and (shots is None or it["iid"] in shots)]
    variant = cond.split("_")[1] if "_" in cond else "v0"
    cond = cond.split("_")[0]
    extra = {"v0": "", "v1": "\n" + CALIB, "v2": "\n" + CALIB + "\n" + PROC,
             "all": "\n" + ALLW.format(full=os.environ.get("WSP_FULL_WEIGHTS", "<full weights dir>"))}[variant]
    fb = ", and the measured effects of earlier edits of the same kind in `feedback.md`" if train else ""
    if cond != "none":
        fb += " (the weight files of those earlier edits are in `weights/` too)" if train else ""
    readme = TASK.format(n=len(test), family=family, nitems=len(test) * len(prm), condition_text=COND[cond].format(fb=fb),
                         feedback_line=", `feedback.md` (measured effects of earlier edits)" if train else "",
                         weights_line=", `weights/` (one file per edit id, plus shared_*.txt)" if cond != "none" else "",
                         arch=ARCH.format(revision=f"revision {REVISION[:10]}"), prompts=prompts_md(prm),
                         required=(REQUIRED if cond != "none" else "") + extra,
                         target_def=("scored token (listed per prompt below; a plausible next token ranked 6 to 10 before the edit, not the top-1)" if ALT
                                     else 'baseline top-1 token (its "target")'))
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(readme)
    with open(os.path.join(out_dir, "test_interventions.md"), "w") as f:
        f.write("# Edits to predict (each applied alone)\n" + "\n".join(f"- **{it['iid']}**: {describe(it)}" + (f" [file weights/{it['iid']}.txt]" if cond != "none" else "") for it in test))
    if train:
        lines = ["# Measured effects of earlier edits of the same kind (dlogp of the target in nats; flip = top-1 changed; new top-1 shown if flipped)"]
        for it in train:
            r = feedback[it["iid"]]
            lines.append(f"\n## {it['iid']}: {describe(it)}" + (f" [file weights/{it['iid']}.txt]" if cond != "none" else ""))
            lines.append(" | ".join(f"{pid} dlogp {x['dlogp']:+.3f}{' FLIP->' + repr(x['new_top1']) if x['flip'] else ''}" for pid, x in r.items()))
            if shots is not None and cond != "none":
                lines.append("\n<details>weights of this example (same file as weights/" + it["iid"] + ".txt)\n\n```\n" +
                             component_files(Ctx(), it, views=(cond == "views")) + "\n```\n</details>")
        with open(os.path.join(out_dir, "feedback.md"), "w") as f:
            f.write("\n".join(lines))
    if cond != "none":
        c = Ctx()
        wd = os.path.join(out_dir, "weights")
        os.makedirs(wd, exist_ok=True)
        for name, txt in shared_files(c).items():
            with open(os.path.join(wd, name), "w") as f:
                f.write(txt)
        for it in test + train:
            with open(os.path.join(wd, f"{it['iid']}.txt"), "w") as f:
                f.write(f"# {it['iid']}: {describe(it)}\n\n" + component_files(c, it, views=(cond == "views")))
    return readme
