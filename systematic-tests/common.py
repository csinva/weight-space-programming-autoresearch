"""Shared helpers: load Qwen3.5-0.8B, fixed prompts, apply weight edits, measure next-token log-probs."""
import json, os
from contextlib import contextmanager
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(ROOT, ".hf_cache"))
MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
torch.set_grad_enabled(False)

_cache = {}


def load():
    if "m" not in _cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).eval()
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        _cache["m"], _cache["tok"] = m, tok
    return _cache["m"], _cache["tok"]


ALT = os.environ.get("WSP_TARGET") == "alt"  # score a lower-ranked token (ranks 6-10, data/alt_targets.json) instead of the top-1


def load_prompts():
    with open(os.path.join(ROOT, "data", "prompts.json")) as f:
        prm = json.load(f)
    if ALT:
        alt = json.load(open(os.path.join(ROOT, "data", "alt_targets.json")))
        for p in prm:
            a = alt[p["pid"]]
            p.update(top1=p["target"], target=a["token"], target_id=a["id"], target_logp=float(__import__("math").log(a["p"])), target_rank=a["rank"])
    return prm


def next_token_logprobs(texts):
    """Log-softmax over the vocab at the last position of each text (no padding: run one by one)."""
    m, tok = load()
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        out.append(torch.log_softmax(m(input_ids=ids).logits[0, -1].float(), -1))
    return torch.stack(out)


def param(name):
    m, _ = load()
    return dict(m.named_parameters())[name]


@contextmanager
def edited(edits):
    """edits: list of (param_name, index_tuple_or_None, fn) where fn maps the old slice to the new slice.
    Applies all edits in place, yields, then restores the exact original values."""
    saved = []
    for name, idx, fn in edits:
        p = param(name)
        sl = p.data if idx is None else p.data[idx]
        saved.append((p, idx, sl.clone()))
        new = fn(sl.clone())
        if idx is None:
            p.data.copy_(new)
        else:
            p.data[idx] = new
    try:
        yield
    finally:
        for p, idx, old in reversed(saved):
            if idx is None:
                p.data.copy_(old)
            else:
                p.data[idx] = old
