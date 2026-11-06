"""Sanity check: compare network features at '=' to a plain-Python reference on hand-made sentences."""
import sys, re
sys.path.insert(0, "/data18/chandan.singh/wsp-nl/sandbox/rec-nli-snli-r1")
import torch
import interpretable_transformer as it
import src.task as T

task = T.get_task("nli-snli")
model = it.build_model(task)


def pad(s):
    return s[:60].ljust(60, "_")


def ref(p, h):
    P, H = pad(p), pad(h)
    full = P + "|" + H + "="
    seg = "|" + H + "="
    votes = [0.0, 0.0, 0.0]
    hyp_words = [mt.group(1) for mt in re.finditer(r"(?<=[ |])([a-z']+)(?=[ _=])", seg)]
    for w in hyp_words:
        if w in it.VOTES:
            for c in range(3):
                votes[c] += it.VOTES[w][c]
    for mt in re.finditer(r"(?<=[ |])([a-z']+)", seg):
        for k, v in it.VOTES.items():
            if k.endswith("*") and mt.group(1).startswith(k[:-1]):
                for c in range(3):
                    votes[c] += v[c]
    prem_words = re.findall(r"(?<= )([a-z']+)(?=[ _])", P)
    pcls = {it.WORD2CLASS[w] for w in prem_words if w in it.WORD2CLASS}
    nconf = 0
    for w in hyp_words:
        if w in it.WORD2CLASS:
            c = it.WORD2CLASS[w]
            g = it.CLASSES[c][0]
            if any(it.CLASSES[c2][0] == g and c2 != c for c2 in pcls):
                nconf += 1
    ptri = {P[i - 2:i + 1] for i in range(2, 60)}
    valid = [H[i - 2:i + 1] for i in range(2, 60) if re.fullmatch("[a-z]{3}", H[i - 2:i + 1])]
    cov = sum(t in ptri for t in valid) / len(valid) if valid else 0.0
    return dict(votes=votes, nconf=nconf, cov=round(cov, 3), lenp=len(P.rstrip("_")), lenh=len(H.rstrip("_"))), full


cases = [
    ("a man is sitting on a bench in the park", "a woman is sleeping on a bench"),
    ("two dogs run through the grass outside", "the animals are outdoors"),
    ("a girl in a red dress dances on stage", "a sad girl is not dancing"),
    ("people are swimming in the pool", "nobody is in the pool"),
    ("a man rides a bike down the street near a white car", "a man rides a motorcycle and a blue car"),
    ("a woman is cooking in the kitchen", "a woman is trying to cook dinner for her family at home"),
    ("a b", "x"),
    ("a man is standing in front of a large building with many windows", "a person stands near a tall building with lots of windows to"),
]
for p, h in cases:
    r, full = ref(p, h)
    ids = torch.tensor([task.encode(full)])
    with torch.no_grad():
        x = model.token_emb(ids) + model.pos_emb(torch.arange(ids.shape[1]))[None]
        for b in model.blocks:
            x = b(x)
        z = x[0, -1]
        logits = model.head(z)
    got = dict(votes=[round(z[i].item(), 3) for i in it.SVOTE], nconf=round(z[it.NCONF].item(), 3),
               cov=round(z[it.COV].item(), 3), lenp=round(z[it.LEN_P].item(), 2), lenh=round(z[it.LEN_H].item(), 2))
    print("REF", r)
    print("NET", got, "pred", task.itos[int(logits.argmax())])
print("params", sum(p.numel() for p in model.parameters()))
