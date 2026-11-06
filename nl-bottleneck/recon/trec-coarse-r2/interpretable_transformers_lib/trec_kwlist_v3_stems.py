"""
Snapshot trec_kwlist_v3_stems (accuracy 0.845 on trec-coarse, 200 examples, seed 0).
Same as v2, but topical nouns of >= 5 letters are stems (leading boundary only).

Hand-written keyword decision list transformer (no training):
  Embedding: 5-bit (+-1) letter code, flags {space, pad, apostrophe}, start flag,
             constant, position j, j^2, salience channel preloaded with -3*j.
  Block 1:   20 Gaussian-kernel copy heads (head h copies token i-h); MLP of exact
             AND-gate neurons, one per keyword pattern, writing 300*priority salience
             and a class vote.
  Block 2:   '=' attends to max salience and copies the vote; default ENTY.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_qk: int, d_v: int):
        super().__init__()
        self.n_heads, self.d_qk, self.d_v = n_heads, d_qk, d_v
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, self.d_qk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.d_qk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.d_v).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * self.d_v)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_qk, d_v, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_qk, d_v)
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None

    def forward(self, x):
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


N_FEAT = 9
F_CODE, F_SPACE, F_PAD, F_APOS, F_START = 0, 5, 6, 7, 8
D_CONST, D_POS, D_POS2, D_SAL = 9, 10, 11, 12
D_VSTORE = 13
D_VOUT = 19
N_COPY = 20
D_COPY = 25
D_MODEL = D_COPY + N_COPY * N_FEAT

ABBR, ENTY, DESC, HUM, LOC, NUM = range(6)


def _words(s):
    return s.split()

def build_patterns():
    P = []

    def add(pats, cls, pri):
        for p in pats:
            P.append((p, cls, pri))

    add([" stands for ", " stand for ", " abbreviat"], ABBR, 12)
    add([" comes from "], DESC, 11)
    add(["^who "], HUM, 10)
    add([" where "], LOC, 10)
    add(["^when "], NUM, 10)
    add([" why "], DESC, 10)
    add([" mean", " origin", " reason", " defin", " caus", " purpose",
         " difference ", " explain", " describe", " effect"], DESC, 9)
    add([" how did ", " how does ", " how is ", " how do ", " how can "], DESC, 8)
    add([" how "], NUM, 7)
    P.append(("#SHORTWHATIS", DESC, 6))
    hum = _words("""person persons people man men woman women president presidents
        actor actors actress author authors writer writers inventor king kings queen
        singer singers player players team teams company companies leader leaders
        scientist scientists artist artists emperor director poet composer character
        captain founder explorer painter astronaut wife husband son daughter father
        mother boy girl owner hero coach general minister senator governor pope
        prince princess chairman ceo band group organization architect designer
        lawyer doctor manufacturer creator comedian musician novelist playwright
        athlete boxer golfer pitcher quarterback winner champion spy villain
        dictator saint god goddess pilot soldier sheriff detective nurse teacher
        professor student brother sister tribe dynasty airline corporation agency
        firm club astronomer mathematician physicist philosopher sculptor
        cartoonist hostess host anchor""")
    loc = _words("""city cities country countries state states capital capitals
        continent island islands river rivers mountain mountains lake lakes ocean
        oceans sea seas place places province county town towns nation nations desert
        park region planet street airport border coast valley peninsula canal bay
        port harbor volcano hemisphere territory village district hometown
        birthplace headquarters address bridge museum stadium cave forest beach
        resort castle palace temple church cathedral tower monument landmark site
        colony republic empire avenue highway prison""")
    num = _words("""year years date dates population percentage percent temperature
        distance speed number numbers cost costs price weight height age century
        degrees miles size area salary rate average day days month months time
        length width depth amount radius diameter volume dollars budget value
        birthday anniversary lifespan calories acres feet inches pounds tons hour
        hours minutes seconds score decade phone telephone fahrenheit celsius
        kilometers meters yards gallons liters ounces grams watts ratio odds
        probability frequency quantity total worth income wage fee expectancy
        duration period""")
    enty = _words("""animal animals dog dogs bird birds fish plant plants flower
        flowers tree trees food foods drink drinks color colors colour book books
        movie movies film films song songs game games sport sports language
        languages disease diseases instrument instruments car cars word words
        letter letters novel show product products brand currency religion award
        prize medal drug drugs element elements material substance fruit fruits
        vegetable vegetables ship ships boat plane aircraft invention weapon
        weapons symbol breed music magazine newspaper series cartoon comic toy
        organ bone bones blood gas metal stone gem computer software program
        insect insects mammal mammals reptile dish sauce cheese wine beer candy
        horse horses cat cats snake flag title war battle event ingredient chemical
        vitamin mineral virus illness cancer dance painting vehicle term sign
        alphabet tool machine device engine trademark logo costume clothing dress
        shoes hat coin coins sound shape fabric wood tea coffee bread meat soup
        dessert cocktail spice herb seed grass crop grain rice oil milk sugar acid
        protein gene muscle syndrome disorder bacteria medicine opera ballet poem
        story album television radio channel website mascot nickname""")

    def nouns(ws):
        ws = list(dict.fromkeys(ws))
        stems = [w for w in ws if len(w) >= 5]
        out = []
        for w in ws:
            if len(w) >= 5:
                if any(s != w and w.startswith(s) for s in stems):
                    continue
                out.append(f" {w}")
            else:
                out.append(f" {w} ")
        return out
    add(nouns(hum), HUM, 5)
    add(nouns(loc) + [" locat"], LOC, 5)
    add(nouns(num) + [" zip code "], NUM, 5)
    add(nouns(enty), ENTY, 5)
    add([" called ", " call ", " what kind ", " what type ", " what sort "], ENTY, 4)
    add(["^what are the "], ENTY, 3)
    add(["^what is ", "^what's "], DESC, 2)
    return P


def pattern_elements(pat):
    if pat == "#SHORTWHATIS":
        els = []
        s = "what is "
        for idx, ch in enumerate(s):
            off = 16 - idx
            els.append((off, "space", None) if ch == " " else (off, "letter", ch))
        els.append((16, "start", None))
        els.append((0, "pad", None))
        return els
    caret = pat.startswith("^")
    lead = pat.startswith(" ")
    body = pat[1:] if (caret or lead) else pat
    tail = body.endswith(" ")
    core = body[:-1] if tail else body
    chars = list(core) + ([None] if tail else [])
    L = len(chars)
    els = []
    for idx, ch in enumerate(chars):
        off = L - 1 - idx
        if ch is None:
            els.append((off, "tail", None))
        elif ch == " ":
            els.append((off, "space", None))
        else:
            els.append((off, "letter", ch))
    if caret:
        els.append((L - 1, "start", None))
    if lead:
        els.append((L, "lead", None))
    assert max(e[0] for e in els) < N_COPY, pat
    return els


PATTERNS = build_patterns()


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, **_):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = D_MODEL
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList([
            Block(D_MODEL, N_COPY, 2, N_FEAT, len(PATTERNS)),
            Block(D_MODEL, 1, 1, 6, 0),
        ])
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def letter_code(ch):
    k = ord(ch) - ord("a") + 1
    return [1.0 if (k >> b) & 1 else -1.0 for b in range(5)]


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    stoi = task.stoi
    E, Pe = model.token_emb.weight, model.pos_emb.weight
    for ch, i in stoi.items():
        if "a" <= ch <= "z":
            E[i, F_CODE:F_CODE + 5] = torch.tensor(letter_code(ch))
        elif ch == " ":
            E[i, F_SPACE] = 1
        elif ch == "_":
            E[i, F_PAD] = 1
        elif ch == "'":
            E[i, F_APOS] = 1
    eq = stoi["="]
    E[eq, D_SAL] = 270.0
    E[eq, D_VSTORE + ENTY] = 1.0
    for j in range(model.max_seq_len):
        Pe[j, D_CONST] = 1.0
        Pe[j, D_POS] = float(j)
        Pe[j, D_POS2] = float(j * j)
        Pe[j, D_SAL] = -3.0 * j
    Pe[0, F_START] = 1.0

    a = 20.0
    att = model.blocks[0].attn
    for h in range(N_COPY):
        att.W_q.weight[2 * h + 0, D_POS] = 2 * a
        att.W_q.weight[2 * h + 0, D_CONST] = -2 * a * h
        att.W_q.weight[2 * h + 1, D_CONST] = -a
        att.W_k.weight[2 * h + 0, D_POS] = 1.0
        att.W_k.weight[2 * h + 1, D_POS2] = 1.0
        for f in range(N_FEAT):
            att.W_v.weight[N_FEAT * h + f, f] = 1.0
            att.W_o.weight[D_COPY + N_FEAT * h + f, N_FEAT * h + f] = 1.0

    mlp = model.blocks[0].mlp
    W1, b1, W2 = mlp.fc1.weight, mlp.fc1.bias, mlp.fc2.weight

    def c(off, f):
        return D_COPY + N_FEAT * off + f

    for n, (pat, cls, pri) in enumerate(PATTERNS):
        total = 0.0
        for off, kind, val in pattern_elements(pat):
            if kind == "letter":
                for b, s in enumerate(letter_code(val)):
                    W1[n, c(off, F_CODE + b)] = s
                total += 5
            elif kind == "space":
                W1[n, c(off, F_SPACE)] = 2; total += 2
            elif kind == "pad":
                W1[n, c(off, F_PAD)] = 2; total += 2
            elif kind == "start":
                W1[n, c(off, F_START)] = 2; total += 2
            elif kind == "tail":
                W1[n, c(off, F_SPACE)] = 2
                W1[n, c(off, F_PAD)] = 2
                W1[n, c(off, F_APOS)] = 2
                total += 2
            elif kind == "lead":
                W1[n, c(off, F_SPACE)] = 2
                W1[n, c(off - 1, F_START)] += 2
                total += 2
        b1[n] = -(total - 1.0)
        W2[D_SAL, n] = 300.0 * pri
        W2[D_VSTORE + cls, n] = 1.0

    att2 = model.blocks[1].attn
    att2.W_q.weight[0, D_CONST] = 5.0
    att2.W_k.weight[0, D_SAL] = 1.0
    for k in range(6):
        att2.W_v.weight[k, D_VSTORE + k] = 1.0
        att2.W_o.weight[D_VOUT + k, k] = 1.0

    H = model.head.weight
    for i in range(model.vocab_size):
        H[i, D_CONST] = -10.0
    for k in range(6):
        H[stoi[str(k)], D_VOUT + k] = 10.0
        H[stoi[str(k)], D_CONST] = 0.0


model_shorthand_name = "trec_kwlist_v3_stems"
model_description = ("2-block hand-written keyword decision list: 20 Gaussian copy heads + ~400 AND-gate "
                     "keyword neurons, '=' attends to max 300*priority-3*pos salience and copies class vote; default ENTY.")


def upsert_overall_results(rows: list[dict], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["task"]) for r in rows}
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_shorthand_name"), row.get("task")) not in new_keys:
                    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows])
    print(f"Overall results saved → {path}")


def build_model(task) -> SimpleTransformer:
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task name (see src/task.py)", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task":        args.task,
        "accuracy":    f"{accuracy:.4f}",
        "status":      "",
        "model_shorthand_name":  model_shorthand_name,
        "n_params":    f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
