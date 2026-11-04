"""
Interpretable transformer for trec-coarse: a hand-written keyword-voting circuit.

  Block 1: attention heads read the character d steps back (a sliding window);
           each MLP hidden unit is an AND over window slots = one keyword detector;
           fc2 turns detections into six class votes.
  Block 2: uniform attention sums votes over the question; head reads the argmax.

No weight is trained: RULES below is written by hand.
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

CLASSES = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
PRIOR = {"ABBR": 0, "ENTY": 1, "DESC": 0, "HUM": 0, "LOC": 0, "NUM": 0}

# "CLASS weight pattern pattern ..."  ~ = word boundary, ^ = start of question.
RULES = """
HUM 30 ^who~ ^whose~ ^whom~
LOC 16 ^where~
NUM 16 ^when~
DESC 20 ^why~
DESC 6 ^how~
NUM 14 ^how~many ^how~much ^how~long ^how~far ^how~old ^how~tall ^how~big ^how~high ^how~fast ^how~deep ^how~wide ^how~hot ^how~cold ^how~heavy ^how~often ^how~large ^how~short
DESC 3 ^what~is~a~ ^what~is~an~
DESC 2 ^what~is~
DESC 0.3 <_
DESC 5 ^what~causes ^what~caused ^what~happened ^what~prompted
ENTY 5 ^what~is~the~name~of ^what~was~the~name~of
DESC 22 ~come~from ~comes~from ~came~from
HUM 8 ~first~name ~full~name ~last~name ~maiden~name
ENTY 8 ~fear~of
ENTY 14 ~type~of ~kind~of ~sort~of
ENTY 3 ~most~ ~best~ ~largest~ ~biggest~ ~highest~ ~longest~ ~oldest~ ~smallest~ ~fastest~ ~first~ ~only~
ABBR 14 ~stand~for ~stands~for ~abbreviat ~full~form ~acronym
DESC 4 ~mean ~definition ~difference~between ~origin ~define ~explain ~describe ~famous~for ~caused~
LOC 4 ~city~ ~cities~ ~countr ~state~ ~states~ ~capital~ ~continent ~island ~river ~mountain ~lake~ ~ocean ~sea~ ~county~ ~province ~nation~ ~town~ ~place~ ~planet~ ~desert~ ~located~ ~stadium ~librar ~street ~building ~bridge ~cathedral ~hotel ~home~of ~seaport ~airport ~museum ~home~page ~address
NUM 4 ~year ~date~ ~percentage ~population ~number~ ~cost~ ~distance ~height ~temperature ~speed ~day~ ~month~ ~century ~age~ ~price ~weight ~length ~size~ ~salary
HUM 4 ~real~name ~president ~actor ~actress ~singer ~author ~writer ~king~ ~queen~ ~company ~team ~person ~man~ ~woman ~star~ ~character ~artist ~director ~scientist ~leader ~player ~composer ~poet ~inventor~ ~nickname ~pseudonym ~organization ~band~ ~wife ~husband ~son~ ~daughter ~mother ~father ~brother ~minister ~general~ ~explorer ~astronaut ~profession ~manufacturer ~firm~ ~lawyer ~painter ~model~ ~professor ~judge ~comedi ~novelist ~tribe ~spy~ ~senator ~governor ~congressman ~mayor ~doctor ~coach
ENTY 4 ~animal ~color ~book ~movie ~film ~song ~tv~ ~series ~show~ ~sport ~game ~language ~food ~drink ~word~ ~bird ~fish ~dog ~car~ ~instrument ~religion ~disease ~plant ~flower ~tree ~fruit ~magazine ~currency ~term~ ~substance ~element ~product ~brand ~breed ~letter ~novel
"""

EARLY_END = 24  # keywords ending at or before this position count as "near the wh-word"
EARLY_BONUS = {"LOC": 6, "NUM": 6, "HUM": 6, "ENTY": 6}  # extra vote for head-noun rules

KEYWORDS = [(c, float(w), p) for line in RULES.strip().splitlines()
            for c, w, *ps in [line.split()] for p in ps]
# A head-noun rule (weight 4) gets a second unit, marked '<', that also requires EARLY.
KEYWORDS += [(c, EARLY_BONUS[c], "<" + p) for c, w, p in KEYWORDS if w == 4 and c in EARLY_BONUS]
WINDOW = max(len(p) for _, _, p in KEYWORDS)
N_CHANNELS = 38  # 36 characters + start-of-question flag + EARLY flag
OMEGA = 2 * math.pi / 160  # position is a phase; 160 > 2 * sequence length
GAIN = 4e4  # sharpens cos(phase difference) into exact selection of one position


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Per-head queries; keys and values are shared across heads. Heads are concatenated."""

    def __init__(self, d_in: int, n_heads: int, d_key: int, d_val: int, d_out: int | None = None):
        super().__init__()
        self.H, self.dk, self.dv = n_heads, d_key, d_val
        self.W_q = nn.Linear(d_in, n_heads * d_key, bias=False)
        self.W_k = nn.Linear(d_in, d_key, bias=False)
        self.W_v = nn.Linear(d_in, d_val, bias=False)
        self.W_o = nn.Linear(n_heads * d_val, d_out, bias=False) if d_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.H, self.dk).transpose(1, 2)
        k = self.W_k(x)[:, None]
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        out = (attn @ self.W_v(x)[:, None]).transpose(1, 2).reshape(B, T, self.H * self.dv)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_in: int, d_ff: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_ff)
        self.fc2 = nn.Linear(d_ff, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 82):
        super().__init__()
        d_in, n_cls = N_CHANNELS + 2, len(CLASSES)
        self.token_emb = nn.Embedding(vocab_size, d_in)
        self.pos_emb = nn.Embedding(max_seq_len, d_in)
        self.window_attn = CausalSelfAttention(d_in, WINDOW, 2, N_CHANNELS)
        self.keyword_mlp = MLP(WINDOW * N_CHANNELS, len(KEYWORDS), n_cls)
        self.pool_attn = CausalSelfAttention(n_cls, 1, 1, n_cls, n_cls)
        self.head = nn.Linear(n_cls, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        votes = self.keyword_mlp(self.window_attn(h))
        return self.head(self.pool_attn(votes))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    s = task.stoi
    SPACE, START, EARLY, COS, SIN = s[" "], 36, 37, 38, 39
    T = model.pos_emb.num_embeddings

    # Embeddings: one-hot characters; padding and '=' also count as word boundaries.
    for i in range(task.vocab_size):
        model.token_emb.weight[i, i] = 1
    model.token_emb.weight[s["_"], SPACE] = 1
    model.token_emb.weight[s["="], SPACE] = 1
    for t in range(T):
        model.pos_emb.weight[t, COS] = math.cos(OMEGA * t)
        model.pos_emb.weight[t, SIN] = math.sin(OMEGA * t)
        model.pos_emb.weight[t, EARLY] = float(t <= EARLY_END)
    model.pos_emb.weight[0, SPACE] += 1
    model.pos_emb.weight[0, START] = 1

    # Window: head d's query is the position phase rotated back d steps, so it
    # peaks on the key at t-d. Before the start of the text it clamps to position 0.
    a = model.window_attn
    r = math.sqrt(GAIN)
    a.W_k.weight[0, COS] = a.W_k.weight[1, SIN] = r
    for d in range(WINDOW):
        c, n = r * math.cos(OMEGA * d), r * math.sin(OMEGA * d)
        a.W_q.weight[2 * d, COS], a.W_q.weight[2 * d, SIN] = c, n
        a.W_q.weight[2 * d + 1, COS], a.W_q.weight[2 * d + 1, SIN] = -n, c
    a.W_v.weight[:, :N_CHANNELS] = torch.eye(N_CHANNELS)

    # Keyword detectors: unit n fires (=1) iff all of its characters are in place.
    m = model.keyword_mlp
    for n, (cls, w, pat) in enumerate(KEYWORDS):
        chars = pat.lstrip("<")
        for i, ch in enumerate(chars):
            d = len(chars) - 1 - i
            chan = {"~": SPACE, "^": START}.get(ch, s.get(ch))
            m.fc1.weight[n, d * N_CHANNELS + chan] = 1
        if pat.startswith("<"):
            m.fc1.weight[n, EARLY] = 1  # delay-0 slot of the EARLY flag
        m.fc1.bias[n] = -(len(pat) - 1)
        m.fc2.weight[CLASSES.index(cls), n] = w

    # Pool: all-zero keys give uniform attention, so 81 x mean = sum of votes.
    # Prior + 1000 per class (spread over 81 positions) keeps digits above letters.
    for c, name in enumerate(CLASSES):
        m.fc2.bias[c] = (1000 + PRIOR[name]) / 81
        model.head.weight[s[str(c)], c] = 1
    model.pool_attn.W_v.weight[:] = torch.eye(len(CLASSES))
    model.pool_attn.W_o.weight[:] = 81 * torch.eye(len(CLASSES))

    dev_report(model, task)


@torch.no_grad()
def dev_report(model, task, n_show: int = 24) -> None:
    """Rule debugging on the pool minus the eval sample. Eval questions are never printed."""
    def predict(exs):
        out = []
        for i in range(0, len(exs), 256):
            ids = torch.tensor([task.encode(e.prompt) for e in exs[i:i + 256]])
            out += model(ids)[:, -1].argmax(-1).tolist()
        return [task.itos[j] for j in out]

    ev = task.generate_examples(200, seed=0)
    seen, dev = {e.prompt for e in ev}, []
    for e in task.generate_examples(2000, seed=1):
        if e.prompt not in seen:
            seen.add(e.prompt)
            dev.append(e)
    pred = predict(dev)
    ok = sum(p == e.answer for p, e in zip(pred, dev))
    print(f"DEV acc {ok / len(dev):.4f} ({ok}/{len(dev)}); keywords {len(KEYWORDS)}; window {WINDOW}")
    print(f"EVAL-CHECK acc {sum(p == e.answer for p, e in zip(predict(ev), ev)) / len(ev):.4f}")
    conf = torch.zeros(6, 6, dtype=torch.int64)
    for p, e in zip(pred, dev):
        conf[int(e.answer), int(p) if p.isdigit() else 0] += 1
    print("confusion (row=true):", CLASSES)
    print(conf)
    bad = [(p, e) for p, e in zip(pred, dev) if p != e.answer]
    for p, e in bad[::max(1, len(bad) // n_show)]:
        print(f"{CLASSES[int(e.answer)]}->{CLASSES[int(p)]}: {e.prompt.rstrip('_=')}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "SuperlativeV10"
model_description = "V8 (V9's 292 extra units gave only +1 dev question, dropped) plus superlatives voting ENTY: definitions do not use most/largest/first."


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

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
