"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) 3-layer character-level transformer for `nli-snli`.

Algorithm (decided at the '=' position):
  L1 attn : 13 heads, head k copies the token at offset t-k into slot k.
  L1 MLP  : exact word / prefix detectors (preceded by ' ' or '|', followed by
            ' ', '_' or '='), gated by premise / hypothesis region:
              - hypothesis lexicon votes (E, N, C)
              - exclusive-group class indicators (premise and hypothesis)
            plus premise-gated trigram keys and a hypothesis all-letter-trigram flag.
  L2 attn : (A) trigram lookup: each hypothesis trigram attends to an identical
            premise trigram if one exists, else to a sink at '|';
            (B) mean over premise positions (pad count, group classes);
            (C) mean over hypothesis positions (pad count, votes, group classes).
  L2 MLP  : match_t = found AND all-letter; clamp class counts to {0,1}.
  L3 attn : coverage = mean of match_t over all-letter hypothesis trigrams.
  L3 MLP  : soft conflicts: hyp class whose group has a different class in premise.
  Head    : E = 16cov - 6.4 - 0.08(D+12) + 2vE ; N = 8cov - 1.2 + 0.08(D+12) + 2vN ;
            C = 4*conf + 2vC   (conf = 0.5 per conflicting hyp class).
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


# ---------------------------------------------------------------------------
# Lexicons (written from general knowledge, guided by the description)
# ---------------------------------------------------------------------------

# (surface form, is_prefix, E, N, C)  -- raw votes; doubled in the readout
VOTES = []
for w in ["no", "not", "nobody", "never"]:
    VOTES.append((w, False, 0, 0, 6))
for w, p in [("sleep", True), ("asleep", False), ("nap", False), ("naps", False),
             ("napping", False), ("alone", False), ("empty", False), ("cat", False),
             ("cats", False), ("tv", False), ("television", False)]:
    VOTES.append((w, p, 0, 0, 6))
for w in ["naked", "nude", "bed", "couch", "sofa", "home", "swimming", "flying"]:
    VOTES.append((w, False, 0, 0, 3))
for w in ["tall", "sad", "trying", "winning"]:
    VOTES.append((w, False, 0, 4, 0))
for w, p in [("happy", False), ("angry", False), ("upset", False), ("excit", True),
             ("tired", False), ("proud", False), ("scared", False),
             ("mother", False), ("father", False), ("mom", False), ("dad", False),
             ("son", False), ("daughter", False), ("wife", False), ("husband", False),
             ("brother", False), ("sister", False), ("famil", True), ("friend", True),
             ("girlfriend", False), ("boyfriend", False),
             ("compet", True), ("champion", True), ("win", False), ("wins", False),
             ("race", False), ("first", False), ("vacation", False), ("try", False),
             ("tries", False), ("because", False), ("birthday", False)]:
    VOTES.append((w, p, 0, 3, 0))
for w in ["people", "someone", "outside", "animal"]:
    VOTES.append((w, False, 4, 0, 0))

# exclusive groups: group -> list of classes, each class = list of surface forms
GROUPS = {
    "gender": [["man", "men", "guy", "guys", "gentleman"], ["woman", "women", "lady", "ladies"],
               ["boy", "boys"], ["girl", "girls"], ["male"], ["female"]],
    "verb": [["sitting"], ["standing"], ["running"], ["walking"], ["sleeping"], ["eating"],
             ["swimming"], ["dancing"], ["jumping"], ["riding"], ["laying", "lying"],
             ["driving"], ["cooking"], ["reading"], ["singing"], ["talking"],
             ["climbing"], ["skiing"], ["surfing"], ["sleeps"], ["sits"], ["runs"], ["walks"]],
    "animal": [["dog", "dogs"], ["cat", "cats"], ["horse", "horses"], ["bird", "birds"],
               ["cow", "cows"], ["sheep"], ["fish"], ["elephant", "elephants"]],
    "place": [["beach"], ["park"], ["street"], ["city"], ["restaurant"], ["kitchen"],
              ["store"], ["mountain"], ["office"], ["pool"], ["field"], ["school"],
              ["church"], ["library"], ["desert"], ["ocean"], ["lake"], ["river"]],
    "vehicle": [["car"], ["bike", "bicycle"], ["bus"], ["truck"], ["boat"], ["train"],
                ["motorcycle"], ["plane"], ["skateboard"]],
    "colour": [["red"], ["blue"], ["green"], ["black"], ["white"], ["yellow"], ["orange"],
               ["pink"], ["purple"], ["brown"], ["gray", "grey"]],
    "inout": [["inside", "indoors"], ["outside", "outdoors"]],
}
CLASSES = []   # (group, class_idx_in_group, forms)
for g, cl in GROUPS.items():
    for i, forms in enumerate(cl):
        CLASSES.append((g, i, forms))

VOCAB = list("abcdefghijklmnopqrstuvwxyz '_|=012")
LETTERS = "abcdefghijklmnopqrstuvwxyz"
SENT_LEN = 60
NSLOT = 13


class Layout:
    def __init__(self, V, T):
        self.n = 0
        def a(k):
            s = self.n; self.n += k; return s
        nC = len(CLASSES)
        self.nC = nC
        self.TOK = a(V); self.CONST = a(1); self.POS = a(T)
        self.ISP = a(1); self.ISH = a(1); self.ISBAR = a(1)
        self.SLOT = [None] + [a(V) for _ in range(NSLOT)]
        self.TRIGK = a(78); self.ALLLET = a(1); self.VOTE = a(3)
        self.HCNT = a(nC); self.PCNT = a(nC)
        self.FOUND = a(1); self.PADP = a(1); self.PADH = a(1); self.VSUM = a(3)
        self.HSUM = a(nC); self.PSUM = a(nC)
        self.MATCH = a(1); self.HC = a(nC); self.PC = a(nC)
        self.COV = a(1); self.CONF = a(1)
        self.D = self.n
        n_votes = len(set((w, p) for w, p, *_ in VOTES))
        n_forms = sum(len(f) for _, _, f in CLASSES)
        self.ff1 = n_votes + 2 * n_forms + 78 + 1
        self.ff2 = 1 + 4 * nC
        self.ff3 = 2 * nC


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with free per-head q/k and v sizes, no 1/sqrt(d) scaling."""
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
    """Pre-norm-free residual block (no LayerNorm, so hand-set features stay exact)."""
    def __init__(self, d_model, n_heads, d_qk, d_v, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_qk, d_v)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, **_):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        L = Layout(vocab_size, max_seq_len)
        self.L = L
        D = L.D
        self.d_model = D
        self.token_emb = nn.Embedding(vocab_size, D)
        self.pos_emb = nn.Embedding(max_seq_len, D)
        dv2 = 1 + 3 + L.nC
        self.blocks = nn.ModuleList([
            Block(D, NSLOT, max_seq_len, vocab_size, L.ff1),
            Block(D, 3, 79, dv2, L.ff2),
            Block(D, 1, 1, 1, L.ff3),
        ])
        self.head = nn.Linear(D, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    L = model.L
    V, T = model.vocab_size, model.max_seq_len
    st = {c: i for i, c in enumerate(task.vocab)}
    for p in model.parameters():
        p.zero_()
    BETA = 50.0
    P_START, P_END = 0, SENT_LEN            # premise positions [0, 60)
    BAR = SENT_LEN                          # '|' at 60
    H_START, H_END = SENT_LEN + 1, 2 * SENT_LEN + 2   # hyp positions 61..121 (incl. '=')
    n_prem = P_END - P_START
    n_hyp = H_END - H_START

    # ---- embeddings ----
    E = model.token_emb.weight
    for i in range(V):
        E[i, L.TOK + i] = 1.0
        E[i, L.CONST] = 1.0
    PE = model.pos_emb.weight
    for t in range(T):
        PE[t, L.POS + t] = 1.0
        if P_START <= t < P_END:
            PE[t, L.ISP] = 1.0
        if H_START <= t < H_END:
            PE[t, L.ISH] = 1.0
        if t == BAR:
            PE[t, L.ISBAR] = 1.0

    # ---- layer 1 attention: head k copies token at t-k (fallback position 0) ----
    a1 = model.blocks[0].attn
    for k in range(1, NSLOT + 1):
        h = k - 1
        for t in range(T):
            a1.W_q.weight[h * T + max(t - k, 0), L.POS + t] = BETA
            a1.W_k.weight[h * T + t, L.POS + t] = 1.0
        for c in range(V):
            a1.W_v.weight[h * V + c, L.TOK + c] = 1.0
            a1.W_o.weight[L.SLOT[k] + c, h * V + c] = 1.0

    # ---- layer 1 MLP: word detectors, trigram keys, all-letter flag ----
    m1 = model.blocks[0].mlp
    W1, b1, W2 = m1.fc1.weight, m1.fc1.bias, m1.fc2.weight
    n = [0]

    def detector(word, prefix, gate):
        j = n[0]; n[0] += 1
        Lw = len(word)
        assert Lw + 1 <= NSLOT
        req = 0
        if not prefix:
            for c in " _=":
                W1[j, L.TOK + st[c]] = 1.0
            req += 1
        for k in range(1, Lw + 1):
            W1[j, L.SLOT[k] + st[word[Lw - k]]] = 1.0
        req += Lw
        for c in " |":
            W1[j, L.SLOT[Lw + 1] + st[c]] = 1.0
        req += 1
        W1[j, gate] = 1.0
        req += 1
        b1[j] = -(req - 1)
        return j

    vote_tab = {}
    for w, p, e, nn_, c in VOTES:
        key = (w, p)
        if key in vote_tab:
            vote_tab[key] = [x + y for x, y in zip(vote_tab[key], (e, nn_, c))]
        else:
            vote_tab[key] = [e, nn_, c]
    for (w, p), vals in vote_tab.items():
        j = detector(w, p, L.ISH)
        for d in range(3):
            W2[L.VOTE + d, j] = float(vals[d])
    for ci, (_, _, forms) in enumerate(CLASSES):
        for f in forms:
            j = detector(f, False, L.ISH)
            W2[L.HCNT + ci, j] = 1.0
            j = detector(f, False, L.ISP)
            W2[L.PCNT + ci, j] = 1.0
    # premise-gated trigram key: [slot2 letter | slot1 letter | current letter]
    srcs = [L.SLOT[2], L.SLOT[1], L.TOK]
    for part in range(3):
        for li, ch in enumerate(LETTERS):
            j = n[0]; n[0] += 1
            W1[j, srcs[part] + st[ch]] = 1.0
            W1[j, L.ISP] = 1.0
            b1[j] = -1.0
            W2[L.TRIGK + part * 26 + li, j] = 1.0
    # hypothesis all-letter trigram flag
    j = n[0]; n[0] += 1
    for s in srcs:
        for ch in LETTERS:
            W1[j, s + st[ch]] = 1.0
    W1[j, L.ISH] = 1.0
    b1[j] = -3.0
    W2[L.ALLLET, j] = 1.0
    assert n[0] == L.ff1, (n[0], L.ff1)

    # ---- layer 2 attention ----
    a2 = model.blocks[1].attn
    dqk, dv = 79, a2.d_v
    # head 0: trigram lookup
    for part in range(3):
        for li, ch in enumerate(LETTERS):
            a2.W_q.weight[part * 26 + li, srcs[part] + st[ch]] = BETA
            a2.W_k.weight[part * 26 + li, L.TRIGK + part * 26 + li] = 1.0
    a2.W_q.weight[78, L.CONST] = BETA
    a2.W_k.weight[78, L.ISBAR] = 2.5
    a2.W_v.weight[0, L.ISP] = 1.0
    a2.W_o.weight[L.FOUND, 0] = 1.0
    # head 1: mean over premise positions
    a2.W_q.weight[dqk + 0, L.CONST] = BETA
    a2.W_k.weight[dqk + 0, L.ISP] = 1.0
    a2.W_v.weight[dv + 0, L.TOK + st["_"]] = 1.0
    a2.W_o.weight[L.PADP, dv + 0] = float(n_prem)
    for ci in range(L.nC):
        a2.W_v.weight[dv + 1 + ci, L.PCNT + ci] = 1.0
        a2.W_o.weight[L.PSUM + ci, dv + 1 + ci] = float(n_prem)
    # head 2: mean over hypothesis positions
    a2.W_q.weight[2 * dqk + 0, L.CONST] = BETA
    a2.W_k.weight[2 * dqk + 0, L.ISH] = 1.0
    a2.W_v.weight[2 * dv + 0, L.TOK + st["_"]] = 1.0
    a2.W_o.weight[L.PADH, 2 * dv + 0] = float(n_hyp)
    for d in range(3):
        a2.W_v.weight[2 * dv + 1 + d, L.VOTE + d] = 1.0
        a2.W_o.weight[L.VSUM + d, 2 * dv + 1 + d] = float(n_hyp)
    for ci in range(L.nC):
        a2.W_v.weight[2 * dv + 4 + ci, L.HCNT + ci] = 1.0
        a2.W_o.weight[L.HSUM + ci, 2 * dv + 4 + ci] = float(n_hyp)

    # ---- layer 2 MLP: match flag, clamp class counts ----
    m2 = model.blocks[1].mlp
    W1, b1, W2 = m2.fc1.weight, m2.fc1.bias, m2.fc2.weight
    W1[0, L.FOUND] = 1.0; W1[0, L.ALLLET] = 1.0; b1[0] = -1.0
    W2[L.MATCH, 0] = 1.0
    j = 1
    for src, dst in [(L.HSUM, L.HC), (L.PSUM, L.PC)]:
        for ci in range(L.nC):
            W1[j, src + ci] = 1.0
            W1[j + 1, src + ci] = 1.0; b1[j + 1] = -1.0
            W2[dst + ci, j] = 1.0; W2[dst + ci, j + 1] = -1.0
            j += 2

    # ---- layer 3 attention: coverage ----
    a3 = model.blocks[2].attn
    a3.W_q.weight[0, L.CONST] = BETA
    a3.W_k.weight[0, L.ALLLET] = 1.0
    a3.W_v.weight[0, L.MATCH] = 1.0
    a3.W_o.weight[L.COV, 0] = 1.0

    # ---- layer 3 MLP: soft conflicts ----
    m3 = model.blocks[2].mlp
    W1, b1, W2 = m3.fc1.weight, m3.fc1.bias, m3.fc2.weight
    for ci, (g, _, _) in enumerate(CLASSES):
        K = float(len(GROUPS[g]))
        for jj in (2 * ci, 2 * ci + 1):
            W1[jj, L.HC + ci] = K
            for cj, (g2, _, _) in enumerate(CLASSES):
                if g2 == g and cj != ci:
                    W1[jj, L.PC + cj] = 1.0
        b1[2 * ci] = -K
        b1[2 * ci + 1] = -K - 1.0
        W2[L.CONF, 2 * ci] = 0.5
        W2[L.CONF, 2 * ci + 1] = -0.5

    # ---- readout ----
    Hd = model.head.weight
    for i in range(V):
        Hd[i, L.CONST] = -1000.0
    e, nu, c = st["0"], st["1"], st["2"]
    # Delta = len_h - len_p = PADP - PADH
    Hd[e, L.CONST] = -6.4 - 0.08 * 12
    Hd[e, L.COV] = 16.0
    Hd[e, L.PADP] = -0.08; Hd[e, L.PADH] = 0.08
    Hd[e, L.VSUM + 0] = 2.0
    Hd[nu, L.CONST] = -1.2 + 0.08 * 12
    Hd[nu, L.COV] = 8.0
    Hd[nu, L.PADP] = 0.08; Hd[nu, L.PADH] = -0.08
    Hd[nu, L.VSUM + 1] = 2.0
    Hd[c, L.CONST] = 0.0
    Hd[c, L.CONF] = 4.0
    Hd[c, L.VSUM + 2] = 2.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "nli_lexcov_v1"
model_description = ("3-layer char transformer: 13 offset heads, MLP word/prefix detectors for hyp lexicon votes "
                     "and exclusive-group classes, trigram-coverage attention, soft conflicts, length delta; "
                     "linear logits per description.")


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
