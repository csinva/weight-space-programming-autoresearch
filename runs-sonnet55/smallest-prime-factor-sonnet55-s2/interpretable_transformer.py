"""
Interpretable transformer for character-level sequence tasks.

Task: smallest-prime-factor. prompt "64507=" -> answer "00251" (spf, zero padded;
a prime maps to itself).

Circuit (no training, no LayerNorm): a "scan over primes in groups" design.
  Block 0 attention: heads 0-4 read digit i of N into DIG_i; head 5 (query pos 5+j) copies digit j into CV.
  For each group of GROUP consecutive primes (in increasing order), two MLP-only blocks:
    bump block  : for each prime p, s_p = sum_i w_i DIG_i (w_i = u*10^(4-i) mod p); 3 relu neurons per multiple
                  k*p form a bump at s_p = k*p, written into the shared channels D_0..D_{g-1}  (D_i = [p_i | N]).
    select block: F_i = relu(D_i - sum_{q<i} D_q - G)  (first dividing prime of the group, only if none found yet);
                  A_j += digit_j(p_i) * F_i ; G += F_i ; D_i is cleared so the channels can be reused.
  Final MLP block: at output position 5+j gated bump neurons turn the scalar A_j into a one-hot logit; if G=0
                   (nothing <=313 divides N, so N is prime) the copied digit CV is written instead.
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
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int = 0, d_head: int = 5):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head) if n_heads > 0 else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


def _primes(n):
    return [i for i in range(2, n + 1) if all(i % q for q in range(2, int(i ** 0.5) + 1))]


PRIMES = _primes(313)  # every composite <= 99999 has a prime factor <= 313
GROUP = 4     # primes handled per (bump block, select block) pair
GROUPS = [PRIMES[i:i + GROUP] for i in range(0, len(PRIMES), GROUP)]


def _layout():
    names = {}
    n = 0

    def add(key, size):
        nonlocal n
        names[key] = n
        n += size
    add("VAL", 1); add("P", 10); add("DIG", 5); add("CV", 1); add("D", GROUP)
    names["G"] = names["VAL"]  # found-flag G reuses VAL, which block 0's MLP clears once attention has read it
    names["A"] = names["P"]  # answer scalars A_0..4 live in P_0..4, which are zero at output positions (>=5)
    # one-hot logit channels OUT_1..9 reuse the (cleared) D channels; digit 0 is the default logit
    # beyond the GROUP D channels, OUT_c also reuses DIG_0..4 (cleared by the last select block), then fresh channels
    scratch = [names["D"] + i for i in range(GROUP)] + [names["DIG"] + i for i in range(5)]
    extra = max(0, 9 - len(scratch))
    add("OUTX", extra)
    scratch += [names["OUTX"] + i for i in range(extra)]
    names["OUT"] = {c: scratch[c - 1] for c in range(1, 10)}
    return names, n


LAY, D_MODEL = _layout()


def _bumps(p):
    """(weight vector, k list) detecting 'p divides sum_i w_i d_i' for 5 digits; unit u chosen for fewest bumps."""
    best = None
    for u in range(1, p):
        w = []
        for i in range(5):
            r = (u * pow(10, 4 - i, p)) % p
            if r > p // 2:
                r -= p
            w.append(r)
        lo = 9 * sum(x for x in w if x < 0)
        hi = 9 * sum(x for x in w if x > 0)
        ks = list(range(-(-lo // p), hi // p + 1))
        if best is None or len(ks) < len(best[1]):
            best = (w, ks)
    return best


BUMPS = {p: _bumps(p) for p in PRIMES}


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        d = D_MODEL
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_seq_len, d)
        blocks = []
        for t, grp in enumerate(GROUPS):
            n_bump = sum(len(BUMPS[p][1]) for p in grp)
            blocks.append(Block(d, 3 * n_bump + (1 if t == 0 else 0), n_heads=6 if t == 0 else 0))  # bump block
            last = t == len(GROUPS) - 1
            blocks.append(Block(d, 2 * len(grp) + (5 if last else 0)))     # select block (last one also clears DIG)
        blocks.append(Block(d, 5 * 9 * 3 + 9 * 3))                          # decode block
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(d, vocab_size, bias=False)

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

def write_weights(model: SimpleTransformer, task) -> None:
    L = LAY
    BUMP = ((1, 1.0), (0, -2.0), (-1, 1.0))  # relu(x+1) - 2 relu(x) + relu(x-1) == [x == 0] on integers
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()
        for c in range(10):
            model.token_emb.weight[c, L["VAL"]] = float(c)
        for pos in range(model.max_seq_len):
            if pos < 10:
                model.pos_emb.weight[pos, L["P"] + pos] = 1.0

        # ---- attention (block 0)
        at = model.blocks[0].attn
        B = 80.0
        dh = at.d_head
        for h in range(5):  # head h reads input digit h
            for i in range(10):  # constant query 1 == sum of the position one-hot
                at.W_q.weight[h * dh, L["P"] + i] = 1.0
            at.W_k.weight[h * dh, L["P"] + h] = B
            at.W_v.weight[h * dh, L["VAL"]] = 1.0
            at.W_o.weight[L["DIG"] + h, h * dh] = 1.0
        for j in range(5):  # head 5: query pos 5+j looks at pos j
            at.W_q.weight[5 * dh + j, L["P"] + 5 + j] = 1.0
            at.W_k.weight[5 * dh + j, L["P"] + j] = B
        at.W_v.weight[5 * dh, L["VAL"]] = 1.0
        at.W_o.weight[L["CV"], 5 * dh] = 1.0

        # ---- per group: bump block + select block
        for t, grp in enumerate(GROUPS):
            m = model.blocks[2 * t].mlp
            r = 0
            for gi, p in enumerate(grp):
                w, ks = BUMPS[p]
                for k in ks:
                    for off, coef in BUMP:
                        for i in range(5):
                            m.fc1.weight[r, L["DIG"] + i] = float(w[i])
                        m.fc1.bias[r] = float(-k * p + off)
                        m.fc2.weight[L["D"] + gi, r] = coef
                        r += 1
            if t == 0:  # clear VAL (digit values >= 0) so that channel can serve as the flag G
                m.fc1.weight[r, L["VAL"]] = 1.0
                m.fc2.weight[L["VAL"], r] = -1.0
            m = model.blocks[2 * t + 1].mlp
            for gi, p in enumerate(grp):
                # first dividing prime of the group, and nothing found earlier
                m.fc1.weight[gi, L["D"] + gi] = 1.0
                for q in range(gi):
                    m.fc1.weight[gi, L["D"] + q] = -1.0
                m.fc1.weight[gi, L["G"]] = -1.0
                digs = f"{p:05d}"
                for j in range(5):
                    m.fc2.weight[L["A"] + j, gi] = float(digs[j])
                m.fc2.weight[L["G"], gi] = 1.0
                # clear D_gi (it is 0/1) so the channel can be reused
                m.fc1.weight[len(grp) + gi, L["D"] + gi] = 1.0
                m.fc2.weight[L["D"] + gi, len(grp) + gi] = -1.0
            if t == len(GROUPS) - 1:  # DIG is no longer needed: clear it so DIG_i can carry an OUT logit
                for i in range(5):
                    m.fc1.weight[2 * len(grp) + i, L["DIG"] + i] = 1.0
                    m.fc2.weight[L["DIG"] + i, 2 * len(grp) + i] = -1.0

        # ---- decode block
        m = model.blocks[-1].mlp
        M = 10.0
        r = 0
        for j in range(5):  # output position 5+j: scalar A_j -> one-hot
            for c in range(1, 10):
                for off, coef in BUMP:
                    m.fc1.weight[r, L["A"] + j] = 1.0
                    m.fc1.weight[r, L["P"] + 5 + j] = M
                    m.fc1.bias[r] = float(-c + off - M)
                    m.fc2.weight[L["OUT"][c], r] = coef
                    r += 1
        for c in range(1, 10):  # N prime (G=0): copy the input digit
            for off, coef in BUMP:
                m.fc1.weight[r, L["CV"]] = 1.0
                m.fc1.weight[r, L["G"]] = -M
                m.fc1.bias[r] = float(-c + off)
                m.fc2.weight[L["OUT"][c], r] = coef
                r += 1

        for c in range(1, 10):
            model.head.weight[c, L["OUT"][c]] = 1.0
        for i in range(5, 10):  # default logit for digit 0 (wins only if no other digit fires)
            model.head.weight[0, L["P"] + i] = 0.5


model_shorthand_name = "GroupScan4_Final"
model_description = ("Final: scan the 65 primes<=313 in groups of 4 (bump block + select block each, d_model 21, 35 blocks). Bump neurons detect "
                     "p | sum w_i d_i; select picks the first divisor not yet found into scalars A_j (stored in P_0..4), G flag in VAL; "
                     "decode block turns A_j or the copied digit into logits. B=80, M=10. Seed 6, 5000 samples.")


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
