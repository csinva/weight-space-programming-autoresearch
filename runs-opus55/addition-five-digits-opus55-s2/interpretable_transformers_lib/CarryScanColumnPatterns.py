"""Hand-written transformer for 5-digit addition. No training anywhere."""

from __future__ import annotations

import argparse
import csv
import itertools
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
    """One head of cross-attention to a slice of the prompt."""

    def __init__(self, d_model: int, start: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.start, self.stop = start, 11   # token range this head may read

    def forward(self, keys: torch.Tensor, values: torch.Tensor, salience=None) -> torch.Tensor:
        T, lo, hi = keys.shape[1], self.start, self.stop
        scores = self.W_q(keys) @ keys[:, lo:hi].transpose(-2, -1)
        if salience is not None:    # content term: how much each token wants to be read
            scores = scores + salience[:, lo:hi].transpose(-2, -1)
        mask = torch.triu(torch.ones(T, hi, dtype=torch.bool, device=keys.device), diagonal=1)[:, lo:hi]
        return scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ values[:, lo:hi]


class SimpleTransformer(nn.Module):
    """Layer 1 sums and classifies each column; layer 2 finds the decisive one."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 6):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, 1)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.pair_attn = CausalSelfAttention(d_model, start=0)
        self.code_fc = nn.Linear(1, 3)      # column sum -> ramps at 8, 9 and 10
        self.code_out = nn.Linear(3, 2)     # -> (generate, decisive)
        self.column_attn = CausalSelfAttention(d_model, start=5)
        self.scan_attn = CausalSelfAttention(d_model, start=5)
        self.mix = nn.Linear(2, 1, bias=False)
        self.wrap_fc = nn.Linear(1, 2)      # two ReLUs detect z >= 10
        self.fold = nn.Linear(3, 1, bias=False)
        self.head = nn.Linear(1, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        keys = self.pos_emb(torch.arange(T, device=ids.device))[None].expand(B, -1, -1)
        sums = self.pair_attn(keys, self.token_emb(ids))    # at b_c: a_c + b_c
        codes = self.code_out(F.relu(self.code_fc(sums)))
        generate, decisive = codes[..., [0]], codes[..., [1]]
        column = self.column_attn(keys, sums)
        carry = self.scan_attn(keys, generate, salience=decisive)   # copy the first decisive g
        z = self.mix(torch.cat([column, carry], dim=-1))    # column sum, 0 to 19
        wraps = F.relu(self.wrap_fc(z))
        return self.head(self.fold(torch.cat([z, wraps], dim=-1)))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    # Circuit: a carry scan, in float16. A column is decisive unless its sum is 9,
    # and a carry reaches column j exactly when the first decisive column below j
    # generates (sum >= 10). Attention does the scan: every column below j scores
    # DECIDE if decisive, less NEAR per step of distance, so the softmax lands on
    # the nearest decisive column and copies its generate bit. No thresholds, no
    # place-value arithmetic, nothing fine-grained for low precision to lose.
    model.half()
    for p in model.parameters():
        p.data.zero_()

    KEY_SINK = 5
    OFF = -80.0     # log-weight of anything that must be ignored
    DECIDE = 60.0   # bonus for a decisive column
    NEAR = 8.0      # penalty per column of distance

    column_table = torch.full((6, 6), OFF, dtype=torch.float64)
    scan_table = torch.full((6, 6), OFF, dtype=torch.float64)
    column_table[0, KEY_SINK] = 0.0     # step 0 has no column: it is only the carry
    for i in range(6):
        j = i - 1
        if j >= 0:
            column_table[i, j] = 0.0    # read the column sum stored at b_j
        for k in range(j + 1, 5):
            scan_table[i, k] = -NEAR * (k - j)
        scan_table[i, KEY_SINK] = -NEAR * 6     # '+' answers "no carry" if all propagate

    pos, tok = model.pos_emb.weight.data, model.token_emb.weight.data
    for c in range(5):
        pos[c, c] = pos[6 + c, c] = 1.0     # a_c and b_c share the key of column c
    pos[5, KEY_SINK] = 1.0
    for i in range(6):
        pos[11 + i, i] = 1.0                # step i reuses key dim i as its query
    for d in range(10):
        tok[d, 0] = 2.0 * d     # the pair head averages two tokens, so this gives a + b

    dtype = model.head.weight.dtype
    model.pair_attn.W_q.weight.data[:] = 20.0 * torch.eye(6, dtype=dtype)   # same column key
    model.column_attn.W_q.weight.data[:] = column_table.T.to(dtype)
    model.scan_attn.W_q.weight.data[:] = scan_table.T.to(dtype)

    # Ramps r8, r9, r10. generate = r9 - r10 is the step at 10, and propagate is the
    # bump r8 - 2 r9 + r10 at 9, so decisive = DECIDE (1 - propagate). The '+' token
    # has sum 0, so it counts as decisive too, but only ever wins when nothing else is.
    model.code_fc.weight.data[:, 0] = 1.0
    model.code_fc.bias.data[:] = torch.tensor([-8.0, -9.0, -10.0], dtype=dtype)
    model.code_out.weight.data[0] = torch.tensor([0.0, 1.0, -1.0], dtype=dtype)
    model.code_out.weight.data[1] = DECIDE * torch.tensor([-1.0, 2.0, -1.0], dtype=dtype)
    model.code_out.bias.data[1] = DECIDE

    model.mix.weight.data[0, 0] = 1.0
    model.mix.weight.data[0, 1] = 1.0

    # Mod 10: relu(z - 9) - relu(z - 10) is 0 up to z = 9 and 1 from z = 10 on.
    model.wrap_fc.weight.data[:, 0] = 1.0
    model.wrap_fc.bias.data[:] = torch.tensor([-9.0, -10.0], dtype=dtype)
    model.fold.weight.data[0] = torch.tensor([1.0, -10.0, 10.0], dtype=dtype)

    # Head: linear templates 2 d R - d^2, whose argmax is the nearest integer d.
    for d in range(10):
        model.head.weight.data[d, 0] = 2.0 * d
        model.head.bias.data[d] = -float(d * d)
    model.head.bias.data[10:] = -1e4    # '+' and '=' are never written

    _stress_test(model)


def _stress_test(model, n_random: int = 50000) -> None:
    """Check the model on random pairs plus hand-picked long carry chains."""
    gen = torch.Generator().manual_seed(1)
    a = torch.randint(0, 100000, (n_random,), generator=gen).tolist()
    b = torch.randint(0, 100000, (n_random,), generator=gen).tolist()
    edges = [0, 1, 9, 10, 99, 100, 999, 1000, 9999, 10000, 49999, 50000, 50001, 89999, 90000, 90001, 99990, 99998, 99999]
    for p in edges:
        for q in edges:
            a.append(p)
            b.append(q)
    for s in (9999, 10000, 19999, 99999, 100000, 150000, 199998):   # complements
        for p in range(max(0, s - 99999), min(s, 99999) + 1, 997):
            a.append(p)
            b.append(s - p)
    for sums in itertools.product((0, 7, 8, 9, 10, 11, 18), repeat=5):  # column patterns
        p, q = 0, 0
        for s in sums:
            d = int(torch.randint(max(0, s - 9), min(s, 9) + 1, (1,), generator=gen))
            p, q = 10 * p + d, 10 * q + s - d
        a.append(p)
        b.append(q)
    wrong, shown = 0, []
    with torch.no_grad():
        for start in range(0, len(a), 4096):
            aa, bb = a[start:start + 4096], b[start:start + 4096]
            ids = torch.tensor([[int(c) for c in f"{p:05d}"] + [10] + [int(c) for c in f"{q:05d}"] + [11]
                                for p, q in zip(aa, bb)])
            for _ in range(6):
                ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
            for row, p, q in zip(ids[:, 12:].tolist(), aa, bb):
                if "".join(map(str, row)) != f"{p + q:06d}":
                    wrong += 1
                    if len(shown) < 5:
                        shown.append((p, q, row))
    print(f"STRESS: {wrong} wrong of {len(a)}; examples: {shown}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "CarryScanColumnPatterns"
model_description = "CarryScanAttnFloat16 rechecked on all 7^5 column-sum patterns over {0,7,8,9,10,11,18}, each randomly split into digits, which exercises every propagate chain."


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
