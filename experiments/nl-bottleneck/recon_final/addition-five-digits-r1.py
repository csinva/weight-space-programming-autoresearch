"""
Interpretable transformer for character-level sequence tasks.

Task: 5-digit addition (`addition-five-digits`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.
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

class SimpleTransformer(nn.Module):
    """One cross-attention read; the sum is an angle on a ten-sector ring.

    Parameters (27 total, float64):
      * pos_emb: (13, 2) table. A fixed (non-learned) index maps each sequence
        position to a row: digits a_c and b_c (c = 0 is the most significant
        digit) share row c, '+' uses row 5 (origin = sink), positions 11..16
        (the positions that predict output digit i = 0..5) use rows 6..11, and
        the final answer position uses row 12 (never used for a prediction).
        The same 2-D vector serves as query and key.
      * value_scale: (1,) scalar; value of a digit token = digit * value_scale.
    Readout: logit_d = cos(z - 2*pi/10*(d + 0.5 - 5e-6)); non-digit tokens masked.
    """

    N_KEYS = 11  # prompt positions 0..10 are the only attendable keys

    def __init__(self, vocab_size: int, max_seq_len: int = 32, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        dt = torch.float64
        self.pos_emb = nn.Parameter(torch.zeros(13, 2, dtype=dt))
        self.value_scale = nn.Parameter(torch.zeros(1, dtype=dt))
        # fixed position -> row index (structure, not parameters)
        rows = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4] + [6 + i for i in range(6)] + [12]
        rows = rows + [12] * max(0, max_seq_len - len(rows))
        self.register_buffer("pos_row", torch.tensor(rows[:max_seq_len], dtype=torch.long))
        # token -> digit value (non-digit tokens have value 0)
        dv = torch.zeros(vocab_size, dtype=dt)
        dmask = torch.zeros(vocab_size, dtype=torch.bool)
        for t in range(min(10, vocab_size)):
            dv[t] = float(t)
            dmask[t] = True
        self.register_buffer("digit_val", dv)
        self.register_buffer("is_digit", dmask)
        # readout sector centres (fixed by the formula)
        phi = torch.zeros(vocab_size, dtype=dt)
        for d in range(10):
            phi[d] = 2 * math.pi / 10 * (d + 0.5 - 5e-6)
        self.register_buffer("phi", phi)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        p = self.pos_emb[self.pos_row[:T]]                  # (T, 2)
        nk = min(self.N_KEYS, T)
        k = p[:nk]                                          # (nk, 2) prompt keys only
        v = self.digit_val[ids[:, :nk]] * self.value_scale  # (B, nk)
        scores = p @ k.T                                    # (T, nk)
        attn = scores.softmax(dim=-1)                       # (T, nk)
        z = v @ attn.T                                      # (B, T)
        logits = torch.cos(z[..., None] - self.phi)         # (B, T, V)
        logits = logits.masked_fill(~self.is_digit, float("-inf"))
        return logits


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Closed-form weights (no training)."""
    ln10 = math.log(10.0)
    S = sum(10.0 ** (-c) for c in range(5))  # per-number sum of weights / t
    with torch.no_grad():
        P = torch.zeros(13, 2, dtype=torch.float64)
        for c in range(5):
            P[c] = torch.tensor([1.0, -c * ln10], dtype=torch.float64)
        P[5] = 0.0  # '+' sink at origin: score 0, value 0
        for i in range(6):
            # weights t*10^-c / (1 + 2*S*t); want t*1e-4/(1+2*S*t) = 1e-5/10^(5-i)
            r = 1e-5 / 10.0 ** (5 - i)
            t = r / (1e-4 - 2 * S * r)
            P[6 + i] = torch.tensor([math.log(t), 1.0], dtype=torch.float64)
        model.pos_emb.copy_(P)
        model.value_scale.fill_(2 * math.pi / 10 / 1e-5)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "RingAngleCrossAttn27"
model_description = ("Single cross-attention read over prompt positions; 2-D pos-embedding keys (1,-c ln10), "
                     "queries (ln t_i,1), '+' sink; value scale 2pi/10/1e-5 turns the shifted sum into an angle; "
                     "cosine sector readout; float64, 27 params.")


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
