"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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
    def __init__(self, d_model: int, n_heads: int, d_key: int = 1):
        super().__init__()
        self.n_heads = n_heads  # each head has a d_key-dim query and key and one scalar value
        self.d_key = d_key
        self.W_q = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_v = nn.Linear(d_model, n_heads, bias=False)
        self.W_o = nn.Linear(n_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, self.d_key).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.d_key).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, 1).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_key: int = 1):
        super().__init__()
        self.ln1 = nn.Identity()  # LayerNorm would rescale the numeric slots
        self.attn = CausalSelfAttention(d_model, n_heads, d_key)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff) if d_ff else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 12,
        n_heads: int = 1,
        n_layers: int = 2,
        d_ff: int = 18,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # (heads, key size) per layer: read a and b, multiply, form the remainder, look up the digit
        specs = [(2, 1), (2, 2), (1, 1), (1, 2)]
        self.blocks = nn.ModuleList([Block(d_model, h, 0, k) for h, k in specs])
        self.final_ln = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Schoolbook multiplication followed by long division, written in closed form.

    Slots of the hidden vector (each has one human meaning):
      D      numeric value of the token
      KA     key of a: log place value on a's cells, far negative on '*'.
             Also its query (-KA/30): +1 on b's cells, -1 from '=' on.
             Constant 30 on every cell that predicts, so KA/30 acts as the bias
      KB     key of b: log place value on b's cells, far negative on '*'
      MK     +60 on cells 10 and 11, the pair the multiplier attends to; -60 elsewhere.
             Also the query of b (MK/60): +1 on '=', -1 on answer cells
      KR     remainder key: ballast on '*', 0 on '=', log place value on answer cells
      ZETA   TAU / Z, the place value being predicted on the remainder's scale
      A, B   a on cell 10 (0 on '=', answer cells); b on '=' (0 on answer cells)
      V      a*b on '=', -d on answer cells: signed contribution to the remainder
      O      next digit in the units place, with R / (1000 Z) below it as a small fraction
      TM,TQ  digit table: cell m stores m and -m^2/2. TM is also +1 / -1 on cells 10 / 11
    Layer 1: two heads read a and b. Softmax is the only nonlinearity in the model.
    Layer 2: two heads multiply through the bilinear attention score.
    Layer 3: one head sums V, giving the remainder.
    Layer 4: one head looks up the digit in the table.
    """
    torch.manual_seed(0)
    model.double()
    for p in model.parameters():
        nn.init.zeros_(p)

    D, KA, KB, MK, KR, ZETA, A, B, V, O, TM, TQ = range(12)
    EPS2 = 1e-15   # sensitivity test: s reaches only 1e-5, a hundredth of the working value
    EPS = math.sqrt(EPS2)
    SHRINK = 1e-3  # keeps R/Z well below 1 so it cannot disturb the digit in the same slot
    SHARP = 2e13   # digit lookup sharpness: half an integer of R is a score gap of at least 40
    LOG10 = math.log(10.0)

    te, pe = model.token_emb.weight.data, model.pos_emb.weight.data
    for d in range(10):
        te[d, D] = d
        te[d, V] = -d
    z = 1.0 + 1e9  # softmax normalizer: 1 for '=' plus the ballast on '*'
    for t in range(pe.shape[0]):
        pe[t, KA] = (4 - t) * LOG10 if t <= 4 else {5: -120.0}.get(t, -30.0 if t <= 10 else 30.0)
        pe[t, KB] = (10 - t) * LOG10 if 6 <= t <= 10 else (-120.0 if t == 5 else -60.0)
        # Cell 0 is lowest: the '*' cell's query is negative, so it lands there, reads table
        # value 0, and keeps V = 0, which the ballast on '*' relies on.
        pe[t, MK] = 60.0 if t in (10, 11) else (-120.0 if t == 0 else -60.0)
        pe[t, KR] = {5: 9 * LOG10, 11: 0.0}.get(t, (21 - t) * LOG10 if 12 <= t <= 20 else -60.0)
        if 11 <= t <= 20:
            if t > 11:
                z += 10.0 ** (21 - t)
            pe[t, ZETA] = SHRINK * 10.0 ** (20 - t) / z
            pe[t, O] = SHRINK * 0.5 / z  # half an integer of R, so a tie falls on the upper digit
        if t <= 9:  # the digit table: cell m stores m and -m^2/2
            pe[t, TM] = t
            pe[t, TQ] = -t * t / 2.0
    # Cells 10 and 11 hold the multiplier's +1 and -1. Cell 10 duplicates table entry 1;
    # cell 11 acts as digit -1, whose score is below digit 0, so it never wins.
    pe[10, TM], pe[11, TM] = 1.0, -1.0
    pe[10, TQ], pe[11, TQ] = -0.5, -0.5

    b1, b2, b3, b4 = model.blocks

    # Layer 1: scores log(10^(4-i)) make softmax weights 10^(4-i)/11111, so each head's
    # output is the number / 11111. A flipped query lands on '*', whose D is 0.
    b1.attn.W_q.weight.data[0, KA] = -1.0 / 30.0
    b1.attn.W_k.weight.data[0, KA] = 1.0
    b1.attn.W_v.weight.data[0, D] = 1.0
    b1.attn.W_o.weight.data[A, 0] = 11111.0
    b1.attn.W_q.weight.data[1, MK] = 1.0 / 60.0
    b1.attn.W_k.weight.data[1, KB] = 1.0
    b1.attn.W_v.weight.data[1, D] = 1.0
    b1.attn.W_o.weight.data[B, 1] = 11111.0

    # Layer 2: the first key dimension is the bilinear product, s = EPS2*a*b on cell 10 and 0
    # on cell 11; the second restricts attention to those two cells. Values +1 / -1 make the
    # head output tanh(s/2). Head 1 doubles s, and (8 tanh(s/2) - tanh(s)) / 3 = s + O(s^5)
    # cancels the cubic error. On answer cells b = 0, so V keeps just -d.
    for head, gain in ((0, 1.0), (1, 2.0)):
        b2.attn.W_q.weight.data[2 * head, B] = gain * EPS
        b2.attn.W_k.weight.data[2 * head, A] = EPS
        b2.attn.W_q.weight.data[2 * head + 1, KA] = 1.0 / 30.0
        b2.attn.W_k.weight.data[2 * head + 1, MK] = 1.0
        b2.attn.W_v.weight.data[head, TM] = 1.0
    b2.attn.W_o.weight.data[V, 0] = 8.0 / (3.0 * EPS2)
    b2.attn.W_o.weight.data[V, 1] = -1.0 / (3.0 * EPS2)

    # Layer 3: a log-place-value head sums V over '=' and the emitted digits, with a ballast
    # on '*' whose V is 0, so O = R / Z on the shrunken scale.
    b3.attn.W_q.weight.data[0, KA] = 1.0 / 30.0
    b3.attn.W_k.weight.data[0, KR] = 1.0
    b3.attn.W_v.weight.data[0, V] = 1.0
    b3.attn.W_o.weight.data[O, 0] = SHRINK

    # Layer 4: score_m = SHARP * (m * (O - ZETA/2) - m^2/2 * ZETA) is a parabola in m whose
    # peak is floor(O / ZETA), so the head reads that digit from the table in cells 0..9.
    # Cells past the table have zero keys and value 0, so they just tie with digit 0.
    b4.attn.W_q.weight.data[0, O] = SHARP
    b4.attn.W_q.weight.data[0, ZETA] = -SHARP / 2.0
    b4.attn.W_q.weight.data[1, ZETA] = SHARP
    b4.attn.W_k.weight.data[0, TM] = 1.0
    b4.attn.W_k.weight.data[1, TQ] = 1.0
    b4.attn.W_v.weight.data[0, TM] = 1.0
    b4.attn.W_o.weight.data[O, 0] = 1.0

    # Head: logit_c = c*d - c^2/2, tangents to the parabola d^2/2, so argmax_c = d.
    h = model.head.weight.data
    for c in range(10):
        h[c, O] = float(c)
        h[c, KA] = -c * c / 60.0
    h[10, KA] = h[11, KA] = -100.0  # '*' and '=' are never emitted

    stress_test(model, task)


def stress_test(model, task, n: int = 300000) -> None:
    """Sanity check only (does not touch the weights): greedy decode on random and edge pairs."""
    g = torch.Generator().manual_seed(123)
    a = torch.randint(0, 100000, (n,), generator=g).tolist()
    b = torch.randint(0, 100000, (n,), generator=g).tolist()
    edge = [0, 1, 9, 10, 99, 100, 9999, 10000, 50000, 99990, 99999]
    pairs = list(zip(a, b)) + [(x, y) for x in edge for y in edge]
    ids = torch.tensor([task.encode(f"{x:05d}*{y:05d}=") for x, y in pairs])
    model.eval()
    with torch.no_grad():
        for _ in range(task.answer_len):
            next_id = model(ids)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    outputs = [task.decode(row[task.prompt_len:].tolist()) for row in ids]
    wrong = [(x, y, o) for (x, y), o in zip(pairs, outputs) if o != f"{x * y:010d}"]
    print(f"STRESS {len(pairs) - len(wrong)}/{len(pairs)} correct; first failures: {wrong[:5]}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "e21_eps2_1e-15_sensitivity"
model_description = "Sensitivity test on e18: EPS2 lowered to 1e-15, where rounding in the softmax weights is predicted to reach about half an integer of the product."


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
