"""
Interpretable transformer for date normalization. All weights are hand-written; nothing is trained.

Tokens are a small readable code: a 4-bit nibble (a digit's value, or a letter's alphabet index
mod 16), an is-digit flag, and flags for '/', '.', '_'. BOS and '=' are the zero vector.

Block 1  attention: 3 heads form a shift register. Positions sit on a circle, and head k rotates the
         query back k steps, so each token sees tokens at offsets 1..3 behind it.
         MLP: one conjunction neuron per parsing rule tags the token that *terminates* a field with
         a compass direction (east YEAR, north MONTH, west DAY); 12 trigram neurons recognise month
         words and overwrite the letter bits in the window with the month's digits.
Block 2  attention only: each answer step queries a direction, and the head for that digit copies it
         from the tagged token's window. Idle heads look south, at '=', whose window is empty.
         Years are 1900-2099, so one bit at offset 3 fixes the century.
Output   nearest code on the 4 output bits; the dashes come straight from the step bits.
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
    def __init__(self, d_model: int, n_heads: int, d_key: int, d_val: int, sum_heads: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_val = d_val
        self.sum_heads = sum_heads
        self.W_q = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_k = nn.Linear(d_model, d_key, bias=False)  # one key per token, shared by all heads
        self.W_v = nn.Linear(d_model, n_heads * d_val, bias=False)
        self.W_o = nn.Linear(d_val if sum_heads else n_heads * d_val, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dk, dv = self.n_heads, self.d_key, self.d_val
        q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.W_k(x)[:, None, :, :]
        v = self.W_v(x).view(B, T, H, dv).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = attn @ v
        if self.sum_heads:
            return self.W_o(out.sum(dim=1))
        return self.W_o(out.transpose(1, 2).contiguous().view(B, T, H * dv))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_key: int, d_val: int, d_ff: int, sum_heads: bool = False):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_key, d_val, sum_heads)
        self.mlp = MLP(d_model, d_ff) if d_ff else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x) if self.mlp else x


class SimpleTransformer(nn.Module):
    """Causal transformer without LayerNorm; a BOS token is prepended inside forward()."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 41,
        n_heads: int = 4,
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

        self.token_emb = nn.Embedding(vocab_size + 1, 8)   # slots 0-7 of the state
        self.pos_emb = nn.Embedding(max_seq_len + 1, 12)   # slots 8-19 of the state
        self.blocks = nn.ModuleList([Block(d_model, 3, 2, 6, d_ff), Block(d_model, n_heads, 2, 4, 0, sum_heads=True)])
        self.head = nn.Linear(6, vocab_size)  # reads the 4 output bits and the 2 dash-step bits
        self.read = []  # state slots the head reads, set in write_weights

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        ids = torch.cat([ids.new_full((B, 1), self.vocab_size), ids], dim=1)
        pos = torch.arange(T + 1, device=ids.device)
        tok, p = self.token_emb(ids), self.pos_emb(pos).expand(B, -1, -1)
        h = torch.cat([tok, p, tok.new_zeros(B, T + 1, self.d_model - 20)], dim=-1)
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, :, self.read])[:, 1:]


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def bits(value: int, width: int) -> list:
    """Most significant bit first."""
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter in closed form. No training."""
    for p in model.parameters():
        nn.init.zeros_(p)
    s = task.stoi

    # Token code fields, as (slot) indices of the state.
    NIB, ISD, SLASH, DOT, UNDER = 0, 4, 5, 6, 7
    COS, SIN, STEP = 8, 9, 10
    NULL = STEP  # the step-0 bit marks '=', whose window holds only padding: idle heads point there
    nib = list(range(NIB, NIB + 4))
    # Which token fields each shift-register offset keeps.
    fields = {1: nib + [ISD], 2: nib + [ISD, SLASH], 3: [ISD, SLASH, DOT, nib[0]]}
    WIN, next_slot = {}, STEP + 10
    for k, f in fields.items():
        WIN[k] = {slot: next_slot + i for i, slot in enumerate(f)}
        next_slot += len(f)
    EAST, NORTH, OUT = next_slot, next_slot + 1, next_slot + 2  # role compass: E=YEAR N=MONTH W=DAY
    assert OUT + 4 == model.d_model
    YEAR, MONTH, DAY = {EAST: 1}, {NORTH: 1}, {EAST: -1}

    with torch.no_grad():
        # Token embedding: the readable code. BOS and '=' stay zero.
        emb = model.token_emb.weight
        for ch, i in s.items():
            if ch.isdigit():
                emb[i, ISD] = 1.0
                emb[i, nib] = torch.tensor(bits(int(ch), 4), dtype=emb.dtype)
            if ch.isalpha():
                emb[i, nib] = torch.tensor(bits((ord(ch) - 96) % 16, 4), dtype=emb.dtype)
            emb[i, SLASH], emb[i, DOT], emb[i, UNDER] = float(ch == "/"), float(ch == "."), float(ch == "_")

        # Position embedding. idx = original position + 1 (BOS is idx 0).
        pe = model.pos_emb.weight
        theta = 2 * math.pi / pe.shape[0]  # one point on the circle per sequence position
        for t in range(pe.shape[0]):
            pe[t, COS - 8], pe[t, SIN - 8] = math.cos(theta * t), math.sin(theta * t)
        for step in range(10):
            pe[32 + step, STEP - 8 + step] = 1.0

        # Block 1 attention: head k-1 rotates the position code back k steps and copies that token.
        a1 = model.blocks[0].attn
        B1 = 40.0
        for k, f in fields.items():
            base = (k - 1) * a1.d_val
            kx, ky = 2 * (k - 1), 2 * (k - 1) + 1
            c, sn = math.cos(k * theta), math.sin(k * theta)
            a1.W_k.weight[0, COS] = a1.W_k.weight[1, SIN] = B1
            a1.W_q.weight[kx, COS], a1.W_q.weight[kx, SIN] = B1 * c, B1 * sn
            a1.W_q.weight[ky, COS], a1.W_q.weight[ky, SIN] = -B1 * sn, B1 * c
            for i, slot in enumerate(f):
                a1.W_v.weight[base + i, slot] = 1.0
                a1.W_o.weight[WIN[k][slot], base + i] = 1.0

        # Block 1 MLP: conjunction neurons.
        fc1, fc2 = model.blocks[0].mlp.fc1, model.blocks[0].mlp.fc2
        neuron = [0]

        def rule(inputs: dict, threshold: float, outputs: dict) -> None:
            n = neuron[0]
            neuron[0] += 1
            for slot, w in inputs.items():
                fc1.weight[n, slot] += w
            fc1.bias[n] = -threshold
            for slot, w in outputs.items():
                fc2.weight[slot, n] += w

        # YEAR: a non-digit token behind three digits (no other field is that long). Its one false
        # tag, on the answer's first dash, comes after the year has already been written.
        rule({WIN[1][ISD]: 1, WIN[2][ISD]: 1, WIN[3][ISD]: 1, ISD: -1}, 2, YEAR)
        # YEAR (mm/dd/yy): padding three after a slash. The '/' has a clear century bit, so the
        # century decodes to 20 with no further wiring.
        rule({UNDER: 1, WIN[3][SLASH]: 1}, 1, YEAR)
        # MONTH: second dot of yyyy.mm.dd. (Every '/' also points north; see block 2.)
        rule({DOT: 1, WIN[3][DOT]: 1}, 1, MONTH)
        # DAY: second slash (which also cancels its north key); end of yyyy.mm.dd;
        # a digit followed by anything but digit / . or _ (vetoed on the steps that read the
        # answer's own dashes, which are the only such tokens after the prompt).
        rule({SLASH: 1, WIN[2][SLASH]: 1, WIN[3][SLASH]: 1}, 1, {**DAY, NORTH: -1})
        rule({UNDER: 1, WIN[3][DOT]: 1}, 1, DAY)
        rule({WIN[1][ISD]: 1, ISD: -1, SLASH: -1, DOT: -1, UNDER: -1, STEP + 5: -1, STEP + 8: -1}, 0, DAY)
        # MONTH words: a trigram detector matches 3 letters, then overwrites the letter bits it
        # knows are in the window with the month number: nibble += digit - letter bits.
        for m, name in enumerate(task.MONTHS, start=1):
            # A leading digit is vetoed: digits hash like letters ("35_" = "sep", "1/2" = "apr").
            inputs, n_ones, outs = {WIN[2][ISD]: -1}, 0, dict(MONTH)
            code2, code1, code0 = (bits((ord(c) - 96) % 16, 4) for c in name[:3])
            for code, slots in ((code2, [WIN[2][x] for x in nib]),
                                (code1, [WIN[1][x] for x in nib]),
                                (code0, nib)):
                n_ones += sum(code)
                inputs.update({slot: 2 * b - 1 for slot, b in zip(slots, code)})
            for k, digit, code in ((2, m // 10, code2), (1, m % 10, code1)):
                outs.update({WIN[k][nib[i]]: bits(digit, 4)[i] - code[i] for i in range(4)})
            rule(inputs, n_ones - 1, outs)
        assert neuron[0] == model.d_ff

        # Block 2 attention. Heads 0-1 write the century, heads 2-3 copy a field's tens and ones
        # digits from offsets 2 and 1. Each answer step sends one head to a role direction.
        a2 = model.blocks[1].attn
        B2 = 20.0
        E, N, W, S = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)
        plan = {0: (0, E), 1: (1, E), 2: (2, E), 3: (3, E), 5: (2, N), 6: (3, N), 8: (2, W), 9: (3, W)}
        a2.W_k.weight[0, EAST] = a2.W_k.weight[1, NORTH] = 1.0
        a2.W_k.weight[1, SLASH] = 1.0   # every '/' points north (MONTH)
        a2.W_k.weight[1, NULL] = -1.0   # '=' points south, where idle heads look
        for h in range(4):
            kx, ky, base = 2 * h, 2 * h + 1, 4 * h
            for step in range(10):
                x, y = plan[step][1] if step in plan and plan[step][0] == h else S
                a2.W_q.weight[kx, STEP + step], a2.W_q.weight[ky, STEP + step] = B2 * x, B2 * y
            for b in range(4):
                if h >= 2:
                    a2.W_v.weight[base + b, WIN[4 - h][nib[b]]] = 1.0
                a2.W_o.weight[OUT + b, b] = 1.0
        # Years are 1900-2099, so one bit fixes the century: the top bit at offset 3 is set for
        # the 9 of 19xx and clear for the 0 of 20xx (or the '/' of mm/dd/yy).
        century = WIN[3][nib[0]]
        a2.W_v.weight[0 + 3, century] = 1.0   # first digit: bit set writes 0001 ('1') ...
        a2.W_v.weight[0 + 2, century] = -1.0  # ... bit clear leaves 0010 ('2') from the YEAR tag
        a2.W_v.weight[0 + 2, EAST] = 1.0
        a2.W_v.weight[4 + 0, century] = a2.W_v.weight[4 + 3, century] = 1.0  # second: 1001 or 0000

        # Unembedding: nearest code on the 4 nibble bits; the dash fires from its two step bits.
        model.read = [OUT, OUT + 1, OUT + 2, OUT + 3, STEP + 4, STEP + 7]
        for d in range(10):
            code = bits(d, 4)
            model.head.weight[s[str(d)], :4] = 2.0 * torch.tensor(code, dtype=emb.dtype)
            model.head.bias[s[str(d)]] = 1.0 - sum(code)  # +1 beats unused tokens' logit of 0
        model.head.weight[s["-"], 4:] = 2.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "final-compass-pointer"
model_description = "Final clean circuit: 3-head shift register, 18 conjunction neurons tagging field terminators with compass directions, then a 4-head role lookup; d_model 41."


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
