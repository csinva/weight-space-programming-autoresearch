"""
Interpretable transformer for date normalization (many text formats -> YYYY-MM-DD).
Snapshot final_v50_anchor_pointer_compact: accuracy 1.0 (200/200; 0/1500 failures in extra debug check), 1.25e5 params.

Hand-written weights, no training, no LayerNorm.  Residual stream = named slices (see LAYOUT).
  Layer 1  prev-char heads (offset -1: 6 second-letters + is-digit; offset -2: is-digit) + MLP:
           J4 = "3rd consecutive digit" flag, month id from the 2nd+3rd letter (an eb ar pr ay un ul ug ep ct ov ec).
  Layer 2  anchor heads: "earliest position with flag X": first digit (A_D1), first '/' (A_S1, F_SL),
           first 3rd-digit-or-'_' (A_P4, F_H4; '_' fallback handles 2-digit years), month id (MIDA, F_LET).
           (head index 3 is an unused leftover)
  Layer 3  5 reader heads: key position s = anchor + offset via score 2*C*s*pos - C*pos^2 (peak at pos = s);
           MLP of one-hot AND units picks, per output step i (G[i] = one-hot of query pos 31+i) and format flags,
           which reader supplies the output digit (tens = first char if second char is a digit else '0', etc.).
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

VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789 ,./-_=")  # same order as task.vocab
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

LAYOUT: dict[str, tuple[int, int]] = {}
_cursor = 0


def _alloc(name: str, size: int) -> None:
    global _cursor
    LAYOUT[name] = (_cursor, size)
    _cursor += size


# Compact token slice: only tokens the circuit ever reads/writes (digits, '-', letters used by the month lookup).
_USED = sorted(set("0123456789-") | {m[1] for m in MONTHS} | {m[2] for m in MONTHS})
TOKIDX = {VOCAB.index(c): k for k, c in enumerate(_USED)}

for _n, _s in [
    ("TOK", len(_USED)), ("CONST", 1), ("POS1", 1), ("POS2", 1), ("G", 10), ("EARLY", 1),
    ("ISDIG", 1), ("ISSL", 1), ("ISUS", 1),
    ("PREV1", 7), ("DIGP2", 1),
    ("J4", 1), ("MIDID", 12),
    ("A_D1", 1), ("A_S1", 1), ("A_P4", 1),
    ("F_SL", 1), ("F_H4", 1), ("F_LET", 1), ("MIDA", 12),
    ("RY", 10), ("RD1a", 10), ("RD1b", 10), ("RS1a", 10), ("RS1b", 10),
]:
    _alloc(_n, _s)
D_MODEL = _cursor
LAYOUT["OUT"] = LAYOUT["TOK"]  # logits are written on top of the (now unused) input one-hot slice


def ix(name: str, i: int = 0) -> int:
    if name in ("TOK", "OUT"):      # i is a token id here
        return LAYOUT[name][0] + TOKIDX[i]
    return LAYOUT[name][0] + i


# (n_heads, d_head, d_ff) per layer
LAYERS = [(2, 7, 13), (5, 13, 0), (5, 10, 140)]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        da = n_heads * d_head
        self.W_q = nn.Linear(d_model, da, bias=False)
        self.W_k = nn.Linear(d_model, da, bias=False)
        self.W_v = nn.Linear(d_model, da, bias=False)
        self.W_o = nn.Linear(da, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
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
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 42, d_model: int = D_MODEL, layers=LAYERS):
        super().__init__()
        self.vocab_size, self.max_seq_len, self.d_model = vocab_size, max_seq_len, d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, h, dh, ff) for h, dh, ff in layers])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def write_weights(model: SimpleTransformer, task) -> None:
    stoi = task.stoi
    DIGITS = [stoi[str(d)] for d in range(10)]
    C_PEAK = 10.0  # sharpness of "attend to position s" parabola
    B_FLAG = 200.0  # flag beats position preference
    C_EARLY = 2.0  # "earliest" preference per position

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # ---- embeddings -------------------------------------------------
        E = model.token_emb.weight
        for ch, t in stoi.items():
            if t in TOKIDX:
                E[t, ix("TOK", t)] = 1
            E[t, ix("CONST")] = 1
            if ch.isdigit():
                E[t, ix("ISDIG")] = 1
            if ch == "/":
                E[t, ix("ISSL")] = 1
            if ch == "_":
                E[t, ix("ISUS")] = 1
        P = model.pos_emb.weight
        for p in range(model.max_seq_len):
            P[p, ix("POS1")] = p
            P[p, ix("POS2")] = p * p
            if p < 2 or p >= 31:   # not a valid "3rd digit" position: first two chars, or answer region
                P[p, ix("EARLY")] = 1
            if 31 <= p <= 40:
                P[p, ix("G", p - 31)] = 1

        # ---- helpers ----------------------------------------------------
        def peak_head(blk, h, s_terms: dict, v_map: dict, o_map: dict):
            """Head h attends to key position s = sum(coef*slice_scalar); value dims v_map {dim: resid idx}
            are written to resid idx via o_map {dim: resid idx}."""
            a = blk.attn
            dh = a.d_head
            sc = math.sqrt(dh)
            b = h * dh
            for name, coef in s_terms.items():
                a.W_q.weight[b + 0, ix(name)] += 2 * C_PEAK * sc * coef
            a.W_q.weight[b + 1, ix("CONST")] = -C_PEAK * sc
            a.W_k.weight[b + 0, ix("POS1")] = 1
            a.W_k.weight[b + 1, ix("POS2")] = 1
            for d, src in v_map.items():
                a.W_v.weight[b + d, src] = 1
            for d, dst in o_map.items():
                a.W_o.weight[dst, b + d] = 1

        def earliest_head(blk, h, flag_srcs: dict, v_map: dict, o_map: dict):
            """Head h attends to earliest key whose flag (sum of flag_srcs {resid idx: w}) is 1."""
            a = blk.attn
            dh = a.d_head
            sc = math.sqrt(dh)
            b = h * dh
            a.W_q.weight[b + 0, ix("CONST")] = B_FLAG * sc
            a.W_q.weight[b + 1, ix("CONST")] = -C_EARLY * sc
            for src, w in flag_srcs.items():
                a.W_k.weight[b + 0, src] = w
            a.W_k.weight[b + 1, ix("POS1")] = 1
            for d, src in v_map.items():
                a.W_v.weight[b + d, src] = 1
            for d, dst in o_map.items():
                a.W_o.weight[dst, b + d] = 1

        # ---- Layer 1: shift heads ----------------------------------------
        b1 = model.blocks[0]
        SECOND = sorted({n[1] for n in MONTHS})   # a c e o p u
        peak_head(b1, 0, {"POS1": 1, "CONST": -1},
                  {**{k: ix("TOK", stoi[c]) for k, c in enumerate(SECOND)}, 6: ix("ISDIG")},
                  {k: ix("PREV1", k) for k in range(7)})   # PREV1[6] = "previous char is a digit"
        peak_head(b1, 1, {"POS1": 1, "CONST": -2}, {0: ix("ISDIG")}, {0: ix("DIGP2")})

        # Layer-1 MLP: unit0 = J4 (3 consecutive digits ending here), units 1..12 = month id from letters 2+3
        m1 = b1.mlp
        w = m1.fc1.weight
        w[0, ix("ISDIG")] = 1
        w[0, ix("DIGP2")] = 1
        w[0, ix("PREV1", 6)] = 1
        w[0, ix("EARLY")] = -3
        m1.fc1.bias[0] = -2
        m1.fc2.weight[ix("J4"), 0] = 1
        for m, name in enumerate(MONTHS):
            u = 1 + m
            w[u, ix("PREV1", SECOND.index(name[1]))] = 1
            w[u, ix("TOK", stoi[name[2]])] = 1
            m1.fc1.bias[u] = -1
            m1.fc2.weight[ix("MIDID", m), u] = 1

        # ---- Layer 2: anchors ---------------------------------------------
        b2 = model.blocks[1]
        earliest_head(b2, 0, {ix("ISDIG"): 1}, {0: ix("POS1")}, {0: ix("A_D1")})
        earliest_head(b2, 1, {ix("ISSL"): 1}, {0: ix("POS1"), 1: ix("ISSL")},
                      {0: ix("A_S1"), 1: ix("F_SL")})
        # year anchor: first 3rd-digit-of-a-run, fallback (2-digit year) = first '_' (value pos-2)
        earliest_head(b2, 2, {ix("J4"): 1, ix("ISUS"): 1.0}, {0: ix("POS1"), 1: ix("J4")},
                      {0: ix("A_P4"), 1: ix("F_H4")})
        b2.attn.W_v.weight[2 * 13 + 0, ix("ISUS")] = -2
        earliest_head(b2, 4, {ix("MIDID", m): 1 for m in range(12)},
                      {m: ix("MIDID", m) for m in range(12)}, {m: ix("MIDA", m) for m in range(12)})
        for m in range(12):
            b2.attn.W_v.weight[4 * 13 + 12, ix("MIDID", m)] = 1
        b2.attn.W_o.weight[ix("F_LET"), 4 * 13 + 12] = 1

        # ---- Layer 3: readers ---------------------------------------------
        b3 = model.blocks[2]
        vd = {d: ix("TOK", DIGITS[d]) for d in range(10)}

        def reader(h, s_terms, dst):
            peak_head(b3, h, s_terms, vd, {d: ix(dst, d) for d in range(10)})

        reader(0, {"A_P4": 1, "POS1": 1, "CONST": -33}, "RY")     # year digit i (also copies position i for YYYY.MM.DD)
        reader(1, {"A_D1": 1}, "RD1a")                            # first digit of first number
        reader(2, {"A_D1": 1, "CONST": 1}, "RD1b")                # char after it
        reader(3, {"A_S1": 1, "CONST": 1}, "RS1a")                # first char after first '/'
        reader(4, {"A_S1": 1, "CONST": 2}, "RS1b")

        # ---- Layer 3 MLP: gated source selection --------------------------
        m3 = b3.mlp
        units = []

        def unit(inp: dict, bias: float, out_tok: int):
            units.append((inp, bias, out_tok))

        G = lambda i: ix("G", i)
        SL, LET, H4 = ix("F_SL"), ix("F_LET"), ix("F_H4")
        T = lambda d: DIGITS[d]
        dash = stoi["-"]

        def digsum(name):
            return {ix(name, d): 1 for d in range(10)}

        def merge(*ds):
            r = {}
            for d in ds:
                for k, v in d.items():
                    r[k] = r.get(k, 0) + v
            return r

        for d in range(10):  # year digits
            unit(merge({ix("RY", d): 1, H4: 1}, {G(i): 1 for i in range(4)}), -2, T(d))
            unit(merge({ix("RY", d): 1, H4: -1}, {G(2): 1, G(3): 1}), -1, T(d))
        unit({G(0): 1, H4: -1}, 0, T(2))
        unit({G(1): 1, H4: -1}, 0, T(0))
        unit({G(4): 1, G(7): 1}, 0, dash)
        for m in range(12):  # month from name
            unit({ix("MIDA", m): 1, G(6): 1, LET: 1}, -2, T((m + 1) % 10))
        late = {ix("MIDA", m): 1 for m in (9, 10, 11)}   # oct nov dec -> tens digit 1
        unit({G(5): 1, LET: 1, **late}, -2, T(1))
        unit({G(5): 1, LET: 1, **{k: -1 for k in late}}, -1, T(0))

        def number(fl, a, b, step_t, step_u):
            digb = digsum(b)
            for d in range(10):
                unit(merge({ix(a, d): 1, G(step_t): 1, fl: 1}, digb), -3, T(d))          # tens = A if B digit
                unit({ix(b, d): 1, G(step_u): 1, fl: 1}, -2, T(d))                        # units = B if B digit
                unit(merge({ix(a, d): 1, G(step_u): 1, fl: 1}, {k: -1 for k in digb}), -2, T(d))  # units = A else
            unit(merge({G(step_t): 1, fl: 1}, {k: -1 for k in digb}), -1, T(0))          # tens = 0 else

        number(SL, "RD1a", "RD1b", 5, 6)    # month, m/d/y
        number(LET, "RD1a", "RD1b", 8, 9)   # day, name formats
        number(SL, "RS1a", "RS1b", 8, 9)    # day, m/d/y
        for d in range(10):                 # YYYY.MM.DD: the year reader (s = P + i - 3 = i) already copies position i
            unit(merge({ix("RY", d): 1, SL: -1, LET: -1}, {G(i): 1 for i in (5, 6, 8, 9)}), -1, T(d))

        assert len(units) <= LAYERS[2][2], len(units)
        for u, (inp, bias, out_tok) in enumerate(units):
            for k, v in inp.items():
                m3.fc1.weight[u, k] = v
            m3.fc1.bias[u] = bias
            m3.fc2.weight[ix("OUT", out_tok), u] = 2

        for t in TOKIDX:
            model.head.weight[t, ix("OUT", t)] = 1


model_shorthand_name = "final_v50_anchor_pointer_compact"
model_description = ("3 layers, no LayerNorm, 5/5/5 heads. L1: prev-char heads + MLP (3rd-digit flag, 2-letter month lookup). "
                     "L2: earliest-flag anchor heads (first digit, first '/', first 3rd-digit-or-'_', month id). "
                     "L3: parabola pointer readers (anchor+offset) + gated one-hot MLP picks the source per output step.")

# (evaluation harness / main block: identical to interpretable_transformer.py in the run folder)
