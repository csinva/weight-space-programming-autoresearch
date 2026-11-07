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
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

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
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        # LayerNorm removed (Identity) so residual-stream arithmetic is exact and hand-designable.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 432,
        n_heads: int = 3,
        n_layers: int = 3,
        d_ff: int = 112,
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
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
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
    """Hand-built NLI circuit: character n-gram overlap + hypothesis length + negation -> label.

    Residual layout (d_model=432; each attention head has dh=144 dims):
      T  [0,34)    current char one-hot          PV  [34,68)   char at p-1
      P  [68,191)  position one-hot              PREM,SEP,HYP,EQ flags at 191..194
      ONE 195      constant 1                    VALID 196     hyp n-gram is all letters
      NEG 197      hyp bigram is "no" / "n'" / "ev"  PRES 198  hyp n-gram occurs in premise
      FF 199       mean PRES over VALID hyp      FN 200        mean NEG over VALID hyp
      PV2 [201,235) char at p-2                  PAD 235       mean '_' over hyp positions
      PADP 236     mean '_' over premise         PV3 [237,271) char at p-3 (only if NG=4)
    Layers (heads 0,1,2 use dims [0,144), [144,288), [288,432)):
      L1: head0 -> PV, head1 -> PV2, head2 -> PV3.  MLP: zero prev-chars at the separator
          (so the separator can act as a null key), compute VALID and NEG.
      L2 head0: query = hyp (T,PV,PV2); keys = premise chars/prev chars; a full NG-char match
          scores NG*B, the separator null scores (NG-0.5)*B, so PRES ~ 1 iff the n-gram occurs.
          Hyp/'=' keys get a -100 penalty so they cannot absorb attention mass.
      L3 head0: mean PRES and NEG over VALID hyp -> FF, FN.   head1: mean '_' over hyp -> PAD.
         head2: mean '_' over premise -> PADP (weight PGAM=0 in the readout).
      Readout: logit0 = A*(FF-TH); logit1 = A*(TH-FF) + C1 - BETA*PAD; logit2 = K*FN + C2.
      Ties and the argmax fall back to class 2 (contradiction) when the other logits are negative.
    """
    D, H = model.d_model, model.n_heads
    dh = D // H
    sq = math.sqrt(dh)
    V = model.vocab_size
    T0, PV0, P0 = 0, 34, 68
    PREM, SEP, HYP, EQ, ONE, VALID, NEG, PRES, FF, FN = range(191, 201)
    PV2, PAD, PADP, PV3 = 201, 235, 236, 237
    H0, H1, H2 = 0, dh, 2 * dh  # head offsets in Q/K/V space
    B1, B2, B3, A, K = 30.0, 20.0, 20.0, 5.0, 5.0
    TH = 0.5  # FF threshold between entailment and neutral
    PGAM = 0.0  # neutral logit weight on premise padding fraction (+/-5 both hurt or were neutral)
    BETA = 20.0  # neutral penalty per unit of hypothesis padding fraction
    NG = 3  # n-gram length used for overlap (3 or 4)
    C1, C2 = 6.0, 0.0  # readout biases for neutral and contradiction logits
    st = task.stoi
    letters = [st[c] for c in "abcdefghijklmnopqrstuvwxyz"]
    SPACE_OK = False  # allow n-grams that contain a space (cross-word n-grams)
    valid_chars = letters + ([st[" "]] if SPACE_OK else [])
    cls_ids = [st["0"], st["1"], st["2"]]

    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()

        # ---- embeddings ----
        for t in range(V):
            model.token_emb.weight[t, T0 + t] = 1.0
            model.token_emb.weight[t, ONE] = 1.0
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, P0 + p] = 1.0
            if p < 60:
                flag = PREM
            elif p == 60:
                flag = SEP
            elif p <= 120:
                flag = HYP
            else:
                flag = EQ
            model.pos_emb.weight[p, flag] = 1.0

        # ---- layer 1: previous-char attention, three heads (lags 1, 2, 3) ----
        a1 = model.blocks[0].attn
        for lag, off, out_base in [(1, H0, PV0), (2, H1, PV2), (3, H2, PV3)]:
            for d in range(122 - (lag - 1)):           # query at p -> key p - lag
                a1.W_q.weight[off + d, P0 + d + lag] = B1 * sq
            for d in range(123):
                a1.W_k.weight[off + d, P0 + d] = 1.0
            for c in range(34):
                a1.W_v.weight[off + c, T0 + c] = 1.0
                a1.W_o.weight[out_base + c, off + c] = 1.0

        # ---- layer 1 MLP ----
        m1 = model.blocks[0].mlp
        u = 0
        for base in (PV0, PV2, PV3):  # zero each prev-char at the separator
            for c in range(34):
                m1.fc1.weight[u, base + c] = 1.0
                m1.fc1.weight[u, SEP] = 1.0
                m1.fc1.bias[u] = -1.0
                m1.fc2.weight[base + c, u] = -1.0
                u += 1
        # VALID = HYP & letter(cur) & letter(prev) & letter(prev2) [& letter(prev3) if NG == 4]
        m1.fc1.weight[u, HYP] = 1.0
        for l in valid_chars:
            m1.fc1.weight[u, T0 + l] = 1.0
            m1.fc1.weight[u, PV0 + l] = 1.0
            m1.fc1.weight[u, PV2 + l] = 1.0
            if NG == 4:
                m1.fc1.weight[u, PV3 + l] = 1.0
        m1.fc1.bias[u] = -float(NG)
        m1.fc2.weight[VALID, u] = 1.0
        u += 1
        # NEG = HYP & bigram in {"no", "n'", "ev"}
        for cur, prev in [("o", "n"), ("'", "n"), ("v", "e")]:
            m1.fc1.weight[u, HYP] = 1.0
            m1.fc1.weight[u, T0 + st[cur]] = 1.0
            m1.fc1.weight[u, PV0 + st[prev]] = 1.0
            m1.fc1.bias[u] = -2.0
            m1.fc2.weight[NEG, u] = 1.0
            u += 1

        # ---- layer 2: NG-char match of hyp query against premise keys (sep = null) ----
        a2 = model.blocks[1].attn
        NULLQ, PENQ = 136, 137  # dims inside head0 used for the null score and the hyp penalty
        match_parts = [(T0, 0), (PV0, 34), (PV2, 68)] + ([(PV3, 102)] if NG == 4 else [])
        for c in range(34):
            for base_res, base_q in match_parts:
                a2.W_q.weight[base_q + c, base_res + c] = B2 * sq
                a2.W_k.weight[base_q + c, base_res + c] = 1.0
        # null score sits between the (NG-1)-match and NG-match scores
        a2.W_q.weight[NULLQ, ONE] = (NG - 0.5) * B2 * sq
        a2.W_k.weight[NULLQ, SEP] = 1.0
        a2.W_q.weight[PENQ, ONE] = -100.0 * sq
        a2.W_k.weight[PENQ, HYP] = 1.0
        a2.W_k.weight[PENQ, EQ] = 1.0
        a2.W_v.weight[0, PREM] = 1.0
        a2.W_o.weight[PRES, 0] = 1.0

        # ---- layer 3: aggregators ----
        a3 = model.blocks[2].attn
        # head0: mean PRES and NEG over VALID hyp positions
        a3.W_q.weight[0, ONE] = B3 * sq
        a3.W_k.weight[0, VALID] = 1.0
        a3.W_v.weight[1, PRES] = 1.0
        a3.W_v.weight[2, NEG] = 1.0
        a3.W_o.weight[FF, 1] = 1.0
        a3.W_o.weight[FN, 2] = 1.0
        # head1: mean '_' indicator over all hypothesis positions
        a3.W_q.weight[H1, ONE] = B3 * sq
        a3.W_k.weight[H1, HYP] = 1.0
        a3.W_v.weight[H1 + 1, T0 + st["_"]] = 1.0
        a3.W_o.weight[PAD, H1 + 1] = 1.0
        # head2: mean '_' indicator over premise positions (premise padding -> premise length)
        a3.W_q.weight[H2, ONE] = B3 * sq
        a3.W_k.weight[H2, PREM] = 1.0
        a3.W_v.weight[H2 + 1, T0 + st["_"]] = 1.0
        a3.W_o.weight[PADP, H2 + 1] = 1.0

        # ---- readout (only the three label tokens can win) ----
        for t in range(V):
            if t not in cls_ids:
                model.head.weight[t, ONE] = -1000.0
        model.head.weight[cls_ids[0], FF] = A
        model.head.weight[cls_ids[0], ONE] = -A * TH
        model.head.weight[cls_ids[1], FF] = -A
        model.head.weight[cls_ids[1], ONE] = A * TH + C1
        model.head.weight[cls_ids[1], PAD] = -BETA
        model.head.weight[cls_ids[1], PADP] = PGAM
        model.head.weight[cls_ids[2], FN] = K
        model.head.weight[cls_ids[2], ONE] = C2


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "trigram-A5-K5-negev-final"
model_description = ("Final: trigram overlap (letters only, NG=3), entail if F>0.5 (5*(F-0.5)), neutral "
                     "5*(0.5-F)+6-20*hyp_padfrac, contradiction 5*neg_frac (negation bigrams no/n'/ev); "
                     "contradiction is the default class.")


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
