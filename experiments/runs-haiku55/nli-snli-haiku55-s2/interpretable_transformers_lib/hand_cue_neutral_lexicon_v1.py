"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. Task format for `nli-snli`:
  prompt = premise[:60] padded with '_' to 60 chars, '|', hypothesis[:60] padded
           with '_' to 60 chars, '='   (122 chars total)
  answer = one digit: '0' (entailment), '1' (neutral), '2' (contradiction)

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task nli-snli --n-samples 500
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
# Note: LayerNorm is removed so that every residual-stream coordinate is
# linear in its inputs and can be set in closed form.

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
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """4-layer causal transformer (no LayerNorm), vocab/seq-len set from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 128,
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
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

# Residual-stream slot layout (d_model = 512)
SLOT_TOK = 0          # 0..33    one-hot of current token (index = token id)
SLOT_POS = 34         # 34..156  one-hot of absolute position
SLOT_PREV1 = 157      # 157..190 one-hot of token at p-1
SLOT_PREV2 = 191      # 191..224 one-hot of token at p-2
SLOT_NEG = 225        # 1 if a negation cue ends at this position (in hypothesis)
SLOT_HYP = 226        # 1 if position is inside the hypothesis
SLOT_LET = 227        # 1 if the current token is a letter a-z
SLOT_CONST = 228      # always 1
SLOT_POOL_NEG = 229   # at '=': number of negation cues in hypothesis
SLOT_POOL_LET = 230   # at '=': fraction of hypothesis positions that are letters
SLOT_PREM = 231       # 1 if position is inside the premise
SLOT_MID = 300        # 300..401 identity (TOK,PREV1,PREV2 one-hots) of the matched premise trigram
SLOT_MS = 402         # number of trigram slots (0..3) equal between query and matched trigram
SLOT_FULL = 403       # 1 if the hypothesis trigram at this position occurs in the premise
SLOT_POOL_OV = 404    # at '=': count of hypothesis positions whose trigram occurs in premise
SLOT_LTRI = 405       # 1 if the trigram ending here is 3 letters (all-letter trigram)
SLOT_POOL_LT = 406    # at '=': count of hypothesis all-letter trigram positions
SLOT_PREV3 = 407      # 407..440 one-hot of token at p-3
SLOT_NEUT = 441       # 1 if a neutral cue word starts at this trigram (in hypothesis)
SLOT_POOL_NEUT = 442  # at '=': count of neutral cue words in hypothesis

OV_RATIO = 0.55       # entailment if overlap count >= OV_RATIO * letter-trigram count
CONTRA_RATIO = 0.0    # 0 disables the low-overlap contradiction band (negation-only contradiction)

# Contradiction cue words (3-letter word prefixes, must start a word in the hypothesis).
# " no"/"n't" are handled separately. Prefixes that are substrings of common other words
# (e.g. "wit" in "with", "alo" in "along") are excluded.
CUE_WORDS = ["nev", "nob", "emp", "onl"]   # never, nobody, empty, only
# Neutral cue words (hand-picked, word-initial 3-letter prefixes): hypotheses that add
# information not in the premise (because, favorite, birthday, wedding, family, tall, young, ...)
NEUT_WORDS = ["bec", "fav", "bir", "wed", "fam", "tal", "you", "old", "pop", "som"]
NEUT_WEIGHT = 10.0    # logit weight of neutral cue count (added to '1', subtracted from '0')
GATE_PREM = 10.0      # score bonus for attending premise positions from hypothesis


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Circuit (4 blocks, each = attention + MLP):
      B0 attn: head0 copies token p-1 -> PREV1 ; head1 -> p-2 ; head2 -> p-3
      B0 MLP : AND-gates for contradiction cues (" no", "n't", word-initial nev/nob/emp/onl) -> NEG
               and neutral cue words (word-initial bec/fav/...) -> NEUT
      B1 attn: each hypothesis position attends to premise positions whose character
               trigram (tok, prev1, prev2) matches its own; copies the matched identity
      B1 MLP : per coordinate AND(query identity, matched identity) -> MS (# equal slots)
      B2 MLP : FULL = 1 iff MS == 3 and trigram has no space/underscore/digit/'='
      B3 attn: '=' pools uniformly over hypothesis -> POOL_NEG, POOL_LET, POOL_OV, POOL_LT, POOL_NEUT
      Head : '2' if any negation cue; else '0' if overlap >= OV_RATIO * letter-trigrams; else '1';
             neutral cue words shift mass from '0' to '1'.
    """
    torch.manual_seed(0)
    S = task.SENT_LEN                  # 60
    ids = task.stoi
    T = model.max_seq_len
    V = model.vocab_size
    dh = model.d_model // model.n_heads
    assert T <= dh and SLOT_POS + T - 1 < SLOT_PREV1
    s = 30.0                           # sharpness of content-matching attention
    scale = math.sqrt(dh)              # nn attention divides scores by sqrt(dh)
    hyp_lo, hyp_hi = S + 1, 2 * S      # hypothesis positions (inclusive)
    nonletter = [ids[c] for c in [" ", "_", "|", "=", "0", "1", "2"]]
    # identity coordinates: TOK(0..33), PREV1(34..67), PREV2(68..101)
    qslot = ([SLOT_TOK + i for i in range(V)]
             + [SLOT_PREV1 + i for i in range(V)]
             + [SLOT_PREV2 + i for i in range(V)])
    n_id = len(qslot)                  # 102

    with torch.no_grad():
        # --- embeddings -----------------------------------------------------
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        for c, i in ids.items():
            model.token_emb.weight[i, SLOT_TOK + i] = 1.0
            if "a" <= c <= "z":
                model.token_emb.weight[i, SLOT_LET] = 1.0
        for p in range(T):
            model.pos_emb.weight[p, SLOT_POS + p] = 1.0
            model.pos_emb.weight[p, SLOT_CONST] = 1.0
            if hyp_lo <= p <= hyp_hi:
                model.pos_emb.weight[p, SLOT_HYP] = 1.0
            if p < S:
                model.pos_emb.weight[p, SLOT_PREM] = 1.0

        # --- block 0: previous tokens + cue AND-gates -----------------------
        blk = model.blocks[0]
        attn = blk.attn
        for m in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
            m.weight.zero_()
        for j in range(T):             # head 0: query p attends to key p-1
            attn.W_k.weight[j, SLOT_POS + j] = 1.0
            if j + 1 < T:
                attn.W_q.weight[j, SLOT_POS + j + 1] = s * scale
        for j in range(T):             # head 1: query p attends to key p-2
            attn.W_k.weight[dh + j, SLOT_POS + j] = 1.0
            if j + 2 < T:
                attn.W_q.weight[dh + j, SLOT_POS + j + 2] = s * scale
        for j in range(T):             # head 2: query p attends to key p-3
            attn.W_k.weight[2 * dh + j, SLOT_POS + j] = 1.0
            if j + 3 < T:
                attn.W_q.weight[2 * dh + j, SLOT_POS + j + 3] = s * scale
        for i in range(V):
            attn.W_v.weight[i, SLOT_TOK + i] = 1.0
            attn.W_v.weight[dh + i, SLOT_TOK + i] = 1.0
            attn.W_v.weight[2 * dh + i, SLOT_TOK + i] = 1.0
            attn.W_o.weight[SLOT_PREV1 + i, i] = 1.0
            attn.W_o.weight[SLOT_PREV2 + i, dh + i] = 1.0
            attn.W_o.weight[SLOT_PREV3 + i, 2 * dh + i] = 1.0

        mlp = blk.mlp
        for m in (mlp.fc1, mlp.fc2):
            m.weight.zero_()
            m.bias.zero_()

        def and_unit(mlp_mod, u: int, slots: list[int]) -> None:
            for sl in slots:
                mlp_mod.fc1.weight[u, sl] = 1.0
            mlp_mod.fc1.bias[u] = -(len(slots) - 1)

        # " no" / "n't" AND-gates (prev2, prev1, tok, in hypothesis)
        u = 0
        for (c2, c1, c0) in [(" ", "n", "o"), ("n", "'", "t")]:
            and_unit(blk.mlp, u, [SLOT_PREV2 + ids[c2], SLOT_PREV1 + ids[c1], ids[c0], SLOT_HYP])
            mlp.fc2.weight[SLOT_NEG, u] = 1.0
            u += 1
        # contradiction cue words: word-initial trigram (char before it is a space or '|')
        for w in CUE_WORDS:
            for start in (" ", "|"):
                and_unit(blk.mlp, u, [SLOT_PREV3 + ids[start], SLOT_PREV2 + ids[w[0]],
                                      SLOT_PREV1 + ids[w[1]], ids[w[2]], SLOT_HYP])
                mlp.fc2.weight[SLOT_NEG, u] = 1.0
                u += 1
        # neutral cue words: same word-initial AND-gates, routed to SLOT_NEUT
        for w in NEUT_WORDS:
            for start in (" ", "|"):
                and_unit(blk.mlp, u, [SLOT_PREV3 + ids[start], SLOT_PREV2 + ids[w[0]],
                                      SLOT_PREV1 + ids[w[1]], ids[w[2]], SLOT_HYP])
                mlp.fc2.weight[SLOT_NEUT, u] = 1.0
                u += 1

        # --- block 1: match hypothesis trigram against premise trigrams ------
        blk = model.blocks[1]
        attn = blk.attn
        for m in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
            m.weight.zero_()
        for j, sl in enumerate(qslot):
            attn.W_q.weight[j, sl] = s * scale
            attn.W_k.weight[j, sl] = 1.0
            attn.W_v.weight[j, sl] = 1.0
            attn.W_o.weight[SLOT_MID + j, j] = 1.0
        # gate: hypothesis queries prefer premise keys (score + GATE_PREM)
        attn.W_q.weight[n_id, SLOT_HYP] = s * GATE_PREM * scale
        attn.W_k.weight[n_id, SLOT_PREM] = 1.0

        mlp = blk.mlp
        for m in (mlp.fc1, mlp.fc2):
            m.weight.zero_()
            m.bias.zero_()
        for j, sl in enumerate(qslot):  # AND(query coord, matched coord) -> MS
            and_unit(blk.mlp, j, [sl, SLOT_MID + j])
            mlp.fc2.weight[SLOT_MS, j] = 1.0

        # --- block 2: FULL trigram-match flag -------------------------------
        blk = model.blocks[2]
        attn = blk.attn
        for m in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
            m.weight.zero_()
        mlp = blk.mlp
        for m in (mlp.fc1, mlp.fc2):
            m.weight.zero_()
            m.bias.zero_()
        # relu(2*MS - 5 - 3*#nonletter coords of matched trigram) == 1 iff MS == 3 and all letters
        mlp.fc1.weight[0, SLOT_MS] = 2.0
        mlp.fc1.bias[0] = -5.0
        for c in nonletter:
            for off in (0, V, 2 * V):
                mlp.fc1.weight[0, SLOT_MID + off + c] = -3.0
        mlp.fc2.weight[SLOT_FULL, 0] = 1.0
        # all-letter trigram flag: relu(L(tok) + L(prev1) + L(prev2) - 2), L = sum of letter coords
        for i in range(26):
            mlp.fc1.weight[1, SLOT_TOK + i] = 1.0
            mlp.fc1.weight[1, SLOT_PREV1 + i] = 1.0
            mlp.fc1.weight[1, SLOT_PREV2 + i] = 1.0
        mlp.fc1.bias[1] = -2.0
        mlp.fc2.weight[SLOT_LTRI, 1] = 1.0

        # --- block 3: pool over hypothesis from '=' -------------------------
        blk = model.blocks[3]
        attn = blk.attn
        for m in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
            m.weight.zero_()
        attn.W_q.weight[0, ids["="]] = s * scale
        attn.W_k.weight[0, SLOT_HYP] = 1.0
        attn.W_v.weight[0, SLOT_NEG] = float(S)      # mean * S = count of negation cues
        attn.W_v.weight[1, SLOT_LET] = 1.0           # mean = letter fraction
        attn.W_v.weight[2, SLOT_FULL] = float(S)     # mean * S = count of overlapping trigrams
        attn.W_v.weight[3, SLOT_LTRI] = float(S)     # mean * S = count of letter trigrams
        attn.W_v.weight[4, SLOT_NEUT] = float(S)     # mean * S = count of neutral cue words
        attn.W_o.weight[SLOT_POOL_NEG, 0] = 1.0
        attn.W_o.weight[SLOT_POOL_LET, 1] = 1.0
        attn.W_o.weight[SLOT_POOL_OV, 2] = 1.0
        attn.W_o.weight[SLOT_POOL_LT, 3] = 1.0
        attn.W_o.weight[SLOT_POOL_NEUT, 4] = 1.0
        mlp = blk.mlp
        for m in (mlp.fc1, mlp.fc2):
            m.weight.zero_()
            m.bias.zero_()

        # --- output head ----------------------------------------------------
        # all linear, no division:
        #   '0' (entail)  logit = 10*(overlap - OV_RATIO*lt) - 0.5 - NEUT_WEIGHT*neut
        #   '2' (contra)  logit = 1000*neg + 10*(CONTRA_RATIO*lt - overlap)
        #   '1' (neutral) logit = 0.5 + NEUT_WEIGHT*neut
        # '1' gets a small constant so that it beats the all-zero logits of unused tokens
        model.head.weight.zero_()
        model.head.weight[ids["0"], SLOT_POOL_OV] = 10.0
        model.head.weight[ids["0"], SLOT_POOL_LT] = -10.0 * OV_RATIO
        model.head.weight[ids["0"], SLOT_CONST] = -0.5
        model.head.weight[ids["1"], SLOT_CONST] = 0.5
        model.head.weight[ids["0"], SLOT_POOL_NEUT] = -NEUT_WEIGHT
        model.head.weight[ids["1"], SLOT_POOL_NEUT] = NEUT_WEIGHT
        model.head.weight[ids["2"], SLOT_POOL_NEG] = 1000.0
        model.head.weight[ids["2"], SLOT_POOL_OV] = -10.0
        model.head.weight[ids["2"], SLOT_POOL_LT] = 10.0 * CONTRA_RATIO


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "hand_cue_neutral_lexicon_v1"
model_description = (
    "4-layer hand-built circuit (no LayerNorm). Layer-0 AND-gates fire on contradiction cues "
    "(' no', n't, word-initial nev/nob/emp/onl) and neutral cue words (bec/fav/bir/wed/fam/tal/you/old/pop/som). "
    "Layer-1 attention matches each hypothesis letter-trigram to premise trigrams (identity copy + ReLU AND); "
    "layer-3 pools counts from '='. Output: cue -> '2'; overlap >= 0.55 * letter-trigrams -> '0'; "
    "else '1', with neutral cue words shifting '0' toward '1'."
)


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
