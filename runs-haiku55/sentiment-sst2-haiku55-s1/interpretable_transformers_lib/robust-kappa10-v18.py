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
# Hand-written sentiment knowledge (general English; not mined from any dataset)
# ---------------------------------------------------------------------------

POS_WORDS = """
good great fine nice love loved lovely beautiful wonderful brilliant excellent
superb terrific fantastic fun funny hilarious delightful charming enjoyable
engaging entertaining compelling smart witty warm touching moving heartfelt
memorable best better perfect strong solid rich sweet pleasure pleasing delight
joy joyful happy hope hopeful masterful masterpiece impressive remarkable
stunning gorgeous graceful deft tender clever intelligent wise enjoy enjoys
enjoyed satisfying satisfyingly winning winner triumph fresh original inventive
lively vibrant captivating riveting thrilling exciting engrossing effective
effortless poignant powerful profound lush amusing playful pleasant recommend
recommended appealing adorable admirable accomplished dazzling genuine honest
humor humorous laugh laughs rewarding romantic charm bright fascinating
favorite greatest
beautifully wonderfully delightfully brilliantly superbly perfectly charmingly
fabulous marvelous marvellous splendid spectacular exceptional extraordinary
captivated entertained enjoyment fascinated uplifting inspiring inspired inspire
inspiration endearing endearingly heartwarming gripping suspenseful elegant
elegance stylish vivid vividly sincere sincerely earnest wonder wondrous marvel
savory sumptuous lovable likable likeable brave bravely courageous noble nobly
funnier smarter greater warmer sweeter richer stronger brighter wiser
""".split()

NEG_WORDS = """
bad worst worse boring bore bored dull dullness stupid awful terrible poor
poorly weak mess messy fails failed failure lame flat tedious bland silly dumb
annoying annoyed painful empty lifeless forgettable mediocre predictable
clumsy trite waste wasted wasteful ugly hollow pointless sloppy tiresome tired
unfunny shallow lousy muddled unpleasant incoherent confusing dreadful dismal
flimsy vapid witless disappointing disappointed lacks lacking unconvincing sad
ridiculous fake phony cheap crude nonsense dragging drag dragged slow sluggish
stale generic cliched contrived overlong unfocused unimaginative uninspired
unsatisfying worthless hate hated dislike disgusting boredom laughable
ludicrous pretentious tasteless mindless hackneyed derivative cloying
lethargic plodding ponderous
badly terribly horribly weakly dully clumsily awfully lamely boringly tediously
flops flop bombs bomb disaster disastrous unwatchable unbearable offensive
obnoxious irritating irritated grating grim gloomy depressing dreary drab dim
soulless insipid silliness idiotic moronic inept incompetent amateurish juvenile
immature vulgar lazy superficial superficially formulaic overwrought bloated
slog slogs cheesy corny tacky sappy saccharine maudlin melodramatic
flawed flaw flaws shoddy uninteresting uneven unengaging dismally dire dimwitted
stilted wooden awkward awkwardly chaotic chaos clunky crap crappy dud duds trashy
sleepy tedium vacuous vacant cringe cringeworthy unappealing unlikable unlikeable
problem problems gross limp lackluster listless monotonous monotony numbing
pathetic preachy pompous redundant repetitive routine scattered shapeless smug
turgid uninvolving unimpressive weary wearisome wretched forced
""".split()

# Words that flip the polarity of a sentiment word when they directly precede it.
NEGATORS = ["not", "n't", "no", "never", "nothing", "nor", "without", "neither"]

# Character set that counts as a word boundary (space, padding, or the '=' separator).
BOUNDARY_CHARS = [" ", "_", "="]
# Require the character before a word to be a boundary (whole-word match)?
LEFT_BOUNDARY = True
# Constant positive evidence added at every position (tie-break toward label '1').
TIE_BIAS = 0.5


POS_WORDS = list(dict.fromkeys(POS_WORDS))
NEG_WORDS = list(dict.fromkeys(NEG_WORDS))
# Coverage knobs (ablation only): keep the first FRAC of each lexicon.
FRAC_POS = 1.0
FRAC_NEG = 1.0
POS_WORDS = POS_WORDS[: round(len(POS_WORDS) * FRAC_POS)]
NEG_WORDS = NEG_WORDS[: round(len(NEG_WORDS) * FRAC_NEG)]


def build_units() -> list[tuple[list, list]]:
    """Return a list of MLP units. Each unit is (pattern, outputs):
      pattern: [(offset, [allowed chars]), ...]  — the unit fires iff every
               offset k holds one of its allowed chars (offset 0 = current char).
      outputs: [(residual_name, desired_value), ...] written when the unit fires.
    """
    units = []
    # Single-word units: [boundary] word [boundary], ending at the boundary.
    for words, dim in ((POS_WORDS, "SP"), (NEG_WORDS, "SN")):
        for w in words:
            L = len(w)
            pat = [(0, BOUNDARY_CHARS)] + [(k, [w[L - k]]) for k in range(1, L + 1)]
            if LEFT_BOUNDARY:
                pat.append((L + 1, BOUNDARY_CHARS))
            units.append((pat, [(dim, 1.0)]))
    # Negated units: [boundary] word ' ' negator [boundary]; cancels the single-word
    # unit's contribution and writes the opposite polarity.
    for neg in NEGATORS:
        Ln = len(neg)
        for words, sign in ((POS_WORDS, +1), (NEG_WORDS, -1)):
            for w in words:
                L = len(w)
                pat = [(0, BOUNDARY_CHARS)] + [(k, [w[L - k]]) for k in range(1, L + 1)]
                pat.append((L + 1, [" "]))
                pat += [(L + 2 + t, [neg[Ln - 1 - t]]) for t in range(Ln)]
                if LEFT_BOUNDARY:
                    pat.append((L + 2 + Ln, BOUNDARY_CHARS))
                if sign > 0:
                    outs = [("SP", -1.0), ("SN", +1.0)]
                else:
                    outs = [("SN", -1.0), ("SP", +1.0)]
                units.append((pat, outs))
    return units


UNITS = build_units()
N_GATHER = max(off for pat, _ in UNITS for off, _ in pat)   # attention offsets 1..N_GATHER
N_UNITS = len(UNITS)


def residual_layout(max_seq_len: int, vocab_size: int, n_gather: int) -> dict:
    """Named residual-stream dimensions (the model's 'bus')."""
    P, V, H = max_seq_len, vocab_size, n_gather
    base = P + V + V * H
    return {
        "POS": 0,              # one-hot position (index i at position i)
        "TOK": P,              # one-hot current character
        "GATH": P + V,         # one-hot character at offset k, block k-1 (k=1..H)
        "SP": base,            # per-position positive evidence (layer-1 MLP)
        "SN": base + 1,        # per-position negative evidence (layer-1 MLP)
        "AP": base + 2,        # summed positive evidence at '=' (layer-2 attention)
        "AN": base + 3,        # summed negative evidence at '=' (layer-2 attention)
        "K": base + 4,         # constant KAPPA at every position: makes LayerNorm statistics fixed
        "Z": base + 5,         # always-zero dimension used as a LayerNorm-compensation column
        "D": base + 6,
    }


# Constant offset written to every position. It dominates the residual's norm so
# LayerNorm's mean/std are the same at every position, even where some attention
# heads have no valid source (the first few characters).
KAPPA = 10.0


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)

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
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """2-layer causal transformer with a one-hot positional basis (d_head = max_seq_len)."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        n_heads: int = N_GATHER,
        n_layers: int = 2,
        d_ff: int = N_UNITS + 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.d_head = max_seq_len
        self.d_model = residual_layout(max_seq_len, vocab_size, n_heads)["D"]

        self.token_emb = nn.Embedding(vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(max_seq_len, self.d_model)
        self.blocks = nn.ModuleList(
            [Block(self.d_model, n_heads, self.d_head, d_ff) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, vocab_size, bias=False)

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

def _ln_stats(n_ones: int, D: int) -> tuple[float, float]:
    """Mean and std LayerNorm uses when the residual is KAPPA plus n_ones one-hot vectors."""
    s = KAPPA + n_ones                  # sum of entries
    sq = KAPPA * KAPPA + n_ones         # sum of squared entries
    mu = s / D
    return mu, math.sqrt(sq / D - mu * mu + 1e-5)


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Circuit (all weights hand-set):
      1. Embeddings: one-hot character + one-hot position.
      2. Layer 1 attention: head k attends exactly to position i-k, copying the
         character at offset k into residual block k. Together the blocks give a
         window of the last H characters ending at the current position.
      3. Layer 1 MLP: one unit per lexicon pattern. A single-word unit fires only
         on an exact whole-word match ending just before the current position and
         writes +1 to SP (positive word) or SN (negative word). Negated units
         (if NEGATORS is non-empty) cancel that and write the flipped polarity.
         One constant unit adds TIE_BIAS to SP at every position.
      4. Layer 2 attention: uniform attention over the prefix. At '=' (last
         position) this averages SP and SN over all positions, giving AP and AN.
      5. Readout: logit('1') = AP - AN, logit('0') = AN - AP.
    """
    torch.manual_seed(0)
    V, P, H, D, dh = model.vocab_size, model.max_seq_len, model.n_heads, model.d_model, model.d_head
    o = residual_layout(P, V, H)
    stoi = task.stoi
    BOUND = [stoi[c] for c in BOUNDARY_CHARS]

    def gath(k: int, c: int) -> int:
        return o["GATH"] + V * (k - 1) + c

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
                m.bias.zero_()

        # 1. Embeddings
        for c in range(V):
            model.token_emb.weight[c, o["TOK"] + c] = 1.0
        for p in range(P):
            model.pos_emb.weight[p, o["POS"] + p] = 1.0
            model.pos_emb.weight[p, o["K"]] = KAPPA

        # 2. Layer 1 attention: gather offsets 1..H
        mu1, s1 = _ln_stats(2, D)            # LN1 input = token + position (2 ones)
        alpha = 20.0 * math.sqrt(dh)         # match score = 20 after 1/sqrt(dh)
        Wq = torch.zeros(H * dh, D)
        Wk = torch.zeros(H * dh, D)
        Wv = torch.zeros(H * dh, D)
        Wo = torch.zeros(D, H * dh)
        for k in range(1, H + 1):
            h = k - 1
            for i in range(P):
                m = i - k
                if 0 <= m < P:
                    Wq[h * dh + m, o["POS"] + i] = alpha
            for j in range(P):
                Wk[h * dh + j, o["POS"] + j] = 1.0
            for c in range(V):
                Wv[h * dh + c, o["TOK"] + c] = 1.0
                Wo[gath(k, c), h * dh + c] = 1.0

        # LayerNorm compensation: rescale by sigma and cancel the mean via the always-zero Z column
        def ln_compensated(W: torch.Tensor) -> torch.Tensor:
            W = s1 * W
            W[:, o["Z"]] = -W.sum(dim=1)
            return W

        a1 = model.blocks[0].attn
        a1.W_q.weight.copy_(ln_compensated(Wq))
        a1.W_k.weight.copy_(ln_compensated(Wk))
        a1.W_v.weight.copy_(ln_compensated(Wv))
        a1.W_o.weight.copy_(Wo)

        # 3. Layer 1 MLP: pattern units + constant bias unit
        mu2, s2 = _ln_stats(2 + H, D)        # LN2 input = token + position + H gathers
        n_units = len(UNITS)
        W1 = torch.zeros(n_units + 1, D)
        b1 = torch.zeros(n_units + 1)
        fc2 = torch.zeros(D, n_units + 1)
        for u, (pat, outs) in enumerate(UNITS):
            for off, chars in pat:
                for c in chars:
                    if off == 0:
                        W1[u, o["TOK"] + stoi[c]] = 1.0
                    else:
                        W1[u, gath(off, stoi[c])] = 1.0
            b1[u] = -len(pat) + 0.5          # all entries matched -> 0.5, any miss -> <= -0.5
            for dim, val in outs:
                fc2[o[dim], u] = 2.0 * val   # relu output is 0.5 on a full match
        bias_u = n_units
        W1[bias_u, o["POS"]:o["POS"] + P] = 1.0  # exactly 1 at every position
        b1[bias_u] = -0.5                        # relu -> 0.5 constant
        # Layer 2 averages over the prompt_len positions the '=' attends to, so a
        # per-position bias of TIE_BIAS / prompt_len contributes exactly TIE_BIAS in total,
        # the same units as one word match (which contributes 1 / prompt_len per position).
        fc2[o["SP"], bias_u] = 2.0 * TIE_BIAS / task.prompt_len
        mlp1 = model.blocks[0].mlp
        mlp1.fc1.weight.copy_(s2 * W1)
        mlp1.fc1.bias.copy_(b1 + mu2 * W1.sum(dim=1))
        mlp1.fc2.weight.copy_(fc2)

        # 4. Layer 2 attention: uniform over prefix (Q=K=0); head 0 averages SP and SN
        a2 = model.blocks[1].attn
        a2.W_v.weight[0, o["SP"]] = 1.0
        a2.W_v.weight[1, o["SN"]] = 1.0
        a2.W_o.weight[o["AP"], 0] = 1.0
        a2.W_o.weight[o["AN"], 1] = 1.0

        # 5. Readout: logit('1') = AP - AN, logit('0') = AN - AP
        model.head.weight[stoi["1"], o["AP"]] = 1.0
        model.head.weight[stoi["1"], o["AN"]] = -1.0
        model.head.weight[stoi["0"], o["AP"]] = -1.0
        model.head.weight[stoi["0"], o["AN"]] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "robust-kappa10-v18"
model_description = (
    "Robustness check of the LayerNorm-constant trick: full lexicon-expanded-v4 model with "
    "KAPPA=10 instead of 100 (a weaker constant offset, so LayerNorm stats vary more by position)."
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
