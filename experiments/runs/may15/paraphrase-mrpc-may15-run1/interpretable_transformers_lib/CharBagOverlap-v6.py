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
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
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
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
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
        self.final_ln = nn.LayerNorm(d_model)
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
    """iter3: Char-bag overlap circuit.

    Idea: compute the L1 distance between the bag-of-chars histograms of s1
    and s2, both normalised to be probability vectors over 33 char-types.
    Predict '1' (paraphrase) when L1 is small (high overlap), else '0'.

    Architecture:
      - d_model=128, n_heads=2 (head1 handles s1, head2 handles s2),
        n_layers=2, d_ff=132 (= 4 ReLU gates per char-class).

    Channel layout in the residual stream:
      - 0..32  : char one-hot (from token_emb).
      - 33     : is_s1 indicator (1 at positions 0..59).
      - 34     : is_s2 indicator (1 at positions 61..120).
      - 35..67 : s1 bag-of-chars (written by block0 attn head 1).
      - 68..100: s2 bag-of-chars (written by block0 attn head 2).
      - 101    : negative L1 distance signal (written by block1 MLP).
      - 127    : constant-1 channel (set by pos_emb on every position).

    Circuit:
      - block0 attention: head1's Q at the '=' position selects all is_s1
        positions; V copies the per-position char one-hot. So head1 averages
        char one-hots over s1 positions, giving the s1 char-frequency vector
        at the '=' position. Head2 does the same for s2.
      - block0 MLP: no-op.
      - block1 attention: no-op.
      - block1 MLP: at the '=' position, computes
            sum_c (relu(s1_c - s2_c) + relu(s2_c - s1_c)) = L1
        and writes -L1 into channel 101.
      - head: logit('1') and logit('0') are linear in chan 101 such that
        logit('1') > logit('0') iff overlap is large.
    """
    torch.manual_seed(0)
    V = task.vocab_size  # 33
    SENT_LEN = 60
    L = task.prompt_len   # 122
    EQ_POS = L - 1        # position of '='  = 121
    SEP_POS = SENT_LEN    # position of '|'  = 60
    eq_idx = task.stoi['=']
    one_idx = task.stoi['1']
    zero_idx = task.stoi['0']
    D = model.d_model     # 128
    H = model.n_heads     # 2
    DH = D // H           # 64
    DFF = model.d_ff      # 132

    # Channel constants
    CH_IS_S1 = 33
    CH_IS_S2 = 34
    CH_S1_BAG = 35        # ..67 inclusive
    CH_S2_BAG = 68        # ..100 inclusive
    CH_OVERLAP = 101
    CH_CONST = 127

    # Head-channel indices used as the routing key inside each attn head.
    # Head 1 lives in dims 0..63, head 2 in dims 64..127.
    H1_ROUTE = 0
    H2_ROUTE = 0
    LARGE = 50.0

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # --- Token embedding: char-c -> e_c on channels 0..32 ---------------
        for c in range(V):
            model.token_emb.weight[c, c] = 1.0

        # --- Position embedding: segment indicators + constant -------------
        for p in range(L):
            if p < SEP_POS:
                model.pos_emb.weight[p, CH_IS_S1] = 1.0
            elif p == SEP_POS:
                pass   # separator '|'
            elif p < EQ_POS:
                model.pos_emb.weight[p, CH_IS_S2] = 1.0
            # constant-1 channel on every position
            model.pos_emb.weight[p, CH_CONST] = 1.0

        # --- LayerNorm settings --------------------------------------------
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)

        # ====================================================================
        # Block 0: collect s1 and s2 char bags via two attention heads.
        # ====================================================================
        b0 = model.blocks[0]

        # W_q (D, D): head1 sees token '=' on channel eq_idx and emits a
        # large query vector along its routing dim H1_ROUTE. Head2 similarly.
        # We rely on the '=' token only existing at position EQ_POS.
        b0.attn.W_q.weight[H1_ROUTE, eq_idx] = LARGE
        b0.attn.W_q.weight[DH + H2_ROUTE, eq_idx] = LARGE

        # W_k (D, D): head1's key along H1_ROUTE reads the is_s1 channel,
        # head2's key along H2_ROUTE reads the is_s2 channel.
        b0.attn.W_k.weight[H1_ROUTE, CH_IS_S1] = 1.0
        b0.attn.W_k.weight[DH + H2_ROUTE, CH_IS_S2] = 1.0

        # W_v (D, D): copy char one-hot (chans 0..V-1) into the first V dims
        # of each head's value space.
        for c in range(V):
            b0.attn.W_v.weight[c, c] = 1.0          # head1 value dims 0..V-1
            b0.attn.W_v.weight[DH + c, c] = 1.0     # head2 value dims DH..DH+V-1

        # W_o (D, D): scatter head1's value back to s1-bag channels, head2's
        # to s2-bag channels.
        for c in range(V):
            b0.attn.W_o.weight[CH_S1_BAG + c, c] = 1.0
            b0.attn.W_o.weight[CH_S2_BAG + c, DH + c] = 1.0

        # Block 0 MLP: zero (no-op). Already zero.

        # ====================================================================
        # Block 1: compute -L1 distance into channel CH_OVERLAP.
        # ====================================================================
        b1 = model.blocks[1]
        # Attention is already zero (no-op).

        # MLP: fc1 has DFF=132 = 4*V rows. For each char c (0..32):
        #   row 4c+0 = relu(s1_c - s2_c)
        #   row 4c+1 = relu(s2_c - s1_c)
        # fc2 row CH_OVERLAP sums all rows with weight -1 PLUS a constant
        # bias K_const, giving chan 101 = K_const - 38.3 * L1.
        # 38.3 ≈ 6.71 / 0.175 is the LN-induced scaling factor at this layer.
        # Boundary chan_101 > ~0 ⇔ L1 < K_const/38.3.  Choose threshold L1=0.51.
        K_CONST = 14.0
        for c in range(V):
            r1 = 4 * c + 0
            r2 = 4 * c + 1
            b1.mlp.fc1.weight[r1, CH_S1_BAG + c] = 1.0
            b1.mlp.fc1.weight[r1, CH_S2_BAG + c] = -1.0
            b1.mlp.fc1.weight[r2, CH_S1_BAG + c] = -1.0
            b1.mlp.fc1.weight[r2, CH_S2_BAG + c] = 1.0
            b1.mlp.fc2.weight[CH_OVERLAP, r1] = -1.0
            b1.mlp.fc2.weight[CH_OVERLAP, r2] = -1.0
        b1.mlp.fc2.bias[CH_OVERLAP] = K_CONST

        # ====================================================================
        # Head: logit('1') - logit('0') = 2*ALPHA * normalize(chan_101).
        # chan_101 > 0 (≈ L1 < 0.51) -> predict '1'; else '0'.
        # ====================================================================
        ALPHA = 5.0
        model.head.weight[one_idx, CH_OVERLAP] = ALPHA
        model.head.weight[zero_idx, CH_OVERLAP] = -ALPHA


model_shorthand_name = "CharBagOverlap-v7"
model_description = "Char-bag overlap circuit (K_CONST=14). Two attn heads accumulate s1/s2 char-frequency vectors at '='; block-1 MLP computes K - 28*L1 into channel 101; head predicts '1' iff channel > final_ln mean. d_model=128 n_heads=2 n_layers=2 d_ff=132."


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
    model = SimpleTransformer(
        vocab_size=task.vocab_size, max_seq_len=max_seq_len,
        d_model=128, n_heads=2, n_layers=2, d_ff=132,
    )
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
