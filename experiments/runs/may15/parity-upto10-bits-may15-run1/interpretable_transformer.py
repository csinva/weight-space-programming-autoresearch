"""
Interpretable transformer for parity-upto10-bits.

Circuit (one attention layer + one MLP, d_model=4, no LayerNorm):

  Token embedding writes "is_one_bit" into channel 0 of the residual stream
  and "is_equals" into channel 1 (other tokens are zero).
  Position embedding is unused (all zeros).

  Attention: W_q = W_k = 0 -> uniform softmax over all visible positions.
  W_v reads channel 0 with gain 11, so the (1/11)-normalized attention
  output at the final '=' position (position 10, attends to all 11 prompt
  positions) is exactly N1 in one channel, where N1 is the number of
  '1' bits. W_o copies that channel into residual channel 2.

  MLP (d_ff = 11) is a hand-built piecewise-linear lookup that maps
  ch2 = N1  ->  ch0 = parity(N1) = (N1 mod 2).
  Hidden neurons are h_k = ReLU(ch2 - k) for k = 0..10. Triangular
  "bumps" centered at each integer k are formed by the second derivative
  pattern (h_{k-1} - 2 h_k + h_{k+1}), and the bumps for odd k are summed.

  The output head maps residual at position 10 to logits:
      logit('0') = +0.5 * is_equals - 10 * parity
      logit('1') = -0.5 * is_equals + 10 * parity
  so the argmax is '0' when parity=0 and '1' when parity=1.

  Architecture: d_model=3, n_layers=1, n_heads=1, d_ff=11, no LayerNorm.
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
# Architecture (LayerNorm replaced with Identity for clean hand-design)
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
        # LayerNorm replaced with Identity so hand-designed weights survive.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 3,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 9,
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
# Hand-built parity circuit
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter in-place to implement parity-upto10-bits."""
    torch.manual_seed(0)

    # Vocab is "01=_" -> token ids 0,1,2,3.
    stoi = task.stoi
    id_0 = stoi["0"]
    id_1 = stoi["1"]
    id_eq = stoi["="]
    id_pad = stoi["_"]

    with torch.no_grad():
        # ---- Token embedding (vocab_size x d_model=3) ----
        # ch0 = "is_one_bit", ch1 = "is_equals", ch2 = scratch (used later
        # to hold the attention count N1 and the MLP-derived parity).
        emb = torch.zeros_like(model.token_emb.weight)
        emb[id_1, 0] = 1.0
        emb[id_eq, 1] = 1.0
        # id_0 and id_pad stay at zero -> they contribute nothing.
        model.token_emb.weight.copy_(emb)

        # ---- Position embedding: unused ----
        model.pos_emb.weight.zero_()

        # ---- Attention: uniform softmax aggregator of "is_one_bit" ----
        attn = model.blocks[0].attn
        attn.W_q.weight.zero_()
        attn.W_k.weight.zero_()
        # V reads ch0 with gain 11. After uniform softmax over the 11 prompt
        # positions, the (1/11)-weighted sum exactly recovers N1.
        Wv = torch.zeros_like(attn.W_v.weight)
        Wv[2, 0] = 11.0
        attn.W_v.weight.copy_(Wv)
        # W_o writes attention's channel 2 into residual channel 2.
        Wo = torch.zeros_like(attn.W_o.weight)
        Wo[2, 2] = 1.0
        attn.W_o.weight.copy_(Wo)

        # After attention at the '=' position (idx 10, attends to all 11
        # prompt positions uniformly), residual channel 2 = N1.

        # ---- MLP: in-place transform ch2: N1 -> parity(N1) via N - 2*floor(N/2) ----
        # floor(N/2) for integer N in 0..10 equals sum_{k=1..5} step(N - 2k + 0.5),
        # which we realise with the ReLU pair (ReLU(N - (2k-1)) - ReLU(N - 2k)).
        # The k=5 pair would need h_10 = ReLU(N - 10), but for N in [0, 10] that
        # term is always 0, so it is dropped. We keep h_1..h_9 (9 neurons).
        # The MLP adds -2 * floor(N/2) to channel 2; since channel 2 already
        # equals N, the post-MLP value is N - 2*floor(N/2) = parity(N).
        mlp = model.blocks[0].mlp
        W1 = torch.zeros_like(mlp.fc1.weight)
        b1 = torch.zeros_like(mlp.fc1.bias)
        # h_j = ReLU(ch2 - j) for j = 1..9 (9 hidden neurons).
        for j in range(1, 10):
            W1[j - 1, 2] = 1.0
            b1[j - 1] = -float(j)
        mlp.fc1.weight.copy_(W1)
        mlp.fc1.bias.copy_(b1)

        # For each k in {1..5}: coefficient -2 on h_{2k-1}, +2 on h_{2k}.
        # k=5 contributes -2*h_9 only (h_10 dropped because it is always 0).
        W2 = torch.zeros_like(mlp.fc2.weight)
        b2 = torch.zeros_like(mlp.fc2.bias)
        for k in range(1, 6):
            W2[2, (2 * k - 1) - 1] = -2.0  # h_{2k-1}
            if 2 * k <= 9:
                W2[2, (2 * k) - 1] = 2.0   # h_{2k}, only when index <= 9
        mlp.fc2.weight.copy_(W2)
        mlp.fc2.bias.copy_(b2)
        # Residual channel 2 at position 10 is now parity(N1) in {0, 1}.

        # ---- Output head ----
        # logit('0') = +0.5 * is_equals - 10 * parity
        # logit('1') = -0.5 * is_equals + 10 * parity
        Wh = torch.zeros_like(model.head.weight)
        Wh[id_0, 1] = 0.5
        Wh[id_0, 2] = -10.0
        Wh[id_1, 1] = -0.5
        Wh[id_1, 2] = 10.0
        model.head.weight.copy_(Wh)


model_shorthand_name = "ParityFloorSubD3F9"
model_description = (
    "d_model=3, 1 head, 1 layer, d_ff=9, no LayerNorm. Same N - 2*floor(N/2) "
    "in-place parity circuit as ParityInPlaceFloorSub, with the always-zero "
    "h_10 neuron dropped (since N1 <= 10 makes ReLU(N1 - 10) trivially 0). "
    "Minimal ReLU width for this approach."
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
