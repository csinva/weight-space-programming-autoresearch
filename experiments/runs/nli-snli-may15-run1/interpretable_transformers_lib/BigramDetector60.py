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
    """Iter 21 — 2-layer bigram-detection circuit.

    Layer 0: shift attention copies prev-char into residual via quadratic
        positional features; MLP fires 60 hyp-bigram detectors as
        ReLU(curr[c2] + prev[c1] - 1).
    Layer 1: uniform-over-hyp attention at LAST_POS counts both char and
        bigram occurrences in the hypothesis.
    Head: in-sample multinomial-LR coefficients over (28 hyp char counts,
        60 hyp bigram counts).
    """
    torch.manual_seed(0)

    for block in model.blocks:
        block.ln1 = nn.Identity()
        block.ln2 = nn.Identity()
    model.final_ln = nn.Identity()

    id_0 = task.stoi["0"]
    id_1 = task.stoi["1"]
    id_2 = task.stoi["2"]

    HYP_START = task.SENT_LEN + 1
    HYP_END   = 2 * task.SENT_LEN + 1
    LAST_POS  = task.prompt_len - 1
    HYP_LEN   = HYP_END - HYP_START

    CHARS = "abcdefghijklmnopqrstuvwxyz _"
    assert len(CHARS) == 28
    char_idx = {c: i for i, c in enumerate(CHARS)}

    # Residual layout (d_model = 224)
    IS_HYP_DIM    = 28
    IS_LAST_DIM   = 29
    POS_LIN_DIM   = 30
    POS_QUAD_DIM  = 31
    POS_CONST_DIM = 32
    PREV_BASE     = 33   # ..60   (28 prev-char dims)
    BIG_BASE      = 61   # ..120  (60 bigram-detector dims)
    HCOUNT_BASE   = 121  # ..148  (28 hyp char counts)
    BCOUNT_BASE   = 149  # ..208  (60 hyp bigram counts)

    BIGRAMS = ['__', 'e ', 'in', ' a', 's ', 'n ', 'a ', 'th', 'he', 'ng', 'g ', ' i', 'an', ' t', 're', 'is', ' w', 'ar', ' s', 'er', ' p', 'ma', ' o', ' m', ' b', ' c', 'le', 'on', 'r ', ' h', 't ', 'wo', 'pl', 'd ', 'at', 'om', 'hi', 'pe', ' f', 'ou', ' d', 'it', 'y ', 'ea', 'o ', 'to', 'en', 'or', 'me', 'ch', ' g', 'st', 'ri', 'al', 'op', 'nd', 'e_', 'ro', 'ti', 'si']
    K = len(BIGRAMS)
    assert K == 60
    CHAR_COEF   = [[0.09697605115972109, -0.12722750396042337, 0.002815168541470877, 0.10222619397417747, -0.0573237425372366, -0.06959964335754515, -0.027029639205940835, -0.09535227860121469, 0.0732140086080797, 0.13646680374339815, -0.038907552524882955, -0.03315101037850439, -0.13265596954315806, -0.032754220490929735, 0.10184235782298195, -0.14844560890376504, -0.5153930589835561, -0.03130830767559911, 0.09869582421233573, 0.036609566547083104, 0.07912697773527673, -0.028755232211535545, 0.018540487776315444, 0.14927769750037048, -0.1652118581524989, -0.5225145577554107, -0.2011436983219262, -0.271506086778272], [0.010309937579132232, -0.20125510813262917, -0.011579470039819182, -0.12454012238676854, -0.06457754736037069, -0.027577805779268758, 0.16497879358675213, 0.10410328653192712, 0.08642804199692578, 0.23026139604515156, 0.00543326146369345, -0.023408867458134187, -0.0034744344444710973, -0.24752601504829702, -0.01472746998106679, 0.09068701412056707, 0.10836308296839996, -0.05202554108051352, 0.03599837934335166, 0.028638133708363184, 0.08561253806512609, 0.03276809105399929, 0.12619407055397358, 0.28614696477634616, 0.16863367018601072, -0.09268408518001883, 0.1321485608451908, -0.007675138692760337], [-0.1072859887388514, 0.3284826120931705, 0.00876430149835493, 0.022313928412631714, 0.12190128989758231, 0.09717744913676496, -0.1379491543809649, -0.00875100793071514, -0.15964205060492928, -0.3667281997884416, 0.03347429106119407, 0.056559877836628976, 0.13613040398750978, 0.2802802355391877, -0.08711488784185277, 0.05775859478336171, 0.4070299760149303, 0.0833338487561021, -0.13469420355564368, -0.06524770025537815, -0.1647395158004179, -0.0040128588424582735, -0.1447345583301426, -0.4354246622766477, -0.003421812033514423, 0.6151986429356003, 0.06899513747682696, 0.2791812254710987]]
    BIGRAM_COEF = [[0.2932573220645418, -0.08708247557453368, -0.2684396775313653, 0.11199648085527046, -0.11668199625962415, -0.0677599887922787, 0.17929493723453555, 0.04707654071521732, 0.008071534853116858, 0.20213473236200646, -0.08944072246985346, 0.37401937096541354, 0.03357215834937039, 0.10757525141390224, 0.23325227564224735, -0.28852847469234644, 0.369521507433304, 0.007361180886623305, -0.02594408930788653, 0.3567187947452136, 0.1200371301928936, 0.2563461047392188, 0.36037800593900954, -0.018985908759805152, 0.21066211634092707, -0.07307516114267969, 0.056112684631840846, -0.10946470939060982, -0.14694916275987988, 0.033988595770092944, -0.12907940195165005, -0.25149919391418135, 0.19176474424254633, 0.05056125097034555, -0.15637848233914925, -0.20140233278422834, 0.3882358837872374, 0.3401735294095877, 0.005534911330894213, 0.15028999008345906, -0.09310035107229465, -0.0013474188774763936, 0.10201027858465664, -0.17055447643026755, 0.07443600379924974, -0.03827684546210715, -0.033731838227524395, 0.01174314967752617, 0.4803246948002901, -0.11292187647989851, 0.12252343142995188, 0.07158347525173019, -0.17016655721550328, -0.0064907042334168535, 0.18039707838963898, -0.09208216585607372, 0.06809442987951947, 0.015690799960050454, -0.02618892972053272, -0.16660214185627206], [-0.010188149732475879, 0.030011894817940022, 0.2422711809827725, -0.07940528075685077, -0.08225605570568707, -0.057155386678720055, -0.05740292515979688, -0.19323129400126168, 0.19944219162689727, -0.2256993876633355, -0.02704225399589943, -0.3350068584859611, 0.21376228586904306, 0.009934291327274074, 0.0028131178128010742, 0.28846197919364236, -0.22296119698952346, 0.1466929879338265, 0.0015882216206717888, -0.08134044797660832, -0.10754484037087075, -0.10932263947588507, -0.31379936817960324, 0.04370944611912507, 0.03392913051601536, 0.048792174070196925, -0.02946628108775209, 0.411042634384334, 0.143151421067416, -0.02601584778544364, 0.062476865155414424, -0.044539383202784985, -0.1396398320806141, -0.002238233581089411, -0.17987862634038454, 0.05208008902849506, -0.35070622591544715, 0.09467730586137213, 0.11420787998834923, 0.0019642454196183187, -0.015404429696641585, -0.16082896909170452, -0.11501529131075802, 0.03926871427935533, -0.1150825716876808, 0.22960187475170404, 0.42927154808145795, 0.22362342860219672, -0.08283648521869207, 0.11145121630229644, -0.24537868535014476, -0.1344200317696371, 0.0006637418979108777, 0.0937755982230314, -0.14212358456196142, 0.1874319574675489, 0.09382454591151135, -0.05519124625735827, 0.12766934288570173, -0.1291620703493923], [-0.2830691723321838, 0.05707058075656785, 0.02616849654877579, -0.03259120009837741, 0.19893805196528622, 0.12491537547104727, -0.12189201207474744, 0.14615475328591535, -0.20751372647982888, 0.02356465530139332, 0.1164829764657808, -0.03901251247896905, -0.24733444421838724, -0.11750954274127566, -0.23606539345527988, 6.649549895372943e-05, -0.14656031044390033, -0.15405416882038936, 0.024355867687211298, -0.27537834676876555, -0.012492289822103665, -0.1470234652635184, -0.046578637759309593, -0.024723537359335464, -0.24459124685681424, 0.024282987072474723, -0.026646403544085574, -0.301577924993655, 0.003797741692436276, -0.007972747984625312, 0.06660253679623744, 0.29603857711705567, -0.0521249121619833, -0.048323017389229436, 0.3362571086796243, 0.1493222437557047, -0.03752965787153441, -0.434850835271083, -0.11974279131922112, -0.15225423550301434, 0.10850478076891816, 0.16217638796898726, 0.013005012726149366, 0.13128576215078488, 0.040646567888406135, -0.19132502928975983, -0.3955397098540131, -0.23536657827968624, -0.39748820958147135, 0.0014706601776532408, 0.12285525392014753, 0.0628365565178933, 0.1695028153175355, -0.08728489398962831, -0.03827349382751286, -0.09534979161156311, -0.16191897579104392, 0.0395004462973175, -0.10148041316524839, 0.29576421220571575]]
    INTERCEPT   = [-0.08421818861867529, -0.03653043480808538, 0.12074862342678566]

    # ---- Token embedding: char one-hot to dims 0..27 ----
    nn.init.zeros_(model.token_emb.weight)
    for i, c in enumerate(CHARS):
        if c in task.stoi:
            model.token_emb.weight.data[task.stoi[c], i] = 1.0

    # ---- Position embedding: hyp/last flags + quadratic pos features ----
    nn.init.zeros_(model.pos_emb.weight)
    for i in range(HYP_START, HYP_END):
        model.pos_emb.weight.data[i, IS_HYP_DIM] = 1.0
    model.pos_emb.weight.data[LAST_POS, IS_LAST_DIM] = 1.0
    for i in range(model.max_seq_len):
        model.pos_emb.weight.data[i, POS_LIN_DIM]   = float(i)
        model.pos_emb.weight.data[i, POS_QUAD_DIM]  = float(i * i)
        model.pos_emb.weight.data[i, POS_CONST_DIM] = 1.0

    # =====================================================================
    # Layer 0: shift attention (q.k peaks at j=i-1) + bigram MLP
    # =====================================================================
    attn0 = model.blocks[0].attn
    nn.init.zeros_(attn0.W_q.weight)
    nn.init.zeros_(attn0.W_k.weight)
    nn.init.zeros_(attn0.W_v.weight)
    nn.init.zeros_(attn0.W_o.weight)

    # q.k = -SCALE * (j - (i-1))**2  using:
    #   Q[i] = [SCALE*1, SCALE*2*(i-1), -SCALE*(i-1)**2]  (3 dims)
    #   K[j] = [-j**2, j, 1]                              (3 dims)
    # Expand (i-1)**2 = i**2 - 2*i + 1.
    SHIFT_SCALE = 100.0
    # Q row 0 = SHIFT_SCALE * pos_const
    attn0.W_q.weight.data[0, POS_CONST_DIM] = SHIFT_SCALE
    # Q row 1 = SHIFT_SCALE * 2*(i-1) = 2*SHIFT_SCALE*pos_lin - 2*SHIFT_SCALE*pos_const
    attn0.W_q.weight.data[1, POS_LIN_DIM]   = 2.0 * SHIFT_SCALE
    attn0.W_q.weight.data[1, POS_CONST_DIM] = -2.0 * SHIFT_SCALE
    # Q row 2 = -SHIFT_SCALE*(i**2 - 2i + 1) = -SHIFT_SCALE*pos_quad + 2*SHIFT_SCALE*pos_lin - SHIFT_SCALE*pos_const
    attn0.W_q.weight.data[2, POS_QUAD_DIM]  = -SHIFT_SCALE
    attn0.W_q.weight.data[2, POS_LIN_DIM]   = 2.0 * SHIFT_SCALE
    attn0.W_q.weight.data[2, POS_CONST_DIM] = -SHIFT_SCALE
    # K row 0 = -j**2
    attn0.W_k.weight.data[0, POS_QUAD_DIM]  = -1.0
    # K row 1 = j
    attn0.W_k.weight.data[1, POS_LIN_DIM]   = 1.0
    # K row 2 = 1
    attn0.W_k.weight.data[2, POS_CONST_DIM] = 1.0

    # V copies char one-hot (dims 0..27) directly
    for d in range(28):
        attn0.W_v.weight.data[d, d] = 1.0
    # W_o routes attn-out dims 0..27 to PREV_BASE..PREV_BASE+27
    for d in range(28):
        attn0.W_o.weight.data[PREV_BASE + d, d] = 1.0

    # ---- Layer 0 MLP: bigram detectors ReLU(curr[c2] + prev[c1] - 1) ----
    mlp0 = model.blocks[0].mlp
    nn.init.zeros_(mlp0.fc1.weight); nn.init.zeros_(mlp0.fc1.bias)
    nn.init.zeros_(mlp0.fc2.weight); nn.init.zeros_(mlp0.fc2.bias)
    for k, big in enumerate(BIGRAMS):
        c1, c2 = big[0], big[1]
        mlp0.fc1.weight.data[k, char_idx[c2]]             = 1.0
        mlp0.fc1.weight.data[k, PREV_BASE + char_idx[c1]] = 1.0
        mlp0.fc1.bias.data[k] = -1.0
        mlp0.fc2.weight.data[BIG_BASE + k, k] = 1.0

    # =====================================================================
    # Layer 1: uniform-over-hyp attention at LAST_POS counts chars+bigrams
    # =====================================================================
    attn1 = model.blocks[1].attn
    nn.init.zeros_(attn1.W_q.weight)
    nn.init.zeros_(attn1.W_k.weight)
    nn.init.zeros_(attn1.W_v.weight)
    nn.init.zeros_(attn1.W_o.weight)

    attn1.W_q.weight.data[0, IS_LAST_DIM] = 100.0
    attn1.W_k.weight.data[0, IS_HYP_DIM]  = 1.0

    # V output dim d (0..27)   = HYP_LEN * char_one_hot[d]
    # V output dim 28+k         = HYP_LEN * bigram_det[k]
    for d in range(28):
        attn1.W_v.weight.data[d, d] = float(HYP_LEN)
    for k in range(K):
        attn1.W_v.weight.data[28 + k, BIG_BASE + k] = float(HYP_LEN)
    for d in range(28):
        attn1.W_o.weight.data[HCOUNT_BASE + d, d] = 1.0
    for k in range(K):
        attn1.W_o.weight.data[BCOUNT_BASE + k, 28 + k] = 1.0

    # Layer 1 MLP: zero (passthrough)
    mlp1 = model.blocks[1].mlp
    nn.init.zeros_(mlp1.fc1.weight); nn.init.zeros_(mlp1.fc1.bias)
    nn.init.zeros_(mlp1.fc2.weight); nn.init.zeros_(mlp1.fc2.bias)

    # =====================================================================
    # Head: hand-coded multinomial LR over (char counts, bigram counts)
    # =====================================================================
    SCALE = 10.0
    nn.init.zeros_(model.head.weight)
    for k, label_id in enumerate([id_0, id_1, id_2]):
        for d in range(28):
            model.head.weight.data[label_id, HCOUNT_BASE + d] = SCALE * CHAR_COEF[k][d]
        for b in range(K):
            model.head.weight.data[label_id, BCOUNT_BASE + b] = SCALE * BIGRAM_COEF[k][b]
        # Intercept routed via POS_CONST_DIM (=1 at every position incl. LAST)
        model.head.weight.data[label_id, POS_CONST_DIM] = SCALE * INTERCEPT[k]


model_shorthand_name = "BigramDetector60"
model_description = "Iter21: 2-layer bigram-detection circuit. L0 shift-attn writes prev-char via quadratic pos features; L0 MLP fires 60 ReLU bigram detectors. L1 uniform-over-hyp attn counts chars+bigrams. Head=multinomial-LR. d_model=224, 1 head, 2 layers, d_ff=60."


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
        d_model=224, n_heads=1, n_layers=2, d_ff=60,
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
