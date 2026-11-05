"""
Snapshot SinkCount_final -- digit-counting-10 (accuracy 1.0 on the 200-example eval).
Condensed copy: architecture + write_weights + constants. The evaluation harness is
identical to the one in interpretable_transformer.py and is omitted here.

Task: prompt = 11 digits + '=', answer = 2-digit count of the OTHER digits equal to the last digit.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1), float("-inf"))
        return self.W_o((s.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, D))


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None
        if self.mlp is not None:
            self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=64, n_heads=1, n_layers=2, d_ff=(0, 20)):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff[i]) for i in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        for b in self.blocks:
            h = b(h)
        return self.head(self.final_ln(h))


SQ, V_POS, GGAIN, A = 0.37, 4.3243, 0.1, 300.0
L1S, EXCL, HEADPEN = 4.0, 3.0, 10.0
# layer-normed sink-mass reading for count m: h(m) = BETA + GAMMA/(m + R)  (fitted, matches m=1..6 exactly)
BETA, GAMMA, R = -0.28503, 1.107, 6.158
THETA = [BETA + GAMMA / (u + 1 + R) for u in range(10)]


def write_weights(model, task):
    # Residual layout: 0-9 token digit | 10 '=' | 11-26 position one-hot
    #                  27-36 copy of LAST digit | 37 sink mass g | 38-47 logit for digit u
    for p in model.parameters():
        p.data.zero_()
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm):
            mod.weight.data.fill_(1.0)
    D = 10
    TOK, POS, COPY, G, OUT = 0, 11, 27, 37, 38
    with torch.no_grad():
        model.token_emb.weight[torch.arange(11), torch.arange(11)] = 1.0
        model.pos_emb.weight[torch.arange(16), POS + torch.arange(16)] = 1.0
        # L1: each position attends to itself (p<=10) or to the last digit (p>=11); copy token to COPY.
        a1 = model.blocks[0].attn
        for p in range(16):
            a1.W_q.weight[p if p <= 10 else 10, POS + p] = L1S
            a1.W_k.weight[p, POS + p] = 1.0
        for j in range(D):
            a1.W_v.weight[j, TOK + j] = 1.0
            a1.W_o.weight[COPY + j, j] = 0.2
        # L2: query = copied digit, key = digit token -> matches score higher; '=' (pos 11) is a sink
        # with value 1, so its attention mass g = 1/(c + m(E-1)) encodes the match count m.
        a2 = model.blocks[1].attn
        for j in range(D):
            a2.W_q.weight[j, COPY + j] = SQ
            a2.W_k.weight[j, TOK + j] = 1.0
        a2.W_q.weight[10, POS + 11] = EXCL
        a2.W_q.weight[10, POS + 12] = EXCL
        a2.W_k.weight[10, POS + 12] = -EXCL      # mask pos-12's own key
        a2.W_v.weight[0, POS + 11] = 1.0
        a2.W_o.weight[G, 0] = GGAIN
        # MLP: nearest-neighbour decode of g -> m; two relus per class u=m-1 give -|h - theta_u|.
        mlp = model.blocks[1].mlp
        for u in range(D):
            mlp.fc1.weight[2 * u, G], mlp.fc1.bias[2 * u] = A, -A * THETA[u]
            mlp.fc1.weight[2 * u + 1, G], mlp.fc1.bias[2 * u + 1] = -A, A * THETA[u]
            for k in (2 * u, 2 * u + 1):
                mlp.fc1.weight[k, POS + 12] = A / 30      # gate: off at pos 11, ~0 at pos 12
                mlp.fc1.bias[k] -= (A / 30) * V_POS
                mlp.fc2.weight[OUT + u, k] = -1.0
        for u in range(D):
            model.head.weight[u, OUT + u] = 1.0
        model.head.weight[0, 10] = 1.0               # pos 11 ('='): tens digit is 0
        model.head.weight[10, POS + 12] = -HEADPEN   # never emit '=' at pos 12


model_shorthand_name = "SinkCount_final"
