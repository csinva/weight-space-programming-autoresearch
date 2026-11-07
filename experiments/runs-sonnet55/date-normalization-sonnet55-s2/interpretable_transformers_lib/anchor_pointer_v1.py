"""Snapshot anchor_pointer_v1: accuracy 1.0, 5.05e5 params (full model code; harness main block omitted here, identical to the run file)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789 ,.-/_=")
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
LAYOUT = {}
_cursor = 0
def _alloc(name, size):
    global _cursor
    LAYOUT[name] = (_cursor, size); _cursor += size
for _n, _s in [
    ("TOK", 43), ("CONST", 1), ("POS1", 1), ("POS2", 1), ("G", 10), ("EARLY", 1),
    ("ISDIG", 1), ("ISLET", 1), ("ISSL", 1), ("ISUS", 1),
    ("PREV1", 43), ("PREV2", 43), ("DIG3", 1),
    ("J4", 1), ("MIDID", 12),
    ("A_D1", 1), ("A_S1", 1), ("A_P4", 1), ("A_L", 1),
    ("F_SL", 1), ("F_H4", 1), ("F_LET", 1), ("MIDA", 12),
    ("RY", 10), ("RYY", 10), ("RD1a", 10), ("RD1b", 10), ("RS1a", 10), ("RS1b", 10), ("RDOT", 10),
    ("OUT", 43)]:
    _alloc(_n, _s)
D_MODEL = _cursor
def ix(name, i=0): return LAYOUT[name][0] + i
LAYERS = [(3, 43, 13), (6, 12, 1), (7, 10, 240)]   # (n_heads, d_head, d_ff)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        da = n_heads * d_head
        self.W_q = nn.Linear(d_model, da, bias=False); self.W_k = nn.Linear(d_model, da, bias=False)
        self.W_v = nn.Linear(d_model, da, bias=False); self.W_o = nn.Linear(da, d_model, bias=False)
    def forward(self, x):
        B, T, _ = x.shape; H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2); k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1), float("-inf"))
        return self.W_o((s.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, H * dh))

class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__(); self.fc1 = nn.Linear(d_model, d_ff); self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_ff):
        super().__init__(); self.attn = CausalSelfAttention(d_model, n_heads, d_head); self.mlp = MLP(d_model, d_ff)
    def forward(self, x):
        x = x + self.attn(x); return x + self.mlp(x)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=42, d_model=D_MODEL, layers=LAYERS):
        super().__init__()
        self.vocab_size, self.max_seq_len, self.d_model = vocab_size, max_seq_len, d_model
        self.token_emb = nn.Embedding(vocab_size, d_model); self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, h, dh, ff) for h, dh, ff in layers])
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, ids):
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        for b in self.blocks: h = b(h)
        return self.head(h)

# write_weights: embeddings (one-hot TOK, CONST, class flags ISDIG/ISLET/ISSL/ISUS; POS1=p, POS2=p^2, EARLY=p<3,
# G[i]=1 at p=31+i); peak_head(): W_q row0 = 2*C*sqrt(dh)*(anchor scalar + offsets), row1 = -C*sqrt(dh)*CONST,
# W_k row0 = POS1, row1 = POS2  -> score = -C (pos_k - s)^2 + const; earliest_head(): q=(B*CONST, -c*CONST),
# k=(flag, POS1).  L1 heads: offsets -1,-2 -> PREV1/PREV2 (token one-hots), -3 -> DIG3; L1 MLP: J4 = relu(4 digits - 3),
# MIDID[m] = relu(PREV2[c1]+PREV1[c2]+TOK[c3]-2).  L2: anchors D1 (ISDIG), S1 (ISSL), P4 (J4), L (ISUS), LET, MID.
# L3 readers: RY s=A_P4+pos-34, RYY s=A_L+pos-35, RD1a s=A_D1, RD1b A_D1+1, RS1a A_S1+1, RS1b A_S1+2, RDOT s=pos-31.
# L3 MLP: gated units relu(source + G[i] + format flags - k) -> OUT[token]; number(): tens = a if b is digit else '0',
# units = b if digit else a.  See interpretable_transformer.py (earlier revision of the run file) for the full code.
