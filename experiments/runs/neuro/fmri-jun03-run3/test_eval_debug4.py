import torch
import numpy as np
from interpretable_transformer import build_embedder, VOCAB

embedder = build_embedder()
model = embedder.model
T = model.max_seq_len

with torch.no_grad():
    input_ids = torch.zeros(1, T, dtype=torch.long, device=embedder.device)
    x0 = model.token_emb(input_ids) + model.pos_emb(torch.arange(T, device=embedder.device).unsqueeze(0))
    ln1 = model.blocks[0].ln1(x0)
    attn_out = model.blocks[0].attn(ln1)
    x1 = x0 + attn_out
    ln2 = model.blocks[0].ln2(x1)
    
    x1_last = x1[0, -1]
    
    print(f"x1_last sum of squares: {torch.sum(x1_last**2).item()}")
    print(f"2*C^2: {2 * 100000.0**2}")
    
    var = torch.var(x1_last, unbiased=False).item()
    std = torch.std(x1_last, unbiased=False).item()
    print(f"x1_last variance: {var}, std: {std}")
    print(f"Calculated S: {100000.0 / std}")

