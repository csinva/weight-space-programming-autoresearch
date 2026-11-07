import torch
from interpretable_transformer import build_embedder

embedder = build_embedder()
model = embedder.model
T = model.max_seq_len

with torch.no_grad():
    input_ids = torch.zeros(1, T, dtype=torch.long, device=embedder.device)
    input_ids[0, -1] = 4 # e
    
    x0 = model.token_emb(input_ids) + model.pos_emb(torch.arange(T, device=embedder.device).unsqueeze(0))
    ln1 = model.blocks[0].ln1(x0)
    attn_out = model.blocks[0].attn(ln1)
    x1 = x0 + attn_out
    
    mean = torch.mean(x1[0, -1]).item()
    print("mean:", mean)

