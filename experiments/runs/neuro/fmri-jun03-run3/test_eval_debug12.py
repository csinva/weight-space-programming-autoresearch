import torch
from interpretable_transformer import build_embedder

embedder = build_embedder()
model = embedder.model
T = model.max_seq_len

with torch.no_grad():
    input_ids = torch.zeros(1, T, dtype=torch.long, device=embedder.device)
    input_ids[0, -1] = 4 # e
    
    x0 = model.token_emb(input_ids) + model.pos_emb(torch.arange(T, device=embedder.device).unsqueeze(0))
    S = 100000.0 * (2.0 / 1020)**0.5
    
    print("x0[4]:", x0[0, -1, 4].item())

