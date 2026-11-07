import torch
from interpretable_transformer import build_embedder

embedder = build_embedder()
model = embedder.model
T = model.max_seq_len

with torch.no_grad():
    input_ids = torch.zeros(1, T, dtype=torch.long, device=embedder.device)
    input_ids[0, -1] = 4 # e
    input_ids[0, -2] = 7 # h
    input_ids[0, -3] = 19 # t
    
    x0 = model.token_emb(input_ids) + model.pos_emb(torch.arange(T, device=embedder.device).unsqueeze(0))
    ln1 = model.blocks[0].ln1(x0)
    attn_out = model.blocks[0].attn(ln1)
    x1 = x0 + attn_out
    ln2 = model.blocks[0].ln2(x1)
    
    S = 100000.0 * (2.0 / 1020)**0.5
    
    # Let's print the actual values of ln2 scaled by S
    v = ln2[0, -1] * S
    print(f"k=0, char=4 ('e'): {v[4].item()}")
    print(f"k=1, char=7 ('h'): {v[50 + 7].item()}")
    print(f"k=2, char=19 ('t'): {v[100 + 19].item()}")
