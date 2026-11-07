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
    
    ao = attn_out[0, -1]
    
    print(f"attn_out k=0, char=4: {ao[4].item()}")
    print(f"attn_out k=1, char=7: {ao[50 + 7].item()}")
    print(f"attn_out k=2, char=19: {ao[100 + 19].item()}")
    
    # Also print attention scores!
    # q, k
    x = ln1
    B, T_dim, D = x.shape
    H, dh = model.blocks[0].attn.n_heads, model.blocks[0].attn.d_head
    q = model.blocks[0].attn.W_q(x).view(B, T_dim, H, dh).transpose(1, 2)
    k = model.blocks[0].attn.W_k(x).view(B, T_dim, H, dh).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / (dh**0.5)
    
    # Look at head k=1 (index 1), query position T-1
    print("Scores for head 1 at last position (last 5 positions):")
    print(scores[0, 1, -1, -5:].tolist())
