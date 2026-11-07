import torch
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
    
    x1_last = x1[0, -1]
    
    print(f"dtype: {x1_last.dtype}")
    
    # Let's write our own var
    mean = torch.mean(x1_last).item()
    print(f"mean: {mean}")
    
    sq_diff = torch.sum((x1_last - mean)**2).item()
    print(f"sum( (x - mean)^2 ): {sq_diff}")
    print(f"PyTorch var: {torch.var(x1_last, unbiased=False).item()}")
    
    # Are there any other large values?
    top_vals, top_idx = torch.topk(torch.abs(x1_last), 5)
    print("Top 5 absolute values:")
    for v, idx in zip(top_vals, top_idx):
        print(f"Index {idx}: {v.item()}")
        
