import torch
import numpy as np
from interpretable_transformer import build_embedder, VOCAB

embedder = build_embedder()
model = embedder.model

s = "this is the"
T = model.max_seq_len
input_ids = torch.zeros(1, T, dtype=torch.long, device=embedder.device)
for j, char in enumerate(s):
    input_ids[0, j] = VOCAB.index(char)

with torch.no_grad():
    x0 = model.token_emb(input_ids) + model.pos_emb(torch.arange(T, device=embedder.device).unsqueeze(0))
    ln1 = model.blocks[0].ln1(x0)
    attn_out = model.blocks[0].attn(ln1)
    x1 = x0 + attn_out
    ln2 = model.blocks[0].ln2(x1)
    
    # Check ln2 values for the last token ('e')
    last_ln2 = ln2[0, len(s)-1]
    
    print("last_ln2 max:", last_ln2.max().item())
    print("last_ln2 min:", last_ln2.min().item())
    
    # Print the values of ln2 at character slots
    # for k=0 (dim 0..49)
    print("k=0 (0..49) max:", last_ln2[0:50].max().item())
    # for k=1 (dim 50..99)
    print("k=1 (50..99) max:", last_ln2[50:100].max().item())
    
    # Now fc1_out
    fc1_out = model.blocks[0].mlp.fc1(ln2)
    print("fc1_out for 'the' (assuming 'the' is top word?):")
    # Actually just print top 5
    print("Top 5 fc1_out values:", torch.topk(fc1_out[0, len(s)-1], 5).values.tolist())

