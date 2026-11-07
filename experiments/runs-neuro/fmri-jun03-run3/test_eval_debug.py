import torch
import numpy as np
from interpretable_transformer import build_embedder, VOCAB

embedder = build_embedder()
model = embedder.model
print("Model built.")

s = "this is the"
print(f"Testing string: '{s}'")
input_strings = [s]
B = len(input_strings)
T = model.max_seq_len
input_ids = torch.zeros(B, T, dtype=torch.long, device=embedder.device)
for i, seq in enumerate(input_strings):
    for j, char in enumerate(seq[-T:]):
        if char in VOCAB:
            input_ids[i, j] = VOCAB.index(char)
        else:
            input_ids[i, j] = VOCAB.index('<unk>')

with torch.no_grad():
    hidden_states = model(input_ids)
    
    # We can inspect intermediate activations!
    x0 = model.token_emb(input_ids) + model.pos_emb(torch.arange(T, device=embedder.device).unsqueeze(0))
    ln1 = model.blocks[0].ln1(x0)
    attn_out = model.blocks[0].attn(ln1)
    x1 = x0 + attn_out
    ln2 = model.blocks[0].ln2(x1)
    
    # Now MLP1
    fc1_out = model.blocks[0].mlp.fc1(ln2)
    relu_out = torch.nn.functional.relu(fc1_out)
    
    last_relu = relu_out[0, -1, :]
    print("Non-zero entries in ReLU output at final position:")
    nz = last_relu.nonzero().squeeze(-1)
    for idx in nz:
        print(f"Index {idx}: value {last_relu[idx].item()}")

    print(f"Max fc1_out at final position: {fc1_out[0, -1].max().item()}")
    print(f"Top 5 fc1_out values: {torch.topk(fc1_out[0, -1], 5).values.tolist()}")
