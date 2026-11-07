import torch
from interpretable_transformer import build_embedder

embedder = build_embedder()
model = embedder.model
T = model.max_seq_len

with torch.no_grad():
    input_ids = torch.zeros(1, T, dtype=torch.long, device=embedder.device)
    # let's set characters 0 to 49
    input_ids[0, -1] = 4 # e
    input_ids[0, -2] = 7 # h
    input_ids[0, -3] = 19 # t
    
    x0 = model.token_emb(input_ids) + model.pos_emb(torch.arange(T, device=embedder.device).unsqueeze(0))
    ln1 = model.blocks[0].ln1(x0)
    attn_out = model.blocks[0].attn(ln1)
    x1 = x0 + attn_out
    ln2 = model.blocks[0].ln2(x1)
    
    # We want to check the dot product of ln2 for the word "the"
    # Word "the" has length 3, expected_sum = 4.
    
    # Wait, how does the fc1 weight match it?
    # W_padded = 'the'
    # chars_from_end = ['e', 'h', 't']
    
    fc1_weight = model.blocks[0].mlp.fc1.weight
    fc1_bias = model.blocks[0].mlp.fc1.bias
    
    # Let's see which neuron corresponds to "the"
    # TOP_WORDS is used. Let's just find the max output over all neurons.
    fc1_out = model.blocks[0].mlp.fc1(ln2[0, -1])
    max_val, max_idx = torch.max(fc1_out, dim=0)
    print(f"Max out: {max_val.item()} at idx {max_idx.item()}")
    
    # Print the sum before bias
    w = fc1_weight[max_idx]
    b = fc1_bias[max_idx]
    
    dot_prod = torch.dot(ln2[0, -1], w).item()
    print(f"Dot product: {dot_prod}")
    print(f"Bias: {b.item()}")
    
