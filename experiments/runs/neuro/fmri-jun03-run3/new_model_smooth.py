import math
import torch
import torch.nn as nn
from interpretable_transformer import SimpleTransformer, VOCAB

def write_weights(model: SimpleTransformer) -> None:
    """Populate `model`'s parameters in-place. No training allowed."""
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.zeros_(p)
    
    with torch.no_grad():
        d_model = model.d_model
        max_seq_len = model.max_seq_len
        vocab_size = model.vocab_size

        # Dim 0: is_space
        # Dim 1: is_letter
        # Dim 2-27: letter identity (a-z)
        # Dim 28: position index (scaled)
        
        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = 1.0
            elif c.isalpha():
                token_emb[i, 1] = 1.0
                token_emb[i, 2 + (ord(c) - ord('a'))] = 1.0
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 28] = p / max_seq_len
            pos_emb[p, 29] = 1.0 # Bias for attention
        model.pos_emb.weight.data.copy_(pos_emb)

        # Layer 1 Attention: Look back to find the most recent space
        # We will use all 10 heads to look back different amounts.
        n_heads = model.blocks[0].attn.n_heads
        d_head = model.blocks[0].attn.d_head
        
        l1_attn = model.blocks[0].attn
        
        for k in range(n_heads):
            # Different heads have different position bias scales
            lookback = 1.0 + k * 1.5
            l1_attn.W_q.weight[k*d_head + 0, 29] = lookback
            l1_attn.W_k.weight[k*d_head + 0, 28] = 1.0
            
            # They also pay attention to spaces
            l1_attn.W_k.weight[k*d_head + 1, 0] = 5.0
            l1_attn.W_q.weight[k*d_head + 1, 29] = 1.0
            
            # Pass through features to different output dims
            for i in range(28):
                l1_attn.W_v.weight[k*d_head + i, i] = 1.0
                l1_attn.W_o.weight[30 + k*28 + i, k*d_head + i] = 1.0
            
        # MLP1: non-linear combinations of the current char (0-27) and the recent context
        mlp1 = model.blocks[0].mlp
        nn.init.normal_(mlp1.fc1.weight, std=0.5)
        nn.init.normal_(mlp1.fc2.weight, std=0.5)
        
        # Layer 2 Attention: pass through
        l2_attn = model.blocks[1].attn
        for k in range(n_heads):
            for i in range(d_head):
                l2_attn.W_q.weight[k*d_head + i, k*d_head + i] = 1.0
                l2_attn.W_k.weight[k*d_head + i, k*d_head + i] = 1.0
                l2_attn.W_v.weight[k*d_head + i, k*d_head + i] = 1.0
                l2_attn.W_o.weight[k*d_head + i, k*d_head + i] = 1.0
            
        # MLP2: more random non-linear combinations
        mlp2 = model.blocks[1].mlp
        nn.init.normal_(mlp2.fc1.weight, std=0.5)
        nn.init.normal_(mlp2.fc2.weight, std=0.5)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.ones_(block.ln2.weight)
                
        nn.init.ones_(model.final_ln.weight)

model_shorthand_name = "SmoothWordBoundaryHash"
model_description = "A scaled and slightly modified version of WordBoundaryFeatures (which scored 0.0405) that uses all 10 heads to gather a multi-scale view of the recent word boundaries and chars, feeding into a deep random MLP hash."
