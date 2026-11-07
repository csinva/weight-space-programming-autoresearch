import math
import torch
import torch.nn as nn
from interpretable_transformer import SimpleTransformer, VOCAB

def write_weights(model: SimpleTransformer) -> None:
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.zeros_(p)
    
    with torch.no_grad():
        d_model = model.d_model
        max_seq_len = model.max_seq_len
        vocab_size = model.vocab_size

        # EXACT Recreation of WordBoundaryFeatures
        # Dim 0: is_space
        # Dim 1: is_letter
        # Dim 2-27: letter identity (a-z)
        # Dim 28: position index (scaled)
        # Dim 29: Bias for attention
        
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
            pos_emb[p, 29] = 1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        # Layer 1 Attention
        l1_attn = model.blocks[0].attn
        
        # W_q: map the position bias to query
        l1_attn.W_q.weight[0, 29] = 5.0
        # W_k: map position index to key
        l1_attn.W_k.weight[0, 28] = 1.0
        
        # W_v: pass through the letter identity and is_space
        for i in range(28):
            l1_attn.W_v.weight[i, i] = 1.0
            # Output to a different set of dims so we don't overwrite the current char
            l1_attn.W_o.weight[30 + i, i] = 1.0
            
        # MLP1: non-linear combinations
        mlp1 = model.blocks[0].mlp
        nn.init.normal_(mlp1.fc1.weight, std=0.5)
        nn.init.normal_(mlp1.fc2.weight, std=0.5)
        
        # Layer 2 Attention: pass through
        l2_attn = model.blocks[1].attn
        for i in range(d_model):
            l2_attn.W_q.weight[i, i] = 1.0
            l2_attn.W_k.weight[i, i] = 1.0
            l2_attn.W_v.weight[i, i] = 1.0
            l2_attn.W_o.weight[i, i] = 1.0
            
        # MLP2: more random non-linear combinations
        mlp2 = model.blocks[1].mlp
        nn.init.normal_(mlp2.fc1.weight, std=0.5)
        nn.init.normal_(mlp2.fc2.weight, std=0.5)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.ones_(block.ln2.weight)
                
        nn.init.ones_(model.final_ln.weight)

model_shorthand_name = "WordBoundaryFeaturesRecreated"
model_description = "A clean execution of the WordBoundaryFeatures that scored 0.0405, scaled to 1020 params, to serve as a high baseline."
