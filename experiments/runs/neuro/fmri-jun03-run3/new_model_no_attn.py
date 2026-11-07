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
        d_ff = model.blocks[0].mlp.fc1.out_features
        n_heads = model.blocks[0].attn.n_heads
        d_head = model.blocks[0].attn.d_head

        C = 10000.0
        S = C * math.sqrt(2.0 / d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            token_emb[i, i] = 1.0 # 0..49
        model.token_emb.weight.data.copy_(token_emb)

        # No pos emb needed if we only look at local token
        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1004] = 1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        # No attention, just pass through the local token representation!
        l1_attn = model.blocks[0].attn
        for i in range(d_model):
            # attend to self
            l1_attn.W_q.weight[i, 1004] = S * 1.0
            l1_attn.W_k.weight[i, 1004] = S * 1.0
            l1_attn.W_v.weight[i, i] = S * 1.0
            l1_attn.W_o.weight[i, i] = 1.0
                
        # MLP acts as a random hash over current character
        mlp1 = model.blocks[0].mlp
        nn.init.normal_(mlp1.fc1.weight, std=1.0)
        nn.init.normal_(mlp1.fc2.weight, std=1.0)
        
        # L2 attention also just passes through
        l2_attn = model.blocks[1].attn
        for i in range(d_model):
            l2_attn.W_q.weight[i, 1004] = S * 1.0
            l2_attn.W_k.weight[i, 1004] = S * 1.0
            l2_attn.W_v.weight[i, i] = S * 1.0
            l2_attn.W_o.weight[i, i] = 1.0
            
        mlp2 = model.blocks[1].mlp
        nn.init.normal_(mlp2.fc1.weight, std=1.0)
        nn.init.normal_(mlp2.fc2.weight, std=1.0)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "SingleCharHash"
model_description = "Eliminates attention lookback entirely. Just hashes the current character to see what the baseline is for no context."
