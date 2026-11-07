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

        # We know WordBoundaryFeatures logic is solid.
        # But we also know that having 10 heads with different scales helps.
        # And we know low variance helps avoid memorization.
        
        C = 10000.0
        S = C * math.sqrt(2.0 / d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            token_emb[i, i] = 1.0 # 0..49
            if c == ' ':
                token_emb[i, 50] = 1.0
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            pos_emb[p, 1002] = float(p)
            pos_emb[p, 1004] = 1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        # L1 Attention:
        # Seek space token, but also pull in the characters softly
        l1_attn = model.blocks[0].attn
        for k in range(n_heads):
            l1_attn.W_q.weight[k*d_head + 0, 1004] = S * 5.0
            
            if k < 5:
                # 5 heads seek spaces
                l1_attn.W_k.weight[k*d_head + 0, 50] = S * 2.0
            else:
                # 5 heads seek recent characters smoothly
                decay = 1.0 + float(k - 5)
                l1_attn.W_k.weight[k*d_head + 0, 1002] = S * decay
                
            for i in range(50):
                l1_attn.W_v.weight[k*d_head + i, i] = S * 1.0
                l1_attn.W_o.weight[k*50 + i, k*d_head + i] = 1.0
                
        # MLP1: Very wide, moderate variance
        mlp1 = model.blocks[0].mlp
        nn.init.normal_(mlp1.fc1.weight, std=0.2)
        nn.init.constant_(mlp1.fc1.bias, 0.5)
        nn.init.normal_(mlp1.fc2.weight, std=0.2)
        
        # L2 Attention:
        # Pass through
        l2_attn = model.blocks[1].attn
        for i in range(d_model):
            l2_attn.W_q.weight[i, 1004] = S * 5.0
            l2_attn.W_k.weight[i, 1004] = S * 5.0
            l2_attn.W_v.weight[i, i] = S * 1.0
            l2_attn.W_o.weight[i, i] = 1.0
            
        # MLP2: Very wide, moderate variance
        mlp2 = model.blocks[1].mlp
        nn.init.normal_(mlp2.fc1.weight, std=0.2)
        nn.init.constant_(mlp2.fc1.bias, 0.5)
        nn.init.normal_(mlp2.fc2.weight, std=0.2)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "SmoothSpaceHash"
model_description = "Uses 5 heads to seek spaces and 5 heads to smoothly gather characters. Uses low variance (std=0.2) with positive bias in deep MLPs to avoid overfitting."
