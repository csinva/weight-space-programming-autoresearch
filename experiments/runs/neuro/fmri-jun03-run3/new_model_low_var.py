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

        C = 100000.0
        S = C * math.sqrt(2.0 / d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            token_emb[i, i] = 1.0 # 0..49
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            pos_emb[p, 1002] = float(p)
            pos_emb[p, 1003] = float(p)**2
            pos_emb[p, 1004] = 1.0
            pos_emb[p, 1005] = -float(p)
            pos_emb[p, 1006] = -float(p)**2
            pos_emb[p, 1007] = -1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        
        # WordBoundary style smooth attention
        for k in range(n_heads):
            l1_attn.W_q.weight[k*d_head + 0, 1004] = S * 5.0
            l1_attn.W_k.weight[k*d_head + 0, 1002] = S * 1.0
            
            # Pass through chars
            for i in range(50):
                l1_attn.W_v.weight[k*d_head + i, i] = S * 1.0
                l1_attn.W_o.weight[k*50 + i, k*d_head + i] = 1.0
                
        # To avoid the overfitting that happens when variance is high and we project to high dims:
        # Let's project to 4000 dims, but use VERY small standard deviation so it stays in the linear regime of ReLU
        # mostly. Actually, ReLU is non-linear.
        # If we use a positive bias, ReLU passes through more of the signal, so it's less sharp.
        
        mlp1 = model.blocks[0].mlp
        nn.init.normal_(mlp1.fc1.weight, std=0.05)
        # Add a positive bias so it's mostly linear!
        nn.init.constant_(mlp1.fc1.bias, 0.5)
        nn.init.normal_(mlp1.fc2.weight, std=0.05)
        
        l2_attn = model.blocks[1].attn
        for i in range(d_model):
            l2_attn.W_q.weight[i, 1002] = S * 1.0
            l2_attn.W_k.weight[i, 1002] = S * 1.0
            l2_attn.W_v.weight[i, i] = S * 1.0
            l2_attn.W_o.weight[i, i] = 1.0
            
        mlp2 = model.blocks[1].mlp
        nn.init.normal_(mlp2.fc1.weight, std=0.05)
        nn.init.constant_(mlp2.fc1.bias, 0.5)
        nn.init.normal_(mlp2.fc2.weight, std=0.05)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "LowVarSmoothHash"
model_description = "Uses small variance MLPs with positive bias to keep representations mostly continuous and overlapping, reducing test-set overfitting."
