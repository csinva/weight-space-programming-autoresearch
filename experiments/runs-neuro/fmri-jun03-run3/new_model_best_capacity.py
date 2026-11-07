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

        # Let's try to replicate the 0.0405 success exactly but within the 1024-dim model.
        # How? By zeroing out 960 dimensions in the MLP entirely.
        # But we MUST handle LayerNorm properly. If we zero out 960 dims, the variance across 1024
        # dims will be var(64_active) * (64/1024).
        # LayerNorm divides by sqrt(var(64_active) / 16).
        # Which multiplies the 64 active dims by 4.
        # So we just scale the MLP weights by 0.25 to counteract it!
        
        # Actually, let's just make the active dimensions perfectly identical to the 64-dim version.
        
        C = 10000.0

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

        l1_attn = model.blocks[0].attn
        
        # Scale Q/K properly since LayerNorm before Attention ALSO multiplies by 4!
        # Original: W_q=5.0, W_k=1.0. With LN * 4, they become 20.0 and 4.0.
        # We need to divide them by 4 to keep the attention logits the same.
        l1_attn.W_q.weight[0, 29] = 5.0 / 4.0
        l1_attn.W_k.weight[0, 28] = 1.0 / 4.0
        
        for i in range(28):
            l1_attn.W_v.weight[i, i] = 1.0 / 4.0  # divided by 4 because input is 4x larger
            l1_attn.W_o.weight[30 + i, i] = 1.0
            
        mlp1 = model.blocks[0].mlp
        # Input to MLP is also multiplied by 4 by LayerNorm
        nn.init.normal_(mlp1.fc1.weight[:256, :64], std=0.5 / 4.0)
        nn.init.normal_(mlp1.fc2.weight[:64, :256], std=0.5)
        
        l2_attn = model.blocks[1].attn
        for i in range(64):
            l2_attn.W_q.weight[i, i] = 1.0 / 4.0
            l2_attn.W_k.weight[i, i] = 1.0 / 4.0
            l2_attn.W_v.weight[i, i] = 1.0 / 4.0
            l2_attn.W_o.weight[i, i] = 1.0
            
        mlp2 = model.blocks[1].mlp
        nn.init.normal_(mlp2.fc1.weight[:256, :64], std=0.5 / 4.0)
        nn.init.normal_(mlp2.fc2.weight[:64, :256], std=0.5)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.ones_(block.ln2.weight)
                
        nn.init.ones_(model.final_ln.weight)

model_shorthand_name = "ExactSparseWordBoundary"
model_description = "Creates exactly one instance of the 64-dim WordBoundaryFeatures, properly scaled to counteract LayerNorm's variance shift on sparse inputs."
