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

        # LayerNorm pass-through trick
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

        # L1 Attention: Smooth lookback with varying window sizes.
        # HybridSemanticHash scored 0.0340, which is good.
        # But notice how train_corr was 0.0649! It didn't overfit!
        # The key to not overfitting is using smooth distributions and letting the 
        # random MLPs create continuous overlapping representations.
        
        l1_attn = model.blocks[0].attn
        space_idx = VOCAB.index(' ')
        
        # We have 10 heads. Let's make them all look back smoothly, but with different slopes.
        for k in range(n_heads):
            # slope controls how far back the head looks
            # k=0 is very local (steep slope), k=9 is very global (flat slope)
            slope = -5.0 + k * 0.5 
            
            # Q[k] = S * slope
            l1_attn.W_q.weight[k*d_head + 0, 1002] = S * slope
            
            # K[k] = S * 1.0
            l1_attn.W_k.weight[k*d_head + 0, 1002] = S * 1.0
            
            # Add strong affinity to spaces for half the heads
            if k % 2 == 0:
                l1_attn.W_k.weight[k*d_head + 0, space_idx] = S * 5.0
            
            # V passes through characters
            for i in range(50):
                l1_attn.W_v.weight[k*d_head + i, i] = S * 1.0
                l1_attn.W_o.weight[k*50 + i, k*d_head + i] = 1.0

        # L1 MLP: Semantic Projection
        mlp1 = model.blocks[0].mlp
        random_proj = torch.randn(d_ff, 500) * 1.0
        
        for k in range(10):
            for i in range(50):
                mlp1.fc1.weight[:, k*50 + i] = random_proj[:, k*50 + i]
                
        # Small bias
        mlp1.fc1.bias.data.copy_(torch.randn(d_ff) * 0.5)
        mlp1.fc2.weight.data.copy_(torch.randn(d_model, d_ff) * 1.0)
        
        # Layer 2 Attention: pass through
        l2_attn = model.blocks[1].attn
        for i in range(d_model):
            # To make it identity, we can just use the pos_emb to construct a strong diagonal attention
            l2_attn.W_q.weight[i, 1002] = S * 1.0
            l2_attn.W_k.weight[i, 1002] = S * 1.0
            l2_attn.W_v.weight[i, i] = S * 1.0
            l2_attn.W_o.weight[i, i] = 1.0
            
        # MLP2: deep hash
        mlp2 = model.blocks[1].mlp
        nn.init.normal_(mlp2.fc1.weight, std=0.5)
        nn.init.normal_(mlp2.fc2.weight, std=0.5)

        # Set LayerNorms
        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "LocalContextHash"
model_description = "Uses 10 heads with different exponential decay slopes (some with space affinity) to build a multi-scale bag-of-characters over the recent context, feeding into a 2-layer random hashing stack."
