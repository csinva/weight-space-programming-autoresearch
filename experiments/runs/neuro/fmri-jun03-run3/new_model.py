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

        # We will use LayerNorm pass-through trick
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
            
            # Balance out the sum to keep LayerNorm mean ≈ 0
            pos_emb[p, 1005] = -float(p)
            pos_emb[p, 1006] = -float(p)**2
            pos_emb[p, 1007] = -1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        # L1 Attention: Exact Relative Position Extraction (up to k=10)
        l1_attn = model.blocks[0].attn
        beta = 100.0
        
        for k in range(n_heads):
            # Q = S * [2*beta*p, -2*beta*k, 1]
            l1_attn.W_q.weight[k*d_head + 0, 1002] = S * 2.0 * beta
            l1_attn.W_q.weight[k*d_head + 0, 1004] = S * (-2.0 * beta * k)
            l1_attn.W_q.weight[k*d_head + 2, 1004] = S * 1.0
            
            # K = S * [m, 1, -beta*m^2]
            l1_attn.W_k.weight[k*d_head + 0, 1002] = S * 1.0
            l1_attn.W_k.weight[k*d_head + 1, 1004] = S * 1.0
            l1_attn.W_k.weight[k*d_head + 2, 1003] = S * (-beta)
            
            # V = pass through char
            for i in range(50):
                l1_attn.W_v.weight[k*d_head + i, i] = S * 1.0
                l1_attn.W_o.weight[k*50 + i, k*d_head + i] = 1.0

        # L1 MLP: Semantic Projection
        # Instead of matched filter, we just project the exact n-gram characters
        # via random weights. Since there are 10*50 = 500 characters, we can project
        # them to 4000 dimensions randomly, then back to d_model.
        
        mlp1 = model.blocks[0].mlp
        
        # We want relu to actually activate for characters, but there are 0s and 1s.
        # 10 heads * 50 characters = 500 dimensions.
        # These 500 dimensions have values around ~1/S (wait, ln2 is multiplied by S).
        # Ah! So the input to fc1 will be exactly 1.0/S or 0.0, multiplied by the layer norm.
        # Wait, the output of attn is just `v`. We set W_o such that dim `k*50+i` is `1.0`.
        # So x1 = x0 + attn_out.
        # x0 has C at 1000, -C at 1001, etc.
        # After ln2, the values are scaled by S.
        # So the dimension `k*50 + i` in ln2 is equal to `attn_out[k*50 + i] * S` ? 
        # Yes, exactly! And `attn_out` has value 1.0/S. 
        # So `ln2[k*50 + i]` is exactly `1.0` if that character is present!
        
        # Let's project these 500 dimensions randomly
        # We will use large random weights to get a nice high-dimensional semantic hash
        random_proj = torch.randn(d_ff, 500) * 1.0
        
        for k in range(10):
            for i in range(50):
                mlp1.fc1.weight[:, k*50 + i] = random_proj[:, k*50 + i]
                
        # We want roughly half of neurons to activate, so bias=0
        mlp1.fc1.bias.data.zero_()
        
        # Then we project back to dense representations
        mlp1.fc2.weight.data.copy_(torch.randn(d_model, d_ff) * 1.0)
        
        # Disable L2
        # Set LayerNorms
        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "SemanticNGramHash"
model_description = "Uses the exact relative position trick to extract the last 10 characters perfectly, then projects them using a random 2-layer MLP to create a distributed semantic n-gram hash."
