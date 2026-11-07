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

        # LayerNorm ruins sparsity. If we zero out 956 dimensions, LayerNorm computes variance over
        # 1024 dimensions, which is extremely small (var ~ 1/16th of what it would be). 
        # Then it divides by sqrt(var), which MULTIPLIES the 64 active dimensions by ~4!
        # This completely changes the distribution of the activations entering the MLP compared to d_model=64.
        
        # To fix this, we can set the LayerNorm weights to counteract this scaling.
        # But even better, let's go back to an approach that uses the full 1020-dim capacity but 
        # restricts the MLP rank/variance to prevent overfitting.
        
        # Let's try the Hybrid approach: 
        # Explicit Orthographic Features + Shallow Network
        
        # We know WordBoundaryHash got 0.0312.
        # MultiScaleWordBoundary got 0.0347.
        # WordBoundaryFeatures_Scaled_LayerNorm got 0.0350.
        
        # Let's make an ensemble of independent 1-layer MLPs.
        # Deep MLPs overfit massively on exactly extracted text.
        
        C = 10000.0
        S = C * math.sqrt(2.0 / d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            token_emb[i, i] = 1.0 # 0..49
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1002] = float(p)
            pos_emb[p, 1004] = 1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        
        for k in range(n_heads):
            # Scale decays from 1.0 to 10.0 (looking back 1 to 10 characters softly)
            decay_scale = 1.0 + float(k)
            l1_attn.W_q.weight[k*d_head + 0, 1004] = S * 5.0
            l1_attn.W_k.weight[k*d_head + 0, 1002] = S * decay_scale
            
            for i in range(50):
                l1_attn.W_v.weight[k*d_head + i, i] = S * 1.0
                l1_attn.W_o.weight[k*50 + i, k*d_head + i] = 1.0
                
        # MLP1 acts as a massive linear hash (no non-linearity if we don't stack it)
        # Actually ReLU makes it non-linear.
        # Let's use VERY large variance to force sparse activation of ReLUs.
        mlp1 = model.blocks[0].mlp
        nn.init.normal_(mlp1.fc1.weight, std=2.0)
        nn.init.normal_(mlp1.fc2.weight, std=2.0)
        
        # L2 Attention: Pass through
        l2_attn = model.blocks[1].attn
        for k in range(n_heads):
            l2_attn.W_q.weight[k*d_head + 0, 1004] = S * 5.0
            l2_attn.W_k.weight[k*d_head + 0, 1002] = S * 5.0 # High so it only looks at self
            
            for i in range(d_head):
                l2_attn.W_v.weight[k*d_head + i, k*d_head + i] = S * 1.0
                l2_attn.W_o.weight[k*d_head + i, k*d_head + i] = 1.0
                
        # MLP2: Pass through to keep it 1-layer deep functionally!
        mlp2 = model.blocks[1].mlp
        # If we set fc1 to Identity and fc2 to Identity, we bypass the 2nd MLP.
        for i in range(d_model):
            mlp2.fc1.weight[i, i] = 1.0
            # ReLU will zero out negatives, so we need a positive bias
            mlp2.fc1.bias[i] = 10.0
            mlp2.fc2.weight[i, i] = 1.0
            # subtract bias back
            mlp2.fc2.bias[i] = -10.0

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "ShallowMultiScaleHash"
model_description = "Uses multi-scale attention heads, followed by exactly ONE high-variance MLP layer. The second layer is bypassed to prevent deep overfitting."
