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

        # Char_Combinations_Map got 0.0338 but was extremely overfitted (train_corr 0.3060).
        # We need to regularize it. We can regularize the massive random map by constraining 
        # its dimensionality and utilizing the variance anchor trick properly to prevent LayerNorm scaling issues.
        # Let's try combining the "Character-to-Meaning" wide map with strict LayerNorm bypassing.
        
        # 1 network, but we use the variance anchor.
        # We will map characters to a 64-dim subspace, and then expand that to a 4000-dim 
        # random semantic space, heavily regularized.
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 # Space
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 # Alpha
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0 # Identity
                
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            pos_emb[p, 28] = S * (p / max_seq_len) # Global
            pos_emb[p, 29] = S * 1.0 # Constant
            
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        h_start = 0
        d_start = 0
        f_start = 0
        
        dim_net = 128
        ff_net = 3840
        
        # Layer 1 Attention: Pool characters
        l1_attn.W_q.weight[0, 29] = 5.0
        l1_attn.W_k.weight[0, 28] = 3.0 # Strong decay (mostly current word)
        
        for i in range(28):
            l1_attn.W_v.weight[i, i] = 1.0
            l1_attn.W_o.weight[30 + i, i] = S * 1.0
            
        # MLP: Expands 28 char counts to a highly constrained random representation
        # To avoid overfitting, we use lower standard deviation
        std_dev = 0.3
        nn.init.normal_(mlp1.fc1.weight[0:ff_net, 30:58], std=std_dev)
        # Random biases to create varied activation thresholds
        nn.init.normal_(mlp1.fc1.bias[0:ff_net], std=0.5) 
        
        nn.init.normal_(mlp1.fc2.weight[58:58+dim_net, 0:ff_net], std=std_dev * S / math.sqrt(ff_net))

        # Layer 2 Attention: Exponential integration of the semantic features
        l2_attn.W_q.weight[0, 29] = 5.0
        l2_attn.W_k.weight[0, 28] = 0.5 # Slow decay to integrate context
        
        for i in range(dim_net):
            l2_attn.W_v.weight[i, 58 + i] = 1.0
            l2_attn.W_o.weight[58 + i, i] = S * 1.0
            
        # Layer 2 MLP: Just pass through or slight mixing
        nn.init.normal_(mlp2.fc1.weight[0:ff_net, 58:58+dim_net], std=std_dev)
        nn.init.normal_(mlp2.fc2.weight[58:58+dim_net, 0:ff_net], std=std_dev * S / math.sqrt(ff_net))

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "Regularized_Semantic_Map"
model_description = "A refined version of Char_Combinations_Map using a lower standard deviation and proper variance anchors to reduce extreme overfitting while mapping char sums to high-dim semantic space."
