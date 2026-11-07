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

        # Let's combine multiple independent heads that all integrate at DIFFERENT timescales
        # into a massive shared MLP.
        # So far, we've used strict sub-networks (Attn -> MLP -> Attn -> MLP).
        # What if Layer 1 has 16 heads, each pooling characters with a DIFFERENT exponential decay?
        # This provides the MLP with a complete timeline of recent orthography (e.g. current char, 
        # last 2 chars, last 5 chars, last 10 chars).
        # The MLP then mixes these timescales to detect patterns (e.g. "short word" vs "long phrase").
        
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
            
            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0
            
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        num_timescales = 16
        dim_per_head = 28
        
        # --- LAYER 1: Multi-scale temporal pooling ---
        for head in range(num_timescales):
            h_start = head * d_head
            d_out = 30 + head * dim_per_head # Output space for this timescale
            
            # Timescale ranges from 0.5 (slow decay) to 5.0 (fast decay)
            decay_scale = 0.5 + head * 0.3
            
            l1_attn.W_q.weight[h_start + 0, 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, 28] = decay_scale
            
            # Value is just the characters
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, i] = 1.0
                l1_attn.W_o.weight[d_out + i, h_start + i] = S * 1.0
                
        # --- LAYER 1 MLP: Mix timescales and characters ---
        # Input space is 30 to 30 + 16*28 = 478
        d_input_end = 30 + num_timescales * dim_per_head
        
        # Large variance to create sparse feature combinations
        std_dev = 0.5
        nn.init.normal_(mlp1.fc1.weight[0:4000, 30:d_input_end], std=std_dev)
        nn.init.normal_(mlp1.fc2.weight[0:512, 0:4000], std=std_dev * S / math.sqrt(4000))
        
        # --- LAYER 2: Contextualize the mixed features ---
        for head in range(num_timescales):
            h_start = head * d_head
            decay_scale = 0.2 + head * 0.2 # Different set of timescales for high-level features
            
            l2_attn.W_q.weight[h_start + 0, 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, 28] = decay_scale
            
            # Distribute the 512 features across the 16 heads (32 features per head)
            f_start = head * 32
            for i in range(32):
                l2_attn.W_v.weight[h_start + i, f_start + i] = 1.0
                l2_attn.W_o.weight[512 + f_start + i, h_start + i] = S * 1.0
                
        # --- LAYER 2 MLP: Final projection ---
        nn.init.normal_(mlp2.fc1.weight[0:4000, 512:1024], std=std_dev)
        nn.init.normal_(mlp2.fc2.weight[512:1024, 0:4000], std=std_dev * S / math.sqrt(4000))

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "MultiScale_Temporal_Pool"
model_description = "Uses 16 attention heads in L1 to pool characters at 16 different exponential decay rates. The MLP mixes these timescales. L2 repeats this for the higher-level features."
