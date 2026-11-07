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

        # Deep_EnsembleWB_Staggered_4x got 0.0372, slightly worse than Staggered_2x (0.0374).
        # We need to refine the staggered logic.
        # Maybe instead of randomly choosing 4 networks, we choose them sequentially?
        # A true hierarchy!
        # Network 0 (Fastest L2) pulls from L1 nets 0, 1, 2, 3 (The fastest L1s)
        # Network 14 (Slowest L2) pulls from L1 nets 11, 12, 13, 14 (The slowest L1s)
        # This way, fast integrators mix fast features, and slow integrators mix slow features!
        
        num_nets = 15
        dim_per_net = 64
        ff_per_net = 256
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 # Space
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 # Alpha
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0 # Identity
                
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0
            
        for net in range(num_nets):
            start = net * dim_per_net
            pos_emb[:, start+28:start+30] = pos_emb[:, 28:30]
            
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        for net in range(num_nets):
            d_start = net * dim_per_net
            f_start = net * ff_per_net
            h_start = (net % n_heads) * d_head
            
            # --- LAYER 1: Extremely sharp local extraction ---
            l1_decay = 5.0 + float(net) * (25.0 / 14.0) 
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay
            
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
                
            std_dev = 0.5
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            # --- LAYER 2: Hierarchical Staggered Integration ---
            l2_decay = 0.5 + float(net) * (5.0 / 14.0)
            
            l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, d_start + 28] = l2_decay
            
            # Hierarchical selection:
            # We want to pull from 4 networks centered roughly around `net`.
            # If net=0, we pull 0, 1, 2, 3
            # If net=14, we pull 11, 12, 13, 14
            # If net=7, we pull 5, 6, 7, 8
            
            center_net = net
            if center_net < 1: center_net = 1
            if center_net > 13: center_net = 13
            
            pull_nets = [center_net - 1, center_net, center_net + 1, (center_net + 2) if center_net < 13 else center_net - 2]
            
            # Ensure unique
            pull_nets = list(set(pull_nets))
            while len(pull_nets) < 4:
                # Add random if needed (shouldn't happen with the logic above, but just safe guard)
                pull_nets.append((pull_nets[-1] + 1) % 15)
                pull_nets = list(set(pull_nets))
                
            chunk_size = 16
            
            for chunk_idx, net_b in enumerate(pull_nets):
                d_start_b = net_b * dim_per_net
                
                # We pull 16 features from net_b into chunk_idx
                for i in range(chunk_size):
                    val_idx = h_start + chunk_idx * chunk_size + i
                    out_idx = d_start + chunk_idx * chunk_size + i
                    
                    l2_attn.W_v.weight[val_idx, d_start_b + i] = 1.0
                    l2_attn.W_o.weight[out_idx, val_idx] = S * 1.0
                
            std_dev2 = 0.1 + float(net) * 0.1
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "Deep_EnsembleWB_Hierarchical"
model_description = "Uses a hierarchical staggered integration in Layer 2: Fast L2 integrators pull from fast L1 extractors, and slow L2 integrators pull from slow L1 extractors."
