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

        # Deep_EnsembleWB_v3 got 0.0368. Deep_EnsembleWB got 0.0369.
        # This confirms that explicit local L1 extraction + long L2 integration is the best structure so far.
        # How do we push it past 0.0369?
        # Let's combine the varied L1 decay (v3) with varied L2 decay AND varied MLP stdev,
        # but let's also give it the Explicit Extractors (word length, space detector) from earlier!
        
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
                
                if c in 'aeiou':
                    token_emb[i, 30] = S * 1.0 # Vowel
                else:
                    token_emb[i, 31] = S * 1.0 # Consonant
                    
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+32] = token_emb[:, 0:32]
            
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
            
            # Networks 0-3: Explicit features
            if net == 0:
                # Word length: sum chars since last space
                l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
                l1_attn.W_k.weight[h_start + 0, d_start + 0] = -5.0 # Stop at space
                l1_attn.W_v.weight[h_start + 0, d_start + 1] = 1.0 # Any char
                l1_attn.W_o.weight[d_start + 32, h_start + 0] = S * 1.0
            elif net == 1:
                # Space detector
                l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
                l1_attn.W_k.weight[h_start + 0, d_start + 0] = 5.0 # Attract space
                l1_attn.W_v.weight[h_start + 0, d_start + 0] = 1.0
                l1_attn.W_o.weight[d_start + 33, h_start + 0] = S * 1.0
            elif net == 2:
                # Vowel sum
                l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
                l1_attn.W_k.weight[h_start + 0, d_start + 28] = 2.0
                l1_attn.W_v.weight[h_start + 0, d_start + 30] = 1.0
                l1_attn.W_o.weight[d_start + 34, h_start + 0] = S * 1.0
            elif net == 3:
                # Consonant sum
                l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
                l1_attn.W_k.weight[h_start + 0, d_start + 28] = 2.0
                l1_attn.W_v.weight[h_start + 0, d_start + 31] = 1.0
                l1_attn.W_o.weight[d_start + 35, h_start + 0] = S * 1.0
            else:
                # Networks 4-14: Standard local L1 extractors
                l1_decay = 5.0 + float((14 - net) % 3) * 5.0 # Mix of sharp decays (5, 10, 15)
                l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
                l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay
                
                for i in range(28):
                    l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                    l1_attn.W_o.weight[d_start + 36 + i, h_start + i] = S * 1.0
                
            std_dev = 0.5
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            # --- LAYER 2: Long term integration ---
            # Even explicit features need long-term context
            decay_scale = 1.0 + float(net) / 4.0
            
            l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, d_start + 28] = decay_scale
            
            for i in range(dim_per_net):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
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

model_shorthand_name = "Deep_Ensemble_Explicit"
model_description = "Merges Deep_EnsembleWB (L1 local, L2 global integration) with Explicit Extractors (word length, vowels, space), creating a rich hierarchical feature set."
