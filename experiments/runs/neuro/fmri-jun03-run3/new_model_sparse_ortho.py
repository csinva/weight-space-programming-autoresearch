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

        # Char_Combinations_Map got 0.0338 but had a very high train_corr (0.3060).
        # It's overfitting. 
        # Explicit_Extractors got 0.0344.
        # Ngram_Ensemble got 0.0351.
        # EnsembleWB_VarTuned got 0.0363!
        # It seems the key to scoring > 0.035 is using multiple independent "WordBoundary"-style 
        # subnetworks (which sum characters) with varying timescales.
        # Can we enhance the EnsembleWB_VarTuned by changing WHAT is summed?
        # Instead of just summing character identities... what if we sum character identities AND bigram transition indicators?
        # In Layer 1, we can create explicit feature vectors for combinations like "th", "he", "in", etc.
        # But maybe that's too complex to hardcode.
        
        # What if we create a massive *sparse random* orthographic map in the token embedding?
        # Instead of just allocating 1 dimension per character, we project each character into a 
        # random 10-dimensional space.
        # This will create rich "hash" combinations in the summation!
        # Then, we run the EnsembleWB_VarTuned on top of this rich token embedding.
        
        num_nets = 15
        dim_per_net = 64
        ff_per_net = 256
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        
        # We will use 20 dimensions for the "rich character hash"
        # 1 for space, 19 for random character features
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 # space flag
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 # char flag
                # Assign 5 random non-zero entries in the next 18 dimensions to create a sparse hash
                torch.manual_seed(ord(c)) # stable hash per character
                for _ in range(5):
                    idx = torch.randint(2, 20, (1,)).item()
                    sign = torch.randint(0, 2, (1,)).item() * 2 - 1
                    token_emb[i, idx] = S * float(sign)
                
        # Copy to all networks
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+20] = token_emb[:, 0:20]
            
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
        
        torch.manual_seed(42) # reset seed for MLPs
        
        for net in range(num_nets):
            d_start = net * dim_per_net
            f_start = net * ff_per_net
            h_start = (net % n_heads) * d_head
            
            decay_scale = 1.0 + float(net) / 4.0
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = decay_scale
            
            # Extract the 20-dim rich hash
            for i in range(20):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
                
            # Var logic: vary std from 0.1 to 1.5
            std_dev = 0.1 + float(net) * 0.1
            
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            for i in range(dim_per_net):
                l2_attn.W_q.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_k.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "EnsembleWB_RichHash"
model_description = "Like EnsembleWB_VarTuned, but characters are embedded into a dense 18-dimensional random hash space before integration."
