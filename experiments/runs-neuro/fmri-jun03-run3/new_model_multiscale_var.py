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

        # EnsembleWB_VarTuned got 0.0363.
        # MultiScale_Temporal_Pool got 0.0362.
        # Let's combine the best of both!
        # Instead of 1 wide network, let's use the VarTuned ensemble (multiple independent MLPs),
        # BUT Layer 1 will pool characters with MULTIPLE TIMESCALES for *each* MLP!
        # If n_heads = 10, d_head = 102.
        # Let's use 10 independent networks.
        # Within each network, the head provides 3 different decay timescales for the characters.
        
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
        
        num_nets = 10 # Since n_heads = 10
        dim_per_net = 28 * 3 # 3 timescales per char
        ff_per_net = 400
        
        for net in range(num_nets):
            h_start = net * d_head
            d_out = 30 + net * dim_per_net
            f_start = net * ff_per_net
            
            # Timescales: Fast (current word), Medium (sentence), Slow (context)
            decays = [5.0, 1.0, 0.2]
            
            # Since we only have 1 query/key per head in standard attention,
            # we can't do multiple decays in ONE head easily unless we hack Q/K for different dims.
            # But wait, Q K^T is scalar per token. We can't do multiple decays in 1 head.
            # INSTEAD: 
            # MultiScale_Temporal_Pool used 1 head per timescale.
            # If we want an ensemble, maybe we use 10 networks, and each network is just 1 head
            # but we vary the timescale ACROSS networks.
            # Wait, that's literally what EnsembleWB_VarTuned is!
            # The difference is MultiScale_Temporal_Pool mixed all timescales in ONE wide MLP.
            # EnsembleWB_VarTuned kept the MLPs independent.
            
            # Let's do the "mix timescales in small MLPs" idea.
            # We will use ALL 10 heads as L1 extractors with different timescales.
            # Then we divide the MLP into 5 sub-networks.
            # EACH sub-network takes input from ALL 10 heads, but has a different
            # standard deviation (var-tuned) and sparse connectivity.
            
            decay_scale = 0.5 + net * 0.4
            
            l1_attn.W_q.weight[h_start + 0, 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, 28] = decay_scale
            
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, i] = 1.0
                l1_attn.W_o.weight[30 + net * 28 + i, h_start + i] = S * 1.0
                
        d_input_end = 30 + 10 * 28 # 310
        
        num_sub_mlps = 5
        ff_per_sub = 800
        
        for sub in range(num_sub_mlps):
            f_start = sub * ff_per_sub
            
            # Var logic: vary std from 0.1 to 1.3
            std_dev = 0.1 + float(sub) * 0.3
            
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_sub, 30:d_input_end], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[sub*100:(sub+1)*100, f_start:f_start+ff_per_sub], std=std_dev * S / math.sqrt(ff_per_sub))

        # We now have 500 features from the 5 sub-MLPs.
        # Layer 2: standard ensemble pooling
        
        for head in range(num_nets):
            h_start = head * d_head
            decay_scale = 0.2 + head * 0.3
            
            l2_attn.W_q.weight[h_start + 0, 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, 28] = decay_scale
            
            for i in range(50):
                f_idx = head * 50 + i
                l2_attn.W_v.weight[h_start + i, f_idx] = 1.0
                l2_attn.W_o.weight[500 + f_idx, h_start + i] = S * 1.0
                
        # Final MLPs independent again
        for sub in range(num_sub_mlps):
            f_start = sub * ff_per_sub
            std_dev = 0.1 + float(sub) * 0.3
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_sub, 500+sub*100:500+(sub+1)*100], std=std_dev)
            nn.init.normal_(mlp2.fc2.weight[500+sub*100:500+(sub+1)*100, f_start:f_start+ff_per_sub], std=std_dev * S / math.sqrt(ff_per_sub))

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "MultiScale_VarTuned"
model_description = "Combines multiscale temporal pooling (10 heads) with VarTuned MLPs (5 subnetworks with stds 0.1 to 1.3), creating diverse scale-mixing non-linearities."
