import os
import math
import torch
import torch.nn as nn

content = ""
with open("runs-neuro/fmri-jun03-run3/interpretable_transformers_lib/Final_Interpretable_Absolute_SOTA.py", "r") as f:
    content = f.read()

import re
pattern = r"def write_weights\(model: SimpleTransformer\) -> None:.*?(?=model_shorthand_name =)"

replacement = """def write_weights(model: SimpleTransformer) -> None:
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

        num_nets = 15
        dim_per_net = 56
        ff_per_net = 250
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ': token_emb[i, 0] = S
            elif c.isalpha(): token_emb[i, 1] = S; token_emb[i, 2 + (ord(c) - ord('a'))] = S
                
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        ling_start = num_nets * dim_per_net
        char_to_dim = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
        for i, c in enumerate(VOCAB):
            if c in char_to_dim: token_emb[i, ling_start + char_to_dim[c]] = S
            if c.isalpha(): token_emb[i, ling_start + 26] = S
            elif c == ' ': token_emb[i, ling_start + 27] = S
                
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C; pos_emb[p, 1001] = -C
            pos_emb[p, 28] = S * (p / max_seq_len); pos_emb[p, 29] = S
            pos_emb[p, ling_start+28] = S * (p / max_seq_len)
            
        for net in range(num_nets):
            start = net * dim_per_net
            pos_emb[:, start+28:start+30] = pos_emb[:, 28:30]
            
        model.pos_emb.weight.data.copy_(pos_emb)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight); nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight); nn.init.zeros_(block.ln2.bias)
        nn.init.ones_(model.final_ln.weight); nn.init.zeros_(model.final_ln.bias)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        for net in range(num_nets):
            d_start = net * dim_per_net
            f_start = net * ff_per_net
            h_start = (net % n_heads) * d_head
            
            l1_decay = 10.0 + float(net) * (70.0 / 14.0) 
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S
                
            std_dev = 0.5
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            l2_decay = 0.05 + float(net) * (11.95 / 14.0)
            l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, d_start + 28] = l2_decay
            
            net_b, net_c = (net + 6) % 15, (net + 12) % 15
            d_start_b, d_start_c = net_b * dim_per_net, net_c * dim_per_net
            
            for i in range(19): l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0; l2_attn.W_o.weight[d_start + i, h_start + i] = S
            for i in range(18): l2_attn.W_v.weight[h_start + 19 + i, d_start_b + i] = 1.0; l2_attn.W_o.weight[d_start + 19 + i, h_start + 19 + i] = S
            for i in range(18): l2_attn.W_v.weight[h_start + 37 + i, d_start_c + i] = 1.0; l2_attn.W_o.weight[d_start + 37 + i, h_start + 37 + i] = S
                
            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)
            
        d_start = ling_start
        h_start = 0
        l1_attn.W_q.weight[h_start + 0, d_start + 28] = 5.0
        l1_attn.W_k.weight[h_start + 0, d_start + 28] = 20.0
        for i in range(28): 
            l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
            l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S
            
        f_start = num_nets * ff_per_net
        morph_scale = S * 12.0
        
        m_list = [
            "ing ", "ed ", "ly ", "er ", "or ", "ion ", "tion ", "ation ", "ic ", "ment ",
            "ness ", "able ", "ible ", "al ", "ial ", "ful ", "less ", "est ", "ous ",
            "ity ", "ty ", "ive ", " re", " un", " in", " im", " dis", " en", " non", 
            " pre", " pro", " trans", " inter", " sub", " super", " anti", " auto", 
            " out", " over", " s "
        ]
        
        for m_idx, m in enumerate(m_list):
            for char in m:
                if char == ' ':
                    mlp1.fc1.weight[f_start + m_idx, d_start + 30 + 27] = 1.0
                elif char.isalpha():
                    c_idx = ord(char) - ord('a')
                    mlp1.fc1.weight[f_start + m_idx, d_start + 30 + c_idx] = 1.0
            # Set bias appropriately
            mlp1.fc1.bias[f_start + m_idx] = -(len(m) - 1.0)
            mlp1.fc2.weight[d_start + 60 + m_idx, f_start + m_idx] = morph_scale
            
        l2_attn.W_q.weight[h_start + 0, d_start + 28] = 5.0
        l2_attn.W_k.weight[h_start + 0, d_start + 28] = 1.0 
        
        for i in range(len(m_list)):
            l2_attn.W_v.weight[h_start + i, d_start + 60 + i] = 1.0
            l2_attn.W_o.weight[d_start + 60 + i, h_start + i] = S * 1.0

"""
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
new_content = new_content.replace('model_shorthand_name = "Interpretable_Staggered_Absolute_Morph_Peak"', 'model_shorthand_name = "Interpretable_Curated_Morphology_40"')
new_content = new_content.replace('model_description = "The absolute ceiling', 'model_description = "Expanded to 40 curated explicit prefixes and suffixes.')

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(new_content)
