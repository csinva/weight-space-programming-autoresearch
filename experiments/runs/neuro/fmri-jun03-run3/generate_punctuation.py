import os
import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformers_lib/Final_Interpretable_Absolute_SOTA.py", "r") as f:
    content = f.read()

content = content.replace("dim_per_net = 64", "dim_per_net = 60")

# 1. Expand char_to_dim to include punctuation
old_char_to_dim = "char_to_dim = {'i': 0, 'n': 1, 'g': 2, 'e': 3, 'd': 4, 's': 5, 'l': 6, 'y': 7, \n                       'r': 8, 'o': 9, 't': 10, 'a': 11, 'c': 12, 'm': 13}"
new_char_to_dim = "char_to_dim = {'i': 0, 'n': 1, 'g': 2, 'e': 3, 'd': 4, 's': 5, 'l': 6, 'y': 7, \n                       'r': 8, 'o': 9, 't': 10, 'a': 11, 'c': 12, 'm': 13, '.': 16, ',': 17, '?': 18, '!': 18, \"'\": 20}"
content = content.replace(old_char_to_dim, new_char_to_dim)

# 2. Move pos_emb ramp from 19 to 25
content = content.replace("pos_emb[p, ling_start+19] = S * (p / max_seq_len)", "pos_emb[p, ling_start+25] = S * (p / max_seq_len)")
content = content.replace("l1_attn.W_q.weight[h_start + 0, d_start + 19] = 5.0", "l1_attn.W_q.weight[h_start + 0, d_start + 25] = 5.0")
content = content.replace("l1_attn.W_k.weight[h_start + 0, d_start + 19] = 20.0", "l1_attn.W_k.weight[h_start + 0, d_start + 25] = 20.0")

content = content.replace("l2_attn.W_q.weight[h_start + 0, d_start + 19] = 5.0", "l2_attn.W_q.weight[h_start + 0, d_start + 25] = 5.0")
content = content.replace("l2_attn.W_k.weight[h_start + 0, d_start + 19] = 1.0", "l2_attn.W_k.weight[h_start + 0, d_start + 25] = 1.0")

# 3. Increase L1 V/O extraction width from 16 to 25
old_l1_vo = """        for i in range(16): 
            l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
            l1_attn.W_o.weight[d_start + 20 + i, h_start + i] = S * 1.0"""
new_l1_vo = """        for i in range(25): 
            l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
            l1_attn.W_o.weight[d_start + 26 + i, h_start + i] = S * 1.0"""
content = content.replace(old_l1_vo, new_l1_vo)

# 4. Modify the morphological indexing
for i in range(16):
    content = content.replace(f"d_start + 20 + {i}]", f"d_start + 26 + {i}]")

# 5. Inject Punctuation Trackers into mlp1
old_morph_end = "mlp1.fc2.weight[d_start + 50, f_start + 10] = morph_scale"
new_morph_end = """mlp1.fc2.weight[d_start + 50, f_start + 10] = morph_scale
        
        # Punctuation Trackers
        mlp1.fc1.weight[f_start + 11, d_start + 26 + 16] = 1.0  # Period
        mlp1.fc1.bias[f_start + 11] = -0.5
        mlp1.fc2.weight[d_start + 51, f_start + 11] = morph_scale
        
        mlp1.fc1.weight[f_start + 12, d_start + 26 + 17] = 1.0  # Comma
        mlp1.fc1.bias[f_start + 12] = -0.5
        mlp1.fc2.weight[d_start + 52, f_start + 12] = morph_scale
        
        mlp1.fc1.weight[f_start + 13, d_start + 26 + 18] = 1.0  # Question/Exclamation
        mlp1.fc1.bias[f_start + 13] = -0.5
        mlp1.fc2.weight[d_start + 53, f_start + 13] = morph_scale
        
        mlp1.fc1.weight[f_start + 14, d_start + 26 + 20] = 1.0  # Quote
        mlp1.fc1.bias[f_start + 14] = -0.5
        mlp1.fc2.weight[d_start + 54, f_start + 14] = morph_scale
"""
content = content.replace(old_morph_end, new_morph_end)

# 6. Increase L2 V/O mapping to include the 4 new dimensions
old_l2_vo = """        for i in range(11):
            l2_attn.W_v.weight[h_start + i, d_start + 40 + i] = 1.0
            l2_attn.W_o.weight[d_start + 40 + i, h_start + i] = S * 1.0"""
new_l2_vo = """        for i in range(15):
            l2_attn.W_v.weight[h_start + i, d_start + 40 + i] = 1.0
            l2_attn.W_o.weight[d_start + 40 + i, h_start + i] = S * 1.0"""
content = content.replace(old_l2_vo, new_l2_vo)

content = content.replace('model_shorthand_name = "Interpretable_Staggered_Absolute_Morph_Peak"', 'model_shorthand_name = "Interpretable_Punctuation_Enhanced"')
content = content.replace('model_description = "The absolute ceiling', 'model_description = "SOTA augmented with 4 explicit punctuation / syntactic boundary trackers.')

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)
