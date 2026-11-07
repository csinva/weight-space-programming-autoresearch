import os
import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformers_lib/Final_Interpretable_Absolute_SOTA.py", "r") as f:
    content = f.read()

content = content.replace("dim_per_net = 64", "dim_per_net = 60")

# 1. char_to_dim
old_char_to_dim = "char_to_dim = {'i': 0, 'n': 1, 'g': 2, 'e': 3, 'd': 4, 's': 5, 'l': 6, 'y': 7, \n                       'r': 8, 'o': 9, 't': 10, 'a': 11, 'c': 12, 'm': 13}"
new_char_to_dim = "char_to_dim = {chr(i + ord('a')): i for i in range(26)}"
content = content.replace(old_char_to_dim, new_char_to_dim)

# 2. alpha / space fallback
old_fallback = """            if c.isalpha():
                token_emb[i, ling_start + 14] = S * 1.0
            elif c == ' ':
                token_emb[i, ling_start + 15] = S * 1.0"""
new_fallback = """            elif c == ' ':
                token_emb[i, ling_start + 26] = S * 1.0"""
content = content.replace(old_fallback, new_fallback)

# 3. pos_emb ramp
content = content.replace("pos_emb[p, ling_start+19] = S * (p / max_seq_len)", "pos_emb[p, ling_start+27] = S * (p / max_seq_len)")

# 4. l1_attn query/key
content = content.replace("l1_attn.W_q.weight[h_start + 0, d_start + 19] = 5.0", "l1_attn.W_q.weight[h_start + 0, d_start + 27] = 5.0")
content = content.replace("l1_attn.W_k.weight[h_start + 0, d_start + 19] = 20.0", "l1_attn.W_k.weight[h_start + 0, d_start + 27] = 20.0")

# 5. l1_attn V/O
old_vo = """        for i in range(16): 
            l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
            l1_attn.W_o.weight[d_start + 20 + i, h_start + i] = S * 1.0"""
new_vo = """        for i in range(27): 
            l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
            l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0"""
content = content.replace(old_vo, new_vo)

# 6. morphology logic
morph_pattern = r"        f_start = 15 \* 256.*?for i in range\(11\):"
replacement = """        f_start = 15 * 256
        morph_scale = S * 12.0
        
        # 5 Clean Phonetic Bins
        phonetics = [
            "aeiouy",        # Vowels
            "fvshxz",        # Fricatives
            "pbtdkgcq",      # Plosives
            "mn",            # Nasals
            "lrwj"           # Approximants
        ]
        
        for p_idx, phoneme_group in enumerate(phonetics):
            for char in phoneme_group:
                c_idx = ord(char) - ord('a')
                mlp1.fc1.weight[f_start + p_idx, d_start + 30 + c_idx] = 1.0
            mlp1.fc1.bias[f_start + p_idx] = -0.5
            mlp1.fc2.weight[d_start + 60 + p_idx, f_start + p_idx] = morph_scale

        l2_attn.W_q.weight[h_start + 0, d_start + 27] = 5.0
        l2_attn.W_k.weight[h_start + 0, d_start + 27] = 1.0 
        
        for i in range(5):"""
content = re.sub(morph_pattern, replacement, content, flags=re.DOTALL)

# 7. L2 V/O
old_l2_vo = """            l2_attn.W_v.weight[h_start + i, d_start + 40 + i] = 1.0
            l2_attn.W_o.weight[d_start + 40 + i, h_start + i] = S * 1.0"""
new_l2_vo = """            l2_attn.W_v.weight[h_start + i, d_start + 60 + i] = 1.0
            l2_attn.W_o.weight[d_start + 60 + i, h_start + i] = S * 1.0"""
content = content.replace(old_l2_vo, new_l2_vo)

content = content.replace('model_shorthand_name = "Interpretable_Staggered_Absolute_Morph_Peak"', 'model_shorthand_name = "Interpretable_Phonetic_Bins"')
content = content.replace('model_description = "The absolute ceiling', 'model_description = "Replaced 11 morphology bins with 5 cleanly defined Phonetic Bins (Vowels, Fricatives, Plosives, Nasals, Approximants).')

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)
