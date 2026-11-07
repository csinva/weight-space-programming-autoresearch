import os
import sys
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Replace the EXACT loop inside write_weights that injects character features
old_injection = """        token_emb = torch.zeros(vocab_size, d_model)
        
        for i, c in enumerate(VOCAB):
            if c == '<pad>' or c == '<unk>':
                continue
            elif len(c) == 1 and c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0
            else:
                token_emb[i, 0] = S * 1.0

        for net_idx in range(num_nets):
            start_dim = net_idx * dim_per_net
            # Inject character feature
            token_emb[:, start_dim:start_dim+28] = token_emb[:, 0:28]
            # Inject positional token feature for queries
            token_emb[:, start_dim+29] = 1.0"""

new_injection = """
        PHONETIC_CLASSES = {
            'plosive': ['p', 'b', 't', 'd', 'k', 'g'],
            'fricative': ['f', 'v', 's', 'z', 'h'],
            'nasal': ['m', 'n'],
            'liquid': ['l', 'r'],
            'glide': ['w', 'y', 'j'],
            'vowel_front': ['i', 'e'],
            'vowel_back': ['u', 'o'],
            'vowel_central': ['a'],
            'silent': [' ', '<pad>', '.', ',', '!', '?']
        }

        token_emb = torch.zeros(vocab_size, d_model)
        
        for i, c in enumerate(VOCAB):
            c_lower = c.lower()
            if c_lower in PHONETIC_CLASSES['plosive']:
                token_emb[i, 2] = S * 1.0
            elif c_lower in PHONETIC_CLASSES['fricative']:
                token_emb[i, 3] = S * 1.0
            elif c_lower in PHONETIC_CLASSES['nasal']:
                token_emb[i, 4] = S * 1.0
            elif c_lower in PHONETIC_CLASSES['liquid']:
                token_emb[i, 5] = S * 1.0
            elif c_lower in PHONETIC_CLASSES['glide']:
                token_emb[i, 6] = S * 1.0
            elif c_lower in PHONETIC_CLASSES['vowel_front']:
                token_emb[i, 7] = S * 1.0
            elif c_lower in PHONETIC_CLASSES['vowel_back']:
                token_emb[i, 8] = S * 1.0
            elif c_lower in PHONETIC_CLASSES['vowel_central']:
                token_emb[i, 9] = S * 1.0
            elif c_lower in PHONETIC_CLASSES['silent']:
                token_emb[i, 0] = S * 1.0
            else:
                token_emb[i, 1] = S * 1.0

        for net_idx in range(num_nets):
            start_dim = net_idx * dim_per_net
            # Inject phonetic features EXACTLY into the exact same dimensions the network routes
            token_emb[:, start_dim:start_dim+28] = token_emb[:, 0:28]
            # Inject positional token feature for queries
            token_emb[:, start_dim+29] = 1.0"""

content = content.replace(old_injection, new_injection)

content = content.replace("Deep_Ensemble_0421_Master", "Pure_Phonetic_Fixed")
content = content.replace("Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures.", "Properly routes phonetic classes through the exact 0421 structural decay networks.")

with open(filepath, "w") as f:
    f.write(content)
print("Generated Fixed Phonetic Extractor")
