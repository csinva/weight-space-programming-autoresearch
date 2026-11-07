import os
import sys
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Replace write_weights
new_write_weights = """
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

def write_weights(model):
    S = 20.0
    
    # Write token embeddings (ONLY PHONETICS)
    for i, c in enumerate(VOCAB):
        c_lower = c.lower()
        if c_lower in PHONETIC_CLASSES['plosive']:
            model.token_emb.weight.data[i, 0] = S
        elif c_lower in PHONETIC_CLASSES['fricative']:
            model.token_emb.weight.data[i, 1] = S
        elif c_lower in PHONETIC_CLASSES['nasal']:
            model.token_emb.weight.data[i, 2] = S
        elif c_lower in PHONETIC_CLASSES['liquid']:
            model.token_emb.weight.data[i, 3] = S
        elif c_lower in PHONETIC_CLASSES['glide']:
            model.token_emb.weight.data[i, 4] = S
        elif c_lower in PHONETIC_CLASSES['vowel_front']:
            model.token_emb.weight.data[i, 5] = S
        elif c_lower in PHONETIC_CLASSES['vowel_back']:
            model.token_emb.weight.data[i, 6] = S
        elif c_lower in PHONETIC_CLASSES['vowel_central']:
            model.token_emb.weight.data[i, 7] = S
        elif c_lower in PHONETIC_CLASSES['silent']:
            model.token_emb.weight.data[i, 8] = S
        else:
            model.token_emb.weight.data[i, 9] = S
            
    # L1: Staggered extraction of ONLY phonetic features over time
    l1_attn = model.blocks[0].attn
    d_head = model.blocks[0].attn.d_head
    for h in range(16):
        d_start = h * d_head
        h_start = h * d_head
        # Pass phonetic features
        for i in range(10):
            l1_attn.W_v.weight.data[h_start + i, i] = 1.0
            l1_attn.W_o.weight.data[d_start + i, h_start + i] = S * (0.85 ** h)  # Decay older contexts

    # L2: Route the smeared features to output
    l2_attn = model.blocks[1].attn
    d_head = model.blocks[1].attn.d_head
    for h in range(16):
        d_start = h * d_head
        h_start = h * d_head
        for i in range(10):
            l2_attn.W_v.weight.data[h_start + i, d_start + i] = 1.0
            l2_attn.W_o.weight.data[d_start + i, h_start + i] = S
            
    # Positive bias for Ridge regression stability
    model.final_ln.bias.data += 1.18
"""

content = re.sub(r"def write_weights\(model\):.*?model_shorthand_name =", new_write_weights + "\n\nmodel_shorthand_name =", content, flags=re.DOTALL)
content = content.replace("Deep_Ensemble_0421_Master", "Pure_Phonetic_Extractor")
content = content.replace("Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures.", "Explicitly maps characters to broad phonetic classes (plosives, fricatives, vowels) ONLY, entirely removing character identity. Tests if brain prediction is phonetic.")

with open(filepath, "w") as f:
    f.write(content)

print("Generated Pure Phonetic Extractor")
