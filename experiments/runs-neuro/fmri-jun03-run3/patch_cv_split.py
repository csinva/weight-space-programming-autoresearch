import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# We want to modify write_weights to split networks into Vowel and Consonant processors.
emb_old = """        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]"""

emb_new = """        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
            # Consonant-Vowel Split
            if net < 7:
                # Vowel networks: Zero out consonants
                for i, c in enumerate(VOCAB):
                    if c.isalpha() and c not in 'aeiou':
                        token_emb[i, start+2 + (ord(c) - ord('a'))] = 0.0
            else:
                # Consonant networks: Zero out vowels
                for i, c in enumerate(VOCAB):
                    if c.isalpha() and c in 'aeiou':
                        token_emb[i, start+2 + (ord(c) - ord('a'))] = 0.0"""

content = content.replace(emb_old, emb_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Consonant_Vowel_Orthogonal_Split"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Consonant-Vowel Stream Hypothesis: Tests if the brain processes vowels (prosody/syllable nuclei) and consonants (articulation/boundaries) in independent streams. Splits the 15 networks so 7 only see vowels and 8 only see consonants."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Consonant-Vowel split.")
