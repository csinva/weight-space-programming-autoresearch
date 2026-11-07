import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# We'll inject a "bracket tracker" to approximate syntactic depth
# When we see punctuation, we reset the baseline. But wait, in the string, there's no brackets.
# Let's approximate clause depth by the number of words since the last period/comma.
# We can do this in the position embedding or the attention mask.
# Let's dynamically shift the token embeddings for punctuation.

emb_old = """        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0"""

emb_new = """        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0
            elif c in '.,;!?': # Clause boundaries
                # Huge magnitude to act as a reset/shock to the system
                token_emb[i, 0] = S * -5.0 
                token_emb[i, 1] = S * -5.0"""

content = content.replace(emb_old, emb_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Clause_Boundary_Shock"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Clause Boundary Shock Hypothesis: Tests if syntactic phrase boundaries (.,;!?) act as massive reset signals to the temporal integrator. Injects heavy negative magnitudes into punctuation embeddings to force the leaky integrators to flush their accumulated history."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Clause Boundary Shock.")
