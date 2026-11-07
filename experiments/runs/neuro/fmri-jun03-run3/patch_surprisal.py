import re
import math

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Replace token embedding initialization
emb_old = """        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0"""

emb_new = """        # Character frequencies in English
        freqs = {
            'e': 12.7, 't': 9.1, 'a': 8.1, 'o': 7.5, 'i': 7.0, 'n': 6.7, 's': 6.3, 'h': 6.1,
            'r': 6.0, 'd': 4.3, 'l': 4.0, 'c': 2.8, 'u': 2.8, 'm': 2.4, 'w': 2.4, 'f': 2.2,
            'g': 2.0, 'y': 2.0, 'p': 1.9, 'b': 1.5, 'v': 1.0, 'k': 0.8, 'j': 0.15, 'x': 0.15,
            'q': 0.10, 'z': 0.07
        }
        total_freq = sum(freqs.values())
        
        for i, c in enumerate(VOCAB):
            if c == ' ':
                # Space is very common, low surprisal
                surprisal = -math.log2(0.20) # Approx 20% of chars are spaces
                token_emb[i, 0] = S * surprisal 
            elif c.isalpha():
                prob = (freqs.get(c, 1.0) / total_freq) * 0.80 # 80% non-space
                surprisal = -math.log2(prob)
                
                token_emb[i, 1] = S * 1.0 # Alpha indicator standard
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * (surprisal / 4.0) # Normalized roughly around 1.0"""

content = content.replace(emb_old, emb_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Info_Theoretic_Surprisal"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Information-Theoretic Surprisal Hypothesis: Scales character embedding magnitudes proportionally to their negative log probability (surprisal) in English. Tests if the brain responds more strongly to high-information (rare) characters (like Z, X, J) than common ones (like E, T, A)."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Surprisal model.")
