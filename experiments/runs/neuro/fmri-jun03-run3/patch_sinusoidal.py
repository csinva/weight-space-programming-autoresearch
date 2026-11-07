import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# I need to modify the forward pass of the attention to inject a sinusoidal oscillation based on the relative position.
# To do this cleanly, I'll modify the mask in CausalSelfAttention to include a sinusoidal ripple on top of the exponential decay.

attn_old = """        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)"""

# Add a cosine ripple to the attention scores based on distance
# distance matrix: dist[i, j] = i - j
# sin_ripple = torch.cos(dist * frequency)
# we can use the head index to determine frequency
attn_new = """        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        
        # Inject Brain-Wave style Resonances (Sinusoidal ripples on top of decay)
        dist = torch.arange(T, device=x.device).unsqueeze(1) - torch.arange(T, device=x.device).unsqueeze(0)
        dist = dist.unsqueeze(0).unsqueeze(2).expand(B, T, H, T)
        
        # Frequencies spread across heads, mimicking Theta/Alpha/Beta/Gamma bands
        # Normalized relative to the max_seq_len (e.g. 64)
        frequencies = torch.linspace(0.1, 3.14, H, device=x.device).view(1, 1, H, 1)
        
        # Add resonance. amplitude 2.0 to make it noticeable.
        resonance = 2.0 * torch.cos(dist * frequencies)
        
        # Apply only to lower triangle
        resonance = resonance.masked_fill(mask.unsqueeze(0).unsqueeze(2), 0.0)
        
        scores = scores + resonance
        
        attn = scores.softmax(dim=-1)"""

content = content.replace(attn_old, attn_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Sinusoidal_Brainwave_Resonance"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Brainwave Resonance Hypothesis: Added sinusoidal ripples (cosines of varying frequencies) to the attention scores before softmax to mimic neural oscillatory bands (Theta, Alpha, Beta, Gamma) riding on top of the exponential decay."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched sinusoidal resonance.")
