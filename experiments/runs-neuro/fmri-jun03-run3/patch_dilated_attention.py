import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

attn_old = """        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)"""

attn_new = """        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        
        # Dilated Temporal Attention Hypothesis
        # Brains might process time not continuously, but in discretized rhythmic chunks (e.g. Theta/Gamma phase locking).
        # We apply a dilation mask: certain attention heads only look at past tokens spaced by a specific stride.
        # Dilation factors spread across the H heads (1, 2, 3, ..., H)
        dist = torch.arange(T, device=x.device).unsqueeze(1) - torch.arange(T, device=x.device).unsqueeze(0)
        dist = dist.unsqueeze(0).unsqueeze(2).expand(B, T, H, T)
        
        dilations = torch.arange(1, H + 1, device=x.device).view(1, 1, H, 1)
        
        # A token is visible only if (i - j) % dilation == 0
        # Create a boolean mask for invalid dilation steps
        dilation_mask = (dist % dilations) != 0
        
        # Apply strict causal mask
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(2), float("-inf"))
        
        # Apply dilation mask
        scores = scores.masked_fill(dilation_mask, float("-inf"))
        
        attn = scores.softmax(dim=-1)"""

content = content.replace(attn_old, attn_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Dilated_Temporal_Attention"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Dilated Temporal Attention Hypothesis: Brains may discretize time via oscillatory phase-locking (e.g., Theta/Gamma rhythms). Tested if the networks operate via Dilated Attention, where different heads strictly attend only to past tokens spaced by specific geometric strides (e.g., every 2nd token, every 3rd token), forcing a sparse hierarchical temporal integration rather than continuous dense integration."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Dilated Attention.")
