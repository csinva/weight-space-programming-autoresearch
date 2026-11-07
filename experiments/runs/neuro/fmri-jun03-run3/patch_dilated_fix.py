import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Fix the mask broadcasting.
# scores shape is (B, H, T, T)
# mask is (T, T)
# dist is shape (1, 1, T, T) before expand. Let's fix dist shape.

attn_old = """        dist = torch.arange(T, device=x.device).unsqueeze(1) - torch.arange(T, device=x.device).unsqueeze(0)
        dist = dist.unsqueeze(0).unsqueeze(2).expand(B, T, H, T)
        
        dilations = torch.arange(1, H + 1, device=x.device).view(1, 1, H, 1)
        
        # A token is visible only if (i - j) % dilation == 0
        # Create a boolean mask for invalid dilation steps
        dilation_mask = (dist % dilations) != 0
        
        # Apply strict causal mask
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(2), float("-inf"))"""

attn_new = """        dist = torch.arange(T, device=x.device).unsqueeze(1) - torch.arange(T, device=x.device).unsqueeze(0)
        dist = dist.view(1, 1, T, T)
        
        # We need H dilations
        dilations = torch.arange(1, H + 1, device=x.device).view(1, H, 1, 1)
        
        dilation_mask = (dist % dilations) != 0
        
        # Apply strict causal mask
        scores = scores.masked_fill(mask.view(1, 1, T, T), float("-inf"))"""

content = content.replace(attn_old, attn_new)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Dilated Attention Fix.")
