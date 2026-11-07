import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Modify the causal mask to allow a lookahead
# mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
mask_old = "mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)"
mask_new = "mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=6) # 5 characters of predictive lookahead"
content = content.replace(mask_old, mask_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Predictive_Coding_Lookahead"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Predictive Coding Anticipation Hypothesis: Relaxes the strict causal attention mask to allow a 5-character lookahead (diagonal=6). Tests if the BOLD signal is better modeled by including representations of immediately anticipated future sounds before they fully arrive in the hemodynamic response."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Predictive Lookahead.")
