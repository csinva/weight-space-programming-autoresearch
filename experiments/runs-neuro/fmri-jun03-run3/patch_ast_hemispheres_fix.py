import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# I need to fix the indentation.
# Instead of string replace, I will restore and re-patch carefully.
with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

l1_old = "l1_decay = 15.0 + float(net) * (65.0 / 14.0)"
l1_new = "l1_decay = 80.0 if net < 8 else 15.0"
content = content.replace(l1_old, l1_new)

l2_old = "l2_decay = 0.01 + float(net) * (13.99 / 14.0)"
l2_new = "l2_decay = 14.0 if net < 8 else 0.01"
content = content.replace(l2_old, l2_new)

desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Hemispheric_Asymmetric_Sampling"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Asymmetric Sampling in Time (AST) Hypothesis: Models left/right brain hemispheric lateralization by replacing the uniform continuous spectrum of timescales with a stark bimodal split. Left Hemisphere (8 nets) uses ultra-fast decays; Right Hemisphere (7 nets) uses ultra-slow decays."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched AST Hemispheres safely.")
