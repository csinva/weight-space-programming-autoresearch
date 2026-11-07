import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Modify Layer 1 decay
l1_old = "l1_decay = 15.0 + float(net) * (65.0 / 14.0)"
l1_new = """            # Hemispheric Lateralization (AST Hypothesis)
            # Left Hemisphere (nets 0-7): Fast sampling (High decay)
            # Right Hemisphere (nets 8-14): Slow sampling (Low decay)
            if net < 8:
                l1_decay = 80.0  # Fast
            else:
                l1_decay = 15.0  # Slow"""
content = content.replace(l1_old, l1_new)

# Modify Layer 2 decay
l2_old = "l2_decay = 0.01 + float(net) * (13.99 / 14.0)"
l2_new = """            if net < 8:
                l2_decay = 14.0  # Fast integration
            else:
                l2_decay = 0.01  # Slow integration"""
content = content.replace(l2_old, l2_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Hemispheric_Asymmetric_Sampling"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Asymmetric Sampling in Time (AST) Hypothesis: Models left/right brain hemispheric lateralization by replacing the uniform continuous spectrum of timescales with a stark bimodal split. Left Hemisphere (8 nets) uses ultra-fast decays; Right Hemisphere (7 nets) uses ultra-slow decays."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched AST Hemispheres.")
