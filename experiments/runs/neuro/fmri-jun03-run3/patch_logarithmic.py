import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Replace linear Layer 1 decay
l1_old = "l1_decay = 15.0 + float(net) * (65.0 / 14.0)"
l1_new = """
            log_start_1 = math.log(15.0)
            log_end_1 = math.log(80.0)
            l1_decay = math.exp(log_start_1 + float(net) * (log_end_1 - log_start_1) / 14.0)
"""
content = content.replace(l1_old, l1_new.strip())

# Replace linear Layer 2 decay
l2_old = "l2_decay = 0.01 + float(net) * (13.99 / 14.0)"
l2_new = """
            log_start_2 = math.log(0.01)
            log_end_2 = math.log(14.0)
            l2_decay = math.exp(log_start_2 + float(net) * (log_end_2 - log_start_2) / 14.0)
"""
content = content.replace(l2_old, l2_new.strip())

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Weber_Fechner_Logarithmic_Decay"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Weber-Fechner Law Hypothesis: Biological systems perceive time logarithmically. Replaced the linear distribution of temporal decay horizons across the 15 networks with a logarithmic distribution."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched logarithmic decays.")
