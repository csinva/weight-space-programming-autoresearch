import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Replace num_nets = 15 with 30
# dim_per_net = 64 with 32
# ff_per_net = 256 with 128
content = content.replace("num_nets = 15", "num_nets = 30")
content = content.replace("dim_per_net = 64", "dim_per_net = 32")
content = content.replace("ff_per_net = 256", "ff_per_net = 128")

# L1 decay logic
content = content.replace(
    "l1_decay = 15.0 + float(net) * (65.0 / 14.0)",
    "l1_decay = 15.0 + float(net) * (65.0 / 29.0)"
)
# L2 decay logic
content = content.replace(
    "l2_decay = 0.01 + float(net) * (13.99 / 14.0)",
    "l2_decay = 0.01 + float(net) * (13.99 / 29.0)"
)
# Staggered logic: 3-way split
# (net + 6) % 15 -> (net + 12) % 30
# (net + 12) % 15 -> (net + 24) % 30
content = content.replace("net_b = (net + 6) % 15", "net_b = (net + 12) % 30")
content = content.replace("net_c = (net + 12) % 15", "net_c = (net + 24) % 30")

# std_dev2 logic
content = content.replace(
    "std_dev2 = 0.01 + float(net) * (3.99 / 14.0)",
    "std_dev2 = 0.01 + float(net) * (3.99 / 29.0)"
)

# For the 3-way split, the ranges were 22, 21, 21 out of 64.
# For 32 dimensions, let's do 11, 11, 10
content = content.replace("for i in range(22):", "for i in range(11):")
content = content.replace("for i in range(21):", "for i in range(11):", 1) # First 21 replacement
content = content.replace("for i in range(21):", "for i in range(10):", 1) # Second 21 replacement

# Update offsets
content = content.replace("h_start + 22 + i", "h_start + 11 + i")
content = content.replace("d_start + 22 + i", "d_start + 11 + i")

content = content.replace("h_start + 43 + i", "h_start + 22 + i")
content = content.replace("d_start + 43 + i", "d_start + 22 + i")

content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_30_Nets")

with open(filepath, "w") as f:
    f.write(content)
print("Applied 30 Nets patch")
