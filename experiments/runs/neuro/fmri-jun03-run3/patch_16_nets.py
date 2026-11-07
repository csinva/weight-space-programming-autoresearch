import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Replace num_nets = 15 with 16
# dim_per_net = 64 with 60
# ff_per_net = 256 with 240
content = content.replace("num_nets = 15", "num_nets = 16")
content = content.replace("dim_per_net = 64", "dim_per_net = 60")
content = content.replace("ff_per_net = 256", "ff_per_net = 240")

# L1 decay logic
content = content.replace(
    "l1_decay = 15.0 + float(net) * (65.0 / 14.0)",
    "l1_decay = 15.0 + float(net) * (65.0 / 15.0)"
)
# L2 decay logic
content = content.replace(
    "l2_decay = 0.01 + float(net) * (13.99 / 14.0)",
    "l2_decay = 0.01 + float(net) * (13.99 / 15.0)"
)
# Staggered logic: 3-way split
# (net + 6) % 15 -> (net + 6) % 16
# (net + 12) % 15 -> (net + 12) % 16
content = content.replace("net_b = (net + 6) % 15", "net_b = (net + 6) % 16")
content = content.replace("net_c = (net + 12) % 15", "net_c = (net + 12) % 16")

# std_dev2 logic
content = content.replace(
    "std_dev2 = 0.01 + float(net) * (3.99 / 14.0)",
    "std_dev2 = 0.01 + float(net) * (3.99 / 15.0)"
)

# For 60 dimensions, let's do 20, 20, 20
content = content.replace("for i in range(22):", "for i in range(20):")
content = content.replace("for i in range(21):", "for i in range(20):") 

# Update offsets
content = content.replace("h_start + 22 + i", "h_start + 20 + i")
content = content.replace("d_start + 22 + i", "d_start + 20 + i")

content = content.replace("h_start + 43 + i", "h_start + 40 + i")
content = content.replace("d_start + 43 + i", "d_start + 40 + i")

content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_16_Nets")

with open(filepath, "w") as f:
    f.write(content)
print("Applied 16 Nets patch")
