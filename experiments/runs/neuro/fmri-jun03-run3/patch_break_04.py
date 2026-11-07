import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to the 0.0399 Masterpiece (L1 Scale 15-80)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# We are at 0.0399 on multiple models. To break 0.0400, we need one final combination.
# Let's combine:
# 1. L1 scale 15-80 (already there from the script)
# 2. L2 scale tightened: instead of 0.05-12.0, let's try 0.1-10.0 (slightly tighter long range to avoid overfitting).
content = content.replace("l2_decay = 0.05 + float(net) * (11.95 / 14.0)", "l2_decay = 0.1 + float(net) * (9.9 / 14.0)")

# 3. Variance scaling: 1x, 1.25x, 1.5x on the staggered portions.
replacement_var = """            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+22], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+22, f_start:f_start+ff_per_net], std=std_dev2 * S)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start+22:d_start+43], std=std_dev2 * 1.25)
            nn.init.normal_(mlp2.fc2.weight[d_start+22:d_start+43, f_start:f_start+ff_per_net], std=std_dev2 * 1.25 * S)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start+43:d_start+dim_per_net], std=std_dev2 * 1.5)
            nn.init.normal_(mlp2.fc2.weight[d_start+43:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * 1.5 * S)"""

content = re.sub(
    r'            std_dev2 = 0\.01 \+ float\(net\) \* \(3\.99 \/ 14\.0\)\n\n            nn\.init\.normal_\(mlp2\.fc1\.weight\[f_start:f_start\+ff_per_net, d_start:d_start\+dim_per_net\], std=std_dev2\)\n            nn\.init\.normal_\(mlp2\.fc2\.weight\[d_start:d_start\+dim_per_net, f_start:f_start\+ff_per_net\], std=std_dev2 \* S\)',
    replacement_var,
    content,
    flags=re.DOTALL
)

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_Break_04")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80", "from a 3-way split (+0, +6, +12) combining L1 15-80, L2 0.1-10.0, and 1.25/1.5x split variance")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
