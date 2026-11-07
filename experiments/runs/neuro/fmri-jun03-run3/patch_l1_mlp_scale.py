import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# Replace std_dev = 0.5 with a scaled version
replacement = """                
            std_dev = 0.1 + float(net) * (0.9 / 14.0)
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)"""

content = re.sub(
    r'                \n            std_dev = 0\.5\n            nn\.init\.normal_\(mlp1\.fc1\.weight\[f_start:f_start\+ff_per_net, d_start:d_start\+dim_per_net\], std=std_dev\)\n            nn\.init\.normal_\(mlp1\.fc2\.weight\[d_start:d_start\+dim_per_net, f_start:f_start\+ff_per_net\], std=std_dev \* S\)',
    replacement,
    content,
    flags=re.DOTALL
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_L1_MLP_Scale_01_10")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80.", "from a 3-way split (+0, +6, +12) with L1 MLP variance scaled from 0.1 to 1.0 instead of fixed 0.5.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
