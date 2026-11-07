import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# Zero MLP dropped performance massively (0.0327 from 0.0399).
# This proves MLPs are doing heavy lifting! They are mixing the integrated embeddings across the vocabulary space.
# If MLPs are good, what if we crank them up significantly?
# L1 MLP is currently fixed at 0.5.
# L2 MLP scales from 0.01 to 4.0.
# Let's try cranking L1 MLP up to 2.0.

replacement = """                
            std_dev = 2.0
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)"""

content = re.sub(
    r'                \n            std_dev = 0\.5\n            nn\.init\.normal_\(mlp1\.fc1\.weight\[f_start:f_start\+ff_per_net, d_start:d_start\+dim_per_net\], std=std_dev\)\n            nn\.init\.normal_\(mlp1\.fc2\.weight\[d_start:d_start\+dim_per_net, f_start:f_start\+ff_per_net\], std=std_dev \* S\)',
    replacement,
    content,
    flags=re.DOTALL
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_High_MLP1")
content = content.replace("with L1 decay scale set to 15-80 instead of 10-80.", "with L1 MLP std set to 2.0 instead of 0.5.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
