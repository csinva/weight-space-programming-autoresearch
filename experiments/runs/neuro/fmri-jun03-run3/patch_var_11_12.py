import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to the 0.0399 Masterpiece (L1 Scale 15-80)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# We are at 0.0399. Split variance 1.5/2.0 caused a drop to 0.0399 in Final Master (vs 0.0399 without it), and 1.2/1.5 in Master2 also didn't break 0.0400.
# Let's try an extremely gentle variance split: 1.1x and 1.2x.

replacement_var = """            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+22], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+22, f_start:f_start+ff_per_net], std=std_dev2 * S)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start+22:d_start+43], std=std_dev2 * 1.1)
            nn.init.normal_(mlp2.fc2.weight[d_start+22:d_start+43, f_start:f_start+ff_per_net], std=std_dev2 * 1.1 * S)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start+43:d_start+dim_per_net], std=std_dev2 * 1.2)
            nn.init.normal_(mlp2.fc2.weight[d_start+43:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * 1.2 * S)"""

content = re.sub(
    r'            std_dev2 = 0\.01 \+ float\(net\) \* \(3\.99 \/ 14\.0\)\n\n            nn\.init\.normal_\(mlp2\.fc1\.weight\[f_start:f_start\+ff_per_net, d_start:d_start\+dim_per_net\], std=std_dev2\)\n            nn\.init\.normal_\(mlp2\.fc2\.weight\[d_start:d_start\+dim_per_net, f_start:f_start\+ff_per_net\], std=std_dev2 \* S\)',
    replacement_var,
    content,
    flags=re.DOTALL
)

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_Var_11_12")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80", "from a 3-way split (+0, +6, +12) combining L1 15-80 and a very gentle 1.1/1.2x split variance")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
