import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

content = content.replace("l1_decay = 10.0 + float(net) * (70.0 / 14.0)", "l1_decay = 15.0 + float(net) * (70.0 / 14.0)")

# Add 0_7_14 stagger because it gave 0.0394 and maybe combined with these scales it helps
content = content.replace("net_b = (net + 6) % 15", "net_b = (net + 7) % 15")
content = content.replace("net_c = (net + 12) % 15", "net_c = (net + 14) % 15")

replacement_var = """            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+22], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+22, f_start:f_start+ff_per_net], std=std_dev2 * S)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start+22:d_start+43], std=std_dev2 * 1.5)
            nn.init.normal_(mlp2.fc2.weight[d_start+22:d_start+43, f_start:f_start+ff_per_net], std=std_dev2 * 1.5 * S)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start+43:d_start+dim_per_net], std=std_dev2 * 2.0)
            nn.init.normal_(mlp2.fc2.weight[d_start+43:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * 2.0 * S)"""

content = re.sub(
    r'            std_dev2 = 0\.01 \+ float\(net\) \* \(3\.99 \/ 14\.0\)\n\n            nn\.init\.normal_\(mlp2\.fc1\.weight\[f_start:f_start\+ff_per_net, d_start:d_start\+dim_per_net\], std=std_dev2\)\n            nn\.init\.normal_\(mlp2\.fc2\.weight\[d_start:d_start\+dim_per_net, f_start:f_start\+ff_per_net\], std=std_dev2 \* S\)',
    replacement_var,
    content,
    flags=re.DOTALL
)

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_Final_Master3")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split (+0, +6, +12) with combined L1 15-85, Split Variance 1.5x/2.0x, and 0_7_14 Stagger")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
