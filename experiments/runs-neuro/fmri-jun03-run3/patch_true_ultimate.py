import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

# 1. Change L1 scale from 10.0-80.0 to 15.0-85.0
content = content.replace("l1_decay = 10.0 + float(net) * (70.0 / 14.0)", "l1_decay = 15.0 + float(net) * (70.0 / 14.0)")

# 2. Add Split Variance
replacement_var = """            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+22], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+22, f_start:f_start+ff_per_net], std=std_dev2 * S)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start+22:d_start+43], std=std_dev2 * 2.0)
            nn.init.normal_(mlp2.fc2.weight[d_start+22:d_start+43, f_start:f_start+ff_per_net], std=std_dev2 * 2.0 * S)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start+43:d_start+dim_per_net], std=std_dev2 * 4.0)
            nn.init.normal_(mlp2.fc2.weight[d_start+43:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * 4.0 * S)"""

# I need a robust replacement
old_var_code = """            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)"""

content = content.replace(old_var_code, replacement_var)

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_True_Ultimate")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split (+0, +6, +12) with perfectly combined L1 15-85 and Split Variance 2.0x/4.0x")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
