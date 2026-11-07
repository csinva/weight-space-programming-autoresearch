import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

# We are going to apply different variances to the different stagger chunks in L2.
# Currently, it initializes std_dev2 = 0.01 + net * (3.99 / 14) and applies to all of d_start:d_start+dim_per_net.
# We will replace the std initialization for MLP2 to do chunks differently.

replacement_var = """            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            # Local chunk gets normal variance
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+22], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+22, f_start:f_start+ff_per_net], std=std_dev2 * S)
            
            # Medium chunk (+6) gets 2x variance to encourage non-linear jumps
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start+22:d_start+43], std=std_dev2 * 2.0)
            nn.init.normal_(mlp2.fc2.weight[d_start+22:d_start+43, f_start:f_start+ff_per_net], std=std_dev2 * 2.0 * S)
            
            # Far chunk (+12) gets 4x variance for extreme long range feature discovery
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start+43:d_start+dim_per_net], std=std_dev2 * 4.0)
            nn.init.normal_(mlp2.fc2.weight[d_start+43:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * 4.0 * S)"""

content = re.sub(
    r'std_dev2 = 0\.01 \+ float\(net\) \* \(3\.99 \/ 14\.0\).*?std=std_dev2 \* S\)',
    replacement_var,
    content,
    flags=re.DOTALL
)

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_SplitVar")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split (+0, +6, +12) with split variance increasing for further chunks")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
