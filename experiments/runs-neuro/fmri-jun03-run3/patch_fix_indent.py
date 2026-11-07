import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    lines = f.readlines()

out = []
for line in lines:
    if line.startswith("            std_dev2 = 0.01 + float(net)"):
        # We might have replaced it with too much indentation
        pass
    out.append(line)

# Wait, let's just do it cleanly again
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")
