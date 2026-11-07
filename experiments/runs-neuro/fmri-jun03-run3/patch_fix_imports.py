import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    lines = f.readlines()

out = []
for line in lines:
    if line.strip() == "import math":
        continue
    out.append(line)

out.insert(6, "import math\n")

with open(filepath, "w") as f:
    f.writelines(out)
print("Updated successfully")
