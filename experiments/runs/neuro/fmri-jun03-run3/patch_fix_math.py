import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    content = f.read()

content = content.replace("import math\\n        l2_log_start", "l2_log_start")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
