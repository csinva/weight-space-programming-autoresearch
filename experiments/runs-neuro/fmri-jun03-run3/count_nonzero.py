import torch
import math
import torch.nn as nn
import os
import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer_temp.py", "r") as f:
    content = f.read()

import sys
from interpretable_transformer_temp import SimpleTransformer, VOCAB, write_weights

model = SimpleTransformer(vocab_size=len(VOCAB), max_seq_len=64, d_model=1020, n_heads=15, n_layers=2, d_ff=4000)
write_weights(model)

total_params = 0
nonzero_params = 0

for name, p in model.named_parameters():
    total_params += p.numel()
    nonzero_params += (p != 0.0).sum().item()

print(f"Total params: {total_params:,}")
print(f"Non-zero params: {nonzero_params:,}")
print(f"Sparsity: {100.0 * (1.0 - nonzero_params / total_params):.4f}%")
