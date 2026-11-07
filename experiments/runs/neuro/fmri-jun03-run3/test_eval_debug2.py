import torch
import numpy as np
from interpretable_transformer import build_embedder, VOCAB

embedder = build_embedder()
model = embedder.model

with torch.no_grad():
    print(f"fc1 bias max: {model.blocks[0].mlp.fc1.bias.max().item()}")
    print(f"fc1 weight max: {model.blocks[0].mlp.fc1.weight.max().item()}")
    print(f"LN2 weight: {model.blocks[0].ln2.weight[:5].tolist()}")
    print(f"LN2 bias: {model.blocks[0].ln2.bias[:5].tolist()}")

