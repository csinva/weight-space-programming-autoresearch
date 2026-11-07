import torch
from interpretable_transformer import build_embedder

embedder = build_embedder()
model = embedder.model
print(sum(p.numel() for p in model.parameters()))
