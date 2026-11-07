import torch
from interpretable_transformer import build_embedder

embedder = build_embedder()
model = embedder.model
print("d_model:", model.d_model)
