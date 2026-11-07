import torch
from interpretable_transformer import SimpleTransformer, VOCAB, write_weights, InterpretableEmbedder
model = SimpleTransformer(len(VOCAB), 1020, 15, 4000, 2, 100)
write_weights(model)
embedder = InterpretableEmbedder(model, device='cpu')
out = embedder([" the", " and", " i", " bat"])
print(out.shape)
print("std:", out.std(dim=0).mean().item())
print("max:", out.max().item())
print("min:", out.min().item())

# check specific dims 400 to 899
print("dims 400-899 std:", out[:, 400:900].std(dim=0).mean().item())
