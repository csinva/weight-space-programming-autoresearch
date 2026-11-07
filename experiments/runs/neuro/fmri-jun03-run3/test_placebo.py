import math
import torch
import torch.nn as nn

from interpretable_transformer import SimpleTransformer, VOCAB, write_weights, InterpretableEmbedder
model = SimpleTransformer(vocab_size=len(VOCAB), d_model=1020, n_heads=15, d_ff=4000, n_layers=2, max_seq_len=64)
write_weights(model)
embedder = InterpretableEmbedder(model, device='cpu')
out = embedder([" testing", " walking", " looked", " carefully"])
print("Morphology dimensions (960 to 1011) max:", out[:, 960:1012].max().item())
print("Morphology dimensions (960 to 1011) min:", out[:, 960:1012].min().item())
print("Morphology dimensions (960 to 1011) sum:", out[:, 960:1012].sum().item())
