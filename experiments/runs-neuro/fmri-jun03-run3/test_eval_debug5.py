import torch
import numpy as np

# Let's just create a dummy vector like x1_last and check PyTorch's var
x = torch.zeros(2048)
x[1000] = 100000.0
x[1001] = -100000.0

print(f"var: {torch.var(x, unbiased=False).item()}")
print(f"var * N: {torch.var(x, unbiased=False).item() * 2048}")
print(f"sum of squares: {torch.sum(x**2).item()}")

# Now LayerNorm
ln = torch.nn.LayerNorm(2048)
y = ln(x)
print(f"max y: {y.max().item()}")

