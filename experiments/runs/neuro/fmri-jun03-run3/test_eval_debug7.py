import torch

# Let's recreate x1_last exactly
x1_last = torch.zeros(2048)
x1_last[1001] = -100000.0
x1_last[1000] = 100000.0
x1_last[1003] = 3969.0
x1_last[1002] = 63.0
x1_last[1] = 13.39

print("Manual:", torch.sum((x1_last - x1_last.mean())**2) / 2048)
print("PyTorch:", torch.var(x1_last, unbiased=False))

