"""Image-only subprocess entry point. No private scoring modules are available."""
import importlib.util
import json
import os
import resource

import numpy as np
import torch

BATCH_TOLERANCE = 0.01  # Approximate independence check; allow 1% vector L2 roundoff.

resource.setrlimit(resource.RLIMIT_FSIZE, (2_000_000_000, 2_000_000_000))
spec = importlib.util.spec_from_file_location('candidate', '/model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = module.MODEL
if isinstance(model, torch.nn.Module):
    model.eval()
device = os.environ['NEURO_CANDIDATE_DEVICE']
if device.startswith('cuda') and not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable inside candidate isolation; refusing device fallback')
images = torch.from_numpy(np.load('/images.npy', allow_pickle=False)).to(device)
with torch.no_grad():
    output = model(images)
    # Fixed-seed, sample-independent representations must agree across batching.
    for index in range(min(3, len(images))):
        single = torch.as_tensor(model(images[index:index + 1]))
        batched = torch.as_tensor(output[index:index + 1])
        if batched.shape != single.shape:
            raise ValueError('Representation shape depends on batch composition')
        a, b = batched.double(), single.double()
        difference = torch.linalg.vector_norm(a - b)
        scale = torch.maximum(torch.linalg.vector_norm(a), torch.linalg.vector_norm(b))
        if not torch.isfinite(difference) or difference > BATCH_TOLERANCE * scale:
            raise ValueError('Representation depends on batch composition or mutable state')
if torch.is_tensor(output):
    output = output.detach().cpu().numpy()
output = np.asarray(output)
np.save('/output/features.npy', output, allow_pickle=False)
seen = set()
def count(value, depth=0):
    if depth > 8 or id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, dict):
        return sum(count(v, depth + 1) for v in value.values())
    if isinstance(value, (tuple, list, set)):
        return sum(count(v, depth + 1) for v in value)
    if hasattr(value, '__dict__'):
        return count(vars(value), depth + 1)
    return 0
metadata = {key: str(getattr(model, key, ''))[:600] for key in ('name', 'description', 'insight', 'idea')}
metadata['n_params'] = max(int(getattr(model, 'n_params', 0)), count(model))
with open('/output/metadata.json', 'w') as file:
    json.dump(metadata, file)
