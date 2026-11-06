"""
Snapshot rotcopy_tags_v1 (198/200 on seed 0).

Identical to rotcopy_tags_v2_sharp40.py except for layer-2 sharpness:
    S2 = 20.0 * math.sqrt(4)     # effective logit 20 per unit tag (v2 uses 40)
With effective sharpness 20 the "=" fallback only beats untagged tokens by 5 logits,
so inactive layer-2 heads leak ~20% attention onto digit tokens; two 19xx years
became 21xx. To reproduce, import v2 and override S2.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rotcopy_tags_v2_sharp40 as _v2
from rotcopy_tags_v2_sharp40 import *  # noqa: F401,F403

model_shorthand_name = "rotcopy_tags_v1"


def write_weights(model, task) -> None:
    import torch
    _v2.write_weights(model, task)
    with torch.no_grad():
        A2 = model.attn2
        # rescale every layer-2 query from 40*sqrt(4) to 20*sqrt(4)
        A2.W_q.mul_(0.5)
