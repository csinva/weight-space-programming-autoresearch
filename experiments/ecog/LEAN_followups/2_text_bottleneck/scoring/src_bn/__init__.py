"""multisubj (from newtest) copy of the (empty) harness package init. Switches PyTorch to its
expandable-segments CUDA allocator: the streamed fit keeps XtX and tune rows on the GPU
across chunks, which fragmented memory enough to OOM the large BIOPHYS encoders on 80 GB.
This changes memory placement only, never any computed value."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

torch.cuda.memory._set_allocator_settings("expandable_segments:True")
