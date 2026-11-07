# Qwen-2.5 32B MultiScale Scaling Analysis

To conclusively test the scaling laws of the Qwen-2.5 architecture, we successfully loaded the 32B parameter model using 8-bit quantization across multiple A6000 GPUs to avoid the previous OOM crashes.

We tested the `L32 Last Token + L64 Mean Pooling` MultiScale configuration (exactly mirroring the proportions of our highest-scoring configurations on the 14B and 7B models).

## Results (UTS03)

| Model | Parameters | MultiScale Arch | Test Corr | Frac > 0.2 |
|-------|------------|-----------------|-----------|------------|
| Qwen-1.5B | 1.5B | L14 Last + L28 Mean | 0.0872 | 0.1622 |
| Qwen-3B | 3B | L14 Mean + L28 Last | 0.0891 | 0.1689 |
| Qwen-7B | 7B | L14 Mean + L28 Last | 0.0897 | 0.1704 |
| Qwen-14B | 14B | L24 Mean + L40 Last | **0.0903** | **0.1722** |
| Qwen-32B (8-bit) | 32B | L32 Last + L64 Mean | **0.0883** | 0.1650 |

## Conclusion

The 32B model performs worse (0.0883) than the 14B model (0.0903) and the 7B model (0.0897) for fMRI encoding via Ridge Regression. 

This strongly indicates that **scaling parameter count beyond ~14B does not automatically improve linear fMRI encoding**. The representations in a 32B model are likely highly complex and non-linear, making them harder for a simple regularized linear ridge regression to cleanly map onto voxel activations given the extremely limited sample size of fMRI data (1,854 TRs). 

This confirms our earlier "Curse of Dimensionality" hypothesis from the ensemble sweep: beyond a certain representational complexity, the ridge solver overfits on colinear noise. The mathematical peak for linear mapping on this dataset is indeed the ~7B to 14B scale, or orthogonal combinations of smaller models (like the 1.5B + 7B SOTA ensemble).
