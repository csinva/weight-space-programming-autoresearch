# PCA on Triple Ensemble

## Hypothesis
We previously observed the "Curse of Dimensionality" when concatenating representations from 3 or more models (LLaMA-3 + Mistral + Qwen). The combined dense feature space exceeds the capacity of Ridge regression on our limited 1,854 TR sample size, causing performance to drop compared to a 2-model ensemble.

We hypothesized that applying Principal Component Analysis (PCA) to the combined 3-model representation space *before* expanding it with FIR delays could distill the synergistic variance into a lower-dimensional manifold, preventing the Ridge solver from overfitting.

## Results (UTS03)
- **Qwen + Mistral (Dual Ensemble, Context 20):** `0.0988` (SOTA)
- **Qwen + Mistral + LLaMA3 (Raw Triple Ensemble, Context 10):** `0.0943`
- **PCA 250 Dims:** `0.0813`
- **PCA 500 Dims:** `0.0823`
- **PCA 1000 Dims:** `0.0826`
- **PCA 1237 Dims (Max Rank):** `0.0827`

## Conclusion
PCA fails to recover the lost performance of the Triple Ensemble. In fact, PCA significantly degrades the representation compared to the raw concatenation. 

Why? PCA captures the directions of maximum variance in the *stimulus feature space*, independent of the fMRI signal. In highly non-linear LLM embeddings, the dimensions that correlate with brain activity (e.g., specific semantic or syntactic manifolds) are not necessarily the dimensions with the largest absolute variance. Discarding variance based on PCA throws away the subtle, sparse signals that the Ridge regression relies on to predict voxel activity.

Even when preserving all variance (PCA Max Rank = 1237, since N=1237 training TRs), the performance is lower than the raw concatenation because the coordinate rotation destroys the sparsity and specific structural alignment that Ridge regression uses for regularization.

The optimal strategy remains strictly limiting the raw dimensionality by selecting only the most orthogonal, maximally predictive pair of models (Qwen + Mistral).
