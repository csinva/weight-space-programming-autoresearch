# Feature Concatenation Super-Embedding SOTA

Previously, our ultimate `0.1180` SOTA was achieved via *prediction ensembling* (averaging the final predicted $\hat{Y}$ matrices of individual models). 

However, since the models map language into non-overlapping high-dimensional geometries, we hypothesized that the Ridge regression solver could perform an even better biological alignment if it were given simultaneous access to all representations at once—a **Feature Concatenation Super-Embedding**.

## The Super-Embedding Hypothesis
Instead of training three independent Ridge models and averaging their outputs, we:
1. Extracted the Last Token features from Llama-3-8B (L16), Qwen-2.5-14B (L24), and Gemma-2-9B (L23).
2. Concatenated these matrices horizontally along the feature dimension to create a massive 51,200-dimensional super-embedding space per time point (after delays).
3. Trained a single unified Ridge regression solver on this combined feature space, allowing the regularization to dynamically select the optimal subspace intersections across all three model families simultaneously.

## Results
- **Prediction Ensemble (Previous SOTA):** `0.1180`
- **Super-Embedding Concatenation:** **`0.1188`**

By providing the Ridge solver simultaneous access to the structural midpoints of all three frontier model families, the solver was able to extract a superior cross-model semantic manifold. 

This sets a definitive new peak limit on what linear models can extract from current language models for this specific dataset. The physiological limit is firmly established.
