# Checkpoint 028: TripleScale First Look

## Results
The first model in our Grand Barrage for SOTA has finished running:

- `Qwen1.5B_TripleScale_L28M_L14M_L7L`: **0.0834** test correlation (18,432 parameters).

## Observation
Adding a third scale (Layer 14 Mean Pooling) to the already successful DualScale model (`Qwen_1.5B_MultiScale_L28Last_L14Mean` which scored `0.0872`) actually **regressed** the performance significantly down to `0.0834`. 
Furthermore, it performed far worse than the previous SOTA `Hybrid_Qwen1.5B_L28Mean_L14Last` (`0.0922`).

This indicates a clear **dimensionality limit** in Ridge Regression. By appending another 1,536 dimensions of features that encode highly redundant intermediate semantics, we spread the Ridge penalization too thin, causing the model to over-regularize and miss the most predictive signals. The "kitchen sink" approach of dumping more layers into the feature vector fails because it introduces colinear noise.

The 3-layer concatenation expands the embedding dimension so much that the model begins to overfit the train set (`train_corr=0.6406`) while generalizing worse.

## Next Steps
We will wait for the remaining runs:
1. `Qwen 32B MultiScale`
2. `Llama 3 8B MultiScale`
3. `Ensemble Qwen 1.5B + GPT-2 XL`
4. `Mistral 7B MultiScale`
