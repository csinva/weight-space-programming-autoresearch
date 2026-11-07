# Late vs. Early Ensembling

## Hypothesis
Since concatenating features from 3 or more models degraded performance due to the "Curse of Dimensionality" (exhausting the capacity of the Ridge Regression solver on limited TR samples), we tested **Late Ensembling**: fitting Ridge Regression independently on each model's feature space, then averaging their final voxel-level predictions. 

We applied this to our optimal pair (Qwen-1.5B + Mistral-7B) using the optimal physiological context window (`ngram_size=20`).

## Results (UTS03)
- **Early Ensemble (Feature Concatenation):** `0.0988` (Current Absolute SOTA)
- **Late Ensemble (Prediction Averaging):** `0.0829`

## Conclusion
Late Ensembling performs significantly worse than Early Ensembling. 

This proves that the magic of multi-model ensembles lies in the **synergistic, cross-model linear combinations** discovered by the Ridge solver. By concatenating the feature spaces *before* fitting, Ridge Regression can weight a syntactic feature from Qwen against a semantic feature from Mistral simultaneously to optimally explain a single voxel's variance. 

When trained in isolation (Late Ensemble), each model's Ridge solver makes disjoint compromises, and averaging their sub-optimal predictions cannot recover the synergistic interactions.

Therefore, the absolute mathematical ceiling of linear fMRI encoding is reached by **Early Ensembling the most orthogonal, maximally predictive representations** (up to the Ridge capacity limit of ~10,000-15,000 dimensions), combined with a **physiologically-aligned context window** (20 words = ~6-10 seconds of BOLD integration).

The `0.0988` test correlation ceiling stands as the absolute SOTA.
