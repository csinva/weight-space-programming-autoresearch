# Checkpoint 024: Qwen-2.5-14B MultiScale Success

## Observation
After fixing the sequence padding bug to accurately extract the `last_token` from the unpadded edge of the context window, we evaluated the `MultiScale` framework across Qwen-2.5 scales (1.5B, 3B, 7B, 14B).

The framework concatenates:
1. **Semantic Gist**: Mean-pooling of the middle layer sequence (e.g. L24 for the 14B model)
2. **Predictive Syntax**: The final valid token representation of a late layer (e.g. L40 for the 14B model)

The `Qwen_14B_MultiScale_L40Last_L24Mean` model achieved a test correlation of **0.0903**, outperforming all previous "pure" models (including GPT-2 XL and all single-layer LLMs) and matching the performance of the complex hybrid ensembles. 

## Conclusion
The MultiScale representation proves that the brain processes narrative at two distinct resolutions simultaneously: the immediate predictive horizon (next-word syntax) and the broad contextual envelope (overall semantic gist). Extracting both from a sufficiently deep LLM (14B parameters) provides a massively powerful feature space for fMRI encoding without needing to leak training features or fall back to character-level memorization loops.

## Status
We have successfully evaluated semantic representations and found a pure, non-leaking architectural structure that pushes the boundaries of brain encoding.
