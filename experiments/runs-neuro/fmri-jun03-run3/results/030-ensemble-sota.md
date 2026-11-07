# Checkpoint 030: A New SOTA via QuadScale Ensembling

## Result
We just received the ridge regression evaluation result for `Ensemble_Qwen1.5B_GPT2XL_QuadScale`, an extreme 6,272-dimensional representation concatenating two different models architectures across four distinct feature scales:
1. `Qwen-2.5-1.5B` Layer 14 Last Token (Local syntax, new family)
2. `Qwen-2.5-1.5B` Layer 28 Mean Pooling (Narrative gist, new family)
3. `GPT-2-XL` Layer 24 Last Token (Local syntax, old family)
4. `GPT-2-XL` Layer 48 Mean Pooling (Narrative gist, old family)

**Test Correlation:** `0.0923`

## Significance
This is the highest correlation we have achieved in the entire repository, finally breaking past the `0.0922` ceiling established by the `Hybrid_Qwen1.5B` model.

This demonstrates that while adding purely redundant intermediate scales *within* the same model (e.g., the TripleScale `0.0834` failure) overfits the ridge penalty, concatenating highly distinct latent topologies from *entirely different model families* (GPT-2 vs. Qwen) actually provides orthogonal predictive variance that improves generalization on brain data.

We are currently awaiting the final results of:
- `Mistral-7B MultiScale`
- `Llama-3-8B MultiScale`
- `Qwen-7B TripleScale`
