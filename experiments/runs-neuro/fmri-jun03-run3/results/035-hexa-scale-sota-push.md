# Checkpoint 035: Pushing for the Ultimate HexaScale SOTA

## Theory
We achieved an absolute SOTA of `0.0939` by concatenating Mistral-7B's DualScale (Middle Last Token + Final Mean Pool) with GPT-2-XL's DualScale. 

If adding orthogonal topological representations from different model families linearly increases encoding performance by capturing disjoint semantic variances in the brain, then combining all three of our best-performing standalone architectures should push performance even higher.

## Action
We launched two extreme-scale ensembling experiments:
1. `Ensemble_Mistral7B_Llama3_QuadScale`: Combining the two absolute best standalone architectures (Mistral 7B at `0.0936` and LLaMA-3 8B at `0.0917`).
2. `Ensemble_Mistral_Llama_GPT2_HexaScale`: A massive 6-scale concatenation combining Mistral, LLaMA-3, and GPT-2-XL. This results in a massive feature dimension spanning three completely different semantic embedding families.

These scripts are currently running on our multi-GPU cluster.
