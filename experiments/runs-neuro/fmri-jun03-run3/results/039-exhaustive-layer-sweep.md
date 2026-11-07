# Checkpoint 039: The Exhaustive Layer Sweep

## Motivation
While we have discovered the `0.0939` SOTA by ensembling Mistral and GPT-2, and we have proven that random topologies score around `0.040` vs their trained `0.094` counterparts, we still lack a cohesive, rigorous picture of exactly how parameters vs. topology transform brain correlations across the *depth* of a single model.

Are middle layers better simply because they have more nonlinearities than early layers, or because the specific parameters encode syntax? How exactly does Mean vs Last Token extraction evolve across the layers in both random and trained states?

## Action
We launched an exhaustive sweep on Qwen-2.5-1.5B (the fastest model with high performance):
- **Layers**: `0`, `7`, `14`, `21`, `28` (spanning early syntax to deep semantics)
- **Pooling**: `Last Token` vs `Mean Pooling`
- **State**: `Trained` vs `Random Initialization`

This produces 20 granular data points that will definitively map the transformation of linguistic features into brain-like semantics across an LLM, separating the structural impact from the learned knowledge.
