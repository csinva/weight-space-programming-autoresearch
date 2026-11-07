# Checkpoint 037: The Final Structural Control

## Context
We found that the quad-scale combination of Mistral-7B and GPT-2-XL yielded our absolute maximum SOTA of `0.0939`.

But a critical, lingering question remains from our earlier experiments: How much of this correlation is derived from the *learned parameters* (the trillions of tokens Mistral and GPT-2 were trained on), and how much is derived simply from the *topological structure* of cascading self-attention across these disparate graphs?

Earlier, we saw that an untrained 0-layer continuous-time word reservoir achieved `0.0388`, indicating that structure provides baseline meaning.

## The Test
We have launched `Ensemble_Untrained_Mistral_GPT2`. This script instantiates the *exact same architecture* as our 0.0939 SOTA (Mistral L16 Last + Mistral L32 Mean + GPT-2 L24 Last + GPT-2 L48 Mean), but it uses `AutoModel.from_config()` to load **completely random weights**. 

This will isolate the purely structural contribution of the SOTA architecture. If this randomly initialized model scores high, it means the brain is highly correlated with the mere routing topology of transformers. If it crashes to 0, it means the correlation is almost entirely parameter-driven.
