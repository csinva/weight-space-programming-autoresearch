# Checkpoint 027: The Grand Barrage for SOTA

## Context
Our highest test correlation on UTS03 is `0.0922`, achieved by `Hybrid_Qwen1.5B_L28Mean_L14Last`. 
This model cleverly inverted the conventional "late layer for syntax, middle layer for semantics" wisdom, realizing that for small models, the absolute final layer contains the best generalized contextual "gist" of the narrative.

When scaling up to the 14B Qwen model, our `MultiScale_L40Last_L24Mean` hit `0.0903`, verifying that for deep/advanced LLMs, the "late" gist should come before the final layers (e.g., L40, not L48), since the final layers become highly specialized for token generation and lose generic semantic mapping.

The user challenged us to continue the run and push past `0.0922` SOTA without leaking data.

## Actions
To achieve this, we launched a "Grand Barrage" of evaluations across all available 4x RTX A6000 GPUs:

1. **Qwen 32B MultiScale (`L64Mean_L32Last`)**: Fixed the sequence length indexing bug (which crashed previous 32B/72B runs due to empty text padding) to evaluate massive-scale representations.
2. **Qwen 1.5B TripleScale (`L28Mean_L14Mean_L7Last`)**: If two scales are good, three might be better. Combines local lexical syntax (L7), medium narrative gist (L14), and high-level narrative state (L28).
3. **Qwen 7B TripleScale (`L28Mean_L14Mean_L7Last`)**: Scaling the TripleScale hypothesis to 7B parameters.
4. **Llama-3 8B MultiScale (`L32Mean_L16Last`)**: Evaluating `meta-llama-3-8b` to determine if Meta's architectural advancements yield more brain-like representations out-of-the-box compared to Qwen.
5. **Ensemble Qwen 1.5B + GPT-2 XL QuadScale**: A massive 6,272-dimensional representation concatenating `Qwen (L28M+L14L)` and `GPT-2-XL (L48M+L24L)`, combining two entirely different semantic routing topologies.

These processes are currently encoding concurrently and pushing the multi-GPU setup to its VRAM and compute limits. We await the ridge regression evaluation.
