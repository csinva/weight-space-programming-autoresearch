# Checkpoint 025: Inverting the MultiScale Hypothesis

## Observation
While reviewing the sweep configs that produced the previous state-of-the-art score (0.0922 from `Hybrid_Qwen1.5B`), we noticed an indexing flip: the sweep evaluated `Hybrid_Qwen1.5B_L28Mean_L14Last` by mapping `layer_mean = 28` (late layer) and `layer_last = 14` (middle layer).
Our initial Qwen-14B implementation manually set `layer_mean = 24` (middle layer) and `layer_last = 40` (late layer), which still performed spectacularly well (0.0903) but didn't quite crack the 0.0922 ceiling.

## Hypothesis
If the original 1.5B Hybrid achieved its peak by extracting the predictive syntax context (`last_token`) from the **middle** of the network and the broad semantic gist (`mean-pooling`) from the **end** of the network, then we should match that layer allocation pattern for the larger models. 

This implies that the abstract, broad contextual envelope (the narrative "gist") solidifies at the end of the transformer, whereas the immediate syntactic predictive machinery peaks in the middle layers before flattening out.

## Action
We launched an exhaustive test of the "Late Mean / Middle Last" formulation for Qwen 7B, 14B, and 32B models:
- Qwen 7B: `L28Mean_L14Last`
- Qwen 14B: `L48Mean_L24Last`
- Qwen 32B: `L64Mean_L32Last`

These evaluators are currently running. If the hypothesis holds, extracting the broad gist from the late layers of a 14B or 32B model should push the test correlation well beyond 0.0922.
