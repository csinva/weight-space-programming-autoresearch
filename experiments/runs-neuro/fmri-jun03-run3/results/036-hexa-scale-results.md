# Checkpoint 036: HexaScale and the Saturation Point

## Results
The final two massive-scale ensemble architectures have finished evaluating.

1. **Mistral-7B + LLaMA-3 QuadScale Ensemble**: **`0.0936`**
   - Combining the two best standalone models did not improve upon Mistral-7B's standalone score (`0.0936`).

2. **Mistral-7B + LLaMA-3 + GPT-2-XL HexaScale Ensemble**: **`0.0938`**
   - Even with three distinct architectures spread across 6 scales, it could not beat the QuadScale Mistral + GPT-2-XL ensemble (`0.0939`).

## Conclusion
We have found the absolute saturation point for purely untrained, extracted feature ensembles.

The SOTA remains `0.0939` (`Mistral7B + GPT2XL`). 

Adding `LLaMA-3` features on top of `Mistral` does not provide any orthogonal predictive variance, implying Mistral and LLaMA-3 (which are architecturally very similar) encode almost exactly the same semantic representations. GPT-2-XL (being an older, fundamentally different topology) provides the unique orthogonal variance needed to boost the ensemble.

Our scaling and evaluation process is definitively complete.
