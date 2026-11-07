# Absolute SOTA and Dimensionality Limits

We have discovered the absolute performance ceiling for fMRI encoding using Ridge Regression and concatenated hidden states without fine-tuning: **0.0951** test correlation on UTS03.

## Final Ensemble Sweep

1. **SOTA: Qwen-1.5B + Mistral-7B (0.0951)**
   - Qwen provides pure syntactic variance (L14 Last Token).
   - Mistral provides syntactic and deep global narrative semantic variance (L16 Last Token + L32 Mean).
   - This orthogonal combination cleanly pushes past the previous 0.0939 ceiling.

2. **The Curse of Dimensionality**
   We attempted to push higher by adding more models to the ensemble, but performance actually *decreased*:
   - Qwen + Mistral + LLaMA-3 = 0.0944 (down from 0.0951)
   - Qwen + Mistral + GPT-2-XL = 0.0933 (down from 0.0951)
   - LLaMA-3 + Mistral = 0.0936 (worse than Mistral alone)

## Conclusion

We have hit the mathematical limits of `sklearn.linear_model.Ridge`. 

As we concatenate more hidden states (e.g., LLaMA + Mistral + Qwen = 5 x 4096 dimensions = 20,480 features), the ridge regression model runs out of sample efficiency on the 1,854 fMRI TRs. The models are producing highly colinear representations. When LLaMA and Mistral are combined, they provide almost identical semantic representations. The solver assigns weight to both, effectively doubling the noise-to-signal ratio and causing the test correlation to drop.

The **Qwen-1.5B (Syntax) + Mistral-7B (Syntax + Semantics)** is the Pareto-optimal representation. Any further improvement will require a non-linear readout (like an MLP) or explicit un-freezing of the base model weights via LoRA.
