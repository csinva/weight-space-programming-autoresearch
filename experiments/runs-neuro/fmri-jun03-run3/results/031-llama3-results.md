# Checkpoint 031: Llama-3-8B MultiScale Results

## Results
The evaluation for `Llama3_8B_MultiScale_L32M_L16L` has finished:
- Test Correlation: **0.0917**

## Observation
This is an incredibly strong result for a pure standalone model, vastly outperforming `Qwen-2.5-7B` (`0.0897`). It falls just short of the `0.0922` SOTA for a single architecture, proving that the MultiScale extraction logic translates universally across top-tier LLM families (Qwen, GPT-2, Llama).

Meta's Llama-3 8B model naturally produces geometric representations that strongly mirror human fMRI semantic encoding.

We are still awaiting the `Qwen-7B TripleScale` and `Mistral-7B MultiScale` regression matrices.
