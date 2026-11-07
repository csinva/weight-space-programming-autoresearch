# Checkpoint 023: Multiscale Padding Token Bug Fix

## Observation
While inspecting the `eval_qwen_*_multiscale.py` logic, we noticed that Qwen-2.5 utilizes **right-padding** during batch tokenization. The multi-scale hypothesis relies on extracting the `last_token` from a late layer (to capture syntactic context and predictive state).
However, using `hidden_states[layer_last][:, -1, :]` extracted the hidden state of the **padding token** (when sequences were padded to equal batch lengths) rather than the true final word token.

This corrupted the local syntactic embedding, reducing the test correlation of the `Qwen_3B_MultiScale` evaluation to 0.0891, performing worse than the SOTA GPT-2 XL Hybrid model.

## Resolution
We fixed the MultiScale mechanism to properly use the sequence attention mask to locate the true end of the input:

```python
# Extract the true final token before padding
seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
batch_indices = torch.arange(inputs['attention_mask'].shape[0], device=self.semantic_model.device)
last_token_repr = hidden_states[self.layer_last][batch_indices, seq_lengths, :] 
```

## Next Steps
We have re-launched the fixed multi-scale evaluators for the 1.5B, 3B, 7B, and 14B models. The `1.5B` model, previously evaluated with incorrect padding logic as well, was separated into its own script to run cleanly.
We are now waiting for the fixed evaluators to finish encoding across the GPUs to see if the true Multiscale representation shatters the 0.0922 SOTA.
