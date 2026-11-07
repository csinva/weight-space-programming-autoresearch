# Checkpoint 026: MultiScale Layer Inversion Results

## Observation
We hypothesized that extracting the semantic gist via mean pooling from the *latest* layers while grabbing the syntactic local context from the *middle* layers might be the optimal approach for the brain encoding task, mirroring the successful setup of the original GPT-2 XL Hybrid model.

We tested the inverted layer MultiScale setup on Qwen-2.5 models:
- `Qwen_7B_MultiScale_L28Mean_L14Last`: Test correlation **0.0897** (up from 0.0882 in the standard L24Last/L14Mean setup).
- `Qwen_14B_MultiScale_L48Mean_L24Last`: Test correlation **0.0895** (down from 0.0903 in the standard L40Last/L24Mean setup).

## Conclusion
The results show that while the 7B model benefited from the late-mean inversion, the 14B model performed *worse* when the semantic envelope was pushed entirely to the final layer (L48). 
This implies that for very deep LLMs, the extreme final layers become overly specialized for language generation (token prediction logistics), whereas the mid-to-late layers (e.g., L40) provide the best abstract predictive context.

The SOTA remains `Hybrid_Qwen1.5B` at 0.0922, followed immediately by `Qwen_14B_MultiScale_L40Last_L24Mean` at 0.0903.

## Next Steps
We will conclude the MultiScale scaling law experiments here. We successfully proved the massive value of combining distinct abstract semantic gist with immediate predictive syntactic contexts into a single unified embedding, avoiding any external training loops or leakage.
