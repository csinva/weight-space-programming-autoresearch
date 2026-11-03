I had no weights, so these predictions come from reasoning about how each edit works. None of them are measured.

**Embedding edits (SC037, SC042, SC045, SC048):** These change one entry in one target token's embedding. Because the embeddings are tied, they mainly shift that token's output logit, and only on the prompt where it is the target. The logit moves by −4 times the final normalized hidden value in that dimension, whose sign I don't know. So I predict a moderate expected drop on that prompt and almost no change on the other seven.

**Single-weight and norm-gain edits (all others):** Most of these should have small effects. I gave larger effects to:
- multiplying a gain by 10, especially in early layers;
- adding ±4 to a down_proj entry;
- zeroing the largest-magnitude weight in layer 13's down_proj.

I made the effect bigger on prompts where the target's lead is small (P7 most of all, then P4 and P3) and smaller on very confident prompts (P6, P5).

```json
{"predictions": [
{"iid":"SC036","pid":"P0","dlogp":-0.08,"p_flip":0.02},{"iid":"SC036","pid":"P1","dlogp":-0.1,"p_flip":0.03},{"iid":"SC036","pid":"P2","dlogp":-0.08,"p_flip":0.03},{"iid":"SC036","pid":"P3","dlogp":-0.1,"p_flip":0.03},{"iid":"SC036","pid":"P4","dlogp":-0.12,"p_flip":0.04},{"iid":"SC036","pid":"P5","dlogp":-0.06,"p_flip":0.01},{"iid":"SC036","pid":"P6","dlogp":-0.03,"p_flip":0.01},{"iid":"SC036","pid":"P7","dlogp":-0.15,"p_flip":0.1},
{"iid":"SC037","pid":"P0","dlogp":0.0,"p_flip":0.0},{"iid":"SC037","pid":"P1","dlogp":0.0,"p_flip":0.0},{"iid":"SC037","pid":"P2","dlogp":0.0,"p_flip":0.0},{"iid":"SC037","pid":"P3","dlogp":-0.5,"p_flip":0.2},{"iid":"SC037","pid":"P4","dlogp":0.0,"p_flip":0.0},{"iid":"SC037","pid":"P5","dlogp":0.0,"p_flip":0.0},{"iid":"SC037","pid":"P6","dlogp":0.0,"p_flip":0.0},{"iid":"SC037","pid":"P7","dlogp":0.0,"p_flip":0.0},
{"iid":"SC038","pid":"P0","dlogp":-0.04,"p_flip":0.01},{"iid":"SC038","pid":"P1","dlogp":-0.05,"p_flip":0.02},{"iid":"SC038","pid":"P2","dlogp":-0.04,"p_flip":0.02},{"iid":"SC038","pid":"P3","dlogp":-0.05,"p_flip":0.02},{"iid":"SC038","pid":"P4","dlogp":-0.07,"p_flip":0.03},{"iid":"SC038","pid":"P5","dlogp":-0.03,"p_flip":0.01},{"iid":"SC038","pid":"P6","dlogp":-0.02,"p_flip":0.0},{"iid":"SC038","pid":"P7","dlogp":-0.08,"p_flip":0.06},
{"iid":"SC039","pid":"P0","dlogp":0.0,"p_flip":0.0},{"iid":"SC039","pid":"P1","dlogp":0.0,"p_flip":0.0},{"iid":"SC039","pid":"P2","dlogp":0.0,"p_flip":0.0},{"iid":"SC039","pid":"P3","dlogp":0.0,"p_flip":0.0},{"iid":"SC039","pid":"P4","dlogp":0.0,"p_flip":0.0},{"iid":"SC039","pid":"P5","dlogp":0.0,"p_flip":0.0},{"iid":"SC039","pid":"P6","dlogp":0.0,"p_flip":0.0},{"iid":"SC039","pid":"P7","dlogp":0.0,"p_flip":0.01},
{"iid":"SC040","pid":"P0","dlogp":-0.02,"p_flip":0.01},{"iid":"SC040","pid":"P1","dlogp":-0.02,"p_flip":0.01},{"iid":"SC040","pid":"P2","dlogp":-0.02,"p_flip":0.01},{"iid":"SC040","pid":"P3","dlogp":-0.02,"p_flip":0.01},{"iid":"SC040","pid":"P4","dlogp":-0.03,"p_flip":0.01},{"iid":"SC040","pid":"P5","dlogp":-0.01,"p_flip":0.0},{"iid":"SC040","pid":"P6","dlogp":-0.01,"p_flip":0.0},{"iid":"SC040","pid":"P7","dlogp":-0.04,"p_flip":0.03},
{"iid":"SC041","pid":"P0","dlogp":-0.1,"p_flip":0.02},{"iid":"SC041","pid":"P1","dlogp":-0.12,"p_flip":0.04},{"iid":"SC041","pid":"P2","dlogp":-0.1,"p_flip":0.04},{"iid":"SC041","pid":"P3","dlogp":-0.13,"p_flip":0.04},{"iid":"SC041","pid":"P4","dlogp":-0.15,"p_flip":0.06},{"iid":"SC041","pid":"P5","dlogp":-0.07,"p_flip":0.01},{"iid":"SC041","pid":"P6","dlogp":-0.04,"p_flip":0.01},{"iid":"SC041","pid":"P7","dlogp":-0.2,"p_flip":0.12},
{"iid":"SC042","pid":"P0","dlogp":0.0,"p_flip":0.0},{"iid":"SC042","pid":"P1","dlogp":0.0,"p_flip":0.0},{"iid":"SC042","pid":"P2","dlogp":-0.5,"p_flip":0.2},{"iid":"SC042","pid":"P3","dlogp":0.0,"p_flip":0.0},{"iid":"SC042","pid":"P4","dlogp":0.0,"p_flip":0.0},{"iid":"SC042","pid":"P5","dlogp":0.0,"p_flip":0.0},{"iid":"SC042","pid":"P6","dlogp":0.0,"p_flip":0.0},{"iid":"SC042","pid":"P7","dlogp":0.0,"p_flip":0.0},
{"iid":"SC043","pid":"P0","dlogp":-0.12,"p_flip":0.03},{"iid":"SC043","pid":"P1","dlogp":-0.14,"p_flip":0.05},{"iid":"SC043","pid":"P2","dlogp":-0.12,"p_flip":0.05},{"iid":"SC043","pid":"P3","dlogp":-0.15,"p_flip":0.05},{"iid":"SC043","pid":"P4","dlogp":-0.18,"p_flip":0.07},{"iid":"SC043","pid":"P5","dlogp":-0.08,"p_flip":0.02},{"iid":"SC043","pid":"P6","dlogp":-0.05,"p_flip":0.01},{"iid":"SC043","pid":"P7","dlogp":-0.24,"p_flip":0.14},
{"iid":"SC044","pid":"P0","dlogp":-0.01,"p_flip":0.0},{"iid":"SC044","pid":"P1","dlogp":-0.01,"p_flip":0.0},{"iid":"SC044","pid":"P2","dlogp":-0.01,"p_flip":0.0},{"iid":"SC044","pid":"P3","dlogp":-0.01,"p_flip":0.0},{"iid":"SC044","pid":"P4","dlogp":-0.01,"p_flip":0.0},{"iid":"SC044","pid":"P5","dlogp":0.0,"p_flip":0.0},{"iid":"SC044","pid":"P6","dlogp":0.0,"p_flip":0.0},{"iid":"SC044","pid":"P7","dlogp":-0.01,"p_flip":0.01},
{"iid":"SC045","pid":"P0","dlogp":0.0,"p_flip":0.0},{"iid":"SC045","pid":"P1","dlogp":0.0,"p_flip":0.0},{"iid":"SC045","pid":"P2","dlogp":0.0,"p_flip":0.0},{"iid":"SC045","pid":"P3","dlogp":0.0,"p_flip":0.0},{"iid":"SC045","pid":"P4","dlogp":0.0,"p_flip":0.0},{"iid":"SC045","pid":"P5","dlogp":-0.4,"p_flip":0.06},{"iid":"SC045","pid":"P6","dlogp":0.0,"p_flip":0.0},{"iid":"SC045","pid":"P7","dlogp":0.0,"p_flip":0.0},
{"iid":"SC046","pid":"P0","dlogp":-0.15,"p_flip":0.04},{"iid":"SC046","pid":"P1","dlogp":-0.18,"p_flip":0.06},{"iid":"SC046","pid":"P2","dlogp":-0.15,"p_flip":0.06},{"iid":"SC046","pid":"P3","dlogp":-0.2,"p_flip":0.06},{"iid":"SC046","pid":"P4","dlogp":-0.22,"p_flip":0.09},{"iid":"SC046","pid":"P5","dlogp":-0.1,"p_flip":0.02},{"iid":"SC046","pid":"P6","dlogp":-0.06,"p_flip":0.01},{"iid":"SC046","pid":"P7","dlogp":-0.3,"p_flip":0.17},
{"iid":"SC047","pid":"P0","dlogp":-0.01,"p_flip":0.0},{"iid":"SC047","pid":"P1","dlogp":-0.01,"p_flip":0.0},{"iid":"SC047","pid":"P2","dlogp":-0.01,"p_flip":0.0},{"iid":"SC047","pid":"P3","dlogp":-0.01,"p_flip":0.0},{"iid":"SC047","pid":"P4","dlogp":-0.01,"p_flip":0.0},{"iid":"SC047","pid":"P5","dlogp":0.0,"p_flip":0.0},{"iid":"SC047","pid":"P6","dlogp":0.0,"p_flip":0.0},{"iid":"SC047","pid":"P7","dlogp":-0.01,"p_flip":0.01},
{"iid":"SC048","pid":"P0","dlogp":0.0,"p_flip":0.0},{"iid":"SC048","pid":"P1","dlogp":0.0,"p_flip":0.0},{"iid":"SC048","pid":"P2","dlogp":0.0,"p_flip":0.0},{"iid":"SC048","pid":"P3","dlogp":-0.5,"p_flip":0.2},{"iid":"SC048","pid":"P4","dlogp":0.0,"p_flip":0.0},{"iid":"SC048","pid":"P5","dlogp":0.0,"p_flip":0.0},{"iid":"SC048","pid":"P6","dlogp":0.0,"p_flip":0.0},{"iid":"SC048","pid":"P7","dlogp":0.0,"p_flip":0.0},
{"iid":"SC049","pid":"P0","dlogp":-0.04,"p_flip":0.01},{"iid":"SC049","pid":"P1","dlogp":-0.05,"p_flip":0.02},{"iid":"SC049","pid":"P2","dlogp":-0.04,"p_flip":0.02},{"iid":"SC049","pid":"P3","dlogp":-0.05,"p_flip":0.02},{"iid":"SC049","pid":"P4","dlogp":-0.07,"p_flip":0.03},{"iid":"SC049","pid":"P5","dlogp":-0.03,"p_flip":0.01},{"iid":"SC049","pid":"P6","dlogp":-0.02,"p_flip":0.0},{"iid":"SC049","pid":"P7","dlogp":-0.08,"p_flip":0.06},
{"iid":"SC050","pid":"P0","dlogp":-0.02,"p_flip":0.01},{"iid":"SC050","pid":"P1","dlogp":-0.02,"p_flip":0.01},{"iid":"SC050","pid":"P2","dlogp":-0.02,"p_flip":0.01},{"iid":"SC050","pid":"P3","dlogp":-0.02,"p_flip":0.01},{"iid":"SC050","pid":"P4","dlogp":-0.03,"p_flip":0.01},{"iid":"SC050","pid":"P5","dlogp":-0.01,"p_flip":0.0},{"iid":"SC050","pid":"P6","dlogp":-0.01,"p_flip":0.0},{"iid":"SC050","pid":"P7","dlogp":-0.04,"p_flip":0.03},
{"iid":"SC051","pid":"P0","dlogp":-0.02,"p_flip":0.01},{"iid":"SC051","pid":"P1","dlogp":-0.03,"p_flip":0.01},{"iid":"SC051","pid":"P2","dlogp":-0.02,"p_flip":0.01},{"iid":"SC051","pid":"P3","dlogp":-0.03,"p_flip":0.01},{"iid":"SC051","pid":"P4","dlogp":-0.04,"p_flip":0.02},{"iid":"SC051","pid":"P5","dlogp":-0.02,"p_flip":0.0},{"iid":"SC051","pid":"P6","dlogp":-0.01,"p_flip":0.0},{"iid":"SC051","pid":"P7","dlogp":-0.05,"p_flip":0.04}
],
"notes":{
"SC036":"Zeroing the largest down_proj weight in a mid layer may remove part of an outlier-dimension write; predicted a small, broad effect.",
"SC037":"Changes only the ' Thursday' unembedding row, so only the P3 logit moves; sign depends on unknown final hidden dim 762.",
"SC038":"+4 on one down_proj entry only matters when neuron 1472 fires; predicted a small effect.",
"SC039":"Zeroing one random entry in an early down_proj is negligible.",
"SC040":"Zeroing one of 1024 input-norm gains in a late linear-attention layer has a small effect.",
"SC041":"Scaling one MLP-input dimension by 10 in layer 18 perturbs that MLP moderately.",
"SC042":"Changes only the ' +' unembedding row, so only P2 is affected; P2 has a strong runner-up '+b', so a flip is possible.",
"SC043":"Scaling one MLP-input dimension by 10 in layer 9 has a moderate effect that spreads to later layers.",
"SC044":"Zeroing one random o_proj entry is negligible.",
"SC045":"Changes only the ' oxygen' unembedding row, so only P5 is affected; the large margin makes a flip unlikely.",
"SC046":"Scaling an MLP-input dimension by 10 in early layer 5 has the most downstream reach of the norm edits.",
"SC047":"Zeroing one random o_proj entry is negligible.",
"SC048":"Changes another dimension of the ' Thursday' unembedding row, so only P3 is affected; sign unknown.",
"SC049":"-4 on one down_proj entry in layer 19 matters only when that neuron is active; predicted a small effect.",
"SC050":"Zeroing one input-norm gain in layer 12 has a small effect.",
"SC051":"Zeroing one MLP-input norm gain in layer 7 has a small effect."
}}
```