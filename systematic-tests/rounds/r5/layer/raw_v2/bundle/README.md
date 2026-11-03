# Task: predict how weight edits change Qwen3.5-0.8B's next-token probabilities

You are given 12 weight edits (interventions) of one kind ("layer" granularity). Each edit is applied **on its own** to the
unedited model, and then the model reads each of 8 fixed prompts. For every (edit, prompt) pair predict:
- `dlogp`: change in the natural-log probability of the prompt's baseline top-1 token (its "target"), edited minus unedited, in nats.
  0 = no change, negative = target becomes less likely. Range clipped to [-20, +3].
- `p_flip`: your probability that the model's top-1 next token is different after the edit.

**Condition: raw weights.** `weights/` has the raw numbers (4 decimals, 16 per line, index of the first value at line start) of each edited component, plus simple statistics (norms, std, max). Shared raw files: target-token embedding rows and all norm gains. Read the weights and use them.

**Required:** before predicting, Read every `weights/<iid>.txt` file of the edits listed in `test_interventions.md` (the shared_*.txt files are optional). In `notes`, cite at least one specific number from each edit's file. Calls that skip a test edit's file are rejected.

**Calibration guidance.** In this model most single edits change a target's log-probability very little: a typical edit moves it by less than 0.05 nats and many by less than 0.01. Large changes (over 1 nat) need a perturbation that is large relative to the residual stream, one aimed at the target token's (un)embedding, or the removal of a component the prediction depends on. Predict values near 0 unless you have a concrete reason for a larger effect, and make the size of each prediction depend on how large the edit is.

**Procedure (do the arithmetic yourself; you cannot run code).** For each edit, before predicting:
1. *Size.* From the weight file, judge how large the change is: the norm of the removed or scaled weights against the other rows, columns or heads of the same matrix (norm tables are given), or a changed scalar against the matrix std. An edit to a small part of a large layer is small.
2. *Path to the output.* Late-layer edits act mostly through the residual stream straight to the logits; early edits pass through many later layers and are usually damped. Final-norm gains and embedding entries act directly on logits.
3. *Direction.* For direct effects, compare the changed write direction with the target's embedding row in `weights/shared_embeddings_targets.txt` and the final-norm gains: estimate the dot product from the largest entries. Positive alignment raises that target's logit when the component's activation is positive.
4. *Number.* Estimate the logit change of each prompt's target and convert it: a logit change d on a token with probability p changes its log-probability by about d(1 - p) when d is small.
Show the key numbers briefly in your reasoning.

You may only read files (no code execution is available). Work by reasoning about the model and the provided material.
Files in this directory: `README.md` (this file), `test_interventions.md` (what to predict), `weights/` (one file per edit id, plus shared_*.txt).

## Model: Qwen3.5-0.8B (text part), revision 2fc0636471
- 24 decoder layers, hidden size 1024, vocab 248,320, **tied embeddings** (`model.embed_tokens.weight` is also the output unembedding).
- Pre-norm residual: h = h + Mixer(RMSNorm_in(h)); h = h + MLP(RMSNorm_post(h)); logits = E · RMSNorm_final(h).
- RMSNorm(x) = x / rms(x) * (1 + w)  (note the **1 + w**: stored w near 0 means gain near 1).
- Token mixer: layers 3, 7, 11, 15, 19, 23 are full softmax attention (8 query heads of dim 256, 2 KV heads, per-head q/k RMSNorm,
  partial RoPE, and a per-head sigmoid output gate computed from the input; o_proj columns [256h, 256h+256) belong to head h).
  All other layers are Gated DeltaNet linear attention (16 heads, head dim 128, short causal conv, learned decay A_log/dt_bias,
  output = gated RMSNorm(read-out) * silu(z); out_proj columns [128h, 128h+128) belong to head h).
- MLP: down_proj( silu(gate_proj x) * up_proj x ), 3,584 neurons; neuron i writes down_proj[:, i].
- Weight edits are applied, then the 8 prompts are run with no other change. "Skip a sublayer" = zero its output projection
  (o_proj/out_proj or down_proj), which removes its residual contribution exactly.

## Prompts and the unedited model's next-token distribution
- **P0** text 'Germany: Berlin\nItaly: Rome\nFrance:'; input tokens ['Germany', ':', ' Berlin', '\n', 'Italy', ':', ' Rome', '\n', 'France', ':'] (ids [48484, 25, 19241, 198, 68121, 25, 21047, 198, 47358, 25])
  target ' Paris' (id 11751), unedited top-10: ' Paris' 0.916, ' London' 0.016, '\n' 0.006, ' Lyon' 0.004, ' L' 0.004, ' Versailles' 0.003, ' Brussels' 0.003, ' Madrid' 0.002, ' Nice' 0.002, ' La' 0.002
- **P1** text 'one, two, three, four,'; input tokens ['one', ',', ' two', ',', ' three', ',', ' four', ','] (ids [588, 11, 1330, 11, 2250, 11, 2943, 11])
  target ' five' (id 4097), unedited top-10: ' five' 0.774, ' and' 0.089, ' or' 0.055, ' etc' 0.026, ' ...' 0.006, '\n' 0.005, ' six' 0.003, '\n\n' 0.002, ' four' 0.002, ' one' 0.001
- **P2** text 'def add(a, b):\n    return a'; input tokens ['def', ' add', '(a', ',', ' b', '):', '\n', '   ', ' return', ' a'] (ids [727, 884, 2784, 11, 292, 1590, 198, 262, 460, 264])
  target ' +' (id 478), unedited top-10: ' +' 0.815, '+b' 0.173, '\n\n' 0.003, ' -' 0.002, ',' 0.002, '+' 0.001, ' *' 0.001, '*b' 0.000, '\n' 0.000, '-b' 0.000
- **P3** text 'Monday, Tuesday, Wednesday,'; input tokens ['Monday', ',', ' Tuesday', ',', ' Wednesday', ','] (ids [34072, 11, 7348, 11, 7670, 11])
  target ' Thursday' (id 7548), unedited top-10: ' Thursday' 0.706, ' and' 0.155, '\n' 0.019, ' or' 0.019, ' Friday' 0.013, ' April' 0.009, '\n\n' 0.006, ' ' 0.006, ' etc' 0.004, ' ...' 0.004
- **P4** text 'The Eiffel Tower is located in the city of'; input tokens ['The', ' Eiff', 'el', ' Tower', ' is', ' located', ' in', ' the', ' city', ' of'] (ids [760, 242476, 300, 21262, 369, 7172, 303, 279, 3177, 314])
  target ' Paris' (id 11751), unedited top-10: ' Paris' 0.469, '\n' 0.071, '<|im_end|>' 0.033, ' **' 0.032, ' the' 0.029, ' which' 0.025, ' ______' 0.022, ',' 0.014, '...' 0.012, '\n\n' 0.012
- **P5** text 'Water is made of hydrogen and'; input tokens ['Water', ' is', ' made', ' of', ' hydrogen', ' and'] (ids [27336, 369, 1801, 314, 33536, 321])
  target ' oxygen' (id 22817), unedited top-10: ' oxygen' 0.927, ' chlorine' 0.008, ' carbon' 0.006, ' helium' 0.005, ' nitrogen' 0.005, ' Oxygen' 0.004, ' ______' 0.003, ' what' 0.003, '\n' 0.002, '/or' 0.002
- **P6** text 'Romeo and Juliet was written by William'; input tokens ['R', 'ome', 'o', ' and', ' Juliet', ' was', ' written', ' by', ' William'] (ids [49, 617, 78, 321, 68613, 557, 5158, 539, 12021])
  target ' Shakespeare' (id 40014), unedited top-10: ' Shakespeare' 0.994, ' Sh' 0.001, ' Words' 0.001, ' Blake' 0.001, ' Lang' 0.000, ' Cong' 0.000, '莎士比亚' 0.000, ' Haz' 0.000, ' Shakespe' 0.000, ' Henry' 0.000
- **P7** text 'The keys to the cabinet'; input tokens ['The', ' keys', ' to', ' the', ' cabinet'] (ids [760, 6679, 310, 279, 21245])
  target ' are' (id 513), unedited top-10: ' are' 0.272, '\n' 0.116, ' were' 0.061, ',' 0.051, ' of' 0.045, '.' 0.045, '\n\n' 0.041, ' door' 0.035, ':' 0.028, ' in' 0.017

## Output format (your final message)
A single fenced ```json block, nothing else after it:
{"predictions": [{"iid": "XX00", "pid": "P0", "dlogp": -0.3, "p_flip": 0.05}, ...  one entry for each of the 96 (edit, prompt) pairs],
 "notes": {"XX00": "one short sentence on the evidence you used", ...}}
Scoring: Spearman correlation and mean absolute error of dlogp against the measured values, and Brier score of p_flip.
