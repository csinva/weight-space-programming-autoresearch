I made all 128 predictions, but they are educated guesses from weight-norm and layer-depth patterns, not computed effects. Without code execution I couldn't calculate neuron activations, so I can't tell whether each neuron actually fires on these 8 prompts, or which way it pushes the target token.

**What I read:** all 16 required weight files. NE116–NE119 I read in full. For NE120–NE131 I read only the header, which has the down_proj norm against the layer median and max and the gate_proj statistics, plus the down_proj summary line. I didn't read the shared embedding or norm files.

**How I ranked the edits:**
- **Zeroing a neuron (NE116, NE118, NE119, NE121, NE122, NE126):** a single neuron removed out of 3,584, so I expect almost no change: |dlogp| of about 0.002–0.03.
- **Multiplying by 5:** larger expected effects. I ranked them higher when:
  - the layer is late (NE129 in layer 23 is the largest);
  - the down_proj column is large (NE124, NE125, NE128 are well above their layer medians);
  - the gate weights have unusually large extreme values (NE131, NE120, NE130). I took that to mean a neuron that fires strongly on specific inputs.

**How I spread effects across prompts:** I scaled each edit's effect by how confident the unedited model is on each prompt.
- P7 ("are", 0.27) and P4 ("Paris", 0.47) get the biggest changes and highest flip probabilities.
- P6 ("Shakespeare", 0.994) gets the smallest.

Every predicted dlogp is negative. In reality some edits will raise the target's probability, and the weights I had don't show which ones.

```json
{"predictions": [
{"iid":"NE116","pid":"P0","dlogp":-0.012,"p_flip":0.002},{"iid":"NE116","pid":"P1","dlogp":-0.02,"p_flip":0.002},{"iid":"NE116","pid":"P2","dlogp":-0.016,"p_flip":0.003},{"iid":"NE116","pid":"P3","dlogp":-0.024,"p_flip":0.004},{"iid":"NE116","pid":"P4","dlogp":-0.03,"p_flip":0.01},{"iid":"NE116","pid":"P5","dlogp":-0.012,"p_flip":0.002},{"iid":"NE116","pid":"P6","dlogp":-0.004,"p_flip":0.002},{"iid":"NE116","pid":"P7","dlogp":-0.026,"p_flip":0.02},
{"iid":"NE117","pid":"P0","dlogp":-0.06,"p_flip":0.005},{"iid":"NE117","pid":"P1","dlogp":-0.1,"p_flip":0.01},{"iid":"NE117","pid":"P2","dlogp":-0.08,"p_flip":0.015},{"iid":"NE117","pid":"P3","dlogp":-0.12,"p_flip":0.02},{"iid":"NE117","pid":"P4","dlogp":-0.15,"p_flip":0.05},{"iid":"NE117","pid":"P5","dlogp":-0.06,"p_flip":0.004},{"iid":"NE117","pid":"P6","dlogp":-0.02,"p_flip":0.002},{"iid":"NE117","pid":"P7","dlogp":-0.13,"p_flip":0.1},
{"iid":"NE118","pid":"P0","dlogp":-0.006,"p_flip":0.002},{"iid":"NE118","pid":"P1","dlogp":-0.01,"p_flip":0.002},{"iid":"NE118","pid":"P2","dlogp":-0.008,"p_flip":0.002},{"iid":"NE118","pid":"P3","dlogp":-0.012,"p_flip":0.002},{"iid":"NE118","pid":"P4","dlogp":-0.015,"p_flip":0.005},{"iid":"NE118","pid":"P5","dlogp":-0.006,"p_flip":0.002},{"iid":"NE118","pid":"P6","dlogp":-0.002,"p_flip":0.002},{"iid":"NE118","pid":"P7","dlogp":-0.013,"p_flip":0.01},
{"iid":"NE119","pid":"P0","dlogp":-0.012,"p_flip":0.002},{"iid":"NE119","pid":"P1","dlogp":-0.02,"p_flip":0.002},{"iid":"NE119","pid":"P2","dlogp":-0.016,"p_flip":0.003},{"iid":"NE119","pid":"P3","dlogp":-0.024,"p_flip":0.004},{"iid":"NE119","pid":"P4","dlogp":-0.03,"p_flip":0.01},{"iid":"NE119","pid":"P5","dlogp":-0.012,"p_flip":0.002},{"iid":"NE119","pid":"P6","dlogp":-0.004,"p_flip":0.002},{"iid":"NE119","pid":"P7","dlogp":-0.026,"p_flip":0.02},
{"iid":"NE120","pid":"P0","dlogp":-0.072,"p_flip":0.006},{"iid":"NE120","pid":"P1","dlogp":-0.12,"p_flip":0.012},{"iid":"NE120","pid":"P2","dlogp":-0.096,"p_flip":0.018},{"iid":"NE120","pid":"P3","dlogp":-0.144,"p_flip":0.024},{"iid":"NE120","pid":"P4","dlogp":-0.18,"p_flip":0.06},{"iid":"NE120","pid":"P5","dlogp":-0.072,"p_flip":0.005},{"iid":"NE120","pid":"P6","dlogp":-0.024,"p_flip":0.002},{"iid":"NE120","pid":"P7","dlogp":-0.156,"p_flip":0.12},
{"iid":"NE121","pid":"P0","dlogp":-0.012,"p_flip":0.002},{"iid":"NE121","pid":"P1","dlogp":-0.02,"p_flip":0.002},{"iid":"NE121","pid":"P2","dlogp":-0.016,"p_flip":0.003},{"iid":"NE121","pid":"P3","dlogp":-0.024,"p_flip":0.004},{"iid":"NE121","pid":"P4","dlogp":-0.03,"p_flip":0.01},{"iid":"NE121","pid":"P5","dlogp":-0.012,"p_flip":0.002},{"iid":"NE121","pid":"P6","dlogp":-0.004,"p_flip":0.002},{"iid":"NE121","pid":"P7","dlogp":-0.026,"p_flip":0.02},
{"iid":"NE122","pid":"P0","dlogp":-0.012,"p_flip":0.002},{"iid":"NE122","pid":"P1","dlogp":-0.02,"p_flip":0.002},{"iid":"NE122","pid":"P2","dlogp":-0.016,"p_flip":0.003},{"iid":"NE122","pid":"P3","dlogp":-0.024,"p_flip":0.004},{"iid":"NE122","pid":"P4","dlogp":-0.03,"p_flip":0.01},{"iid":"NE122","pid":"P5","dlogp":-0.012,"p_flip":0.002},{"iid":"NE122","pid":"P6","dlogp":-0.004,"p_flip":0.002},{"iid":"NE122","pid":"P7","dlogp":-0.026,"p_flip":0.02},
{"iid":"NE123","pid":"P0","dlogp":-0.048,"p_flip":0.004},{"iid":"NE123","pid":"P1","dlogp":-0.08,"p_flip":0.008},{"iid":"NE123","pid":"P2","dlogp":-0.064,"p_flip":0.012},{"iid":"NE123","pid":"P3","dlogp":-0.096,"p_flip":0.016},{"iid":"NE123","pid":"P4","dlogp":-0.12,"p_flip":0.04},{"iid":"NE123","pid":"P5","dlogp":-0.048,"p_flip":0.003},{"iid":"NE123","pid":"P6","dlogp":-0.016,"p_flip":0.002},{"iid":"NE123","pid":"P7","dlogp":-0.104,"p_flip":0.08},
{"iid":"NE124","pid":"P0","dlogp":-0.048,"p_flip":0.004},{"iid":"NE124","pid":"P1","dlogp":-0.08,"p_flip":0.008},{"iid":"NE124","pid":"P2","dlogp":-0.064,"p_flip":0.012},{"iid":"NE124","pid":"P3","dlogp":-0.096,"p_flip":0.016},{"iid":"NE124","pid":"P4","dlogp":-0.12,"p_flip":0.04},{"iid":"NE124","pid":"P5","dlogp":-0.048,"p_flip":0.003},{"iid":"NE124","pid":"P6","dlogp":-0.016,"p_flip":0.002},{"iid":"NE124","pid":"P7","dlogp":-0.104,"p_flip":0.08},
{"iid":"NE125","pid":"P0","dlogp":-0.06,"p_flip":0.005},{"iid":"NE125","pid":"P1","dlogp":-0.1,"p_flip":0.01},{"iid":"NE125","pid":"P2","dlogp":-0.08,"p_flip":0.015},{"iid":"NE125","pid":"P3","dlogp":-0.12,"p_flip":0.02},{"iid":"NE125","pid":"P4","dlogp":-0.15,"p_flip":0.05},{"iid":"NE125","pid":"P5","dlogp":-0.06,"p_flip":0.004},{"iid":"NE125","pid":"P6","dlogp":-0.02,"p_flip":0.002},{"iid":"NE125","pid":"P7","dlogp":-0.13,"p_flip":0.1},
{"iid":"NE126","pid":"P0","dlogp":-0.012,"p_flip":0.002},{"iid":"NE126","pid":"P1","dlogp":-0.02,"p_flip":0.002},{"iid":"NE126","pid":"P2","dlogp":-0.016,"p_flip":0.003},{"iid":"NE126","pid":"P3","dlogp":-0.024,"p_flip":0.004},{"iid":"NE126","pid":"P4","dlogp":-0.03,"p_flip":0.01},{"iid":"NE126","pid":"P5","dlogp":-0.012,"p_flip":0.002},{"iid":"NE126","pid":"P6","dlogp":-0.004,"p_flip":0.002},{"iid":"NE126","pid":"P7","dlogp":-0.026,"p_flip":0.02},
{"iid":"NE127","pid":"P0","dlogp":-0.036,"p_flip":0.003},{"iid":"NE127","pid":"P1","dlogp":-0.06,"p_flip":0.006},{"iid":"NE127","pid":"P2","dlogp":-0.048,"p_flip":0.009},{"iid":"NE127","pid":"P3","dlogp":-0.072,"p_flip":0.012},{"iid":"NE127","pid":"P4","dlogp":-0.09,"p_flip":0.03},{"iid":"NE127","pid":"P5","dlogp":-0.036,"p_flip":0.002},{"iid":"NE127","pid":"P6","dlogp":-0.012,"p_flip":0.002},{"iid":"NE127","pid":"P7","dlogp":-0.078,"p_flip":0.06},
{"iid":"NE128","pid":"P0","dlogp":-0.054,"p_flip":0.0045},{"iid":"NE128","pid":"P1","dlogp":-0.09,"p_flip":0.009},{"iid":"NE128","pid":"P2","dlogp":-0.072,"p_flip":0.0135},{"iid":"NE128","pid":"P3","dlogp":-0.108,"p_flip":0.018},{"iid":"NE128","pid":"P4","dlogp":-0.135,"p_flip":0.045},{"iid":"NE128","pid":"P5","dlogp":-0.054,"p_flip":0.0036},{"iid":"NE128","pid":"P6","dlogp":-0.018,"p_flip":0.002},{"iid":"NE128","pid":"P7","dlogp":-0.117,"p_flip":0.09},
{"iid":"NE129","pid":"P0","dlogp":-0.12,"p_flip":0.01},{"iid":"NE129","pid":"P1","dlogp":-0.2,"p_flip":0.02},{"iid":"NE129","pid":"P2","dlogp":-0.16,"p_flip":0.03},{"iid":"NE129","pid":"P3","dlogp":-0.24,"p_flip":0.04},{"iid":"NE129","pid":"P4","dlogp":-0.3,"p_flip":0.1},{"iid":"NE129","pid":"P5","dlogp":-0.12,"p_flip":0.008},{"iid":"NE129","pid":"P6","dlogp":-0.04,"p_flip":0.004},{"iid":"NE129","pid":"P7","dlogp":-0.26,"p_flip":0.2},
{"iid":"NE130","pid":"P0","dlogp":-0.072,"p_flip":0.006},{"iid":"NE130","pid":"P1","dlogp":-0.12,"p_flip":0.012},{"iid":"NE130","pid":"P2","dlogp":-0.096,"p_flip":0.018},{"iid":"NE130","pid":"P3","dlogp":-0.144,"p_flip":0.024},{"iid":"NE130","pid":"P4","dlogp":-0.18,"p_flip":0.06},{"iid":"NE130","pid":"P5","dlogp":-0.072,"p_flip":0.005},{"iid":"NE130","pid":"P6","dlogp":-0.024,"p_flip":0.002},{"iid":"NE130","pid":"P7","dlogp":-0.156,"p_flip":0.12},
{"iid":"NE131","pid":"P0","dlogp":-0.09,"p_flip":0.0075},{"iid":"NE131","pid":"P1","dlogp":-0.15,"p_flip":0.015},{"iid":"NE131","pid":"P2","dlogp":-0.12,"p_flip":0.0225},{"iid":"NE131","pid":"P3","dlogp":-0.18,"p_flip":0.03},{"iid":"NE131","pid":"P4","dlogp":-0.225,"p_flip":0.075},{"iid":"NE131","pid":"P5","dlogp":-0.09,"p_flip":0.006},{"iid":"NE131","pid":"P6","dlogp":-0.03,"p_flip":0.003},{"iid":"NE131","pid":"P7","dlogp":-0.195,"p_flip":0.15}
],
"notes": {
"NE116": "Single-neuron ablation at layer 8; down column L2=0.3527 is near the layer max 0.3624, but one neuron out of 3584 gives a tiny effect.",
"NE117": "x5 at layer 20 with an ordinary down norm (L2=0.3010 vs median 0.2969); moderate late-layer effect assumed.",
"NE118": "Layer-10 ablation with a weak gate (std 0.00764) and below-median down L2=0.2488; near-zero effect.",
"NE119": "Layer-20 ablation, down L2=0.3124 (median 0.2969); small effect.",
"NE120": "x5 at layer 16; gate row has large extremes (min -0.1069, max +0.0957), suggesting a sharply selective neuron; moderate-to-large effect.",
"NE121": "Layer-1 ablation, down L2=0.2840; early single neuron, tiny effect.",
"NE122": "Layer-2 ablation; gate std 0.02217 is large but the edit only removes one neuron; tiny effect.",
"NE123": "x5 at layer 18, down L2=0.2807 (about median); moderate effect.",
"NE124": "x5 at layer 12; down L2=0.3674 is near the layer max 0.3873; moderate effect.",
"NE125": "x5 at layer 13; down L2=0.3991 (median 0.2562); moderate effect.",
"NE126": "Layer-22 ablation, down L2=0.3259; small effect.",
"NE127": "x5 at layer 3, down L2=0.2657; early-layer scaling is largely washed out downstream; small-to-moderate effect.",
"NE128": "x5 at layer 13; down L2=0.3869 with gate max +0.0693; moderate effect.",
"NE129": "x5 at layer 23 (last layer), down L2=0.3530 vs median 0.2700; acts directly on the logits, so the largest expected effect.",
"NE130": "x5 at layer 19; gate max +0.0869 and std 0.02049 suggest a selective neuron; moderate-to-large effect.",
"NE131": "x5 at layer 17; gate extremes -0.1211/+0.1006 are the largest seen; fairly large effect, though down L2=0.2502 is below median."
}}
```