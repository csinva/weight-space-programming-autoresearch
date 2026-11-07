I couldn't make evidence-based predictions here. The edits are listed only by layer and neuron index, and without weights or code execution nothing tells these 16 neurons apart. So every edit gets the same fallback answer.

- **`up` tokens:** the same list for all 16 edits. It is made of common function-word and punctuation tokens, which tend to show up when a direction in the residual stream is pushed hard. I only gave token ids that appear in the prompt inputs; the rest are `null`.
- **`dlogp`:**
  - The size of the drop is set by how confident the unedited model was on each prompt. P6 (0.994 on ' Shakespeare') gets the smallest drop, and P4 (0.469) and P7 (0.272) get the largest.
  - The drop is also scaled up for neurons in later layers (20–23), which sit closer to the output, and down for those in layers 12–14.

Expect accuracy close to chance on `up`, and only rough accuracy on `dlogp`.

```json
{"edits": [
{"iid": "XNE00", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.5, "P1": -0.6, "P2": -0.4, "P3": -0.6, "P4": -0.8, "P5": -0.4, "P6": -0.15, "P7": -0.5}},
{"iid": "XNE01", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.3, "P1": -0.4, "P2": -0.3, "P3": -0.4, "P4": -0.5, "P5": -0.25, "P6": -0.1, "P7": -0.35}},
{"iid": "XNE02", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.6, "P1": -0.7, "P2": -0.5, "P3": -0.7, "P4": -0.9, "P5": -0.5, "P6": -0.2, "P7": -0.6}},
{"iid": "XNE03", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.3, "P1": -0.4, "P2": -0.3, "P3": -0.4, "P4": -0.5, "P5": -0.25, "P6": -0.1, "P7": -0.35}},
{"iid": "XNE04", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.6, "P1": -0.7, "P2": -0.5, "P3": -0.7, "P4": -0.9, "P5": -0.5, "P6": -0.2, "P7": -0.6}},
{"iid": "XNE05", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.5, "P1": -0.6, "P2": -0.4, "P3": -0.6, "P4": -0.8, "P5": -0.4, "P6": -0.15, "P7": -0.5}},
{"iid": "XNE06", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.35, "P3": -0.5, "P4": -0.7, "P5": -0.35, "P6": -0.12, "P7": -0.45}},
{"iid": "XNE07", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.8, "P1": -0.9, "P2": -0.7, "P3": -0.9, "P4": -1.1, "P5": -0.7, "P6": -0.3, "P7": -0.7}},
{"iid": "XNE08", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.35, "P3": -0.5, "P4": -0.7, "P5": -0.35, "P6": -0.12, "P7": -0.45}},
{"iid": "XNE09", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.5, "P1": -0.6, "P2": -0.4, "P3": -0.6, "P4": -0.8, "P5": -0.4, "P6": -0.15, "P7": -0.5}},
{"iid": "XNE10", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.6, "P1": -0.7, "P2": -0.5, "P3": -0.7, "P4": -0.9, "P5": -0.5, "P6": -0.2, "P7": -0.6}},
{"iid": "XNE11", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.3, "P1": -0.4, "P2": -0.3, "P3": -0.4, "P4": -0.5, "P5": -0.25, "P6": -0.1, "P7": -0.35}},
{"iid": "XNE12", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.8, "P1": -0.9, "P2": -0.7, "P3": -0.9, "P4": -1.1, "P5": -0.7, "P6": -0.3, "P7": -0.7}},
{"iid": "XNE13", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.35, "P3": -0.5, "P4": -0.7, "P5": -0.35, "P6": -0.12, "P7": -0.45}},
{"iid": "XNE14", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.5, "P1": -0.6, "P2": -0.4, "P3": -0.6, "P4": -0.8, "P5": -0.4, "P6": -0.15, "P7": -0.5}},
{"iid": "XNE15", "up": [{"tok": "\n", "id": 198}, {"tok": ",", "id": 11}, {"tok": " the", "id": 279}, {"tok": " and", "id": 321}, {"tok": " of", "id": 314}, {"tok": " a", "id": 264}, {"tok": " in", "id": 303}, {"tok": ":", "id": 25}, {"tok": " is", "id": 369}, {"tok": " to", "id": 310}], "dlogp": {"P0": -0.3, "P1": -0.4, "P2": -0.3, "P3": -0.4, "P4": -0.5, "P5": -0.25, "P6": -0.1, "P7": -0.35}}
],
 "notes": {
"XNE00": "No neuron-specific info; generic frequent-token guess, drop sized by layer 20 and baseline confidence.",
"XNE01": "No neuron-specific info; earlier layer 13 assumed weaker effect.",
"XNE02": "No neuron-specific info; late layer 21 assumed stronger effect.",
"XNE03": "No neuron-specific info; earlier layer 13 assumed weaker effect.",
"XNE04": "No neuron-specific info; late layer 21 assumed stronger effect.",
"XNE05": "No neuron-specific info; layer 20 moderate-strong effect assumed.",
"XNE06": "No neuron-specific info; layer 18 moderate effect assumed.",
"XNE07": "No neuron-specific info; final layer 23 writes most directly to logits, largest drop assumed.",
"XNE08": "No neuron-specific info; layer 18 moderate effect assumed.",
"XNE09": "No neuron-specific info; layer 20 moderate-strong effect assumed.",
"XNE10": "No neuron-specific info; late layer 21 assumed stronger effect.",
"XNE11": "No neuron-specific info; earlier layer 12 assumed weaker effect.",
"XNE12": "No neuron-specific info; final layer 23 writes most directly to logits, largest drop assumed.",
"XNE13": "No neuron-specific info; layer 17 moderate effect assumed.",
"XNE14": "No neuron-specific info; layer 20 moderate-strong effect assumed.",
"XNE15": "No neuron-specific info; earlier layer 14 assumed weaker effect."
}}
```