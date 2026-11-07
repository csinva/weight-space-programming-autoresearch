I had no weights and couldn't run code, so these predictions are educated guesses, not inferences from the actual edits. Every edit gets the same answer because nothing I had separates one dimension from another.

**What each edit does:** Multiplying the final-norm gain on dimension d by 30 adds about 29 · x̂_d · E[:, d] to the logits. Here x̂_d is the normalised hidden value in dimension d, and E[:, d] is that dimension's column of the shared embedding/unembedding matrix. Which tokens go up depends on the sign of x̂_d on each prompt and on the values in E[:, d]. I didn't have either, so I can't name the actual winning tokens for any edit.

**`up` tokens:** I listed generic tokens that already have some probability across several prompts: line breaks, punctuation, `' and'`, `' the'`, `' ______'`, `'<|im_end|>'`, `' etc'`. These easily pass the 0.001 threshold and tend to rise when an edit flattens the distribution. The real top-10 lists are probably full of tokens tied to each dimension, such as rare or non-English tokens, and I can't predict those. Expect low scores on this part.

**`dlogp`:** I predicted small drops in the target token's log-probability:
- smallest for P6 (−0.1), since `' Shakespeare'` is at 0.994;
- larger for the less confident prompts, P4 (−0.6) and P7 (−0.4);
- −0.3 to −0.5 for the rest.

These fit the README's stated KL range of 0.1 to 2 nats. Dimensions that carry very large activations, which could include dimension 0, might cause much bigger drops, but I had no evidence for that either.

```json
{"edits": [
{"iid": "XGA00", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA01", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA02", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA03", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA04", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA05", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA06", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA07", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA08", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA09", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA10", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA11", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA12", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA13", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA14", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}},
{"iid": "XGA15", "up": [{"tok": "\n", "id": 198}, {"tok": "\n\n", "id": null}, {"tok": ",", "id": 11}, {"tok": " and", "id": 321}, {"tok": " the", "id": 279}, {"tok": "...", "id": null}, {"tok": " ______", "id": null}, {"tok": " **", "id": null}, {"tok": "<|im_end|>", "id": null}, {"tok": " etc", "id": null}], "dlogp": {"P0": -0.4, "P1": -0.5, "P2": -0.4, "P3": -0.5, "P4": -0.6, "P5": -0.3, "P6": -0.1, "P7": -0.4}}
],
"notes": {
"XGA00": "No weights; generic prior: tokens that already have some probability on several prompts, and small drops in the target that scale with how unconfident the unedited model is.",
"XGA01": "Same generic prior; the effect depends on the unknown embedding column 370.",
"XGA02": "Same generic prior; the effect depends on the unknown embedding column 565.",
"XGA03": "Same generic prior; the effect depends on the unknown embedding column 365.",
"XGA04": "Same generic prior; the effect depends on the unknown embedding column 77.",
"XGA05": "Same generic prior; the effect depends on the unknown embedding column 625.",
"XGA06": "Same generic prior; the effect depends on the unknown embedding column 736.",
"XGA07": "Same generic prior; dim 0 might carry very large activations, but there was no evidence for that.",
"XGA08": "Same generic prior; the effect depends on the unknown embedding column 75.",
"XGA09": "Same generic prior; the effect depends on the unknown embedding column 311.",
"XGA10": "Same generic prior; the effect depends on the unknown embedding column 348.",
"XGA11": "Same generic prior; the effect depends on the unknown embedding column 128.",
"XGA12": "Same generic prior; the effect depends on the unknown embedding column 530.",
"XGA13": "Same generic prior; the effect depends on the unknown embedding column 71.",
"XGA14": "Same generic prior; the effect depends on the unknown embedding column 624.",
"XGA15": "Same generic prior; the effect depends on the unknown embedding column 827."
}}
```