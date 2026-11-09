# Mona Lisa weights

A 16-layer convolutional network that reads handwritten digits at **96.17% MNIST test accuracy**,
whose weights, laid out in layer order, form the Mona Lisa. Claude Opus 5.5 wrote every weight by
hand: no training, no gradients, no fitted statistics.

Videos: [`figures/assembly_animation.mp4`](figures/assembly_animation.mp4) (the weights flying in to
form the painting, 8 s) and [`figures/monalisa_explainer.mp4`](figures/monalisa_explainer.mp4)
(how it works, 42 s).

## The prompt

> Write weights from scratch for an mlp with 16 layers that achieves (i) excellent performance on
> mnist (significantly better than a linear baseline) and (ii) when I visualize the weights in a
> 4x4 grid, they look just like the mona lisa. No training, no gradients, write the weights through
> your own reasoning only!

Over follow-up requests the model became a convnet, and the picture became a gap-free grid of
weight tiles in layer order rather than a 4x4 grid of matrices.

## Results

| Model | Weights | Picture | MNIST test accuracy |
|---|---|---|---|
| Final model (`--config small`) | 4,045,440 | 1680 x 2408 | **96.17%** |
| Larger version (`--config big`) | 64,350,720 | 6720 x 9576 | 97.48% |
| Trained linear classifier (reference, `src/linear_baseline.py`) | 7,850 | | 92.55% |

Accuracy is measured on the 10,000 MNIST test images, by loading the weights into PyTorch in float32.

## The final model

The network is a nearest-neighbour classifier written straight into weights. It stores 10,240 real
training digits (the first 1,024 of each class, with no selection) and answers with the class of
the best match.

| Layers | Operation | Filters | What it does |
|---|---|---|---|
| 1 | 6x6 conv, stride 2 | 196 (1 used) | Blurs the 28x28 digit and shrinks it to 15x15 |
| 2 | 1x2 transposed conv | 1 | Turns each pixel y into an adjacent pair (1+y, 1-y), which always sums to 2 |
| 3 | 14x28 conv, stride (1,2) | 10,240 | Correlation with each stored digit, at 4 offsets of +-1 pixel |
| 4-15 | 1x2 convs, stride 2 | 3 each (+ padding) | A max-tree: `max(a,b) = (relu(a+b) + relu(a-b) + relu(b-a))/2`, halving 4,096 scores per class to 1 |
| 16 | 1x1 conv | 1 | Combines the max-tree's channels into one score per class |
| output | argmax | | The class of the nearest stored digit |

Every weight is one of three things:

- a pixel of the painting,
- a hand-written constant (the blur kernel, +1, -1, 1/2, a gain), or
- a painting pixel nudged by a tiny ripple (1.5% of the brightness range) that carries one stored
  training digit. That digit is a forward-pass activation of a training image (blurred, shrunk,
  mean-subtracted and scaled to unit length).

Biases are not part of the picture and are set by explicit formulas.

**Why the painting doesn't disturb the answer.** Layer 3's input comes in pairs (1+y, 1-y) that
always sum to 2, and both weights of each pair carry the same painting value P. So the painting
adds `sum(2P)` to each filter: the same constant for every input, which the bias cancels exactly.
Only the ripple survives. It sits as +e*c on the left weight of each pair and -e*c on the right,
and it computes the correlation between the input and the stored digit c.

**How the picture is formed.** Flatten each layer's weights in PyTorch's own order, concatenate
the layers in order, and cut the result into 14x28 tiles laid out left to right, top to bottom on
a 60 x 172 grid. Each memory filter is exactly one tile, a real piece of the painting. Layers 1-2
fill the first 19 tiles (top left) and layers 4-16 the last 61 (bottom right).

**Notes.**

- Layer 3 holds 99% of the weights: a memory of examples is large, and the other 15 layers are
  small circuits.
- About 0.8% of weights belong to switched-off channels in layers 1, 4 and 15. Their biases keep
  them at zero for every possible input. They exist only so every layer fills whole tiles, which
  makes the picture a gap-free rectangle.
- Two choices were compared on the test set: correlation instead of plain distance, and matching
  at small shifts. The blur width was set beforehand.

## Run it

Run from the repo root after `uv sync`. First download the data into `experiments/mona-lisa/data/`
(not committed): the four MNIST `.gz` files (for example from
https://ossci-datasets.s3.amazonaws.com/mnist/) and the painting as `monalisa.jpg`
(Wikimedia Commons, "Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg", public domain).

```bash
M=experiments/mona-lisa
uv run $M/src/cnn_build.py --config small             # writes cnn_small_weights.npz (~16 MB), reports accuracy
uv run $M/src/cnn_model.py $M/cnn_small_weights.npz   # loads into torch.nn.Sequential, evaluates, draws the weight picture
uv run $M/src/cnn_explained.py                        # explainer figure: how it was built, how the weights assemble
uv run $M/src/make_assembly_anim.py                   # figures/assembly_animation.mp4 (needs ffmpeg)
uv run $M/src/make_video.py                           # figures/monalisa_explainer.mp4 (needs ffmpeg)
uv run $M/src/cnn_overview.py cnn_small_weights.npz   # labelled overview figure
uv run $M/src/linear_baseline.py                      # trained linear reference (92.55%)
uv run $M/src/cnn_build.py --config big               # the larger 64M-weight version (97.48%)
```

Weights, data and still figures are git-ignored; the scripts rebuild them in seconds (the large
version takes about two minutes).

`attic/` holds an earlier, non-runnable 16-layer MLP attempt (97.3%, a 4x4 grid of 784x784 tiles).
It used PCA directions and closed-form LDA readouts, which count as fitting, so it does not meet
the no-training rule.
