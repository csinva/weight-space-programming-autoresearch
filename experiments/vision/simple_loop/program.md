# Interpretable weight-space programming: a visual cortex encoding model

You are building, by hand, an image-to-feature function whose features predict neural responses in visual cortex.
Nothing is trained: you write every filter and computation yourself, so the final model can be read and understood.
This run's dataset: **{dataset}**, region **{region}**, metric **{metric}** ({cohort}).

## How you are scored
`./eval.py --idea "<plain-language label>"` scores the current `model.py`. The evaluator extracts your features for every image, fits a
ridge readout per subject on that subject's own training images (cross-validated penalty), and reports the Pearson correlation between
predicted and measured responses on a fixed **validation** set of images, averaged over voxels or units and then over the subjects.
Higher is better. A separate hidden set of images and subjects is held out and is never shown to you; it will judge the final model, so
prefer ideas that should generalize over tricks tuned to the validation score.

`./eval.py --check` runs `model.py` on 8 synthetic images on the CPU (output shape, finiteness, time) without scoring; use it to catch
bugs before a scoring call. One scoring call takes a few minutes on a shared GPU. Results accumulate in `results/overall_results.csv`
and every scored model is saved as `models_lib/<id>.py`.

## What to build
Start from the **known interpretable features** of early and mid-level vision, written as explicit, readable computations:
- oriented edge filters (Gabor-like simple cells) and phase-invariant energy (complex cells) at several scales and orientations;
- center-surround (difference of Gaussians) and colour-opponent channels (red-green, blue-yellow, luminance);
- divisive normalization and local contrast gain control; pooling over position at several grain sizes (receptive field sizes grow along the stream).

Then go beyond them. Create **new interpretable motifs** and recombine features in interesting ways, in the spirit of the circuits that
were found inside trained vision networks:
- **high-low frequency detectors** (https://distill.pub/2020/circuits/frequency-edges/): units that respond where high spatial frequency
  on one side meets low frequency on the other, i.e. boundaries between textured and smooth regions, oriented like edges;
- **curve detectors** (https://distill.pub/2020/circuits/curve-circuits/): units built by combining oriented edge detectors along an arc,
  tuned to curvature and orientation, with opposite curves inhibiting each other;
- and your own: e.g. corner and junction detectors, symmetry, texture boundaries, closed contours, figure-ground cues, shape
  fragments, second-order (texture-defined) edges, colour-texture conjunctions, spatial layout statistics.
Each feature family should have a clear meaning you can state in a sentence. Combine stages (filter -> rectify -> normalize -> combine
-> pool) explicitly, and say what each new motif detects. Compare against your own strong known-feature baseline: a new motif earns its
place by improving the score beyond it, and an informative failure is worth recording.

## Rules (the evaluator enforces them)
- Edit `model.py` (self-contained). First line `# idea: <label>`; a class `Model` with attributes `name`, `idea`, `description`,
  `insight`, `n_params` and a method `__call__(self, X)`; last line `MODEL = Model()`.
- Input `X`: a float tensor (N, {imsize}*{imsize}*3) of flattened HWC RGB pixels in [0, 1] ({imsize} x {imsize} images). Return finite
  features (N, D) with D < 10,000. Each image's features must depend only on that image.
- No training, no learned or pretrained weights, no batch statistics or dataset adaptation, no file or network access. Weights must be
  analytic or from a fixed random seed, retained on `MODEL`, at most 100,000 stored parameters, declared honestly in `n_params`.
- Use torch; work in chunks if memory is large (budget about 20 GB GPU memory); evaluation times out after 7 minutes.
- The only command you can run is `./eval.py`. Do not try to read neural data or evaluator internals.

## Loop
1. Read `scratch/notes.md` and `results/overall_results.csv` to see where things stand.
2. Make one change with a stated hypothesis, run `./eval.py --check`, then `./eval.py --idea "<label>"`.
3. Interpret the result and update `scratch/notes.md` (a short rolling summary: best id and score, what helped, what did not, next idea).
4. Repeat. Keep at least {target} scored experiments; keep the best model's code readable and its metadata accurate.
