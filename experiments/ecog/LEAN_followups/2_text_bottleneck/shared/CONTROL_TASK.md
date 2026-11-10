# Task: build an acoustic feature space (one shot, no feedback)

An encoding-model study predicts human intracranial brain activity (ECoG high-gamma power, 30 Hz) from the sound a patient hears, using a fixed acoustic feature space and a linear readout. Build that feature space: an interpretable acoustic feature space that you expect to predict
high-gamma as well as possible.

## Rules
- **Closed-form signal processing only.** No learning or fitting of any kind (no PCA, no clustering, no
  trained or pretrained models, nothing estimated from data). Fixed parameters only.
- **One shot.** You will not receive any performance feedback, and nothing in this environment contains
  brain data or results. Your file will be scored exactly once, after you finish.
- Work only inside `/work`. Python 3.10 with numpy 2.2, scipy 1.15 and torch 2.10 (CPU here; scoring may
  use a GPU through torch). You may install any package from PyPI with `pip install <package>`: packages go
  into `/work/pkgs`, which is on the Python path here and when your feature space is scored (numpy, scipy
  and torch stay at the versions above). Internet access is for installing packages only.

## How the feature space is used (fixed; cannot be changed)
- Audio: mono, 44,100 Hz, naturalistic recordings from a hospital room (conversation, television, music,
  ambient noise), in segments of about 2 minutes.
- The feature space is called **once per segment** with all of its frames: `windows` is a float32 array of
  shape `(T, win)`. Row `i` is `win` audio samples centered on frame `i`; frames are spaced 1/30 s apart
  (1,470 samples), so neighboring windows overlap when `win > 1470`. `win` is chosen by the feature space
  (at most 4096). Information across frames within the segment may be used.
- It returns a float32 array of shape `(T, H)`, finite values, with `H <= 4000` (a compute limit).
- The harness then, for every segment: appends copies of the features delayed by 1, 2 and 3 frames
  (0–100 ms of history), aligns them to the neural data with a fixed 100 ms lag, and z-scores every column
  within the segment. A separate ridge regression per electrode (regularization chosen by cross-validation
  on the training data) maps features to high-gamma. About 16 hours of training audio.
- Scoring: Pearson correlation between predicted and actual high-gamma on held-out recordings.

## Deliverables
1. `/work/feature_space.py` defining:
   ```python
   class FeatureSpace:
       win = 2048                      # analysis window in samples (<= 4096)
       def __init__(self, device='cpu'): ...
       def __call__(self, windows):    # np.ndarray (T, win) float32 -> np.ndarray (T, H) float32
           ...
   ```
2. `/work/FEATURES.md`: each feature group, its parameters and its number of dimensions.

Check that it runs with `python /work/check.py` (example audio in `/work/examples/`, no brain data): it
verifies shape, dtype, finite values, `H <= 4000`, and speed (must take under 60 s per 2-minute segment
here). When it passes and FEATURES.md is written, you are done.
