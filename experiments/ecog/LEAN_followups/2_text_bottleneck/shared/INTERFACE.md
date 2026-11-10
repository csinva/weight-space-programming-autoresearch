# How the feature space is used

An encoding-model study predicts human intracranial brain activity (ECoG high-gamma power, 30 Hz) from the sound a patient hears, using a fixed acoustic feature space and a linear readout.

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
