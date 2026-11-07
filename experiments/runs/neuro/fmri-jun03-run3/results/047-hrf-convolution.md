# HRF Convolution vs. FIR Delays

## Hypothesis
Our current SOTA (`0.0988`) uses Finite Impulse Response (FIR) delays. For every representation vector, `ndelays=4` copies are concatenated and shifted by 1, 2, 3, and 4 TRs (2s, 4s, 6s, 8s). This quadruples the feature dimensionality. Given the "Curse of Dimensionality" observed with multi-model ensembles, we hypothesized that replacing FIR delays with a predefined canonical Hemodynamic Response Function (HRF) convolution would reduce the feature space by 4x, allowing the Ridge solver to accommodate more models (or perform better on the dual-ensemble) by reducing colinearity.

## Results (UTS03)
- **FIR Delays (ndelays=4):** `0.0988` (Current Absolute SOTA)
- **Canonical HRF Convolution:** `0.0753`

## Conclusion
Convolving features with a canonical HRF significantly degrades performance compared to FIR delays. 

Why? The canonical HRF assumes that every semantic and syntactic feature evokes the exact same temporal blood oxygenation profile across all brain regions. FIR delays, however, allow the Ridge regression to learn a **feature-specific and voxel-specific HRF**. 

For example, a syntactic feature from Qwen might peak in the language network at 4s, while a high-level semantic feature from Mistral might have a broader integration window peaking at 6s in the default mode network. FIR delays allow the linear solver to assign different weights to different delays for each feature, effectively discovering the optimal temporal dynamics from the data itself.

The dimensionality penalty of FIR delays is vastly outweighed by the benefit of learning flexible, feature-specific hemodynamic responses. FIR delays remain strictly optimal.
