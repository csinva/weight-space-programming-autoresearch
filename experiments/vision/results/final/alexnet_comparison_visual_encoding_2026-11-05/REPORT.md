# NSD per-ROI encoding: search champion erjpu vs ImageNet AlexNet

- Champion: erjpu (search run laion__visual__encoding__2026-11-05), frozen sha256 cef6828c..., LAION search score 0.4875; official NSD pooled 0.3395.
- AlexNet: ImageNet IMAGENET1K_V1, 224px input, each layer adaptively pooled to <10k features (search budget). Layer chosen per ROI by highest LAION score under the same pooled early+ventral encoding protocol; NSD then scored with that layer only.
- NSD: final protocol (8 subjects, ncsnr>0.2, native split, 5-fold ridge, per-voxel Pearson r). Areas = named-area voxels intersected with pooled early+ventral units; score = mean voxel r. Pooled/stream values reproduce the official final results exactly.
- Final-only diagnostic computed after selection; not image-held-out from LAION (NSD-source images are in LAION search). AlexNet gets per-ROI layer choice; the champion is one fixed model.

| ROI | AlexNet layer | Champion | AlexNet | Champion - AlexNet (sd over subjects) | Champion wins | Voxels/subject |
|---|---|---|---|---|---|---|
| V1 | conv3 | 0.480 | 0.470 | +0.009 (0.014) | 5/8 | 712-1291 |
| V2 | conv3 | 0.433 | 0.429 | +0.004 (0.010) | 5/8 | 712-1269 |
| V3 | conv3 | 0.380 | 0.389 | -0.008 (0.008) | 1/8 | 379-976 |
| hV4 | conv3 | 0.371 | 0.393 | -0.022 (0.006) | 0/8 | 4-183 |
| FFA | fc8 | 0.283 | 0.370 | -0.086 (0.016) | 0/8 | 291-761 |
| PPA | fc7 | 0.365 | 0.426 | -0.061 (0.008) | 0/8 | 527-893 |
| early visual stream | conv3 | 0.402 | 0.404 | -0.002 (0.011) | 4/8 | 2415-4834 |
| ventral visual stream | fc6 | 0.280 | 0.349 | -0.069 (0.009) | 0/8 | 2554-4437 |
| early and ventral visual streams | conv4 | 0.339 | 0.366 | -0.026 (0.008) | 0/8 | 5069-9271 |

## Per-subject

| ROI | model | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | champion | 0.546 | 0.570 | 0.486 | 0.457 | 0.513 | 0.426 | 0.437 | 0.403 |
| V1 | AlexNet conv3 | 0.514 | 0.546 | 0.473 | 0.460 | 0.520 | 0.427 | 0.430 | 0.393 |
| V2 | champion | 0.494 | 0.485 | 0.424 | 0.416 | 0.453 | 0.395 | 0.423 | 0.372 |
| V2 | AlexNet conv3 | 0.476 | 0.470 | 0.417 | 0.421 | 0.465 | 0.396 | 0.420 | 0.364 |
| V3 | champion | 0.469 | 0.484 | 0.346 | 0.332 | 0.394 | 0.318 | 0.373 | 0.326 |
| V3 | AlexNet conv3 | 0.474 | 0.487 | 0.349 | 0.345 | 0.419 | 0.328 | 0.380 | 0.325 |
| hV4 | champion | 0.419 | 0.476 | 0.354 | 0.256 | 0.386 | 0.286 | 0.415 | 0.376 |
| hV4 | AlexNet conv3 | 0.443 | 0.498 | 0.375 | 0.282 | 0.414 | 0.300 | 0.441 | 0.389 |
| FFA | champion | 0.325 | 0.353 | 0.271 | 0.250 | 0.303 | 0.243 | 0.291 | 0.232 |
| FFA | AlexNet fc8 | 0.402 | 0.423 | 0.365 | 0.332 | 0.418 | 0.344 | 0.377 | 0.299 |
| PPA | champion | 0.373 | 0.440 | 0.349 | 0.337 | 0.452 | 0.321 | 0.349 | 0.300 |
| PPA | AlexNet fc7 | 0.441 | 0.498 | 0.407 | 0.398 | 0.529 | 0.374 | 0.411 | 0.350 |
| early visual stream | champion | 0.488 | 0.485 | 0.397 | 0.364 | 0.420 | 0.332 | 0.379 | 0.350 |
| early visual stream | AlexNet conv3 | 0.476 | 0.478 | 0.394 | 0.374 | 0.439 | 0.344 | 0.382 | 0.346 |
| ventral visual stream | champion | 0.298 | 0.343 | 0.262 | 0.252 | 0.340 | 0.234 | 0.289 | 0.221 |
| ventral visual stream | AlexNet fc6 | 0.372 | 0.411 | 0.325 | 0.325 | 0.425 | 0.298 | 0.354 | 0.278 |
| early and ventral visual streams | champion | 0.398 | 0.403 | 0.327 | 0.302 | 0.381 | 0.285 | 0.336 | 0.283 |
| early and ventral visual streams | AlexNet conv4 | 0.411 | 0.428 | 0.351 | 0.336 | 0.420 | 0.315 | 0.358 | 0.305 |
