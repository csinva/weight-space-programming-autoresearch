# Text bottleneck for LEAN-v20 (paper Fig. 2g protocol)

A fresh Opus 5.5 session described LEAN-v20 in 197 words (`frozen/DESCRIPTION.md`); 5 fresh sessions rebuilt it from that description and the task alone; 5 controls got the identical task with no description (PROTOCOL.md; sandbox and audits in AUDIT.md, INVARIANCE.md). Every model scored once with the LEAN harness on six held-out sessions. Grid electrodes; test r = mean over held-out chunks.

## 1. Reproduction of the original

| Subject | original (this wrapper, CPU features) | stored LEAN-v20 (GPU features) | difference |
|---|---|---|---|
| patient1 (all 493) | 0.06880 | 0.06897 | -0.00017 |
| patient1 (primary 317) | 0.07336 | 0.07352 | -0.00016 |
| patient2 | 0.06433 | 0.06431 | +0.00002 |
| patient3 | 0.06025 | 0.06038 | -0.00013 |
| patient4 | 0.05782 | 0.05781 | +0.00001 |
| patient5 | 0.06366 | 0.06367 | -0.00001 |
| patient6 | 0.06147 | 0.06145 | +0.00002 |

## 2. Held-out performance

| Subject | original | rebuilds (5) | rebuild mean | controls (5) | control mean | rebuild − original | control − original | rebuild − control |
|---|---|---|---|---|---|---|---|---|
| patient1 (all 493) | 0.0688 | 0.0646 0.0669 0.0669 0.0669 0.0641 | 0.0659 | 0.0614 0.0611 0.0632 0.0616 0.0566 | 0.0608 | -0.0029 [-0.0036, -0.0022] | -0.0080 [-0.0091, -0.0070] | +0.0051 [+0.0041, +0.0061] |
| patient1 (primary 317) | 0.0734 | 0.0696 0.0717 0.0710 0.0711 0.0692 | 0.0705 | 0.0657 0.0658 0.0675 0.0660 0.0612 | 0.0652 | -0.0028 [-0.0036, -0.0020] | -0.0081 [-0.0094, -0.0068] | +0.0053 [+0.0041, +0.0067] |
| patient2 | 0.0643 | 0.0674 0.0699 0.0684 0.0694 0.0666 | 0.0683 | 0.0643 0.0615 0.0638 0.0639 0.0577 | 0.0622 | +0.0040 [+0.0029, +0.0050] | -0.0021 [-0.0038, -0.0003] | +0.0061 [+0.0049, +0.0072] |
| patient3 | 0.0602 | 0.0595 0.0604 0.0591 0.0622 0.0575 | 0.0597 | 0.0615 0.0612 0.0608 0.0625 0.0564 | 0.0605 | -0.0005 [-0.0015, +0.0005] | +0.0002 [-0.0015, +0.0020] | -0.0007 [-0.0025, +0.0012] |
| patient4 | 0.0578 | 0.0564 0.0568 0.0547 0.0580 0.0558 | 0.0563 | 0.0532 0.0503 0.0543 0.0538 0.0488 | 0.0521 | -0.0015 [-0.0028, -0.0002] | -0.0058 [-0.0076, -0.0039] | +0.0043 [+0.0020, +0.0064] |
| patient5 | 0.0637 | 0.0614 0.0648 0.0633 0.0625 0.0628 | 0.0630 | 0.0613 0.0597 0.0626 0.0618 0.0565 | 0.0604 | -0.0007 [-0.0016, +0.0002] | -0.0033 [-0.0047, -0.0019] | +0.0026 [+0.0013, +0.0039] |
| patient6 | 0.0615 | 0.0606 0.0590 0.0593 0.0601 0.0579 | 0.0594 | 0.0598 0.0567 0.0582 0.0577 0.0555 | 0.0576 | -0.0021 [-0.0030, -0.0012] | -0.0039 [-0.0052, -0.0025] | +0.0018 [+0.0004, +0.0033] |

## 3. Closeness to the original

Prediction correlation = Pearson r between a model's held-out grid predictions and the original's, per chunk and electrode, averaged (patient1, 493 chunks). Dimensionality ratio = H / 3,852.

| Model | prediction correlation with original | dims (H) | H / 3,852 |
|---|---|---|---|
| rebuild_1 | 0.609 | 3837 | 0.996 |
| rebuild_2 | 0.658 | 3852 | 1.000 |
| rebuild_3 | 0.633 | 3852 | 1.000 |
| rebuild_4 | 0.651 | 3769 | 0.978 |
| rebuild_5 | 0.614 | 3852 | 1.000 |
| control_1 | 0.510 | 710 | 0.184 |
| control_2 | 0.501 | 1117 | 0.290 |
| control_3 | 0.513 | 1478 | 0.384 |
| control_4 | 0.506 | 892 | 0.232 |
| control_5 | 0.484 | 382 | 0.099 |

- Prediction correlation, mean: rebuilds 0.633 (range 0.609–0.658) vs controls 0.503 (range 0.484–0.513).
- Dimensionality ratio, geometric mean: rebuilds 0.995× vs controls 0.216× the original.
