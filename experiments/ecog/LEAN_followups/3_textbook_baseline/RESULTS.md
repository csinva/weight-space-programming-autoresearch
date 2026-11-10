# Textbook baseline vs the search (grid electrodes, held-out sessions)

Five isolated agents (claude-opus-4-7, the search model) each built a one-shot "textbook" feature space from the literature, with no access to the search or its results and no performance feedback (PROTOCOL.md, AUDIT.md). Scored once each with the LEAN harness. Test r = mean over held-out chunks of the per-chunk Pearson r, averaged over grid electrodes. Differences: paired chunk bootstrap, 95% CI.

| Subject | agent 1 | agent 2 | agent 3 | agent 4 | agent 5 | agent mean | trial 52 | v16 | v20 | kept | Whisper |
|---|---|---|---|---|---|---|---|---|---|---|---|
| patient1 (all 493) | 0.0548 | 0.0562 | 0.0542 | 0.0538 | 0.0557 | 0.0549 | 0.0626 | 0.0649 | 0.0690 | 0.0691 | 0.0752 |
| patient1 (primary 317) | 0.0587 | 0.0599 | 0.0578 | 0.0571 | 0.0595 | 0.0586 | 0.0664 | 0.0693 | 0.0735 | 0.0737 | 0.0819 |
| patient2 | 0.0591 | 0.0583 | 0.0609 | 0.0607 | 0.0602 | 0.0598 | 0.0660 | 0.0661 | 0.0643 | 0.0641 | 0.0776 |
| patient3 | 0.0556 | 0.0564 | 0.0558 | 0.0517 | 0.0563 | 0.0551 | 0.0605 | 0.0592 | 0.0604 | 0.0603 | 0.0710 |
| patient4 | 0.0509 | 0.0494 | 0.0505 | 0.0489 | 0.0539 | 0.0507 | 0.0580 | 0.0575 | 0.0578 | 0.0575 | 0.0591 |
| patient5 | 0.0558 | 0.0561 | 0.0554 | 0.0540 | 0.0584 | 0.0559 | 0.0612 | 0.0628 | 0.0637 | 0.0634 | 0.0678 |
| patient6 | 0.0521 | 0.0505 | 0.0497 | 0.0495 | 0.0546 | 0.0513 | 0.0600 | 0.0606 | 0.0614 | 0.0615 | 0.0649 |

## Search minus textbook (agent mean), with 95% CI

| Subject | trial 52 − textbook | v16 − textbook | v20 − textbook | kept − textbook | Whisper − textbook | best agent | trial 52 − best agent |
|---|---|---|---|---|---|---|---|
| patient1 (all 493) | +0.0076 [+0.0065, +0.0087] | +0.0099 [+0.0086, +0.0112] | +0.0140 [+0.0127, +0.0153] | +0.0142 [+0.0129, +0.0154] | +0.0202 [+0.0184, +0.0220] | 2 (0.0562) | +0.0063 [+0.0052, +0.0074] |
| patient1 (primary 317) | +0.0078 [+0.0065, +0.0091] | +0.0107 [+0.0091, +0.0121] | +0.0149 [+0.0134, +0.0165] | +0.0152 [+0.0135, +0.0166] | +0.0233 [+0.0212, +0.0253] | 2 (0.0599) | +0.0065 [+0.0051, +0.0078] |
| patient2 | +0.0062 [+0.0048, +0.0075] | +0.0063 [+0.0047, +0.0078] | +0.0045 [+0.0029, +0.0062] | +0.0043 [+0.0026, +0.0059] | +0.0177 [+0.0158, +0.0196] | 3 (0.0609) | +0.0051 [+0.0035, +0.0066] |
| patient3 | +0.0054 [+0.0042, +0.0064] | +0.0041 [+0.0023, +0.0058] | +0.0052 [+0.0034, +0.0070] | +0.0052 [+0.0032, +0.0072] | +0.0159 [+0.0128, +0.0187] | 2 (0.0564) | +0.0041 [+0.0031, +0.0052] |
| patient4 | +0.0072 [+0.0056, +0.0088] | +0.0068 [+0.0046, +0.0090] | +0.0071 [+0.0048, +0.0093] | +0.0068 [+0.0045, +0.0092] | +0.0083 [+0.0040, +0.0130] | 5 (0.0539) | +0.0041 [+0.0021, +0.0061] |
| patient5 | +0.0053 [+0.0040, +0.0065] | +0.0068 [+0.0053, +0.0084] | +0.0077 [+0.0062, +0.0092] | +0.0075 [+0.0060, +0.0089] | +0.0119 [+0.0089, +0.0148] | 5 (0.0584) | +0.0028 [+0.0014, +0.0043] |
| patient6 | +0.0088 [+0.0075, +0.0101] | +0.0093 [+0.0075, +0.0112] | +0.0102 [+0.0085, +0.0117] | +0.0102 [+0.0087, +0.0119] | +0.0136 [+0.0111, +0.0159] | 5 (0.0546) | +0.0054 [+0.0044, +0.0066] |

## Search trials needed to beat the textbook mean

First trial (Figure 1 order, of 593) at which the model the search was keeping (best so far on the patient1 old test) scores above the 5-agent textbook mean on that subject, and the trial from which it stays above.

- patient1 (all 493): first above at trial 37 (Gammatone-BEST+HL+flux); stays above from trial 37
- patient1 (primary 317): first above at trial 37 (Gammatone-BEST+HL+flux); stays above from trial 37
- patient2: first above at trial 10 (Gammatone-mod-bank); stays above from trial 41
- patient3: first above at trial 41 (Gammatone-BEST+HL+flux+modbank-multi); stays above from trial 41
- patient4: first above at trial 10 (Gammatone-mod-bank); stays above from trial 41
- patient5: first above at trial 10 (Gammatone-mod-bank); stays above from trial 41
- patient6: first above at trial 6 (Gammatone-multi-tau); stays above from trial 6
