# Conductor strategy (keep short — this is your durable plan)

## Hypothesis
The proven log-polar + steerable-pyramid texture family tops ventral (~0.325). The open
frontier is high-level cortex (FFA / PPA / VWFA), where untrained models still lag a
trained AlexNet — that is where a regime change has to come from.

## Try next
- Default round: fleet 10, topK 5, randK 5, directions mostly null (open discovery).
- If `deltaBest` < 0.005 for ~3 rounds: raise `randK` and add 1–2 vague frontier nudges.

## Knobs
fleet=10 topK=5 randK=5

## Stop when
budget reached, or ventral test_corr plateaus near the noise ceiling (~0.50).
