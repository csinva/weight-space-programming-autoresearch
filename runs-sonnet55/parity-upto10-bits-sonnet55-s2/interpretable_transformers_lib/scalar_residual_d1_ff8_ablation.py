# Snapshot: scalar_residual_d1_ff8_ablation (200/200 on the 200-sample eval, 26 params). NOT a valid solution.
# Same as scalar_residual_d1_softmax but d_ff=8 and the knot loop is range(1, 9). The triangle wave then continues
# with slope +1 past k=8, so f(10)=2 and prompts with exactly 10 ones get the wrong parity; those are too rare to appear in 200 samples.
