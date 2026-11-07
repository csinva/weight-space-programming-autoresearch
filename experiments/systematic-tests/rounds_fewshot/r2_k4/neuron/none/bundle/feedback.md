# Measured effects of earlier edits of the same kind (dlogp of the target in nats; flip = top-1 changed; new top-1 shown if flipped)

## NE04: MLP neuron 2298 in layer 10: multiply its down_proj column by 5
P0 dlogp +0.001 | P1 dlogp -0.002 | P2 dlogp +0.001 | P3 dlogp -0.001 | P4 dlogp -0.000 | P5 dlogp +0.002 | P6 dlogp +0.000 | P7 dlogp +0.006

## NE10: MLP neuron 1102 in layer 20: zero the neuron (its down_proj column)
P0 dlogp +0.000 | P1 dlogp -0.000 | P2 dlogp -0.000 | P3 dlogp +0.000 | P4 dlogp +0.001 | P5 dlogp -0.000 | P6 dlogp -0.000 | P7 dlogp -0.067

## NE18: MLP neuron 480 in layer 10: multiply its down_proj column by 5
P0 dlogp +0.001 | P1 dlogp +0.004 | P2 dlogp +0.002 | P3 dlogp +0.010 | P4 dlogp -0.011 | P5 dlogp +0.001 | P6 dlogp +0.000 | P7 dlogp -0.033

## NE21: MLP neuron 2974 in layer 20: multiply its down_proj column by 5
P0 dlogp -0.001 | P1 dlogp -0.000 | P2 dlogp -0.001 | P3 dlogp -0.002 | P4 dlogp -0.008 | P5 dlogp +0.000 | P6 dlogp +0.000 | P7 dlogp -0.003