# Measured effects of earlier edits of the same kind (dlogp of the target in nats; flip = top-1 changed; new top-1 shown if flipped)

## LA00: layer 12: skip the MLP sublayer
P0 dlogp -0.012 | P1 dlogp +0.048 | P2 dlogp +0.070 | P3 dlogp -0.084 | P4 dlogp +0.531 | P5 dlogp +0.047 | P6 dlogp -0.019 | P7 dlogp +0.016

## LA01: layer 14: skip the token mixer (attention / DeltaNet) sublayer
P0 dlogp -0.140 | P1 dlogp -0.055 | P2 dlogp -0.000 | P3 dlogp -0.088 | P4 dlogp +0.416 | P5 dlogp +0.013 | P6 dlogp -0.001 | P7 dlogp +0.093

## LA02: layer 1: skip the MLP sublayer
P0 dlogp -0.055 | P1 dlogp +0.104 | P2 dlogp -0.010 | P3 dlogp +0.125 | P4 dlogp +0.019 | P5 dlogp -0.076 | P6 dlogp -0.019 | P7 dlogp -0.423

## LA03: layer 17: skip the whole block (mixer and MLP)
P0 dlogp -0.164 | P1 dlogp -0.032 | P2 dlogp -0.076 | P3 dlogp -0.597 | P4 dlogp -0.821 | P5 dlogp -0.115 | P6 dlogp -0.105 | P7 dlogp -0.310

## LA04: layer 8: skip the MLP sublayer
P0 dlogp +0.017 | P1 dlogp -0.089 | P2 dlogp -0.001 | P3 dlogp -0.104 | P4 dlogp -0.082 | P5 dlogp +0.014 | P6 dlogp -0.001 | P7 dlogp +0.123

## LA05: layer 7: skip the whole block (mixer and MLP)
P0 dlogp -0.031 | P1 dlogp +0.101 | P2 dlogp -0.008 | P3 dlogp -0.208 | P4 dlogp +0.485 | P5 dlogp -0.342 | P6 dlogp -0.001 | P7 dlogp -0.315

## LA06: layer 20: skip the whole block (mixer and MLP)
P0 dlogp -0.031 | P1 dlogp -0.192 | P2 dlogp +0.124 | P3 dlogp +0.134 | P4 dlogp -0.560 | P5 dlogp -0.020 | P6 dlogp -0.351 | P7 dlogp -0.193

## LA07: layer 9: skip the MLP sublayer
P0 dlogp -0.054 | P1 dlogp +0.053 | P2 dlogp +0.017 | P3 dlogp -0.039 | P4 dlogp -0.337 | P5 dlogp -0.000 | P6 dlogp -0.001 | P7 dlogp -0.075

## LA08: layer 1: skip the token mixer (attention / DeltaNet) sublayer
P0 dlogp +0.000 | P1 dlogp +0.024 | P2 dlogp -0.088 | P3 dlogp +0.226 | P4 dlogp -0.387 | P5 dlogp +0.019 | P6 dlogp -0.005 | P7 dlogp -0.303

## LA09: layer 12: skip the whole block (mixer and MLP)
P0 dlogp -0.069 | P1 dlogp +0.026 | P2 dlogp +0.095 | P3 dlogp -0.126 | P4 dlogp +0.374 | P5 dlogp +0.033 | P6 dlogp -0.025 | P7 dlogp -0.163

## LA10: layer 14: skip the whole block (mixer and MLP)
P0 dlogp -0.168 | P1 dlogp +0.065 | P2 dlogp +0.021 | P3 dlogp +0.030 | P4 dlogp +0.187 | P5 dlogp +0.011 | P6 dlogp -0.002 | P7 dlogp +0.186

## LA11: layer 13: skip the whole block (mixer and MLP)
P0 dlogp -0.041 | P1 dlogp +0.034 | P2 dlogp -0.011 | P3 dlogp +0.005 | P4 dlogp +0.320 | P5 dlogp -0.025 | P6 dlogp -0.004 | P7 dlogp -0.195