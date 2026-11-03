# Measured effects of earlier edits of the same kind (dlogp of the target in nats; flip = top-1 changed; new top-1 shown if flipped)

## LA04: layer 8: skip the MLP sublayer
P0 dlogp +0.017 | P1 dlogp -0.089 | P2 dlogp -0.001 | P3 dlogp -0.104 | P4 dlogp -0.082 | P5 dlogp +0.014 | P6 dlogp -0.001 | P7 dlogp +0.123

## LA10: layer 14: skip the whole block (mixer and MLP)
P0 dlogp -0.168 | P1 dlogp +0.065 | P2 dlogp +0.021 | P3 dlogp +0.030 | P4 dlogp +0.187 | P5 dlogp +0.011 | P6 dlogp -0.002 | P7 dlogp +0.186

## LA18: layer 10: skip the MLP sublayer
P0 dlogp -0.047 | P1 dlogp -0.106 | P2 dlogp -0.129 | P3 dlogp -0.111 | P4 dlogp +0.103 | P5 dlogp -0.043 | P6 dlogp -0.004 | P7 dlogp -0.049

## LA21: layer 16: skip the token mixer (attention / DeltaNet) sublayer
P0 dlogp -0.030 | P1 dlogp -0.015 | P2 dlogp +0.029 | P3 dlogp -0.067 | P4 dlogp -0.616 | P5 dlogp +0.013 | P6 dlogp -0.012 | P7 dlogp -0.191