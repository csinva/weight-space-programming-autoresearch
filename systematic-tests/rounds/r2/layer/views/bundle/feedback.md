# Measured effects of earlier edits of the same kind (dlogp of the target in nats; flip = top-1 changed; new top-1 shown if flipped)

## LA00: layer 12: skip the MLP sublayer [file weights/LA00.txt]
P0 dlogp -0.012 | P1 dlogp +0.048 | P2 dlogp +0.070 | P3 dlogp -0.084 | P4 dlogp +0.531 | P5 dlogp +0.047 | P6 dlogp -0.019 | P7 dlogp +0.016

## LA01: layer 14: skip the token mixer (attention / DeltaNet) sublayer [file weights/LA01.txt]
P0 dlogp -0.140 | P1 dlogp -0.055 | P2 dlogp -0.000 | P3 dlogp -0.088 | P4 dlogp +0.416 | P5 dlogp +0.013 | P6 dlogp -0.001 | P7 dlogp +0.093

## LA02: layer 1: skip the MLP sublayer [file weights/LA02.txt]
P0 dlogp -0.055 | P1 dlogp +0.104 | P2 dlogp -0.010 | P3 dlogp +0.125 | P4 dlogp +0.019 | P5 dlogp -0.076 | P6 dlogp -0.019 | P7 dlogp -0.423

## LA03: layer 17: skip the whole block (mixer and MLP) [file weights/LA03.txt]
P0 dlogp -0.164 | P1 dlogp -0.032 | P2 dlogp -0.076 | P3 dlogp -0.597 | P4 dlogp -0.821 | P5 dlogp -0.115 | P6 dlogp -0.105 | P7 dlogp -0.310

## LA04: layer 8: skip the MLP sublayer [file weights/LA04.txt]
P0 dlogp +0.017 | P1 dlogp -0.089 | P2 dlogp -0.001 | P3 dlogp -0.104 | P4 dlogp -0.082 | P5 dlogp +0.014 | P6 dlogp -0.001 | P7 dlogp +0.123

## LA05: layer 7: skip the whole block (mixer and MLP) [file weights/LA05.txt]
P0 dlogp -0.031 | P1 dlogp +0.101 | P2 dlogp -0.008 | P3 dlogp -0.208 | P4 dlogp +0.485 | P5 dlogp -0.342 | P6 dlogp -0.001 | P7 dlogp -0.315

## LA06: layer 20: skip the whole block (mixer and MLP) [file weights/LA06.txt]
P0 dlogp -0.031 | P1 dlogp -0.192 | P2 dlogp +0.124 | P3 dlogp +0.134 | P4 dlogp -0.560 | P5 dlogp -0.020 | P6 dlogp -0.351 | P7 dlogp -0.193

## LA07: layer 9: skip the MLP sublayer [file weights/LA07.txt]
P0 dlogp -0.054 | P1 dlogp +0.053 | P2 dlogp +0.017 | P3 dlogp -0.039 | P4 dlogp -0.337 | P5 dlogp -0.000 | P6 dlogp -0.001 | P7 dlogp -0.075

## LA08: layer 1: skip the token mixer (attention / DeltaNet) sublayer [file weights/LA08.txt]
P0 dlogp +0.000 | P1 dlogp +0.024 | P2 dlogp -0.088 | P3 dlogp +0.226 | P4 dlogp -0.387 | P5 dlogp +0.019 | P6 dlogp -0.005 | P7 dlogp -0.303

## LA09: layer 12: skip the whole block (mixer and MLP) [file weights/LA09.txt]
P0 dlogp -0.069 | P1 dlogp +0.026 | P2 dlogp +0.095 | P3 dlogp -0.126 | P4 dlogp +0.374 | P5 dlogp +0.033 | P6 dlogp -0.025 | P7 dlogp -0.163

## LA10: layer 14: skip the whole block (mixer and MLP) [file weights/LA10.txt]
P0 dlogp -0.168 | P1 dlogp +0.065 | P2 dlogp +0.021 | P3 dlogp +0.030 | P4 dlogp +0.187 | P5 dlogp +0.011 | P6 dlogp -0.002 | P7 dlogp +0.186

## LA11: layer 13: skip the whole block (mixer and MLP) [file weights/LA11.txt]
P0 dlogp -0.041 | P1 dlogp +0.034 | P2 dlogp -0.011 | P3 dlogp +0.005 | P4 dlogp +0.320 | P5 dlogp -0.025 | P6 dlogp -0.004 | P7 dlogp -0.195

## LA12: layer 23: skip the MLP sublayer [file weights/LA12.txt]
P0 dlogp -0.248 | P1 dlogp +0.242 | P2 dlogp +0.099 | P3 dlogp -0.805 FLIP->' and' | P4 dlogp -2.877 FLIP->'\n' | P5 dlogp +0.045 | P6 dlogp -0.080 | P7 dlogp +1.107

## LA13: layer 20: skip the MLP sublayer [file weights/LA13.txt]
P0 dlogp -0.078 | P1 dlogp -0.186 | P2 dlogp +0.103 | P3 dlogp -0.048 | P4 dlogp -1.080 | P5 dlogp -0.025 | P6 dlogp -0.033 | P7 dlogp -0.161

## LA14: layer 22: skip the whole block (mixer and MLP) [file weights/LA14.txt]
P0 dlogp -0.332 | P1 dlogp -0.838 | P2 dlogp +0.018 | P3 dlogp -0.947 | P4 dlogp -0.843 | P5 dlogp -0.304 | P6 dlogp -0.316 | P7 dlogp -1.730 FLIP->'.'

## LA15: layer 7: skip the MLP sublayer [file weights/LA15.txt]
P0 dlogp -0.051 | P1 dlogp +0.044 | P2 dlogp +0.027 | P3 dlogp -0.049 | P4 dlogp +0.295 | P5 dlogp -0.161 | P6 dlogp -0.001 | P7 dlogp -0.333

## LA16: layer 10: skip the token mixer (attention / DeltaNet) sublayer [file weights/LA16.txt]
P0 dlogp -0.133 | P1 dlogp -0.220 | P2 dlogp -0.025 | P3 dlogp -0.249 | P4 dlogp +0.106 | P5 dlogp +0.002 | P6 dlogp -0.003 | P7 dlogp -0.114

## LA17: layer 6: skip the token mixer (attention / DeltaNet) sublayer [file weights/LA17.txt]
P0 dlogp +0.031 | P1 dlogp +0.168 | P2 dlogp -0.087 | P3 dlogp -0.011 | P4 dlogp +0.293 | P5 dlogp +0.009 | P6 dlogp +0.001 | P7 dlogp +0.037

## LA18: layer 10: skip the MLP sublayer [file weights/LA18.txt]
P0 dlogp -0.047 | P1 dlogp -0.106 | P2 dlogp -0.129 | P3 dlogp -0.111 | P4 dlogp +0.103 | P5 dlogp -0.043 | P6 dlogp -0.004 | P7 dlogp -0.049

## LA19: layer 11: skip the whole block (mixer and MLP) [file weights/LA19.txt]
P0 dlogp -0.021 | P1 dlogp +0.024 | P2 dlogp +0.031 | P3 dlogp +0.101 | P4 dlogp +0.282 | P5 dlogp -0.144 | P6 dlogp -0.001 | P7 dlogp +0.015

## LA20: layer 19: skip the MLP sublayer [file weights/LA20.txt]
P0 dlogp -0.347 | P1 dlogp -0.098 | P2 dlogp -0.024 | P3 dlogp -0.747 | P4 dlogp -0.464 | P5 dlogp -0.122 | P6 dlogp -0.015 | P7 dlogp -0.011

## LA21: layer 16: skip the token mixer (attention / DeltaNet) sublayer [file weights/LA21.txt]
P0 dlogp -0.030 | P1 dlogp -0.015 | P2 dlogp +0.029 | P3 dlogp -0.067 | P4 dlogp -0.616 | P5 dlogp +0.013 | P6 dlogp -0.012 | P7 dlogp -0.191

## LA22: layer 18: skip the MLP sublayer [file weights/LA22.txt]
P0 dlogp -0.392 | P1 dlogp -0.048 | P2 dlogp +0.059 | P3 dlogp -0.076 | P4 dlogp -0.746 | P5 dlogp -0.160 | P6 dlogp -0.005 | P7 dlogp +0.124

## LA23: layer 16: skip the MLP sublayer [file weights/LA23.txt]
P0 dlogp -0.013 | P1 dlogp -0.033 | P2 dlogp +0.050 | P3 dlogp +0.082 | P4 dlogp -0.107 | P5 dlogp -0.028 | P6 dlogp -0.030 | P7 dlogp -0.006