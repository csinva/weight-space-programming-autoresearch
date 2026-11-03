# Measured effects of earlier edits of the same kind (dlogp of the target in nats; flip = top-1 changed; new top-1 shown if flipped)

## SC00: `model.embed_tokens.weight`[11751, 880] (target-token embedding entry): add -4
P0 dlogp +0.008 | P1 dlogp -0.002 | P2 dlogp -0.273 | P3 dlogp -0.155 | P4 dlogp +0.703 | P5 dlogp -0.268 | P6 dlogp -8.736 FLIP->' Paris' | P7 dlogp +0.000

## SC01: RMSNorm `model.layers.20.post_attention_layernorm.weight` dimension 373: set its gain (1+w) to 0 (w := -1)
P0 dlogp -0.001 | P1 dlogp -0.000 | P2 dlogp -0.000 | P3 dlogp -0.002 | P4 dlogp -0.004 | P5 dlogp +0.001 | P6 dlogp -0.000 | P7 dlogp -0.001

## SC02: `model.layers.20.mlp.down_proj.weight`[179, 320] (random down_proj entry): add +4
P0 dlogp -0.019 | P1 dlogp -0.036 | P2 dlogp -0.000 | P3 dlogp -0.102 | P4 dlogp -0.041 | P5 dlogp -0.000 | P6 dlogp -0.002 | P7 dlogp -0.054

## SC03: RMSNorm `model.norm.weight` dimension 160: set its gain (1+w) to 0 (w := -1)
P0 dlogp -0.007 | P1 dlogp -0.011 | P2 dlogp -0.039 | P3 dlogp -0.035 | P4 dlogp +0.025 | P5 dlogp +0.002 | P6 dlogp -0.001 | P7 dlogp +0.045

## SC04: `model.layers.10.linear_attn.out_proj.weight`[0, 332] (largest |w| in matrix): set to 0
P0 dlogp -0.002 | P1 dlogp -0.001 | P2 dlogp -0.000 | P3 dlogp -0.001 | P4 dlogp -0.011 | P5 dlogp -0.004 | P6 dlogp -0.000 | P7 dlogp +0.000

## SC05: `model.layers.10.mlp.gate_proj.weight`[2222, 618] (random entry): set to 0
P0 dlogp +0.000 | P1 dlogp +0.000 | P2 dlogp -0.000 | P3 dlogp +0.000 | P4 dlogp -0.000 | P5 dlogp -0.000 | P6 dlogp +0.000 | P7 dlogp +0.000

## SC06: RMSNorm `model.layers.6.post_attention_layernorm.weight` dimension 931: multiply its gain (1+w) by 10
P0 dlogp +0.004 | P1 dlogp +0.001 | P2 dlogp -0.017 | P3 dlogp +0.038 | P4 dlogp -0.012 | P5 dlogp +0.003 | P6 dlogp -0.001 | P7 dlogp +0.018

## SC07: RMSNorm `model.layers.10.input_layernorm.weight` dimension 950: set its gain (1+w) to 0 (w := -1)
P0 dlogp -0.000 | P1 dlogp -0.000 | P2 dlogp +0.000 | P3 dlogp +0.001 | P4 dlogp -0.008 | P5 dlogp -0.000 | P6 dlogp -0.000 | P7 dlogp +0.003

## SC08: `model.layers.1.mlp.down_proj.weight`[665, 3271] (random entry): set to 0
P0 dlogp -0.000 | P1 dlogp -0.000 | P2 dlogp +0.000 | P3 dlogp +0.000 | P4 dlogp +0.000 | P5 dlogp -0.000 | P6 dlogp +0.000 | P7 dlogp +0.000

## SC09: `model.layers.23.mlp.down_proj.weight`[747, 2266] (random down_proj entry): add +4
P0 dlogp -0.001 | P1 dlogp +0.002 | P2 dlogp -0.018 | P3 dlogp -0.015 | P4 dlogp -0.006 | P5 dlogp -0.003 | P6 dlogp +0.000 | P7 dlogp +0.030

## SC10: `model.layers.10.linear_attn.out_proj.weight`[74, 1912] (random entry): set to 0
P0 dlogp -0.000 | P1 dlogp -0.000 | P2 dlogp +0.000 | P3 dlogp +0.000 | P4 dlogp +0.000 | P5 dlogp -0.000 | P6 dlogp +0.000 | P7 dlogp -0.000

## SC11: RMSNorm `model.layers.12.post_attention_layernorm.weight` dimension 686: set its gain (1+w) to 0 (w := -1)
P0 dlogp -0.000 | P1 dlogp +0.000 | P2 dlogp -0.001 | P3 dlogp +0.005 | P4 dlogp +0.014 | P5 dlogp +0.000 | P6 dlogp +0.000 | P7 dlogp +0.002

## SC12: RMSNorm `model.layers.9.input_layernorm.weight` dimension 235: multiply its gain (1+w) by 10
P0 dlogp -0.006 | P1 dlogp -0.004 | P2 dlogp -0.005 | P3 dlogp -0.007 | P4 dlogp -0.018 | P5 dlogp -0.000 | P6 dlogp -0.000 | P7 dlogp +0.020

## SC13: RMSNorm `model.layers.19.input_layernorm.weight` dimension 140: multiply its gain (1+w) by 10
P0 dlogp -0.025 | P1 dlogp +0.008 | P2 dlogp -0.003 | P3 dlogp -0.129 | P4 dlogp +0.081 | P5 dlogp -0.012 | P6 dlogp +0.000 | P7 dlogp +0.028

## SC14: `model.embed_tokens.weight`[513, 854] (target-token embedding entry): add +4
P0 dlogp +0.000 | P1 dlogp +0.000 | P2 dlogp +0.000 | P3 dlogp +0.000 | P4 dlogp -0.001 | P5 dlogp -0.000 | P6 dlogp -0.000 | P7 dlogp -9.025 FLIP->'\n'

## SC15: `model.layers.0.mlp.down_proj.weight`[538, 2319] (random down_proj entry): add -4
P0 dlogp -0.003 | P1 dlogp +0.021 | P2 dlogp +0.007 | P3 dlogp -0.013 | P4 dlogp +0.012 | P5 dlogp -0.001 | P6 dlogp +0.000 | P7 dlogp -0.049

## SC16: `model.layers.2.mlp.gate_proj.weight`[2965, 980] (random entry): set to 0
P0 dlogp -0.000 | P1 dlogp -0.000 | P2 dlogp +0.000 | P3 dlogp -0.000 | P4 dlogp -0.000 | P5 dlogp +0.000 | P6 dlogp +0.000 | P7 dlogp -0.000

## SC17: `model.layers.9.mlp.gate_proj.weight`[2252, 16] (largest |w| in matrix): set to 0
P0 dlogp -0.000 | P1 dlogp -0.000 | P2 dlogp -0.000 | P3 dlogp -0.000 | P4 dlogp +0.000 | P5 dlogp +0.000 | P6 dlogp -0.000 | P7 dlogp +0.000

## SC18: `model.layers.1.linear_attn.out_proj.weight`[767, 1456] (random entry): set to 0
P0 dlogp -0.000 | P1 dlogp -0.000 | P2 dlogp +0.000 | P3 dlogp -0.000 | P4 dlogp -0.000 | P5 dlogp -0.000 | P6 dlogp +0.000 | P7 dlogp +0.000

## SC19: `model.layers.4.mlp.down_proj.weight`[117, 474] (random down_proj entry): add +4
P0 dlogp +0.005 | P1 dlogp -0.002 | P2 dlogp +0.050 | P3 dlogp +0.002 | P4 dlogp +0.018 | P5 dlogp +0.005 | P6 dlogp -0.001 | P7 dlogp -0.001

## SC20: `model.layers.21.mlp.down_proj.weight`[49, 301] (random down_proj entry): add -4
P0 dlogp -0.023 | P1 dlogp -0.018 | P2 dlogp +0.016 | P3 dlogp +0.039 | P4 dlogp -0.052 | P5 dlogp +0.002 | P6 dlogp -0.001 | P7 dlogp +0.056

## SC21: RMSNorm `model.layers.21.post_attention_layernorm.weight` dimension 495: set its gain (1+w) to 0 (w := -1)
P0 dlogp -0.000 | P1 dlogp +0.001 | P2 dlogp -0.001 | P3 dlogp -0.002 | P4 dlogp -0.000 | P5 dlogp -0.000 | P6 dlogp -0.000 | P7 dlogp +0.002

## SC22: `model.embed_tokens.weight`[11751, 995] (target-token embedding entry): add -4
P0 dlogp +0.087 | P1 dlogp +0.000 | P2 dlogp +0.000 | P3 dlogp -0.018 | P4 dlogp +0.757 | P5 dlogp +0.000 | P6 dlogp +0.000 | P7 dlogp -6.838 FLIP->' Paris'

## SC23: `model.embed_tokens.weight`[11751, 594] (target-token embedding entry): add +4
P0 dlogp -27.440 FLIP->' London' | P1 dlogp +0.000 | P2 dlogp +0.000 | P3 dlogp +0.000 | P4 dlogp -24.891 FLIP->'\n' | P5 dlogp +0.000 | P6 dlogp -0.000 | P7 dlogp +0.000