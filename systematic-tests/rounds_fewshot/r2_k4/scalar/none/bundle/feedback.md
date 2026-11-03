# Measured effects of earlier edits of the same kind (dlogp of the target in nats; flip = top-1 changed; new top-1 shown if flipped)

## SC02: `model.layers.20.mlp.down_proj.weight`[179, 320] (random down_proj entry): add +4
P0 dlogp -0.019 | P1 dlogp -0.036 | P2 dlogp -0.000 | P3 dlogp -0.102 | P4 dlogp -0.041 | P5 dlogp -0.000 | P6 dlogp -0.002 | P7 dlogp -0.054

## SC06: RMSNorm `model.layers.6.post_attention_layernorm.weight` dimension 931: multiply its gain (1+w) by 10
P0 dlogp +0.004 | P1 dlogp +0.001 | P2 dlogp -0.017 | P3 dlogp +0.038 | P4 dlogp -0.012 | P5 dlogp +0.003 | P6 dlogp -0.001 | P7 dlogp +0.018

## SC18: `model.layers.1.linear_attn.out_proj.weight`[767, 1456] (random entry): set to 0
P0 dlogp -0.000 | P1 dlogp -0.000 | P2 dlogp +0.000 | P3 dlogp -0.000 | P4 dlogp -0.000 | P5 dlogp -0.000 | P6 dlogp +0.000 | P7 dlogp +0.000

## SC21: RMSNorm `model.layers.21.post_attention_layernorm.weight` dimension 495: set its gain (1+w) to 0 (w := -1)
P0 dlogp -0.000 | P1 dlogp +0.001 | P2 dlogp -0.001 | P3 dlogp -0.002 | P4 dlogp -0.000 | P5 dlogp -0.000 | P6 dlogp -0.000 | P7 dlogp +0.002