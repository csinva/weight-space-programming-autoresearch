# Natural-language compression (2026-11-07)

Claude cost: $0.07. Retention = (rebuild - gabor_norm) / (original - gabor_norm); CKA = linear CKA with the original's features on 1,000 LAION test images.

## laion__visual__encoding

| model | pooled | early | ventral | retention | CKA |
|---|---|---|---|---|---|
| gabor_norm | 0.430 | 0.543 | 0.310 | -0.00 | nan |
| original | 0.148 | 0.182 | 0.111 | 1.00 | 1.00 |
| rebuild_60w_0 | 0.147 | 0.181 | 0.112 | 1.00 | 1.00 |

Rebuilds failing the contract check: none.

## description_60w

Flatten each image's raw pixels into one vector. If it has 2,048 or fewer values, output it unchanged. Otherwise produce 2,048 features. Each feature sums four pixels, chosen uniformly at random with replacement, each multiplied by a random plus-or-minus-one sign, then divides by 2. The pixel choices and signs are fixed from seed 0 and shared across images. No learning.

