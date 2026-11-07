# Checkpoint 022: Qwen Semantic Multiscale Breakthrough

## Overview
Having exhaustively proven that purely untrained structural representations (Characters, Trigram Word Hashes, CMUDict Phonemes) possess a hard mathematical ceiling of exactly `0.0421` (via 15 staggered continuous-time exponential decay networks), we pivoted back to utilizing pre-trained semantic weights.

We deployed Qwen-2.5-3B and Qwen-2.5-7B, constructing a dual-stream MultiScale representational space.

## Architecture: MultiScale LLM Representation
Brain regions encode language at multiple concurrent timescales:
1. **Immediate Syntactic/Local Context**: Encoded strongly in the final token embedding of late-stage transformer layers.
2. **Global Semantic Gist**: Encoded strongly in mean-pooled embeddings across the entire sequence in middle layers.

To capture both simultaneously, we implemented:
`MultiScaleEmbedder(layer_last, layer_mean)`
- Maps n-gram sequences to their respective HF tokenizations.
- Extracts `hidden_states[layer_last][:, -1, :]` (Last Token Syntax)
- Extracts `mean(hidden_states[layer_mean] * mask)` (Mean-Pooled Semantic Gist)
- Concatenates the two vectors.

## Experimental Variations
### Qwen 3B MultiScale (3 Billion Parameters)
- Last Layer: 28 (Syntax)
- Mean Layer: 14 (Semantics)

### Qwen 7B MultiScale (7 Billion Parameters)
- Last Layer: 24 (Syntax)
- Mean Layer: 14 (Semantics)

We are currently evaluating these dual-stream MultiScale semantic models against the Ridge Regression fMRI pipeline. If successful, they should massively shatter the `0.0421` structural ceiling by effectively utilizing pre-trained semantic routing to map syntactic and conceptual flows.
