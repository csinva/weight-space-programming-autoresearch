# Manuscript review

- Use computational model consistently; retain code terminology only when explaining the implementation.
- Place loop, architecture and results in that order in the main text. Remove obsolete appendix references and extra plots.
- Keep stimulus holdout distinct from individual holdout, and distinguish the fixed candidate from the fitted readout.
- Figure 3 displays held-out NSD and TVSD against all nine saved baselines per dataset. It does not display Triple-N or search cohorts. Distinguish untrained AlexNet, ResNet-50 and ViT-B/16 from pretrained references; no untrained DINOv2 or CLIP results are available.
- Preserve the single-animal limitation for Triple-N EVC and the asymmetry between search and prespecified references.
- Component ablations remain necessary to attribute predictivity to particular operations; the present results establish predictivity, not a biological mechanism.
- Check figure numbering, label legibility and the four-page main-text limit after compiling.

## Figure 3 review, 2026-09-17

- Identify each independently discovered model by its main computations, without implying component-level causal evidence.
- Show zero-based regional bars, all saved references, and individual NSD subjects; do not infer uncertainty from one held-out monkey.
- The NSD V3 mean lead over pretrained AlexNet is small; avoid suggesting a uniform individual-subject advantage.
- Check all plotted values against held-out subject records and inspect the compiled page at manuscript scale.
- The existing manuscript already has Discussion on page 5, exceeding its stated four-page main-text target. A separate manuscript-length edit remains necessary; this figure revision preserves the six-page total.
