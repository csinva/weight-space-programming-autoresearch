# RSA search and cross-dataset encoding migration

Implementation plan, agreed October 21, 2026.

1. Vendor the scientific loaders and RSA/encoding kernels from `looped-networks`
   revision `3ee575baeceb1ad784102162a051b7444182fdaa`, with local import/path
   adaptations and source provenance. Keep model inputs as flattened RGB images.
2. Define versioned protocols: LAION-fMRI (all five subjects) RSA search → NSD
   encoding final evaluation; Triple-N (all five monkeys, available regions) RSA
   search → TVSD (both monkeys) encoding final evaluation.
3. Use the shared LAION image set excluding NSD-source stimuli (1,252 images in
   the current release), and all 1,000 Triple-N images. Exclude the 240 known
   NSD-source LAION images to avoid exposing final-dataset stimuli during search.
4. Match the source metrics: Pearson dissimilarity matrices compared by
   average-rank Spearman for RSA; train-centered features and per-unit five-fold
   ridge alpha selection for held-out Pearson encoding. Average region scores
   within subject, then subjects equally. Fail on incomplete or invalid scores.
5. Replace Copilot execution in proposer, repair, conductor and simplification
   with a shared adapter for the installed Codex/Astra and agy/Argon 4 CLIs.
   Verify actual local model identifiers and CLI behavior before configuring them.
6. Persist protocol, metric, subjects, data configuration and source/model
   provenance. Reject incompatible resumes. Keep final encoding artifacts out
   of the search queue, reward tables and proposer feedback.
7. Validate metric agreement, data alignment, aggregation, provider commands,
   failure handling and search/final separation. Run a local data smoke test;
   document separately any full evaluation or autonomous search not performed.

Parallel ownership: dataset adapters/protocols; scoring kernels/reward;
provider integration. The primary agent handles queue/run integration, final
evaluation, packaging, documentation and end-to-end verification.
