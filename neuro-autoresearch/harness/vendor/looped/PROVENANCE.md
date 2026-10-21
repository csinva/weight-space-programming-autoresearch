# Source provenance

Source: local repository `/home/ymehta3/research/looped-networks`.
Pinned Git revision: `3ee575baeceb1ad784102162a051b7444182fdaa`.
Imported October 21, 2026 for neuro-autoresearch.

The files `metrics/rsa.py` and `metrics/encoding.py` are byte-for-byte copies of
`looped/metrics/rsa.py` and `looped/metrics/encoding.py` at that revision. Data
readers `data/{nsd,laion,triplen,tvsd,stimuli}.py` preserve their source algorithms
with imports redirected into `harness.vendor.looped`. `utils.py` is a minimal
local adaptation for pickle loading, local environment variables and shared
storage defaults. The source repository is not imported at runtime.

The surrounding adapter intentionally changes the experiment, not these
numerical kernels: LAION search uses shared non-NSD-source images only;
Triple-N search combines its source folds; both average primary-region RSA
within subject then subjects equally. Final NSD/TVSD encoding retains source
image splits and the source evaluator's `features='center'` default. Primary
regions are the early and ventral streams. Available coverage, fixed thresholds,
file identities and exact image IDs are recorded by the local dataset adapter.

Prepared files remain in shared storage. Legacy local baked `.pt` formats are
not used by these four datasets. See the root README for runtime paths.
