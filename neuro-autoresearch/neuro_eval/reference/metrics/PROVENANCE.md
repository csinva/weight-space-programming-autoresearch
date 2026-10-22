These two Python modules are copied without modification from Yash Mehta's
`looped-networks` repository at commit
`3ee575baeceb1ad784102162a051b7444182fdaa`:

- `looped/metrics/rsa.py`
- `looped/metrics/encoding.py`

The adapters in `neuro_eval/rsa.py` and `neuro_eval/encoding.py` validate inputs and use the source's
average-rank Spearman RSA implementation and default encoding feature scaling
(`features="center"` in the source `evaluate.py`). They add no noise-ceiling
normalization. Cohort and anatomical-region aggregation are defined by
`neuro_eval/benchmarks.py`.

The source files deliberately retain both historical and current helpers.
Active RSA calls `rdm_chunked`, `compute_rdm`, `triu_ranks`, and
`spearman_from_ranks`; it never calls the ordinal-rank `ranked_triu` helper.
Active encoding calls `design` and `unit_scores`. The archival readout and its
environment switch are not included in this package.
