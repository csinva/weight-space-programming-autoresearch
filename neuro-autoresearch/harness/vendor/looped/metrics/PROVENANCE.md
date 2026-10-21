These two Python modules are copied without modification from Yash Mehta's
`looped-networks` repository at commit
`3ee575baeceb1ad784102162a051b7444182fdaa`:

- `looped/metrics/rsa.py`
- `looped/metrics/encoding.py`

The adapter in `harness/metrics.py` validates inputs and uses the source's
average-rank Spearman RSA implementation and default encoding feature scaling
(`features="center"` in the source `evaluate.py`). It adds no noise-ceiling
normalization. Cohort and anatomical-region aggregation are defined by
`harness/protocols.py`.

The source files deliberately retain both historical and current helpers.
Active RSA calls `rdm_chunked`, `compute_rdm`, `triu_ranks`, and
`spearman_from_ranks`; it never calls the ordinal-rank `ranked_triu` helper.
Active encoding calls `design` and `unit_scores`; it never uses the legacy
`harness.ridge` readout or its environment switch.
