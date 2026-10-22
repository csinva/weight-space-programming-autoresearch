# Search contract validation — 2026-10-22

The isolated evaluator preserved the checked-in sparse random-projection baseline's
LAION RSA exactly: **0.053461991608419225** across all five required subjects.
This validates execution and scoring equivalence for this baseline; it is not a
new model-search result or evidence of final-dataset transfer.

## Reference and execution

- Pre-change reference: `9db62573bb151fb4ee1b5004c9e29655941cf01d`.
- Baseline `model.py` SHA256:
  `3f72c1482a85713a490be06b388c54e47cf480cc9dc6336930ce3756d893c0b7`.
- Candidate and scoring device: NVIDIA GeForce RTX 4090, CUDA 13.0;
  PyTorch `2.14.1+cu130`, NumPy `2.5.4`, SciPy `1.18.1`.
- Input: fixed 128×128 flattened HWC RGB images; 1,252 permitted LAION shared
  images per subject, with NSD-source images excluded.
- Pinned source metrics revision: `3ee575baeceb1ad784102162a051b7444182fdaa`.
  Protocol `looped-cross-dataset-v2` changes access and feedback contracts while
  preserving primary source kernels, cohort, image selection and aggregation.

The reference baseline was explicitly run in-process only for this regression
comparison. Production candidate execution uses the isolated image-only worker.
On identical CUDA inputs, legacy and isolated features were bitwise equal
(maximum absolute difference zero). Original metrics produced identical subject
scores for both feature matrices. The complete new `evaluate_subjects` path,
including batched isolated forwards and source-group diagnostics, reproduced the
same cohort score.

| Subject | Legacy and isolated primary RSA |
| --- | ---: |
| sub-01 | 0.05157554727453957 |
| sub-03 | 0.054411984392809726 |
| sub-05 | 0.06346195516185443 |
| sub-06 | 0.055103083455452846 |
| sub-07 | 0.04275738775743952 |
| Equal-subject mean | **0.053461991608419225** |

An initial sandbox check exposed CPU fallback: the seeded baseline generates
different projections on CPU and CUDA. Its CPU-feature score was
`0.05315925079776549` under the same GPU scoring kernels. The worker now receives
the requested device and refuses unavailable CUDA; required GPU runtime mounts
were corrected. The final comparison above uses matching CUDA execution.

## Diagnostic feedback

Each subgroup recomputes Pearson RDMs and average-rank Spearman RSA on its own
images, then averages regions within subjects and all five subjects equally.
All three permitted groups succeeded for every subject:

| LAION source group | Images per subject | Cohort RSA |
| --- | ---: | ---: |
| `laion` | 641 | -0.009291955838634638 |
| `thingsplus` | 240 | 0.039386591886567165 |
| `ood` | 371 | 0.18904132165936255 |

These values do not replace or reconstruct the primary reward: full-set RSA
also includes between-group image pairs and ranks the full set. Here `ood` names
a permitted LAION stimulus source; it is not NSD or TVSD final evaluation.
A subgroup with an undefined subject score reports unavailable instead of
silently dropping that subject. Primary reward remains unchanged.

Targeted tests cover subgroup alignment and NSD exclusion, complete subgroup
cohorts, Triple-N's differing region coverage, unchanged primary weighting,
batched numeric-array features, and freezing the selection before final loading.
The six-field `VisionSplit` retains source-group metadata through the loader's
memoization; separate checks verified deepcopy and pickle round trips. Existing
file identities and selected stimulus IDs remain in trusted provenance, while
public feedback contains scores, counts and coverage.

Subgroup neural ranks deliberately bypass the full-set rank cache: its keys do
not identify stimulus subsets. This adds computation without introducing a
second scoring kernel or unsafe cache reuse. No runtime speedup is claimed.
The pre-existing full-set cache should be used with a consistent scoring device
within a process; it does not distinguish CPU and CUDA calculations in its key.

No NSD or TVSD data were loaded and no final encoding evaluation was performed.
The numerical regression above did not launch provider requests. A subsequent
live smoke run is recorded below.


## Five-minute operational smoke

On 2026-10-22, run `laion__2099-10-23` exercised the fixed three-provider roster
and Astra conductor for five minutes, using the existing machine logins inside
isolated workspaces. The future date is only an unused smoke-run directory label.
It finished with 12 successful evaluations, six errors and six queued candidates;
shutdown requeued its interrupted evaluation. Best primary RSA was
**0.1176484993247899**, from a 1,296-parameter oriented-energy/color model.
No final datasets or encoding scores were accessed.

All three providers submitted and received feedback. Models included parent-linked
component ablations and alternative representation families. Errors comprised
one invalid feature output, four parameter-cap violations and one batch-consistency
rejection of a convolutional model. The run exposed three practical issues:

- Full nested diagnostics were truncated by CLI tool output, motivating compact
  default feedback with full details available on request.
- CUDA batch-kernel rounding can falsely trigger pointwise equality checks;
  consistency tolerances must distinguish numerical drift from batch dependence.
- Ordinary parent-linked revisions must not consume the automatic repair budget.

An earlier interrupted smoke also led to moving provider logs outside agent
workspaces and fixing the initial overly strict consistency check. The final
compact-feedback and repair-budget fixes were tested separately after this run.
This is an operational smoke, not a controlled comparison of feedback policies,
search efficiency, or evidence of a newly discovered cortical mechanism.


## Final checks after smoke fixes

The complete offline suite passed: **80 tests**. A final five-subject LAION
baseline evaluation again returned exactly `0.053461991608419225`. The valid
96,400-parameter convolutional smoke candidate previously rejected by the
coordinate-wise consistency check now passes, with RSA `0.03443927649839325`.
The revised check uses a 1% relative vector-L2 tolerance and does not change
candidate activations or scoring kernels.

Formatting the recorded smoke champion with the compact feedback view reduced
pretty-printed JSON from 16,963 to 3,959 characters (76.7%). Region coverage and
source-group-by-region means remain in the default response; full subject trees
are available with `status ID --details` or `wait-for-score ID --details`.
These final checks did not launch another live search or access final datasets.
