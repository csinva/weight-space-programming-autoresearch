#!/usr/bin/env bash
# search/evaluator/repair.sh -- the low-effort repair agent. Rather than a deterministic check that
# errors out on a small mistake, a single low-effort agent lightly fixes silly issues in a
# proposed model (bad shapes, typos, import slips, or a too-heavy model that timed out), edits
# the on-disk model.py IN PLACE, and gets it running. Only genuinely broken models fall
# through to status='error'. Wired into the evaluator's error/timeout path, bounded to <=2
# attempts per model there.
#
#   search/evaluator/repair.sh <model_file> <log_file> [model] [effort] [repo]
#
# <model_file>  the model's source on disk (the row's `path`) -- edited in place
# <log_file>    the failed eval's stdout+stderr (its tail holds the traceback)
# model/effort  the repair agent's model + reasoning effort (default: Codex/Astra, LOW effort)
set -uo pipefail
MODEL_FILE="${1:?usage: repair.sh <model_file> <log_file> [model] [effort] [repo]}"
LOG_FILE="${2:?need the eval log/traceback file}"
REPAIR_MODEL="${3:-codex:astra}"
REPAIR_EFFORT="${4:-low}"
REPO="${5:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "$REPO/search/lib.sh"
use_project_python || exit 2

# Tail the log so the prompt stays small -- the tail holds the actual error / traceback.
TRACE="$(tail -c 6000 "$LOG_FILE" 2>/dev/null || true)"

read -r -d '' TASK <<EOF || true
A proposed UNTRAINED neural-representation model failed to evaluate. Fix ONLY small mechanical
problems so it runs -- do NOT redesign it, do NOT change its core idea, do NOT train it.

FILE TO FIX (edit it IN PLACE, save to the same path): ${MODEL_FILE}

Harness contract the file MUST satisfy: a self-contained Python file defining \`class Model\`
with attributes name/idea/description/insight/n_params and \`__call__(self, X)\` mapping a stimulus batch
X of shape (N, D_in) -> activations (N, D_out), ending with \`MODEL = Model()\`. Two HARD rules
stay intact: (1) UNTRAINED -- every weight analytic or random-but-fixed (fixed seeds), never
fit to data, never read the neural target Y; (2) PER-SAMPLE -- an image's activations depend
only on that image (no statistic across the batch).

Typical fixes: a wrong reshape / output dimension, a NameError or typo, a missing/wrong
import, a dtype or device (cpu/cuda) mismatch, an off-by-one dim. If the run TIMED OUT (no
result row, exit ~124), the model is too heavy -- make it LIGHTER and FASTER (smaller D_out,
chunk the forward pass with bs<=128 under torch.no_grad()) WITHOUT changing the idea.

Make the smallest change that fixes it, edit ONLY the named file in place, then STOP. Do NOT run it.
The harness, hub.py, data, search scripts, run contracts/env.json, and score records are immutable.
Never change the reward computation, subject cohort, filters, metric, or final evaluation targets.

EVAL ERROR / TRACEBACK (tail):
${TRACE}
EOF

exec "$PY" "$REPO/search/providers.py" run --agent "$REPAIR_MODEL" --effort "$REPAIR_EFFORT" \
  --role repair --cwd "$(dirname "$MODEL_FILE")" <<< "$TASK"
