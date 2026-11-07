#!/bin/bash
# score all selected original programs (CPU, 12 parallel)
cd "$(dirname "$0")"; mkdir -p preds/orig
export HF_HOME=/data18/chandan.singh/wsp-nl/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DATASETS_VERBOSITY=error
PY=~/weight-space-programming-autoresearch/.venv/bin/python
ls originals/*.py | xargs -P 12 -I{} bash -c 't=$(basename {} .py); '$PY' eval_program.py {} $t preds/orig/$t.json 2>&1 | grep -v "^Using\|^Found"'
