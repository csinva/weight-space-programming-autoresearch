#!/bin/bash
# After the describer: enforce <= 200 words, freeze the description, build rebuild_1..5 and start them.
set -uo pipefail
B=/path/to/ecog_data/lean_text_bottleneck
D=$B/describer/work/DESCRIPTION.md
[ -f "$D" ] || { echo "no DESCRIPTION.md"; exit 1; }
N=$(wc -w < "$D")
echo "description words: $N" | tee -a $B/logs/jobs.txt
[ "$N" -le 200 ] || { echo "description exceeds 200 words; stopping (protocol: no truncation)" | tee -a $B/logs/jobs.txt; exit 1; }
mkdir -p $B/frozen && cp "$D" $B/frozen/DESCRIPTION.md && sha256sum $B/frozen/DESCRIPTION.md >> $B/frozen/HASHES.txt
python3 - <<'PY'
import json, os, random, shutil
B = '/path/to/ecog_data/lean_text_bottleneck'
desc = open(f'{B}/frozen/DESCRIPTION.md').read().strip()
task = open(f'{B}/shared/REBUILD_TASK.md.tmpl').read().replace('@@DESCRIPTION@@', desc)
train = json.load(open('/path/to/ecog_data/circuit-building-autoresearch/runs-wave/newtest/train_frozen.json'))
for k in range(1, 6):
    n = f'{B}/rebuild_{k}'; w = f'{n}/work'
    shutil.rmtree(n, ignore_errors=True); os.makedirs(f'{w}/examples'); os.makedirs(f'{n}/home/.claude')
    open(f'{w}/TASK.md', 'w').write(task); shutil.copy(f'{B}/shared/check.py', f'{w}/check.py')
    for i, c in enumerate(random.Random(2000 + k).sample(train, 3)):       # same clips as control_k
        shutil.copy(f'/path/to/scratch/patient1_wave_aligned/{c}_wave.npy', f'{w}/examples/clip{i + 1}.npy')
PY
for k in 1 2 3 4 5; do sbatch --parsable -J bn_rebuild$k -p cpu_short -c 4 --mem=32G -t 08:00:00 -o $B/logs/slurm_rebuild_$k.out --wrap "$B/run_session.sh rebuild_$k run"; done | tr '\n' ' ' | sed 's/^/rebuild jobs: /' | tee -a $B/logs/jobs.txt
