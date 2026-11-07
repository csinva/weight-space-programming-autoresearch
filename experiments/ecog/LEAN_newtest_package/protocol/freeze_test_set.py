"""Freeze the NEW held-out test set (Nov 6 2026) before any trial is scored on it.

Rule (fixed in advance, no tuning): session 450a36e4 is excluded from training in every
autoresearch run, and only its 23-chunk block 0190-0212 (one podcast segment) was ever used as
the test set. Candidates = every other 450a36e4 chunk in master_list.json with high-gamma and
audio (493). Primary test set = candidates in master_list_speech150.json (>=150 clean words; the
Feb 2026 list, matching the old set's speech-rich selection) and more than 10 chunks away from the
old block (index < 180 or > 222), so no adjacent continuation of the old podcast is reused.
Per-chunk r is saved for all candidates, so the all-candidate set is reported as a secondary.
Training is unchanged: the frozen 500-chunk list the harness built for every trial.
"""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(REPO)
sys.path.insert(0, os.path.join(REPO, 'evolve-wave'))
os.environ['ECOG_TEST_CHUNKS_FILE'] = os.path.join(REPO, 'evolve-wave', 'speech_rich_test_chunks.json')
os.environ['ECOG_EXCLUDE_PREFIX'] = '450a36e4'
from src import data  # noqa: E402

train, old = data.get_chunk_names(num_train=500)
oldset = set(old)
sp = set(json.load(open(os.path.join(ROOT, 'master_list_speech150.json'))))
ex = lambda c, s: os.path.exists(os.path.join(data.WAV_DIR, f'{c}{s}'))
ml = json.load(open(data.MASTER_LIST))
cand = [c for c in ml if c.startswith('450a36e4') and c not in oldset
        and ex(c, '_ecog_hasson_highgamma.npy') and ex(c, '.wav')]
idx = lambda c: int(c.split('_')[1])
prim = [c for c in cand if c in sp and not (180 <= idx(c) <= 222)]
assert not set(cand) & set(train) and not set(cand) & oldset
out = dict(train_frozen=train, test_old=old, test_candidates=cand, test_primary=prim)
for k, v in out.items():
    p = os.path.join(HERE, f'{k}.json')
    json.dump(v, open(p, 'w'), indent=0)
    print(f'{k:16s} n={len(v):4d} sha256={hashlib.sha256(json.dumps(v).encode()).hexdigest()[:16]}')
