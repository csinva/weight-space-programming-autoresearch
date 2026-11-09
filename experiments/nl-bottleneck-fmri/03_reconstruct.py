"""Step 2: a fresh headless Claude Code session (Opus 5.5) rebuilds each model from its description alone.
Sandbox: template model.py, score.py (fit on the fitting stories, report visible correlation), results/.
The session may score up to MAX_EXP experiments. Writes WORK/recon/<name>-r<k>/ (agent log, results, snapshots, meta).

  python 03_reconstruct.py --reps 1 2 --workers 3
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import HERE, MODELS, WORK

PY = sys.executable
MAX_EXP = 6
TOOLS = 'Read,Edit,Write,Glob,Grep,Bash'
PROMPT = """You are rebuilding a hand-written model of brain responses to language. An earlier agent wrote, with no training, a model that turns text into features for predicting fMRI responses. You will not see that model. All you get is the description below, written by another engineer who did read it. Your job is to hand-write a model that computes the same features, so that it predicts the brain about as well as the original and carries the same information.

TASK: Subjects listened to narrative stories while fMRI was recorded. For every word, the model receives the 10-gram ending at that word (the word plus up to 9 preceding words) and returns a feature vector; the model is called once per story with all of that story's 10-grams in order. The features are downsampled to the fMRI time points, z-scored, delayed by 1 to 4 time points, and ridge regression fit on 8 stories maps them to each voxel's response. Performance is the mean correlation between predicted and measured responses across voxels on 2 visible stories. The original model reached {vis:.4f} on these visible stories.

DESCRIPTION OF THE ORIGINAL MODEL ({nw} words):
\"\"\"
{summary}
\"\"\"

Your working folder has model.py (a template; build_embedder() must return an object mapping a list of 10-gram strings to an array of shape (number of 10-grams, number of features)), score.py (read-only scoring script) and results/.
Rules:
- NO training of any kind: no gradient steps, optimizers, fitting loops, least squares or searches over weights. Every parameter and feature is set in closed form from your design.
- Do not use pretrained models, embeddings or tokenizers, and do not fit anything to the fMRI responses or to statistics of the stories. Write any word lists from your own knowledge, guided by the description. You may use numpy, and NLTK's WordNet (available offline) only if the description says the original used it.
- Do not read anything outside this folder, and do not modify score.py.
Procedure: implement the described model in model.py, then score it with exactly `{py} score.py` (set MODEL_NAME, unique per run, and MODEL_DESCRIPTION in model.py before each run; each run appends a row to results/overall_results.csv and saves a snapshot to interpretable_transformers_lib/<MODEL_NAME>.py). Scoring takes a few minutes and may wait for other jobs. Run at most {maxexp} experiments in total. Use the experiments to fix bugs and match the description. Do not tune it to beat the original: the goal is faithfulness, so do not add features or fix limitations beyond what the description says. Finish with a one-line statement naming the MODEL_NAME of your final model."""


def run(job):
    name, k = job
    rd = WORK / 'recon' / f'{name}-r{k}'
    if (rd / 'done').exists():
        return f'{name}-r{k} cached'
    shutil.rmtree(rd, ignore_errors=True); rd.mkdir(parents=True)
    summary = (WORK / 'summaries' / f'{name}.txt').read_text().strip()
    vis = json.loads((WORK / 'originals_eval' / f'{name}.json').read_text())['visible']
    sb = WORK / 'sandbox' / f'rec-{name}-r{k}'
    shutil.rmtree(sb, ignore_errors=True)
    shutil.copytree(HERE / 'sandbox_template', sb)
    (sb / 'results').mkdir(); (sb / 'interpretable_transformers_lib').mkdir()
    prompt = PROMPT.format(vis=vis, nw=len(summary.split()), summary=summary, py=PY, maxexp=MAX_EXP)
    env = dict(os.environ, NLBF_CODE=str(HERE), NLBF_DATA=str(WORK / 'data_visible'), NLBF_WORK=str(WORK),
               NLTK_DATA=str(WORK / 'data_visible' / 'nltk'), OMP_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    cmd = ['claude', '-p', prompt, '--model', 'claude-opus-5-5', '--restricted', '--tools', TOOLS, '--strict-mcp-config',
           '--output-format', 'stream-json', '--verbose', '--allowedTools', f'Bash({PY} score.py)', f'Bash({PY} score.py:*)', 'Edit', 'Write']
    t0 = time.time()
    with (rd / 'agent_log.jsonl').open('w') as fh:
        subprocess.run(cmd, cwd=sb, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env, timeout=4 * 3600)
    res = {}
    for line in (rd / 'agent_log.jsonl').open():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('type') == 'result':
            res = d
    shutil.copytree(sb / 'results', rd / 'results')
    shutil.copytree(sb / 'interpretable_transformers_lib', rd / 'interpretable_transformers_lib')
    shutil.copy(sb / 'model.py', rd / 'model_final_state.py')
    rows = list(csv.DictReader((sb / 'results' / 'overall_results.csv').open())) if (sb / 'results' / 'overall_results.csv').exists() else []
    snaps = {p.stem for p in (sb / 'interpretable_transformers_lib').glob('*.py')}
    text = res.get('result', '') or ''
    named = [s for s in snaps if s in text]
    ok = [r for r in rows if r['status'] == 'success']
    final = max(named, key=len) if named else (ok[-1]['model_name'] if ok else None)
    meta = dict(name=name, rep=k, experiments=len(rows), cost_usd=res.get('total_cost_usd'), turns=res.get('num_turns'),
                seconds=round(time.time() - t0), final=final, final_text=text[-500:])
    (rd / 'meta.json').write_text(json.dumps(meta, indent=1))
    (rd / 'done').write_text('')
    return f"{name}-r{k}: {len(rows)} experiments, final {final}, ${meta['cost_usd'] or 0:.2f}, {meta['seconds'] // 60} min"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    jobs = [(n, k) for k in a.reps for _, n in MODELS if not a.only or n in a.only]
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(run, jobs):
            print(msg, flush=True)
