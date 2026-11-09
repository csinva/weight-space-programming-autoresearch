"""Step 1: a fresh headless Claude Code session (Opus 5.5, Read tool only, a sandbox holding just the model files)
turns each selected model into a description of at most 200 words. Writes WORK/summaries/<name>.txt and .log.json,
retrying in the same session with a shorten request if the text runs over 200 words."""
import json, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor
from common import MODELS, WORK

TASK = """Subjects listened to narrative stories while fMRI was recorded. For every word, the model receives the 10-gram ending at that word (the word plus up to 9 preceding words) and returns a feature vector; the model is called once per story with all of that story's 10-grams in order. The features are downsampled to the fMRI time points, z-scored, delayed by 1 to 4 time points, and ridge regression maps them to each voxel's response. Performance is the mean correlation between predicted and measured responses across voxels on held-out stories."""
PROMPT = """Below is a task, and a file model.py in your working directory (plus any helper files it loads from the same folder). model.py was hand-written by an earlier agent, with no training, to turn text into features that predict fMRI responses.

TASK: {task}

Read model.py and its helpers. Then write a description of what this model computes, for an engineer who will have to rebuild a model whose features predict the brain about as well AND carry the same information, from scratch, without ever seeing the code or its numbers. Rules:
- At most 200 words, plain prose, no code blocks, no pasted lists longer than a few example words.
- Describe the features the model computes (word categories or lists and how they are chosen, how features are pooled over the words of the 10-gram or over the story, time constants, dimensionality and scaling that matter), not the raw code. Practical hints are welcome.
- Say where the model is approximate, and name any external resources or libraries it relies on.
Output only the description."""


def words(s):
    return len(s.split())


def claude(prompt, cwd, resume=None):
    cmd = ['claude', '-p', prompt, '--model', 'claude-opus-5-5', '--restricted', '--tools', 'Read', '--strict-mcp-config', '--output-format', 'json']
    if resume:
        cmd += ['--resume', resume]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1800)
    return json.loads(r.stdout)


def run(name):
    out = WORK / 'summaries' / f'{name}.txt'
    if out.exists():
        return name, 'cached'
    cwd = WORK / 'sandbox' / f'sum-{name}'
    shutil.rmtree(cwd, ignore_errors=True)
    shutil.copytree(WORK / 'originals' / name, cwd)
    logs = [claude(PROMPT.format(task=TASK), cwd)]
    text = logs[-1]['result'].strip()
    for _ in range(3):
        if words(text) <= 200:
            break
        logs.append(claude(f'That is {words(text)} words. Rewrite it in at most 190 words, keeping the most important information. Output only the description.', cwd, logs[0]['session_id']))
        text = logs[-1]['result'].strip()
    out.write_text(text + '\n')
    meta = dict(words=words(text), cost=sum(l.get('total_cost_usd', 0) for l in logs), turns=[l.get('num_turns') for l in logs],
                seconds=sum(l.get('duration_ms', 0) for l in logs) / 1000)
    (WORK / 'summaries' / f'{name}.log.json').write_text(json.dumps(meta))
    return name, f"{meta['words']} words, ${meta['cost']:.2f}"


if __name__ == '__main__':
    (WORK / 'summaries').mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        for n, m in ex.map(run, [n for _, n in MODELS]):
            print(n, m, flush=True)
