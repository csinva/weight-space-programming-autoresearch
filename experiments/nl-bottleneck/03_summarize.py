"""Step 1: Opus 5.5 (headless, Read tool only, sandbox holding just the program file) turns each selected program into a <=200-word description.
Writes summaries/<task>.txt and summaries/<task>.log.json. Retries with a shorten request (same session) if over 200 words."""
import json, os, subprocess, sys, shutil, re
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evolve"))
import src.task as T
SB = "/data18/chandan.singh/wsp-nl/sandbox"
PROMPT = """Below is a task, and a file model.py in your working directory. model.py is a small transformer whose weights were written by hand (no training) to solve the task; the file builds the model and sets every weight in closed form.

TASK: {name}
{doc}
Prompts are fed token by token (character level); the model generates the answer autoregressively with greedy decoding and is scored by exact match.

Read model.py. Then write a description of what this model learned/implements, for an engineer who will have to rebuild a model with similar accuracy AND similar mistakes from scratch, without ever seeing model.py or its numbers. Rules:
- At most 200 words, plain prose, no code blocks, no pasted tensors.
- Describe the algorithm the network implements (what attention does, what the MLP or readout does, how the answer is produced), not the raw weights. Practical hints are welcome (e.g. "look up bigrams", which features or words matter, thresholds, scale constants that matter, how many layers/heads).
- Say where the model is known to be imperfect or approximate, if it is, so its error pattern can be matched.
Output only the description."""


def words(s):
    return len(s.split())


def claude(prompt, cwd, resume=None):
    cmd = ["claude", "-p", prompt, "--model", "claude-opus-5-5", "--restricted", "--tools", "Read", "--strict-mcp-config", "--output-format", "json"]
    if resume:
        cmd += ["--resume", resume]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1800)
    return json.loads(r.stdout)


def run(task):
    out = f"{HERE}/summaries/{task}.txt"
    if os.path.exists(out):
        return task, "cached"
    cwd = f"{SB}/sum-{task}"
    shutil.rmtree(cwd, ignore_errors=True); os.makedirs(cwd)
    shutil.copy(f"{HERE}/originals/{task}.py", f"{cwd}/model.py")
    doc = re.sub(r"\n +", "\n", T.get_task(task).__doc__.strip())
    logs = [claude(PROMPT.format(name=task, doc=doc), cwd)]
    text = logs[-1]["result"].strip()
    for _ in range(3):
        if words(text) <= 200:
            break
        logs.append(claude(f"That is {words(text)} words. Rewrite it in at most 190 words, keeping the most important information. Output only the description.", cwd, logs[0]["session_id"]))
        text = logs[-1]["result"].strip()
    open(out, "w").write(text + "\n")
    json.dump(dict(words=words(text), cost=sum(l.get("total_cost_usd", 0) for l in logs), turns=[l.get("num_turns") for l in logs], seconds=sum(l.get("duration_ms", 0) for l in logs) / 1000), open(f"{HERE}/summaries/{task}.log.json", "w"))
    return task, f"{words(text)} words"


if __name__ == "__main__":
    os.makedirs(f"{HERE}/summaries", exist_ok=True)
    tasks = sorted(json.load(open(f"{HERE}/selected.json")))
    with ThreadPoolExecutor(9) as ex:
        for t, m in ex.map(run, tasks):
            print(t, m, flush=True)
