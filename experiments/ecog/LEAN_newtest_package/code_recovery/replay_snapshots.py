"""Recover the exact wave_encoder.py behind every logged autoresearch trial from the agent logs.

Many trials were never snapshotted (the Jun 6+ grid missions saved only kept versions) and some
snapshots were later overwritten. The stream-json worker logs record every tool call, and every
successful Edit result carries `originalFile` (the full file before the edit); every complete Read
carries the full file. So the file the agent evaluated can be reconstructed by replaying the run:

  state  = content of <run>/wave_encoder.py (None = unknown)
  Edit   -> anchor on originalFile, then apply old->new      Write -> content
  Read   -> anchor (full reads only)                          cp X wave_encoder.py -> content of X
  other Bash writes to wave_encoder.py -> unknown until the next anchor
  python wave_encoder.py ... -> an eval: snapshot = state (or, if unknown, the next anchor
                                provided nothing modified the file in between)

Each anchor also checks the replayed state; an eval counts as `verified` when a later anchor,
with no modification in between, equals its snapshot. The score the agent read back after an
eval (test_corr in run.log / the results row) ties evals to logged rows. Output:
snapshots/<run>/<k>.py and replay_<run>.json; matching to rows happens in make_trials_newtest.py.
"""
import glob
import hashlib
import json
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RW = os.path.join(REPO, 'runs-wave')
LOGS = os.path.join(REPO, 'autoresearch_loop', 'logs')
SNAP = os.path.join(HERE, 'snapshots')

EVAL = re.compile(r'(?:^|[\s;&(])(?:[\w./-]*python3?)\s+((?:\S*/)?wave_encoder\.py)((?:\s+[^\s>|;&]+)*)')
CP = re.compile(r'^\s*cp\s+(?:-\w+\s+)*([^\s;&|]+)\s+([^\s;&|]+)\s*$')
WRITE_HINT = re.compile(r'sed\s+-i|perl\s+-pi|>\s*\S*wave_encoder\.py|\bmv\s|open\([^)]*wave_encoder|write_text|'
                        r'\.write\(|git\s+(checkout|restore|stash)|\bpatch\b|replace\(')
SCORE = re.compile(r'test_corr[:=]\s*(-?\d+\.\d+)')
NAME = re.compile(r'^model_shorthand_name\s*=\s*[\'"]([^\'"]+)[\'"]', re.M)


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(c.get('text', '') for c in content if isinstance(c, dict))
    return ''


def run_of_log(path):
    for line in open(path, errors='ignore'):
        if '"subtype":"init"' in line:
            cwd = json.loads(line).get('cwd', '')
            if '/runs-wave/' in cwd:
                return cwd.rstrip('/').split('/runs-wave/')[1].split('/')[0]
    return None


def jobid(p):
    t = os.path.basename(p).rsplit('_', 1)[-1].split('.')[0]
    return int(t) if t.isdigit() else 0


class Replay:
    def __init__(self, run):
        self.run, self.rd = run, os.path.join(RW, run)
        self.state = None          # wave_encoder.py content
        self.lib = {}              # replayed contents of other files written in the run dir
        self.evals = []
        self.pending = []          # evals whose snapshot awaits the next anchor
        self.clean_since = []      # evals not yet verified, file untouched since
        self.n_diverged = 0

    def rel(self, p):
        p = p.strip('\'"')
        if p.startswith(self.rd + '/'):
            return p[len(self.rd) + 1:]
        return p if not p.startswith('/') else p   # absolute path outside the run dir stays absolute

    def content(self, p):
        r = self.rel(p)
        if r == 'wave_encoder.py':
            return self.state
        if r in self.lib:
            return self.lib[r]
        full = r if r.startswith('/') else os.path.join(self.rd, r)
        return open(full, errors='ignore').read() if os.path.isfile(full) else None

    # --- events -------------------------------------------------------------------------
    def anchor(self, text, source):
        norm = lambda t: None if t is None else t.rstrip('\n')
        for e in self.pending:
            e['content'], e['how'] = text, f'backfill:{source}'
        self.pending = []
        for e in self.clean_since:
            e['verified'] = norm(e['content']) == norm(text)
        self.clean_since = []
        if self.state is not None and norm(self.state) != norm(text):
            self.n_diverged += 1
        self.state = text

    def modified(self):
        for e in self.pending:
            e['how'] = 'lost'
        self.pending = []
        self.clean_since = []

    def on_edit(self, inp, tur):
        if os.path.basename(inp.get('file_path', '')) != 'wave_encoder.py' or self.rel(inp['file_path']) != 'wave_encoder.py':
            return
        orig = (tur or {}).get('originalFile')
        if isinstance(orig, str):
            self.anchor(orig, 'edit')
        cur = self.state
        self.modified()
        if cur is None:
            return
        edits = inp.get('edits') or [dict(old_string=inp.get('old_string', ''), new_string=inp.get('new_string', ''),
                                          replace_all=inp.get('replace_all', False))]
        for ed in edits:
            old, new = ed.get('old_string', ''), ed.get('new_string', '')
            if old not in cur:
                self.state = None
                return
            cur = cur.replace(old, new) if ed.get('replace_all') else cur.replace(old, new, 1)
        self.state = cur

    def on_write(self, inp):
        r = self.rel(inp.get('file_path', ''))
        if r == 'wave_encoder.py':
            self.modified()
            self.state = inp.get('content')
        elif not r.startswith('/'):
            self.lib[r] = inp.get('content')

    def on_read(self, inp, tur):
        f = (tur or {}).get('file') or {}
        if self.rel(f.get('filePath', inp.get('file_path', ''))) != 'wave_encoder.py':
            return
        if f.get('startLine') == 1 and f.get('numLines') == f.get('totalLines') and isinstance(f.get('content'), str):
            c = f['content']
            self.anchor(c if c.endswith('\n') else c + '\n', 'read')

    def on_bash(self, cmd, result, ts):
        for seg in re.split(r'&&|\|\||;|\n', cmd):
            m = CP.match(seg)
            if m:
                src, dst = self.rel(m.group(1)), self.rel(m.group(2))
                if dst in ('.', './') or dst.endswith('/'):
                    dst = os.path.join(dst, os.path.basename(src)).lstrip('./') if dst not in ('.', './') else os.path.basename(src)
                if dst == 'wave_encoder.py':
                    self.modified()
                    self.state = self.content(src)
                elif not dst.startswith('/') and src == 'wave_encoder.py':
                    self.lib[dst] = self.state
                continue
            ev = EVAL.search(seg)
            if ev and self.rel(ev.group(1)) in ('wave_encoder.py', './wave_encoder.py'):
                try:
                    args = shlex.split(ev.group(2))
                except ValueError:
                    args = ev.group(2).split()
                env = dict(re.findall(r'\b([A-Z][A-Z0-9_]+)=(\S+)\s+(?=\S*python)', seg))
                e = dict(k=len(self.evals), args=args, env=env, ts=ts, content=self.state,
                         how='replay' if self.state is not None else None, verified=None,
                         scores=[], crashed=False)
                self.evals.append(e)
                if self.state is None:
                    self.pending.append(e)
                else:
                    self.clean_since.append(e)
                self.last_eval = e
                continue
            if 'wave_encoder.py' in seg and WRITE_HINT.search(seg):
                self.modified()
                self.state = None
        self.note_result(result)

    def note_result(self, result):
        e = getattr(self, 'last_eval', None)
        if e is None or not result:
            return
        for s in SCORE.findall(result):
            e['scores'].append(float(s))
        for line in result.splitlines():
            parts = line.split(',')
            if len(parts) > 6 and parts[0].strip().endswith('patient1'):
                try:
                    e.setdefault('rows_seen', []).append((parts[6], float(parts[1])))
                except ValueError:
                    pass
        if 'Traceback' in result and not e['scores']:
            e['crashed'] = True


def replay_run(run, logs):
    R = Replay(run)
    for path in sorted(logs, key=jobid):
        uses = {}
        for line in open(path, errors='ignore'):
            if not line.startswith('{'):
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get('type') == 'assistant':
                for b in ev['message'].get('content', []):
                    if b.get('type') == 'tool_use':
                        uses[b['id']] = b
            elif ev.get('type') == 'user' and isinstance(ev['message'].get('content'), list):
                for b in ev['message']['content']:
                    if b.get('type') != 'tool_result':
                        continue
                    u = uses.pop(b.get('tool_use_id'), None)
                    if u is None:
                        continue
                    err, inp, tur = b.get('is_error'), u.get('input', {}), ev.get('tool_use_result')
                    tur = tur if isinstance(tur, dict) else None
                    name = u['name']
                    if name in ('Edit', 'MultiEdit') and not err:
                        R.on_edit(inp, tur)
                    elif name == 'Write' and not err:
                        R.on_write(inp)
                    elif name == 'Read' and not err:
                        R.on_read(inp, tur)
                    elif name == 'Bash':
                        R.on_bash(inp.get('command', ''), text_of(b.get('content')), ev.get('timestamp'))
    for e in R.pending:
        e['how'] = 'lost'
    os.makedirs(os.path.join(SNAP, run), exist_ok=True)
    out = []
    for e in R.evals:
        c = e.pop('content')
        e['name'] = None
        if c is not None:
            m = NAME.search(c)
            e['name'] = m.group(1) if m else None
            if '--name' in e['args']:
                e['name'] = e['args'][e['args'].index('--name') + 1]
            e['sha'] = hashlib.sha256(c.encode()).hexdigest()[:16]
            p = os.path.join(SNAP, run, f"{e['k']:04d}_{e['sha']}.py")
            open(p, 'w').write(c)
            e['snapshot'] = p
        out.append(e)
    json.dump(dict(run=run, n_diverged=R.n_diverged, evals=out),
              open(os.path.join(HERE, f'replay_{run}.json'), 'w'), indent=0)
    n = len(out); k = sum(e.get('snapshot') is not None for e in out)
    v = sum(e['verified'] is True for e in out); bad = sum(e['verified'] is False for e in out)
    print(f'{run:24s} evals={n:4d} recovered={k:4d} verified={v:4d} contradicted={bad:3d} '
          f'diverged_anchors={R.n_diverged}', flush=True)


def main():
    runs = sys.argv[1:]
    bylog = {}
    for p in glob.glob(os.path.join(LOGS, 'claude_*.jsonl')):
        r = run_of_log(p)
        if r:
            bylog.setdefault(r, []).append(p)
    for run in runs:
        replay_run(run, bylog.get(run, []))


if __name__ == '__main__':
    main()
