"""Freeze the five agents' deliverables (sha256) and audit their runs, before any scoring.

Freeze: work/feature_space.py, work/FEATURES.md and work/pkgs/ -> frozen/agent_k/, hashes in frozen/HASHES.txt.
Audit (AUDIT.md): every tool call in logs/agent_k.jsonl. Flags (a) any path outside the sandbox view
(/work, /home/agent, /opt, /tmp and system dirs), (b) network use other than `pip install`, (c) any touch of the
credentials file or Claude internals, (d) in feature_space.py: learning/fitting, pretrained models, network
access, hard-coded /work paths (which would not exist at scoring). Also lists installed packages, H, run end
status and the agent's own final summary.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

B = os.path.dirname(os.path.abspath(__file__))
OK_PREFIX = ('/work', '/home/agent', '/opt', '/tmp', '/usr', '/etc', '/proc', '/dev', '/bin', '/lib', '/sbin')
NET = re.compile(r'\b(curl|wget|ssh|scp|sftp|rsync|nc|ncat|telnet|git\s+clone|urllib|requests\.|http\.client|socket\.|'
                 r'aiohttp|huggingface|hf_hub|from_pretrained|torch\.hub)\b')
SECRET = re.compile(r'credentials|\.claude|anthropic|api\.|oauth|token', re.I)
LEARN = re.compile(r'\b(fit|fit_transform|partial_fit|train\(|PCA|KMeans|GaussianMixture|NMF|ICA|torch\.load|'
                   r'load_state_dict|from_pretrained|hub\.load|pretrained|download)\b')
PATHS = re.compile(r'(?<![\w.])(/[A-Za-z_][\w./-]*)')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def tree_hash(d):
    if not os.path.isdir(d):
        return '-'
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs)
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.relpath(f, d).encode()); h.update(sha(f).encode())
    return h.hexdigest()


def main(ks):
    os.makedirs(f'{B}/frozen', exist_ok=True)
    hashes, report = [], ['# Textbook-baseline audit', '']
    for k in ks:
        w, fz = f'{B}/agent_{k}/work', f'{B}/frozen/agent_{k}'
        shutil.rmtree(fz, ignore_errors=True); os.makedirs(fz)
        for f in ('feature_space.py', 'FEATURES.md'):
            if os.path.exists(f'{w}/{f}'):
                shutil.copy2(f'{w}/{f}', f'{fz}/{f}')
        if os.path.isdir(f'{w}/pkgs'):
            shutil.copytree(f'{w}/pkgs', f'{fz}/pkgs', symlinks=True)
        hashes.append(f'agent_{k} feature_space.py {sha(fz + "/feature_space.py") if os.path.exists(fz + "/feature_space.py") else "MISSING"} '
                      f'FEATURES.md {sha(fz + "/FEATURES.md") if os.path.exists(fz + "/FEATURES.md") else "MISSING"} pkgs {tree_hash(fz + "/pkgs")}')
        flags, pips, n_tools, final, end = [], [], 0, '', ''
        for line in open(f'{B}/logs/agent_{k}.jsonl', errors='ignore'):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get('type') == 'result':
                final, end = str(e.get('result', ''))[:1500], f"subtype={e.get('subtype')} is_error={e.get('is_error')} turns={e.get('num_turns')}"
            if e.get('type') != 'assistant':
                continue
            for b in e['message'].get('content', []):
                if b.get('type') != 'tool_use':
                    continue
                n_tools += 1
                inp = b.get('input', {})
                txt = inp.get('command', '') if b['name'] == 'Bash' else inp.get('file_path', '')
                if b['name'] == 'Bash' and re.search(r'(^|[;&|]\s*)pip\s+install', txt):
                    pips.append(re.sub(r'\s+', ' ', txt)[:200])
                for p in PATHS.findall(txt):
                    if not p.startswith(OK_PREFIX):
                        flags.append(f'path outside sandbox view: {b["name"]}: {p}')
                if NET.search(txt) and 'pip install' not in txt:
                    flags.append(f'network-like call: {b["name"]}: {txt[:160]!r}')
                if SECRET.search(txt):
                    flags.append(f'credentials/Claude internals: {b["name"]}: {txt[:160]!r}')
        code = open(f'{fz}/feature_space.py').read() if os.path.exists(f'{fz}/feature_space.py') else ''
        code_flags = [f'line {i + 1}: {l.strip()[:120]}' for i, l in enumerate(code.splitlines())
                      if (LEARN.search(l) or NET.search(l) or '/work' in l) and not l.strip().startswith('#')]
        report += [f'## Agent {k}', '', f'- run: {end}; tool calls: {n_tools}',
                   f'- pip installs: {len(pips)}' + ''.join(f'\n  - `{p}`' for p in pips),
                   f'- sandbox/network/credential flags: {len(flags)}' + ''.join(f'\n  - {f}' for f in sorted(set(flags))[:40]),
                   f'- feature_space.py lines to review (learning/pretrained/network/hard-coded /work): {len(code_flags)}'
                   + ''.join(f'\n  - {f}' for f in code_flags[:40]), '', '**Agent summary:**', '', final, '']
    open(f'{B}/frozen/HASHES.txt', 'w').write('\n'.join(hashes) + '\n')
    open(f'{B}/AUDIT.md', 'w').write('\n'.join(report) + '\n')
    print('\n'.join(hashes)); print(f'audit -> {B}/AUDIT.md')


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]] or [1, 2, 3, 4, 5])
