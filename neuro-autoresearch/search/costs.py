"""Durable, credential-free provider usage and frozen API-price estimates.

Costs are API-equivalent estimates, not subscription bills. Missing telemetry is
never a zero-cost run. Native logs are reduced to numeric usage before persistence.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import threading
import uuid

PRICING = Path(__file__).with_name('pricing.json')
BUCKETS = ('input', 'cached_input', 'cache_write_5m', 'cache_write_1h',
           'cache_write_30m', 'cache_write_12h', 'cache_write_unknown', 'output')


def _atomic(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temp.open('w') as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def _lock(run):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.usage.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def ensure_pricing(run):
    """Freeze the shipped public price snapshot exactly once, including resumes."""
    run = Path(run)
    with _lock(run):
        target = run / 'pricing.json'
        if not target.exists():
            _atomic(target, json.loads(PRICING.read_text()))
        coverage = run / 'usage_coverage.json'
        if not coverage.exists():
            _atomic(coverage, {'legacy_log_bytes': {str(p.relative_to(run)): p.stat().st_size
                for p in (run / 'logs').glob('*') if p.is_file() and p.stat().st_size}})
        return json.loads(target.read_text())


def _int(value):
    return max(0, value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _valid(u, *fields):
    if not isinstance(u, dict):
        return False
    checked = {k: u.get(k) for k in fields}
    checked.update({k: v for k, v in u.items() if k.endswith('_tokens')})
    creation = u.get('cache_creation')
    if creation is not None:
        if not isinstance(creation, dict):
            return False
        checked.update(creation)
    return all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in checked.values())


def _valid_usage(u, provider):
    if not _valid(u, 'input_tokens', 'output_tokens'):
        return False
    output = u['output_tokens']
    if provider == 'codex':
        cached = u.get('cached_input_tokens', u.get('input_cached_tokens', 0))
        if 'cached_input_tokens' in u and 'input_cached_tokens' in u and u['cached_input_tokens'] != u['input_cached_tokens']:
            return False
        return cached + u.get('cache_write_input_tokens', 0) <= u['input_tokens'] and u.get('reasoning_output_tokens', 0) <= output
    if provider == 'claude':
        creation = u.get('cache_creation') or {}
        written = creation.get('ephemeral_5m_input_tokens', 0) + creation.get('ephemeral_1h_input_tokens', 0)
        if 'cache_creation_input_tokens' in u and written > u['cache_creation_input_tokens']:
            return False
        details = u.get('output_tokens_details') or {}
        if not isinstance(details, dict) or not _valid(details):
            return False
        return details.get('thinking_tokens', 0) <= output
    return u.get('thinking_tokens', 0) <= output and ('total_tokens' not in u or u['total_tokens'] == u['input_tokens'] + output)


def _canonical(model, snapshot):
    aliases, seen = snapshot.get('aliases', {}), set()
    while model in aliases and model not in seen:
        seen.add(model)
        model = aliases[model]
    return model


def _usage(u, provider):
    b = dict.fromkeys(BUCKETS, 0)
    if provider == 'claude':
        b['input'] = _int(u.get('input_tokens'))
        b['cached_input'] = _int(u.get('cache_read_input_tokens'))
        creation = u.get('cache_creation') or {}
        b['cache_write_5m'] = _int(creation.get('ephemeral_5m_input_tokens'))
        b['cache_write_1h'] = _int(creation.get('ephemeral_1h_input_tokens'))
        b['cache_write_unknown'] = max(0, _int(u.get('cache_creation_input_tokens')) - b['cache_write_5m'] - b['cache_write_1h'])
        b['output'] = _int(u.get('output_tokens'))
        reasoning = _int((u.get('output_tokens_details') or {}).get('thinking_tokens'))
    elif provider == 'codex':
        b['cached_input'] = _int(u.get('cached_input_tokens', u.get('input_cached_tokens')))
        write = _int(u.get('cache_write_input_tokens'))
        b['cache_write_unknown'] = write
        b['input'] = max(0, _int(u.get('input_tokens')) - b['cached_input'] - write)
        b['output'] = _int(u.get('output_tokens'))
        reasoning = _int(u.get('reasoning_output_tokens'))
    else:
        b['cached_input'] = _int(u.get('cachedContentTokenCount'))
        b['input'] = max(0, _int(u.get('promptTokenCount')) - b['cached_input'])
        reasoning = _int(u.get('thoughtsTokenCount'))
        b['output'] = _int(u.get('candidatesTokenCount')) + reasoning
    return {'tokens': b, 'reasoning_tokens': reasoning,
            'context_tokens': sum(v for k, v in b.items() if k != 'output')}


def _agy_usage(u):
    # Installed agy counters use disjoint uncached input/cache-read buckets.
    row = _usage({'promptTokenCount': u['input_tokens'] + _int(u.get('cache_read_tokens')),
                  'cachedContentTokenCount': _int(u.get('cache_read_tokens')),
                  'candidatesTokenCount': u['output_tokens']}, 'agy')
    row['reasoning_tokens'] = _int(u.get('thinking_tokens'))
    # Native output includes thinking: https://www.antigravity.google/docs/cli/headless/
    # This differs from raw Gemini usageMetadata.candidatesTokenCount.
    row['output_semantics_unverified'] = False
    return row


def _sum(rows):
    result = {'tokens': dict.fromkeys(BUCKETS, 0), 'reasoning_tokens': 0}
    for row in rows:
        for k in BUCKETS:
            result['tokens'][k] += row['tokens'].get(k, 0)
        result['reasoning_tokens'] += row.get('reasoning_tokens', 0)
    return result


def _price(row, model, snapshot, *, aggregate=False):
    if row.get('model_mismatch') or row.get('output_semantics_unverified'):
        return 0.0, False
    model = _canonical(model, snapshot)
    rates = snapshot.get('models', {}).get(model)
    if not isinstance(rates, dict):
        return 0.0, False
    rates = dict(rates)
    tiers = rates.get('tiers', [])
    if aggregate and tiers:
        # Session totals do not reveal any individual request's context tier.
        return 0.0, False
    for tier in sorted(tiers, key=lambda x: x['above_input_tokens']):
        if row.get('context_tokens', 0) > tier['above_input_tokens']:
            rates.update(tier)
    total, priced = 0.0, True
    for bucket, count in row['tokens'].items():
        if not count:
            continue
        rate = rates.get(bucket)
        if not isinstance(rate, (int, float)):
            priced = False
        else:
            total += count * rate / 1_000_000
    return total, priced


class UsageSession:
    """One provider process invocation; every observation is durably replaceable."""
    def __init__(self, run, agent, role, workspace):
        self.run = Path(run)
        self.snapshot = ensure_pricing(run)
        self.provider, self.model = agent.provider, agent.model
        self.id = uuid.uuid4().hex
        self.role = role
        self.workspace = str(workspace)
        self.proposer = os.environ.get('LAB_PROPOSER')
        self._mutex = threading.RLock()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._claude_streams = {}
        self.requests = {}
        self.results = {}
        self.codex_turns = []
        self._codex_seen = set()
        self._codex_previous = None
        self._codex_epoch = 0
        self._codex_epoch_totals = {}
        self._thread = None
        self._turn = 0
        self._terminal = False
        self._code = None
        self._finished = False
        self._invalid_usage = False
        self._save()

    def _accept_usage(self, usage):
        if _valid_usage(usage, self.provider):
            return True
        self._invalid_usage = True
        self._save()
        return False

    def observe(self, event):
        with self._mutex:
            typ = event.get('type', event.get('event'))
            if self.provider == 'codex':
                if typ == 'thread.started':
                    self._thread = event.get('thread_id')
                elif typ == 'turn.started':
                    self._turn += 1
                    self._terminal = False
                elif typ == 'turn.completed' and self._accept_usage(event.get('usage')):
                    key = (self._thread, self._turn)
                    if not any(x['key'] == list(key) for x in self.codex_turns):
                        self.codex_turns.append({'key': list(key), **_usage(event['usage'], 'codex')})
                    self._terminal = True
                else:
                    return
            elif self.provider == 'claude':
                sid = event.get('session_id', event.get('sessionId', 'session'))
                if typ == 'stream_event':
                    stream = event.get('event') or {}
                    kind = stream.get('type')
                    if kind == 'message_start':
                        message = stream.get('message') or {}
                        self._claude_streams[sid] = {'id': message.get('id'), 'model': message.get('model', self.model), 'usage': dict(message.get('usage') or {})}
                    elif kind == 'message_delta' and sid in self._claude_streams:
                        self._claude_streams[sid]['usage'].update(stream.get('usage') or {})
                    elif kind == 'message_stop':
                        return
                    else:
                        return
                    self.observe({'type': 'assistant', 'session_id': sid, 'message': self._claude_streams[sid]})
                    return
                if typ == 'result' and self._accept_usage(event.get('usage')):
                    row = _usage(event['usage'], 'claude')
                    row.update(model=self.model, session=sid, reported_usd=event.get('total_cost_usd'),
                               model_mismatch=any(_canonical(model, self.snapshot) != _canonical(self.model, self.snapshot) for model in (event.get('modelUsage') or {})))
                    self.results[sid] = row
                    self._terminal = True
                elif typ == 'assistant' and isinstance(event.get('message'), dict):
                    message = event['message']
                    if not self._accept_usage(message.get('usage')) or not message.get('id'):
                        return
                    key = f'{sid}:{message["id"]}'
                    row = _usage(message['usage'], 'claude')
                    row.update(model=message.get('model', self.model), session=sid)
                    old = self.requests.get(key)
                    if old:
                        # Blocks repeat initial usage; native transcript may supply final usage.
                        for k in BUCKETS:
                            row['tokens'][k] = max(old['tokens'][k], row['tokens'][k])
                        row['reasoning_tokens'] = max(old['reasoning_tokens'], row['reasoning_tokens'])
                    self.requests[key] = row
                else:
                    return
            else:
                if typ == 'result':
                    result = event.get('result') or {}
                    u = result.get('usage')
                    if not self._accept_usage(u):
                        return
                    self.results['agy'] = {**_agy_usage(u), 'model': self.model}
                    self._terminal = True
                elif typ == 'step_update':
                    step = event.get('step_update') or {}
                    u = step.get('usage')
                    if step.get('state') != 'DONE' or u is None or not self._accept_usage(u):
                        return
                    key = f"{step.get('conversation_id')}:{step.get('step_index')}"
                    self.requests[key] = {**_agy_usage(u), 'model': self.model}
                else:
                    return
            self._save()

    def observe_codex(self, event):
        """Consume native event_msg/token_count records (stdout omits these)."""
        with self._mutex:
            payload = event.get('payload', event)
            if payload.get('type') != 'token_count':
                return
            info = payload.get('info') or {}
            total = info.get('total_token_usage')
            last = info.get('last_token_usage')
            if total is None or last is None:
                return
            if not self._accept_usage(total) or not self._accept_usage(last):
                return
            if self._codex_previous and any(_int(total.get(k)) < _int(self._codex_previous.get(k)) for k in ('input_tokens', 'output_tokens')):
                self._codex_epoch += 1
            stamp = f'{self._codex_epoch}:' + json.dumps(total, sort_keys=True)
            if stamp in self._codex_seen:
                return
            self._codex_seen.add(stamp)
            # Last usage gives the actual per-request prompt size, unlike session totals.
            # Cumulative growth can exceed last when telemetry was missed; flag the gap.
            row = _usage(last, 'codex')
            row.update(model=self.model, cumulative=_usage(total, 'codex'))
            self.requests[f'codex:{len(self.requests)}'] = row
            self._codex_previous = total
            self._codex_epoch_totals[self._codex_epoch] = _usage(total, 'codex')
            self._save()

    def _record(self):
        rows, aggregate = [], False
        complete = self._finished and self._code == 0 and self._terminal and not self._invalid_usage
        if self.provider == 'claude':
            for result in self.results.values():
                result['model_mismatch'] = result.get('model_mismatch', False) or any(r.get('session') == result.get('session') and _canonical(r.get('model'), self.snapshot) != _canonical(self.model, self.snapshot) for r in self.requests.values())
            rows.extend(self.results.values())
            rows.extend(r for r in self.requests.values() if r.get('session') not in self.results)
            # Aggregate result is authoritative; assistant output can be provisional.
            complete = complete and all(r.get('session') in self.results for r in self.requests.values())
            aggregate = True
        elif self.provider == 'codex':
            if self.requests:
                rows = list(self.requests.values())
                native = _sum(rows)
                last_total = _sum(self._codex_epoch_totals.values())
                complete = complete and native['tokens'] == last_total['tokens']
                if self.codex_turns:
                    complete = complete and native['tokens'] == _sum(self.codex_turns)['tokens']
            else:
                rows = self.codex_turns
                aggregate = True
        else:
            rows = list(self.results.values()) if self.results else list(self.requests.values())
        totals = _sum(rows)
        known, priced = 0.0, bool(rows)
        for row in rows:
            amount, ok = _price(row, row.get('model', self.model), self.snapshot, aggregate=aggregate)
            known += amount
            priced = priced and ok
        reported = [r['reported_usd'] for r in self.results.values() if isinstance(r.get('reported_usd'), (int, float))]
        return {'schema_version': 1, 'invocation_id': self.id, 'provider': self.provider,
                'model': self.model, 'role': self.role, 'proposer': self.proposer,
                'started_at': self.started_at, 'updated_at': datetime.now(timezone.utc).isoformat(),
                'provenance': 'native provider usage; frozen run/pricing.json',
                'workspace': self.workspace, 'finished': self._finished, 'exit_code': self._code,
                'complete': bool(complete and rows), 'priced': priced,
                'incomplete_usage': not bool(complete and rows), 'unpriced_usage': not priced,
                'invalid_usage': self._invalid_usage,
                'usd': known if complete and priced and rows else None,
                'known_usd': known, 'reported_usd': sum(reported) if reported else None,
                'requests': rows, **totals,
                'note': 'API-equivalent estimate; known_usd includes observed partial usage, not unobserved usage.'}

    def _save(self):
        _atomic(self.run / 'usage' / f'{self.id}.json', self._record())

    def finish(self, code):
        with self._mutex:
            self._finished, self._code = True, code
            self._save()
            summarize_run(self.run)
            return self._record()


def _aggregate(records):
    totals = _sum(records)
    known = sum(r.get('known_usd', 0) for r in records)
    complete = bool(records) and all(r.get('complete') for r in records)
    priced = bool(records) and all(r.get('priced') for r in records)
    return {**totals, 'invocations': len(records), 'complete': complete, 'priced': priced,
            'known_usd': known, 'usd': known if complete and priced else None}


def summarize_run(run):
    """Recompute from durable invocation records; never add to a previous summary."""
    run = Path(run)
    with _lock(run):
        records = []
        for path in sorted((run / 'usage').glob('*.json')):
            records.append(json.loads(path.read_text()))
        roles, proposers, models, providers = (defaultdict(list) for _ in range(4))
        for r in records:
            roles[r['role']].append(r)
            providers[r['provider']].append(r)
            if r.get('proposer') is not None:
                proposers[r['proposer']].append(r)
            models[r['model']].append(r)
        coverage_path = run / 'usage_coverage.json'
        legacy = json.loads(coverage_path.read_text()).get('legacy_log_bytes', {}) if coverage_path.exists() else {}
        result = {'schema_version': 1, **_aggregate(records),
                  'legacy_untracked_logs': legacy,
                  'roles': {k: _aggregate(v) for k, v in roles.items()},
                  'proposers': {k: _aggregate(v) for k, v in proposers.items()},
                  'models': {k: _aggregate(v) for k, v in models.items()},
                  'providers': {k: _aggregate(v) for k, v in providers.items()},
                  'logs': [str(p.relative_to(run)) for p in sorted((run / 'logs').glob('*')) if p.is_file()],
                  'note': 'Frozen standard API prices; not subscription billing. usd is null when usage or pricing is incomplete.'}
        if legacy:
            result['complete'], result['usd'] = False, None
        _atomic(run / 'usage.json', result)
        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Rebuild a run usage summary from durable invocation records.')
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    if not args.run_directory.is_dir():
        parser.error('run_directory must be an existing directory')
    print(json.dumps(summarize_run(args.run_directory), indent=2, sort_keys=True))
