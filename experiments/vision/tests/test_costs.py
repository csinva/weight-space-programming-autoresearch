import json
from types import SimpleNamespace

import pytest

from search import costs


@pytest.fixture
def prices(tmp_path, monkeypatch):
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'models': {
        'codex-model': {'input': 2, 'cached_input': .2, 'output': 10,
                        'tiers': [{'above_input_tokens': 272000, 'input': 4, 'cached_input': .4, 'output': 15}]},
        'claude-model': {'input': 5, 'cached_input': .5, 'cache_write_5m': 6.25,
                         'cache_write_1h': 10, 'output': 25},
        'agy-model': {'input': 2, 'cached_input': .2, 'output': 12},
    }}))
    monkeypatch.setattr(costs, 'PRICING', source)
    return tmp_path / 'run'


def session(run, provider='codex', role='proposer'):
    return costs.UsageSession(run, SimpleNamespace(provider=provider, model=f'{provider}-model'), role, '/work')


def native(input, cached, output, total=None):
    last = {'input_tokens': input, 'cached_input_tokens': cached, 'output_tokens': output,
            'reasoning_output_tokens': output // 2}
    return {'type': 'event_msg', 'payload': {'type': 'token_count', 'info': {
        'last_token_usage': last, 'total_token_usage': total or last}}}


def test_codex_per_request_tiers_dedup_and_reasoning(prices):
    s = session(prices)
    first = native(272000, 200000, 100)
    s.observe_codex(first)
    s.observe_codex(first)  # repeated rate-limit update, not another billed request
    cumulative = {'input_tokens': 572001, 'cached_input_tokens': 400000,
                  'output_tokens': 300, 'reasoning_output_tokens': 150}
    s.observe_codex(native(300001, 200000, 200, cumulative))
    s.observe({'type': 'turn.completed', 'usage': cumulative})
    result = s.finish(0)
    assert result['complete'] and result['priced']
    assert result['tokens']['output'] == 300
    assert result['reasoning_tokens'] == 150
    assert result['usd'] == pytest.approx((72000*2+200000*.2+100*10+100001*4+200000*.4+200*15)/1e6)
    assert len(result['requests']) == 2


def test_codex_aggregate_cannot_select_context_tier(prices):
    s = session(prices)
    s.observe({'type': 'turn.completed', 'usage': {'input_tokens': 600000, 'output_tokens': 100}})
    result = s.finish(0)
    assert result['complete']
    assert not result['priced'] and result['usd'] is None


def test_codex_interruption_and_missing_native_requests(prices):
    s = session(prices)
    s.observe_codex(native(100, 50, 20))
    result = s.finish(-15)
    assert result['known_usd'] > 0 and result['usd'] is None and not result['complete']
    s = session(prices)
    cumulative = {'input_tokens': 300, 'cached_input_tokens': 150, 'output_tokens': 40}
    s.observe_codex(native(100, 50, 20, cumulative))
    s.observe({'type': 'turn.completed', 'usage': cumulative})
    assert not s.finish(0)['complete']


def test_claude_blocks_dedup_result_supersedes_and_ttls(prices):
    s = session(prices, 'claude')
    message = {'type': 'assistant', 'session_id': 'a', 'message': {
        'id': 'm', 'model': 'claude-model', 'usage': {'input_tokens': 2,
        'cache_read_input_tokens': 100, 'cache_creation_input_tokens': 30,
        'cache_creation': {'ephemeral_5m_input_tokens': 10, 'ephemeral_1h_input_tokens': 20},
        'output_tokens': 3}}}
    s.observe(message)
    s.observe(message)
    assert s._record()['tokens']['input'] == 2
    usage = dict(message['message']['usage'], output_tokens=200, output_tokens_details={'thinking_tokens': 50})
    result_event = {'type': 'result', 'session_id': 'a', 'usage': usage, 'total_cost_usd': .42}
    s.observe(result_event)
    s.observe(result_event)
    result = s.finish(0)
    assert result['tokens']['output'] == 200
    assert result['reasoning_tokens'] == 50
    assert result['reported_usd'] == .42
    assert result['usd'] == pytest.approx((2*5+100*.5+10*6.25+20*10+200*25)/1e6)


def test_claude_native_final_usage_replaces_provisional_without_doublecount(prices):
    s = session(prices, 'claude')
    for usage, sidkey in [(3, 'session_id'), (200, 'sessionId')]:
        s.observe({'type': 'assistant', sidkey: 's', 'message': {'id': 'm', 'usage': {
            'input_tokens': 2, 'output_tokens': usage}}})
    result = s.finish(-15)
    assert result['tokens']['input'] == 2 and result['tokens']['output'] == 200
    assert result['usd'] is None


def test_unavailable_provider_is_unknown_not_free(prices):
    s = session(prices, 'agy')
    s.observe({'event': 'init', 'conversation_id': 'x'})
    result = s.finish(0)
    assert result['usd'] is None and not result['complete'] and not result['priced']


def test_snapshot_idempotent_summaries_restart_and_roles(prices, monkeypatch):
    costs.ensure_pricing(prices)
    frozen = (prices / 'pricing.json').read_bytes()
    costs.PRICING.write_text('{}')
    monkeypatch.setenv('LAB_PROPOSER', 'p01')
    for role in ['proposer', 'coordinator', 'repair']:
        s = session(prices, 'claude', role)
        s.observe({'type': 'result', 'session_id': 'same-id-new-process', 'usage': {'input_tokens': 1, 'output_tokens': 1}})
        s.finish(0)
    a = costs.summarize_run(prices)
    b = costs.summarize_run(prices)
    assert a == b and a['invocations'] == 3
    assert a['tokens']['input'] == 3
    assert a['providers']['claude']['invocations'] == 3
    assert set(a['roles']) == {'proposer', 'coordinator', 'repair'}
    assert a['proposers']['p01']['invocations'] == 3
    assert (prices / 'pricing.json').read_bytes() == frozen
    assert not list(prices.rglob('*.tmp'))


def test_unknown_cache_write_ttl_is_unpriced(prices):
    s = session(prices, 'claude')
    s.observe({'type': 'result', 'session_id': 's', 'usage': {
        'input_tokens': 1, 'cache_creation_input_tokens': 10, 'output_tokens': 1}})
    result = s.finish(0)
    assert result['complete'] and not result['priced'] and result['usd'] is None
    assert result['tokens']['cache_write_unknown'] == 10


def test_malformed_missing_usage_is_not_zero(prices):
    for provider in ['claude', 'codex']:
        s = session(prices, provider)
        s.observe({'type': 'result' if provider == 'claude' else 'turn.completed', 'usage': {}})
        assert s.finish(0)['usd'] is None
    s = session(prices)
    s.observe_codex({'type': 'token_count', 'info': {'last_token_usage': {}, 'total_token_usage': {}}})
    assert not s.finish(0)['complete']


def test_legacy_resume_never_claims_complete_historical_cost(prices):
    (prices / 'logs').mkdir(parents=True)
    (prices / 'logs' / 'p00.jsonl').write_text('{"type":"turn.started"}\n')
    s = session(prices, 'claude')
    s.observe({'type': 'result', 'usage': {'input_tokens': 10, 'output_tokens': 5}})
    s.finish(0)
    result = costs.summarize_run(prices)
    assert result['legacy_untracked_logs']
    assert not result['complete'] and result['usd'] is None
    assert result['known_usd'] > 0


def test_claude_fallback_model_not_priced_as_requested_model(prices):
    s = session(prices, 'claude')
    s.observe({'type': 'assistant', 'message': {'id': 'a', 'model': 'unknown-fallback',
        'usage': {'input_tokens': 10, 'output_tokens': 10}}})
    s.observe({'type': 'result', 'usage': {'input_tokens': 10, 'output_tokens': 10}})
    assert s.finish(0)['usd'] is None


def test_claude_partial_stream_final_output_retained_on_interruption(prices):
    s = session(prices, 'claude')
    s.observe({'type': 'stream_event', 'session_id': 'x', 'event': {'type': 'message_start',
        'message': {'id': 'm', 'model': 'claude-model', 'usage': {'input_tokens': 10, 'output_tokens': 1}}}})
    s.observe({'type': 'stream_event', 'session_id': 'x', 'event': {'type': 'message_delta',
        'usage': {'output_tokens': 50}}})
    s.observe({'type': 'assistant', 'session_id': 'x', 'message': {'id': 'm',
        'usage': {'input_tokens': 10, 'output_tokens': 1}}})
    result = s.finish(-15)
    assert result['tokens']['output'] == 50 and result['tokens']['input'] == 10
    assert not result['complete'] and result['usd'] is None and result['known_usd'] > 0


def test_agy_disjoint_cache_terminal_supersedes_steps(prices):
    s = session(prices, 'agy')
    for index, u in enumerate([
        {'input_tokens': 8727, 'output_tokens': 222, 'thinking_tokens': 33, 'cache_read_tokens': 0},
        {'input_tokens': 2373, 'output_tokens': 323, 'thinking_tokens': 13, 'cache_read_tokens': 6673},
    ]):
        event = {'event': 'step_update', 'step_update': {'conversation_id': 'c',
            'step_index': index, 'state': 'DONE', 'usage': u}}
        s.observe(event)
        s.observe(event)
    assert s._record()['tokens']['input'] == 11100
    assert s._record()['tokens']['cached_input'] == 6673
    s.observe({'event': 'result', 'result': {'usage': {'input_tokens': 11100,
        'output_tokens': 545, 'thinking_tokens': 46, 'cache_read_tokens': 6673}}})
    result = s.finish(0)
    assert result['complete'] and result['tokens']['output'] == 545
    assert result['reasoning_tokens'] == 46
    assert result['usd'] == pytest.approx((11100*2+6673*.2+545*12)/1e6)


def test_codex_native_counter_reset_retains_both_segments(prices):
    s = session(prices)
    s.observe_codex(native(100, 50, 20))
    s.observe_codex(native(50, 25, 10))
    s.observe({'type': 'turn.completed', 'usage': {'input_tokens': 150,
        'cached_input_tokens': 75, 'output_tokens': 30, 'reasoning_output_tokens': 15}})
    result = s.finish(0)
    assert result['complete'] and result['tokens']['input'] == 75
    assert result['tokens']['output'] == 30


def test_claude_result_model_usage_mismatch_unpriced(prices):
    s = session(prices, 'claude')
    s.observe({'type': 'result', 'usage': {'input_tokens': 10, 'output_tokens': 10},
               'modelUsage': {'different-model': {'inputTokens': 10, 'outputTokens': 10}}})
    assert s.finish(0)['usd'] is None


def test_invalid_optional_cache_counter_not_silently_zero(prices):
    s = session(prices, 'claude')
    s.observe({'type': 'result', 'usage': {'input_tokens': 10, 'output_tokens': 10,
                                        'cache_read_input_tokens': 'unknown'}})
    assert s.finish(0)['usd'] is None


@pytest.mark.parametrize('provider,usage', [
    ('codex', {'input_tokens': 10, 'cached_input_tokens': 8, 'cache_write_input_tokens': 3, 'output_tokens': 5}),
    ('codex', {'input_tokens': 10, 'output_tokens': 5, 'reasoning_output_tokens': 6}),
    ('claude', {'input_tokens': 10, 'output_tokens': 5, 'cache_creation_input_tokens': 3,
                'cache_creation': {'ephemeral_5m_input_tokens': 2, 'ephemeral_1h_input_tokens': 2}}),
    ('claude', {'input_tokens': 10, 'output_tokens': 5, 'output_tokens_details': {'thinking_tokens': 6}}),
    ('agy', {'input_tokens': 10, 'output_tokens': 5, 'thinking_tokens': 6}),
])
def test_impossible_counters_do_not_price_clamped_usage(prices, provider, usage):
    s = session(prices, provider)
    if provider == 'codex':
        s.observe_codex({'type': 'token_count', 'info': {'last_token_usage': usage, 'total_token_usage': usage}})
        s.observe({'type': 'turn.completed', 'usage': usage})
    elif provider == 'agy':
        s.observe({'event': 'result', 'result': {'usage': usage}})
    else:
        s.observe({'type': 'result', 'usage': usage})
    result = s.finish(0)
    assert result['invalid_usage'] and not result['complete']
    assert result['usd'] is None and result['known_usd'] == 0


def test_aliases_canonicalized_before_model_mismatch_check(prices):
    source = json.loads(costs.PRICING.read_text())
    source['aliases'] = {'claude-alias': 'claude-model'}
    costs.PRICING.write_text(json.dumps(source))
    s = session(prices, 'claude')
    s.observe({'type': 'assistant', 'message': {'id': 'm', 'model': 'claude-alias',
        'usage': {'input_tokens': 10, 'output_tokens': 5}}})
    s.observe({'type': 'result', 'usage': {'input_tokens': 10, 'output_tokens': 5},
               'modelUsage': {'claude-alias': {'inputTokens': 10, 'outputTokens': 5}}})
    assert s.finish(0)['usd'] is not None


def test_codex_unknown_write_ttl_never_uses_catalog_default(prices):
    s = session(prices)
    usage = {'input_tokens': 100, 'cached_input_tokens': 40,
             'cache_write_input_tokens': 20, 'output_tokens': 5}
    s.observe_codex({'type': 'token_count', 'info': {'last_token_usage': usage, 'total_token_usage': usage}})
    s.observe({'type': 'turn.completed', 'usage': usage})
    result = s.finish(0)
    assert result['tokens']['input'] == 40 and result['tokens']['cache_write_unknown'] == 20
    assert result['complete'] and not result['priced'] and result['usd'] is None


def test_events_without_usage_are_not_corrupt_telemetry(prices):
    s = session(prices, 'agy')
    s.observe({'event': 'step_update', 'step_update': {'state': 'DONE', 'step_type': 'user_input'}})
    s.observe({'event': 'result', 'result': {'usage': {'input_tokens': 10, 'output_tokens': 2}}})
    assert s.finish(0)['complete']
    s = session(prices)
    s.observe_codex({'type': 'token_count', 'info': None})
    assert not s.finish(-15)['invalid_usage']
