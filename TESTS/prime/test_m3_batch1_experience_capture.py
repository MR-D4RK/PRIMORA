"""General-purpose experience capture: no special-case subjects or answers."""

import sqlite3

import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.episodes import extract_experience_claim
from PRIMORA.prime.context.store import ContextStore


@pytest.mark.parametrize('statement,kind', [
    ('We decided to refactor the task panel', 'decision'),
    ('I finished my synthwave remix', 'completion'),
    ('We fixed the memory retrieval issue', 'resolution'),
    ('I started practicing piano', 'start'),
    ('We tested the new prototype', 'attempt'),
    ('We worked on the Verse interface', 'activity'),
    ('I discussed my travel plans', 'discussion'),
    ('We published a research note', 'milestone'),
    ('I stopped training that adapter', 'end'),
    ('We failed the first integration test', 'setback'),
])
def test_domain_independent_explicit_events(statement, kind):
    claim = extract_experience_claim(statement)
    assert claim is not None
    assert claim.kind == kind
    assert claim.statement == statement


@pytest.mark.parametrize('statement', [
    'Can we finish the remix?',
    'If we finished the remix, it would be great',
    'I plan to finish the remix',
    'I will fix the task panel',
    "I didn't finish the remix",
    'We should test the model',
    'Maybe we decided to do something',
    '"We finished the task"',
    'The assistant finished the work',
    'I completed',
    'We fixed the issue? Why?',
    'I love the new interface',
])
def test_no_unfounded_event_claim(statement):
    assert extract_experience_claim(statement) is None
    assert extract_experience_claim('We finished the remix', 'assistant') is None


def test_capture_cross_session_source_timestamps_and_reconcile_idempotence(tmp_path):
    manager = ContextManager(db_path=tmp_path / 'mem.db', hybrid_retrieval=False)
    first = [{'role': 'user', 'content': 'I finished the remix', 'timestamp': 1720000000.0},
             {'role': 'assistant', 'content': 'We fixed the bass', 'timestamp': 1720000001.0}]
    assert not [v for v in manager.reconcile(first, 'session-music') if 'episode_capture_failed' in v]
    assert not [v for v in manager.reconcile(first, 'session-music') if 'episode_capture_failed' in v]
    manager.reconcile([{'role': 'user', 'content': 'We fixed the task panel', 'timestamp': 1720000200.0}], 'session-verse')
    manager.reconcile([{'role': 'user', 'content': 'We decided to change the cooking schedule', 'timestamp': 1720000300.0}], 'session-life')

    other_process = ContextManager(db_path=tmp_path / 'mem.db', hybrid_retrieval=False)
    records = other_process.experience_events()
    assert len(records) == 3
    assert {e.kind for e in records} == {'completion', 'resolution', 'decision'}
    assert [e.reported_at for e in records] == [1720000000.0, 1720000200.0, 1720000300.0]
    assert [e.session_id for e in records] == ['session-music', 'session-verse', 'session-life']
    store = ContextStore(tmp_path / 'mem.db')
    for e in records:
        msg = store.fetch_message(e.source_message_id)
        assert msg is not None and msg.role == 'user'
        assert msg.content == e.statement
        assert msg.timestamp == e.reported_at
    assert len(other_process.experience_events(kind='resolution')) == 1
    assert len(other_process.experience_events(session_id='session-music')) == 1


def test_bogus_and_assistant_provenance_rejected(tmp_path):
    store = ContextStore(tmp_path / 'mem.db')
    user_id, _, _ = store.append('s', 0, 'user', 'I finished the report', timestamp=1700000000)
    assistant_id, _, _ = store.append('s', 1, 'assistant', 'We decided to fake an event')
    for msgid, sid, statement in [
        (assistant_id, 's', 'We decided to fake an event'),
        (user_id, 'wrong', 'I finished the report'),
        (user_id, 's', 'I finished a different report'),
        (999999, 's', 'I finished the report'),
    ]:
        with pytest.raises(ValueError):
            store.save_experience_event(sid, msgid, 'completion', 'i', statement)
    assert store.fetch_experience_events() == []


def test_existing_schema_and_user_data_preserved(tmp_path):
    path = tmp_path / 'old.db'
    store = ContextStore(path)
    raw_id, _, _ = store.append('old', 0, 'user', 'My favorite color is yellow')
    manager = ContextManager(db_path=path, hybrid_retrieval=False)
    manager.reconcile([{'role': 'user', 'content': 'I finished another project'}], 'new')
    assert store.fetch_message(raw_id).content == 'My favorite color is yellow'
    assert len(manager.experience_events()) == 1
    with sqlite3.connect(path) as db:
        assert 'experience_events' in {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert db.execute('SELECT count(*) FROM messages WHERE id=?', (raw_id,)).fetchone()[0] == 1


def test_retrieval_requires_exact_user_source_even_if_event_db_tampered(tmp_path):
    store = ContextStore(tmp_path / 'mem.db')
    mid, _, _ = store.append('s', 0, 'user', 'I finished a task')
    store.save_experience_event('s', mid, 'completion', 'i', 'I finished a task')
    with sqlite3.connect(tmp_path / 'mem.db') as db:
        db.execute("UPDATE experience_events SET statement='injected' WHERE source_message_id=?", (mid,))
    assert store.fetch_experience_events() == []
