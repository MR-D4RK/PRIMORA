"""M3 Batch 2: domain-independent, evidence-only episode organization."""

import sqlite3

import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.episodes import (
    experience_subject,
    organize_experience_events,
)
from PRIMORA.prime.context.store import ContextStore


def add(manager, session, text, timestamp):
    issues = manager.reconcile([{'role': 'user', 'content': text, 'timestamp': timestamp}], session)
    assert not [x for x in issues if 'episode_capture_failed' in x], issues


@pytest.mark.parametrize('object_name', [
    'synthwave remix', 'Verse interface', 'cooking schedule',
    'travel itinerary', 'art installation', 'music metadata catalog',
])
def test_general_episode_links_across_sessions(object_name, tmp_path):
    cm = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False)
    add(cm, 'first', f'I started working on a {object_name}', 1700000000)
    add(cm, 'second', f'We tested the {object_name}', 1700000300)
    add(cm, 'third', f'I finished my {object_name}', 1700000600)
    episodes = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False).experience_episodes()
    assert len(episodes) == 1
    episode, = episodes
    assert len(episode.events) == 3
    assert [e.kind for e in episode.events] == ['start', 'attempt', 'completion']
    assert episode.session_ids == ('first', 'second', 'third')
    assert len(set(episode.source_message_ids)) == 3
    assert episode.first_reported_at == 1700000000
    assert episode.last_reported_at == 1700000600
    for ev in episode.events:
        assert ContextStore(tmp_path / 'memory.db').fetch_message(ev.source_message_id).content == ev.statement


def test_unrelated_experiences_never_merge_due_to_shared_words(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False)
    add(cm, 's1', 'I started the task panel', 1700000000)
    add(cm, 's2', 'I finished the task tracker', 1700000100)
    add(cm, 's3', 'We tested the panel', 1700000200)
    groups = cm.experience_episodes()
    assert len(groups) == 3
    assert len({g.episode_id for g in groups}) == 3
    assert all(len(g.events) == 1 for g in groups)


def test_multiple_projects_different_eras_do_not_blend(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False)
    add(cm, 'old', 'I started my art installation', 1700000000)
    add(cm, 'new', 'I finished the art installation', 1700000000 + 80 * 86400)
    assert len(cm.experience_episodes(max_gap_days=30)) == 2
    assert len(cm.experience_episodes(max_gap_days=90)) == 1


def test_replay_is_idempotent_and_read_only(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False)
    snapshot = [{'role': 'user', 'content': 'We tested the UX prototype', 'timestamp': 1700000000}]
    cm.reconcile(snapshot, 's')
    one, = cm.experience_episodes()
    cm.reconcile(snapshot, 's')
    two, = cm.experience_episodes()
    assert one == two
    with sqlite3.connect(tmp_path / 'memory.db') as con:
        before = con.execute('SELECT count(*) FROM experience_events').fetchone()[0]
    cm.experience_episodes()
    with sqlite3.connect(tmp_path / 'memory.db') as con:
        after = con.execute('SELECT count(*) FROM experience_events').fetchone()[0]
    assert before == after == 1


def test_untrusted_or_corrupt_records_excluded(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False)
    add(cm, 's', 'I finished the project report', 1700000000)
    add(cm, 's2', 'I started the garden restoration', 1700000001)
    stored = cm.experience_events()
    assert len(stored) == 2
    with sqlite3.connect(tmp_path / 'memory.db') as con:
        con.execute('UPDATE experience_events SET kind = ? WHERE source_message_id = ?',
                    ('milestone', stored[0].source_message_id))
        con.execute('UPDATE experience_events SET reported_at = ? WHERE source_message_id = ?',
                    (12345, stored[1].source_message_id))
    assert cm.experience_episodes() == []


def test_role_and_raw_source_required(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False)
    add(cm, 's', 'I finished the prototype', 1700000000)
    assert len(cm.experience_episodes()) == 1
    with sqlite3.connect(tmp_path / 'memory.db') as con:
        con.execute("UPDATE messages SET role='assistant'")
    assert cm.experience_episodes() == []


def test_session_filter_and_event_order(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False)
    add(cm, 's1', 'I started the mural', 1700000000)
    add(cm, 's2', 'I finished the mural', 1700000010)
    add(cm, 's3', 'I finished the paper sculpture', 1700000020)
    assert [len(e.events) for e in cm.experience_episodes()] == [1, 2]
    assert len(cm.experience_episodes(session_id='s1')) == 1
    assert cm.experience_episodes(session_id='missing') == []
    assert cm.experience_episodes(limit=1)[0].subject == 'paper sculpture'


def test_malformed_and_ambiguous_input_not_grouped():
    assert experience_subject('I finished it') is None
    assert experience_subject('I fixed that') is None
    assert experience_subject('I will finish the report') is None
    assert experience_subject('I finished the report') == ('report', 'report')
    with pytest.raises(ValueError):
        organize_experience_events([], max_gap_days=0)
