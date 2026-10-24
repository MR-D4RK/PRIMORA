"""Milestone 3: generic recent likes, evidence-grouping and completeness regressions.

No test subject is a permitted-domain whitelist. All are arbitrary names.
"""
import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.mind import Mind
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.aurora.brain import BrainResult

NOW = 1792800000.0


def say(manager, session, utterance, timestamp):
    manager.reconcile([{"role": "user", "content": utterance,
                        "timestamp": timestamp}], session)


@pytest.mark.parametrize('question', [
    "What have I told you I liked in the last 2 days?",
    "What have I said I loved in the last 2 days?",
    "What have I told you I like in the last 2 days?",
    "Every interest in the last 2 days]",
    "What are all my interests in the last 2 days?",
])
def test_broad_last_two_days_without_category_clarification(tmp_path, monkeypatch, question):
    monkeypatch.setattr('PRIMORA.prime.context.manager.time.time', lambda: NOW)
    cm = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    say(cm, 'old', 'I love Glass Thistle', NOW - 4 * 86400)
    say(cm, 'new', 'I like Copper Atlas', NOW - 86400)
    answer = cm.verified_likes_history(question)
    assert answer is not None
    assert 'Copper Atlas' in answer
    assert 'Glass Thistle' not in answer
    assert 'Which interest' not in answer
    assert 'last 2 days' in answer


def test_qualifier_variants_group_under_one_named_item(tmp_path, monkeypatch):
    monkeypatch.setattr('PRIMORA.prime.context.manager.time.time', lambda: NOW)
    cm = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    store = cm._get_store()
    subjects = [
        'Saffron Atlas regardless of its shape',
        'Saffron Atlas whether its bright or not',
        'Saffron Atlas too like bright Saffron Atlas',
    ]
    # Legacy records with different concept keys are left intact and grouped
    # only at READ time. The user-authored statements remain individually cited.
    for i, name in enumerate(subjects):
        when = NOW - (i+1)*3600
        statement = 'I love ' + name
        mid, *_ = store.append('older', i, 'user', statement,
                               timestamp=when, importance=0.5)
        store.create_state('older', 'global', 'likes for ' + name.casefold(),
                           'loves ' + name.casefold(), [mid], timestamp=when)
    before = len(store.fetch_states(scope='global', status='current'))
    answer = cm.verified_likes_history('What have I told you I liked in the last 2 days?')
    assert answer.count('- loves saffron atlas (') == 1
    for text in subjects:
        assert 'I love ' + text in answer
    assert len(store.fetch_states(scope='global', status='current')) == before


def test_related_but_distinct_subjects_do_not_merge(tmp_path, monkeypatch):
    monkeypatch.setattr('PRIMORA.prime.context.manager.time.time', lambda: NOW)
    cm = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    say(cm, 'a', 'I love Garden Whistle', NOW - 60)
    say(cm, 'b', 'I love Garden Whistle Studio', NOW - 40)
    result = cm.verified_likes_history('What have I told you I liked recently?')
    assert '- loves Garden Whistle (' in result
    assert '- loves Garden Whistle Studio (' in result


def test_new_preference_never_dropped_from_shorter_window(tmp_path, monkeypatch):
    monkeypatch.setattr('PRIMORA.prime.context.manager.time.time', lambda: NOW)
    cm = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    say(cm, 'earlier', 'I love Northern Prism', NOW - 5*86400)
    say(cm, 'recent', 'I like Cedar Regera', NOW - 16*3600)
    say(cm, 'recent2', 'I love Alpine Engineering', NOW - 10*3600)
    # Another six preferences are newer; order must not suppress Cedar Regera.
    for i in range(6):
        say(cm, f'recent-{i}', f'I love Canary Shelf {i}', NOW - (i+1)*100)
    last7 = cm.verified_likes_history('what have I liked recently?')
    last2 = cm.verified_likes_history('every interest in the last 2 days')
    assert 'Cedar Regera' in last7
    assert 'Cedar Regera' in last2
    assert 'Northern Prism' not in last2
    assert 'Alpine Engineering' in last2


def test_old_evidence_does_not_inherit_newer_qualifier_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr('PRIMORA.prime.context.manager.time.time', lambda: NOW)
    cm = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    store = cm._get_store()
    old, *_ = store.append('one', 0, 'user', 'I love Frost Lantern',
                           timestamp=NOW - 20*86400)
    recent, *_ = store.append('two', 0, 'user', 'I love Frost Lantern regardless of its shape',
                              timestamp=NOW - 86400)
    store.create_state('one', 'global', 'likes for frost lantern',
                       'loves frost lantern', [old], timestamp=NOW - 20*86400)
    store.create_state('two', 'global', 'likes for frost lantern regardless of its shape',
                       'loves frost lantern regardless of its shape', [recent],
                       timestamp=NOW - 86400)
    answer = cm.verified_likes_history('What have I told you I liked in the last 2 days?')
    assert 'regardless of its shape' in answer
    assert '"I love Frost Lantern"' not in answer


class NeverRunBrain:
    def __init__(self): self.calls = 0
    def get_model_name(self): return 'test-model'
    def count_tokens(self, msgs): return sum(len(x.get('content',''))//4 + 1 for x in msgs)
    def generate(self, *args, **kwargs):
        self.calls += 1
        return BrainResult.success('Which interest do you mean?', 'test-model')


def test_live_path_does_not_ask_for_unneeded_clarification(tmp_path, monkeypatch):
    monkeypatch.setattr('PRIMORA.prime.context.manager.time.time', lambda: NOW)
    path = tmp_path / 'm.db'
    cm = ContextManager(db_path=path, hybrid_retrieval=False)
    say(cm, 'older', 'I love Dusk Valley', NOW - 3600)
    brain = NeverRunBrain()
    loop = CognitiveLoop(Mind(ContextManager(db_path=path, hybrid_retrieval=False)),brain)
    reply = loop.run([{'role':'system', 'content':'Accurate answers only.'},
                      {'role':'user', 'content': 'What have I told you I liked in the last 2 days?'}],
                     session_id='fresh')
    assert reply.success and 'Dusk Valley' in reply.content
    assert 'Which interest' not in reply.content
    assert brain.calls == 0
