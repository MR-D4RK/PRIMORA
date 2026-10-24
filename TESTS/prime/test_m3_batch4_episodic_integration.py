"""Milestone 3 Batch 4: cross-session episodic integration and safeguards.

No subjects in this file appear in production routing rules. The same parser,
source verification, grouping and bounded rendering apply to all of them.
"""
import sqlite3

import pytest

from PRIMORA.aurora.brain import BrainResult
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.episode_retrieval import parse_episode_question
from PRIMORA.prime.context.mind import Mind


def report(manager, session, statement, timestamp):
    errors = manager.reconcile(
        [{'role': 'user', 'content': statement, 'timestamp': float(timestamp)}],
        session)
    assert not [e for e in errors if 'episode_capture_failed' in e]


@pytest.mark.parametrize('subject', [
    'glass greenhouse', 'planetarium website', 'indigo dance remix',
    'task overview board', 'floating lantern garden', 'fiction research notebook',
])
def test_history_and_completion_with_user_source_across_sessions(tmp_path, subject):
    path = tmp_path / 'aurora.db'
    cm = ContextManager(db_path=path, hybrid_retrieval=False)
    report(cm, 'one', f'I started the {subject}', 1730000000)
    report(cm, 'two', f'We tested the {subject}', 1730003600)
    report(cm, 'three', f'I finished the {subject}', 1730010800)
    reopened = ContextManager(db_path=path, hybrid_retrieval=False)
    answer = reopened.verified_episode_history(f'Do you remember our work on the {subject}?')
    assert subject in answer and 'started' in answer and 'tested' in answer
    assert 'finished' in answer
    assert answer.count('source message ') == 3
    finished = reopened.verified_episode_history(
        'When did we finish it?', context=(f'Do you remember our work on the {subject}?',))
    assert 'finished' in finished and 'started' not in finished
    assert 'UTC' in finished


def test_natural_followup_resolves_only_immediate_named_user_question(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False)
    report(cm, 's1', 'I started the copper sculpture', 1710000000)
    report(cm, 's2', 'We finished the copper sculpture', 1710000600)
    report(cm, 's3', 'We completed the star chart', 1710000900)
    previous = 'What happened with the copper sculpture?'
    reply = cm.verified_episode_history('What happened next?', context=(previous,))
    assert 'copper sculpture' in reply and 'star chart' not in reply
    assert 'started' in reply and 'finished' in reply
    # Neither unrelated intervening user message nor old database records can
    # silently resolve an unbound pronoun in a later conversation.
    unclear = cm.verified_episode_history(
        'What happened next?', context=(previous, 'I like violet',))
    assert 'Which activity' in unclear
    assert 'copper sculpture' not in unclear
    new_session = cm.verified_episode_history('Did we finish it?')
    assert 'Which activity' in new_session


def test_older_matching_experience_survives_newer_episode_volume(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False)
    report(cm, 'old', 'We finished the moonstone journal', 1700000000)
    for i in range(115):
        report(cm, f's{i}', f'I tested the unnamed artifact {i}', 1700000300 + i)
    response = cm.verified_episode_history('What happened with the moonstone journal?')
    assert 'moonstone journal' in response
    assert 'artifact' not in response
    assert 'source message ' in response


def test_unverified_records_cannot_pollute_cross_session_followups(tmp_path):
    db = tmp_path / 'memory.db'
    cm = ContextManager(db_path=db, hybrid_retrieval=False)
    report(cm, 'real', 'We tested the ice bridge prototype', 1720000000)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE experience_events SET statement='We finished the ice bridge prototype'")
    answer = cm.verified_episode_history('What happened with the ice bridge prototype?')
    assert "don't have a verified" in answer
    assert 'finished' not in answer


class FakeBrain:
    def __init__(self):
        self.calls = 0

    def get_model_name(self):
        return 'test-model'

    def generate(self, *args, **kwargs):
        self.calls += 1
        return BrainResult.success('Invented an unrelated experience', 'test-model')

    def count_tokens(self, messages):
        return sum(len(m.get('content', '')) // 4 + 1 for m in messages)


def test_real_loop_answer_and_followup_no_inference_or_research(tmp_path):
    db = tmp_path / 'memory.db'
    cm = ContextManager(db_path=db, hybrid_retrieval=False)
    report(cm, 'past', 'We started the ceramic lantern', 1710000000)
    report(cm, 'past2', 'I finished the ceramic lantern', 1710000100)
    brain = FakeBrain()
    loop = CognitiveLoop(Mind(ContextManager(db_path=db, hybrid_retrieval=False)), brain)
    first = 'What happened with the ceramic lantern?'
    res1 = loop.run([{'role': 'system', 'content': 'Be truthful'},
                     {'role': 'user', 'content': first}], session_id='new')
    assert res1.success and 'ceramic lantern' in res1.content
    second = 'When did we finish it?'
    streamed = []
    res2 = loop.run([{'role': 'system', 'content': 'Be truthful'},
                     {'role': 'user', 'content': first},
                     {'role': 'assistant', 'content': res1.content},
                     {'role': 'user', 'content': second}],
                    session_id='new', on_delta=streamed.append)
    assert res2.success and 'finished the ceramic lantern' in res2.content
    assert 'started the ceramic lantern' not in res2.content
    assert streamed == [res2.content]
    assert brain.calls == 0
    assert len(cm.experience_events()) == 2


def test_unrelated_requests_are_not_hijacked():
    assert parse_episode_question('Find a working music app') is None
    assert parse_episode_question('What do I like recently?') is None
    assert parse_episode_question('How do I install the prototype?') is None
    assert parse_episode_question('When did I tell you about the prototype?') is None
    assert parse_episode_question('Did we finish it?', context=()).needs_subject
