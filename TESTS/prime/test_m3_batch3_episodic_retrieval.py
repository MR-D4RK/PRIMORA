"""M3 Batch 3: general episodic lookup with source-backed, cross-domain facts."""
import sqlite3

import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.episode_retrieval import parse_episode_question
from PRIMORA.prime.context.mind import Mind
from PRIMORA.aurora.brain import BrainResult
from PRIMORA.aurora.loop import CognitiveLoop


def report(manager, session, text, at):
    messages = [{'role': 'user', 'content': text, 'timestamp': float(at)}]
    assert not any('episode_capture_failed' in msg for msg in manager.reconcile(messages, session))


@pytest.mark.parametrize('subject', [
    'synthwave remix', 'Verse interface', 'garden renovation',
    'physics experiment', 'travel itinerary', 'unknown lantern project',
])
def test_cross_session_retrieval_for_unrelated_and_unseen_domains(tmp_path, subject):
    path = tmp_path / 'mem.db'
    cm = ContextManager(db_path=path, hybrid_retrieval=False)
    report(cm, 'a', f'I started working on the {subject}', 1730000000)
    report(cm, 'b', f'We tested the {subject}', 1730000100)
    report(cm, 'c', f'I finished the {subject}', 1730000200)
    separate = ContextManager(db_path=path, hybrid_retrieval=False)
    answer = separate.verified_episode_history(f'Do you remember when we worked on the {subject}?')
    assert answer is not None
    for verb in ['started', 'tested', 'finished']:
        assert verb in answer
    assert subject in answer
    assert 'UTC' in answer and 'source message' in answer
    assert 'not necessarily when they occurred' in answer


def test_only_relevant_episode_and_no_hallucinated_outcome(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)
    report(cm, 'a', 'We tested the task panel', 1710000000)
    report(cm, 'b', 'We finished the task tracker', 1710000100)
    report(cm, 'c', 'I fixed the garden fence', 1710000200)
    got = cm.verified_episode_history('What happened with the task panel?')
    assert 'tested the task panel' in got
    assert 'finished the task tracker' not in got
    assert 'garden fence' not in got
    assert 'finished the task panel' not in got
    assert 'tested' in got


def test_uncertainty_and_no_made_up_history(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)
    report(cm, 'a', 'I tested the notebook interface', 1710000000)
    got = cm.verified_episode_history('What happened with the telescope mount?')
    assert 'don\'t have a verified' in got
    assert 'notebook' not in got


def test_user_provenance_and_read_only(tmp_path):
    path = tmp_path / 'm.db'
    cm = ContextManager(db_path=path, hybrid_retrieval=False)
    report(cm, 'a', 'I completed the coffee table restoration', 1710000000)
    with sqlite3.connect(path) as con:
        before = con.execute('SELECT count(*) FROM experience_events').fetchone()[0]
    cm._get_store().append('a', 3, 'assistant', 'I completed the fake project', timestamp=1710000020)
    answer = cm.verified_episode_history('Remember when I completed the coffee table restoration?')
    assert 'coffee table restoration' in answer
    assert 'fake project' not in answer
    with sqlite3.connect(path) as con:
        after = con.execute('SELECT count(*) FROM experience_events').fetchone()[0]
    assert before == after


def test_no_interception_of_ordinary_facts_or_personal_dates():
    for question in [
        'What are my favorite cars?',
        'When did I tell you about Regera?',
        'Remember when I told you about Apollo IE?',
        'What were we talking about?',
        'Can you research the Verse engine?',
        'I finished my first remix',
        'When is the next meeting?',
    ]:
        assert parse_episode_question(question) is None, question
    assert parse_episode_question('What happened with the remix?')
    assert parse_episode_question('When did we finish the remix?')
    assert parse_episode_question('How did our remix go?')
    assert parse_episode_question('What experiences do you remember?')


def test_general_experience_listing_bounded(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)
    for i in range(12):
        report(cm, f's{i}', f'I finished the activity {i}', 1710000000 + i)
    answer = cm.verified_episode_history('What experiences do you remember?')
    assert answer.count('source message') <= 12
    assert 'activity 11' in answer
    assert len(answer) < 3000


class NoModelNeeded:
    def __init__(self):
        self.calls = 0

    def get_model_name(self):
        return 'test-brain'

    def generate(self, *args, **kwargs):
        self.calls += 1
        return BrainResult.success('model guessed an unrelated answer', 'test-brain')

    def count_tokens(self, messages):
        return sum(len(m['content']) // 4 + 1 for m in messages)


def test_actual_conversation_loop_uses_episodes_and_streams(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)
    report(cm, 'old', 'I started the paper sculpture', 1710000000)
    report(cm, 'second', 'We finished the paper sculpture', 1710000100)
    brain = NoModelNeeded()
    loop = CognitiveLoop(Mind(ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)),
                         brain)
    chunks = []
    result = loop.run([
        {'role': 'system', 'content': 'Be factual.'},
        {'role': 'user', 'content': 'Remember when we finished the paper sculpture?'},
    ], session_id='fresh', on_delta=chunks.append)
    assert result.success and 'paper sculpture' in result.content
    assert 'started' in result.content and 'finished' in result.content
    assert chunks == [result.content]
    assert brain.calls == 0
    # No fabricated autobiographical experience created by Aurora's own answer.
    assert len(cm.experience_events()) == 2


def test_tampered_source_cannot_surface_in_episode_lookup(tmp_path):
    path = tmp_path / 'm.db'
    cm = ContextManager(db_path=path, hybrid_retrieval=False)
    report(cm, 'a', 'We solved the printer configuration issue', 1710000000)
    with sqlite3.connect(path) as con:
        con.execute("UPDATE experience_events SET statement='We solved the spaceship issue'")
    answer = cm.verified_episode_history('What happened with the spaceship issue?')
    assert 'don\'t have a verified' in answer
    answer = cm.verified_episode_history('What happened with the printer configuration issue?')
    assert 'don\'t have a verified' in answer
