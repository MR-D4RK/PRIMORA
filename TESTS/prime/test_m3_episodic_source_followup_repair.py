"""Cross-domain regression for source-bound old/new experience recall."""

from datetime import datetime, timezone

import pytest

from PRIMORA.aurora.brain import BrainResult
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.episode_retrieval import parse_episode_question
from PRIMORA.prime.context.mind import Mind


class NoBrainCalls:
    def __init__(self):
        self.calls = 0

    def get_model_name(self):
        return "isolated-test-brain"

    def generate(self, *args, **kwargs):
        self.calls += 1
        return BrainResult.success("brain invented an answer", self.get_model_name())

    def count_tokens(self, messages):
        return sum(2 + len(str(msg["content"])) // 4 for msg in messages)


@pytest.mark.parametrize("statement,topic,question", [
    ("recently i have watched VIKINGS series", "vikings", "I watched Vikings"),
    ("I visited the planetarium", "planetarium", "I visited the planetarium"),
    ("We attended the makers festival", "makers festival", "We attended the makers festival"),
    ("I read the blue notebook", "blue notebook", "I read the blue notebook"),
    ("I listened to the dusk symphony", "dusk symphony", "I listened to the dusk symphony"),
    ("We played the tabletop game", "tabletop game", "We played the tabletop game"),
])
def test_general_activities_get_exact_report_dates(tmp_path, statement, topic, question):
    manager = ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)
    timestamp = datetime(2026, 10, 20, 15, 42, tzinfo=timezone.utc).timestamp()
    assert not any("episode_capture_failed" in x for x in manager.reconcile(
        [{"role": "user", "content": statement, "timestamp": timestamp}], "first"))
    reopened = ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)
    answer = reopened.verified_episode_history(f"When did I tell you that {question}?")
    assert statement in answer and "2026-10-20 15:42 UTC" in answer
    assert "source message" in answer and "not necessarily when they occurred" in answer


def test_old_raw_user_report_is_accessible_without_backfill(tmp_path):
    manager = ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)
    store = manager._get_store()
    at = datetime(2026, 10, 23, 15, 22, tzinfo=timezone.utc).timestamp()
    event_id, _, _ = store.append('pre-upgrade', 0, 'user',
                                  'recently i have watched VIKINGS series', 0, timestamp=at)
    store.append('pre-upgrade', 1, 'assistant', 'I remember the release date!', 0,
                 timestamp=at+15)
    reaction_id, _, _ = store.append('pre-upgrade', 2, 'user', 'i loved it', 0,
                                    timestamp=at+60)
    assert not store.fetch_experience_events()

    reopened = ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)
    specific = reopened.verified_episode_history(
        'When exactly did I tell you that I watched Vikings?')
    assert '2026-10-23 15:22 UTC' in specific
    assert 'recently i have watched VIKINGS series' in specific
    general = reopened.verified_episode_history(
        'What do you remember about my experience watching Vikings and how I felt about it?')
    assert 'i loved it' in general
    assert f'source message {reaction_id}' in general
    assert 'release date!' not in general
    assert reopened._get_store().fetch_experience_events() == []
    # No auto-backfill of the user's production memory during retrieval.
    assert event_id != reaction_id


def test_date_followup_only_resolves_single_previous_event(tmp_path):
    manager = ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)
    manager.reconcile([{'role':'user','content':'I watched the canyon documentary',
                        'timestamp': 1792700000}], 's1')
    previous = 'What have I watched recently?'
    answer = manager.verified_episode_history('When did I tell you about it?',
                                               context=(previous,))
    assert 'canyon documentary' in answer and 'UTC' in answer
    manager.reconcile([{'role':'user','content':'I watched the forest documentary',
                        'timestamp': 1792701000}], 's2')
    ambiguous = manager.verified_episode_history('When did I tell you about it?',
                                                  context=(previous,))
    assert 'Which activity' in ambiguous
    assert manager.verified_episode_history('When did I tell you about it?') is None


def test_interception_and_source_date_in_cognitive_loop(tmp_path):
    db_path = tmp_path / 'm.db'
    manager = ContextManager(db_path=db_path, hybrid_retrieval=False)
    manager.reconcile([{'role':'user', 'content':'I visited the paper workshop',
                        'timestamp':1792701000}], 'old-session')
    brain = NoBrainCalls()
    loop = CognitiveLoop(Mind(ContextManager(db_path=db_path, hybrid_retrieval=False)), brain)
    streamed = []
    result = loop.run([
        {'role':'system','content':'Remember accurately.'},
        {'role':'user','content':'When exactly did I tell you that I visited the paper workshop?'},
    ], session_id='new-session',on_delta=streamed.append)
    assert result.success and 'paper workshop' in result.content
    assert 'UTC' in result.content and brain.calls == 0
    assert streamed == [result.content]


def test_no_assistant_memory_or_unclear_reaction(tmp_path):
    manager = ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)
    store = manager._get_store()
    store.append('bad',0,'assistant','I watched the imaginary program',0,timestamp=1792000000)
    store.append('good',0,'user','I watched the nature program',0,timestamp=1792000000)
    store.append('good',1,'user','I prefer fresh coffee',0,timestamp=1792000010)
    answer = manager.verified_episode_history(
        'What do you remember about my experience watching the nature program and how I felt about it?')
    assert 'nature program' in answer and 'fresh coffee' not in answer
    assert 'imaginary program' not in answer
    assert parse_episode_question('When did I tell you about Apollo IE?') is None
    assert parse_episode_question('How powerful is the telescope?') is None


def test_ongoing_unconfirmed_experience_not_misreported(tmp_path):
    manager = ContextManager(db_path=tmp_path / 'm.db', hybrid_retrieval=False)
    manager.reconcile([{'role':'user','content':'I am watching the twilight play',
                        'timestamp':1792000000}], 's')
    response = manager.verified_episode_history('When did I tell you that I watched the twilight play?')
    assert "don't have a verified" in response


def test_recent_activity_query_excludes_old_evidence(tmp_path):
    import time
    cm = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    cm.reconcile([{'role':'user','content':'I watched the old stage show',
                   'timestamp':time.time() - 40*86400}], 'past')
    cm.reconcile([{'role':'user','content':'I watched the new stage show',
                   'timestamp':time.time() - 3600}], 'recent')
    answer = cm.verified_episode_history('What have I watched recently?')
    assert 'new stage show' in answer and 'old stage show' not in answer
