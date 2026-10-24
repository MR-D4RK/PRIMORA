"""Regression tests for truthful personal-history answers; isolated SQLite."""
import time

from PRIMORA.aurora.brain import BrainResult
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.prime.context import ContextManager, Mind
from PRIMORA.prime.context.timeline import parse_request
from TESTS.prime.test_slots import _db, _say


class NoResearchBrain:
    def __init__(self):
        self.calls = 0

    def get_model_name(self):
        return "test"

    def generate(self, *_args, **_kwargs):
        self.calls += 1
        return BrainResult.success("Unsupported invented date April 25, 2024", "test")


def _seed(manager, text, timestamp, session="old"):
    _say(manager, [], session, text, timestamp=timestamp)


def _answer(manager, question, context=()):
    return manager.verified_personal_history(question, context=context)


def test_exact_apollo_date_is_original_user_message_not_future_or_assistant():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    timestamp = 1792258781.8769336
    _seed(m, 'I love Apollo IE the most in terms of visual and design', timestamp)
    answer = _answer(m, 'when did i first tell you about Apollo IE?')
    assert 'October 17, 2026' in answer
    assert 'I love Apollo IE the most in terms of visual and design' in answer
    assert '2024' not in answer and 'October 25' not in answer
    assert 'earliest stored evidence' in answer


def test_regera_followup_trims_suffix_and_only_quotes_user():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    timestamp = time.time() - 300
    _seed(m, 'I recently started liking Koenigsegg Regera', timestamp)
    answer = _answer(m, 'when did i tell you about regera just tell me the time of it')
    assert 'Koenigsegg Regera' in answer and 'UTC' in answer
    assert 'couldn\'t verify' not in answer
    claims = _answer(m, 'what did i say about Regera?')
    assert 'I recently started liking Koenigsegg Regera' in claims
    assert 'Apollo IE' not in claims


def test_assistant_claims_never_become_verified_memory():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    store = m._get_store()
    ai, _, _ = store.append('old', 0, 'assistant',
                             'Apollo IE first mentioned April 25, 2024',
                             timestamp=time.time()-200)
    store.create_state('old', 'global', 'likes for apollo ie',
                       'Apollo IE first mentioned April 25, 2024', [ai],
                       authority='assistant', status='unresolved',
                       timestamp=time.time()-200)
    assert "couldn't verify" in _answer(m, 'when did i tell you about Apollo IE?')


def test_future_user_evidence_is_excluded():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    _seed(m, 'I recently started liking Regera', time.time()+86400)
    assert "couldn't verify" in _answer(m, 'when did i tell you about Regera?')


def test_memory_clarification_refers_to_recent_explicit_date_request():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    _seed(m, 'I recently started liking Regera', time.time()-300)
    answer = _answer(m, 'just retrieve it from memory',
                     context=('when did i first tell you about Regera?',))
    assert 'UTC' in answer and 'Regera' in answer


def test_loop_returns_db_answer_without_research_or_model_guess():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    _seed(m, 'I love Apollo IE the most in terms of visual and design',
          1792258781.8769336)
    brain = NoResearchBrain()
    loop = CognitiveLoop(Mind(m), brain)
    answer = loop.run([{'role': 'user', 'content':
                        'When did I first tell you about Apollo IE?'}],
                      session_id='new-personal-history')
    assert answer.success and 'October 17, 2026' in answer.content
    assert brain.calls == 0
    assert loop.last_trace.stop_reason == 'finished'
    turns = m.load_transcript('new-personal-history')
    assert any(t['role'] == 'assistant' for t in turns)


def test_parser_preserves_other_timeline_questions():
    assert parse_request('What cars have I started liking recently?',
                         now=time.time()).mode == 'started'
    assert parse_request('When did I first tell you about Apollo IE?',
                         now=time.time()).target.casefold() == 'apollo ie'
    assert parse_request('What did I say about Regera?',
                         now=time.time()).mode == 'claims'
    assert parse_request('I like Regera', now=time.time()) is None


def test_category_dates_are_ordered_and_source_bound():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    now = time.time()
    h = []
    _say(m, h, 'old', 'my favorite car is Corvette C8', timestamp=now-500)
    _say(m, h, 'old', 'What are my favorite cars?', timestamp=now-450)
    _say(m, h, 'old', 'I love Apollo IE the most in terms of visual and design', timestamp=now-400)
    _say(m, h, 'old', 'my favorite classic car is 1969 Boss Mustang', timestamp=now-300)
    answer = _answer(m, 'what are my favorite cars and when did i tell you about them')
    assert answer.startswith('Here are the earliest stored')
    assert 'Corvette C8' in answer and '1969 Boss Mustang' in answer
    assert 'Apollo IE' in answer
    assert answer.index('Corvette C8') < answer.index('Apollo IE')
    assert answer.index('Apollo IE') < answer.index('1969 Boss Mustang')
    assert 'April 25, 2024' not in answer


def test_user_source_must_mention_named_interest_not_just_bad_state():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    store = m._get_store()
    source_id, _, _ = store.append('old', 1, 'user',
                                    'I love yellow', timestamp=time.time()-400)
    store.create_state('old', 'global', 'likes for apollo ie',
                       'loves Apollo IE', [source_id],
                       authority='user', status='current',
                       timestamp=time.time()-400)
    answer = _answer(m, 'when did I tell you about Apollo IE?')
    assert "couldn't verify" in answer
    assert 'yellow' not in answer
