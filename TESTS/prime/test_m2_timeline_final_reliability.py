"""Milestone 2 targeted repairs: user-source, start-only, causal evidence.

Tests use throwaway SQLite; never connect to a user's production database.
"""
import time
from PRIMORA.prime.context import ContextManager, Mind
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.aurora.brain import BrainResult
from TESTS.prime.test_slots import _db, _say, _budget, _joined


def _seed(mgr, *texts):
    hist = []
    base = time.time() - 500
    for i, text in enumerate(texts):
        _say(mgr, hist, "old", text, timestamp=base + i * 10)


def test_started_only_and_source_ids_are_real():
    mgr = ContextManager(db_path=_db(), hybrid_retrieval=False)
    _seed(mgr, "What are my favorite cars?", "I love Apollo IE",
          "I've recently started liking Koenigsegg Regera",
          "The Regera has advanced engineering")
    output = mgr.verified_personal_history(
        "Which cars have I started liking recently and when and why?")
    assert "koenigsegg regera" in output.lower()
    assert "I love Apollo IE" not in output
    assert "source message" in output
    assert "unknown" not in output
    assert "I don't have a verified reason stored" in output


def test_follows_user_stated_reason_in_a_fresh_session():
    db = _db()
    mgr = ContextManager(db_path=db, hybrid_retrieval=False)
    _seed(mgr, "What are my favorite cars?",
          "I've recently started liking Koenigsegg Regera",
          "it has a great engineering behind it",
          "that's why i love the regera")
    restarted = ContextManager(db_path=db, hybrid_retrieval=False)
    assert restarted.load_transcript("new") == []
    answer = restarted.verified_personal_history(
        "Which cars have I started liking recently and when and why?")
    assert "koenigsegg regera" in answer.lower()
    assert "great engineering" in answer
    reasons = restarted._get_store().fetch_states(
        scope="global", concept="rationale for likes for koenigsegg regera",
        status="current")
    assert len(reasons) == 1 and len(reasons[0].source_ids) == 2
    source = restarted._get_store().fetch_messages_by_ids(reasons[0].source_ids)
    assert all(m.role == "user" for m in source)
    assert 'great engineering' in restarted.verified_personal_history(
        "What have I told you about why I love the Regera?")


def test_no_assistant_motive_leakage():
    mgr = ContextManager(db_path=_db(), hybrid_retrieval=False)
    hist = []
    ts = time.time()-300
    _say(mgr, hist, 'old', "I've recently started liking Koenigsegg Regera", timestamp=ts)
    # Assistant's interpretation is never a stored user reason.
    hist.append({'role': 'assistant',
                 'content': 'it has a great engineering behind it',
                 'timestamp': ts+5})
    mgr.build(hist, 'old', token_counter=lambda ms: sum(
        len(m['content'].split())+2 for m in ms))
    _say(mgr, hist, 'old', "that's why i love the regera", timestamp=ts+10)
    answer = mgr.verified_personal_history(
        "Which cars have I started liking recently and when and why?")
    assert 'Your stated reason: it has a great engineering' not in answer


def test_unrelated_user_turn_is_not_a_reason():
    mgr = ContextManager(db_path=_db(), hybrid_retrieval=False)
    _seed(mgr, "I've recently started liking Koenigsegg Regera",
          'My favorite color is yellow', "that's why i love the regera")
    states = mgr._get_store().fetch_states(scope='global')
    assert not any(st.concept.startswith('rationale for likes for regera')
                   for st in states)


def test_unlinked_and_assistant_states_are_not_verified():
    mgr = ContextManager(db_path=_db(), hybrid_retrieval=False)
    store = mgr._get_store()
    store.create_state('old', 'global', 'likes for unrelated',
                       'likes unrelated', [], authority='user',
                       status='current', timestamp=time.time()-100)
    aid, _, _ = store.append('old', 10, 'assistant',
                             'I recently started liking Apollo IE',
                             timestamp=time.time()-90)
    store.create_state('old', 'global', 'likes for apollo ie',
                       'likes apollo ie', [aid], authority='assistant',
                       status='unresolved', timestamp=time.time()-90)
    assert "couldn't verify" in mgr.verified_personal_history(
        'what cars have i started liking recently?')


def test_normal_timeline_context_only_uses_linked_user_sources():
    mgr = ContextManager(db_path=_db(), hybrid_retrieval=False)
    store = mgr._get_store()
    store.create_state('old', 'global', 'likes for unrecoverable',
                       'likes unrecoverable', [], authority='user',
                       status='current', timestamp=time.time()-100)
    view = _say(mgr, [], 'new', 'What interests have I liked lately?',
                budget=_budget(input_target=480, input_ceiling=850,
                               total_budget=900, output_reserve=30)).view
    text = _joined(view)
    assert 'fact: [likes for unrecoverable' not in text


class _NoBrain:
    def get_model_name(self):
        return "no-brain"
    def generate(self, *_args, **_kwargs):
        raise AssertionError('Started timeline must not consult the model')


def test_started_intent_bypasses_research_and_model():
    mgr = ContextManager(db_path=_db(), hybrid_retrieval=False)
    _seed(mgr, 'What are my favorite cars?',
          "I've recently started liking Koenigsegg Regera", "I love Apollo IE")
    loop = CognitiveLoop(Mind(mgr), _NoBrain())
    result = loop.run([{'role':'user','content':
        'Which cars have I started liking recently and when and why?'}],
        session_id='new')
    assert result.success and 'koenigsegg regera' in result.content.lower()
    assert 'Apollo IE' not in result.content
