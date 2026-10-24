"""Favorite inventory consistency regression, using only temporary SQLite."""
import time

from PRIMORA.prime.context import ContextManager, Mind
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.aurora.brain import BrainResult
from TESTS.prime.test_slots import _db, _say


class NoGuessBrain:
    calls = 0

    def get_model_name(self):
        return "test"

    def generate(self, *_args, **_kwargs):
        self.calls += 1
        return BrainResult.success("Bayern Munich and magic horsepower", "test")


def seed(manager):
    hist = []
    base = time.time() - 10000
    old = "cars-original"
    for i, s in enumerate((
        "I love Apollo IE the most in terms of visual and design",
        "I love the Corvette C8 the most",
        "I like the ZR1 too",
        "my favorite classic car is 1969 Boss Mustang",
        "I've recently started liking the Koenigsegg Regera",
        "it has great engineering behind it",
        "that's why i love the regera",
        "I support Bayern Munich",
        "my favorite color is yellow",
    )):
        _say(manager, hist, old, s, timestamp=base + 60 * i)


def test_five_favorites_are_consistent_and_exclude_unrelated_facts():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    seed(m)
    answer = m.verified_personal_history('Show me my 5 fav cars')
    for car in ('Apollo Ie', 'Corvette C8', 'ZR1', '1969 Boss Mustang', 'Koenigsegg Regera'):
        assert car.casefold() in answer.casefold(), (car, answer)
    assert 'Bayern' not in answer and 'yellow' not in answer
    assert answer.count('source message') == 5


def test_times_reasons_and_no_reason_transfer():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    seed(m)
    answer = m.verified_personal_history('what are my fav cars, when and why?')
    assert 'first verified record' in answer
    assert 'visual and design' in answer
    assert 'great engineering' in answer
    assert answer.count('No verified reason stored') >= 2
    blocks = answer.split('\n- ')
    c8 = next(b for b in blocks if b.startswith('Corvette C8'))
    apollo = next(b for b in blocks if b.lower().startswith('apollo ie'))
    regera = next(b for b in blocks if b.lower().startswith('koenigsegg regera'))
    assert 'great engineering' not in c8 and 'visual and design' not in c8
    assert 'visual and design' in apollo and 'great engineering' not in apollo
    assert 'great engineering' in regera and 'visual and design' not in regera


def test_followup_uses_only_prior_user_category_request():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    seed(m)
    assert m.verified_personal_history(
        'can you mention why i love each of them?',
        context=('show me my 5 favorite cars',)) is not None
    assert m.verified_personal_history('can you mention why i love each of them?') is None


def test_ignores_assistant_memory_and_unrelated_user_source():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    seed(m)
    store = m._get_store()
    mid, _, _ = store.append('old', 999, 'assistant',
                              'Your favorite car is Bayern Munich',
                              timestamp=time.time()-100)
    store.create_state('old', 'global', 'likes for bayern munich',
                       'loves bayern munich', [mid], authority='assistant',
                       status='unresolved', timestamp=time.time()-100)
    wrong, _, _ = store.append('old', 1000, 'user', 'I love yellow',
                               timestamp=time.time()-90)
    store.create_state('old', 'global', 'likes for ferrari', 'loves ferrari',
                       [wrong], authority='user', status='current',
                       timestamp=time.time()-90)
    out = m.verified_personal_history('what are my favorite cars?')
    assert 'Bayern' not in out and 'Ferrari' not in out


def test_nontopic_comparison_question_remains_for_normal_brain():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    assert m.verified_personal_history('Which car is most powerful?') is None


def test_full_favorite_list_bypasses_model_guess_in_new_session():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    seed(m)
    brain = NoGuessBrain()
    loop = CognitiveLoop(Mind(m), brain)
    result = loop.run([{'role':'user','content':'What are my favorite cars?'}],
                      session_id='fresh-inventory')
    assert result.success
    assert 'Apollo' in result.content and 'Regera' in result.content
    assert 'Bayern' not in result.content
    assert brain.calls == 0


def test_explicit_car_category_does_not_reclassify_unrelated_like():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    hist = []
    _say(m, hist, 'old', 'What are my favorite cars?')
    _say(m, hist, 'old', 'I love Bayern Munich')
    _say(m, hist, 'old', 'I like Corvette C8')
    out = m.verified_personal_history('What are my favorite cars?')
    assert 'Corvette' in out and 'Bayern' not in out


def test_followup_rest_of_favorites_inherits_explicit_user_category():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    seed(m)
    result = m.verified_personal_history('what about the rest?',
             context=('what are my favorite cars?',))
    assert result and 'Apollo' in result and 'Regera' in result
    assert m.verified_personal_history('what about the rest?') is None


def test_loop_reasons_followup_uses_prior_user_question_not_assistant():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    seed(m)
    brain = NoGuessBrain()
    loop = CognitiveLoop(Mind(m), brain)
    result = loop.run([
        {'role':'user','content':'show me my five favorite cars'},
        {'role':'assistant','content':'This is only some of them.'},
        {'role':'user','content':'can you mention why i love each of them?'},
    ], session_id='new-followup')
    assert result.success and 'great engineering' in result.content
    assert 'visual and design' in result.content
    assert 'Bayern' not in result.content
    assert brain.calls == 0
