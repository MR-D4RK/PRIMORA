"""Regressions from the user's live favorite-car inventory dialogue."""
import time

from PRIMORA.prime.context import ContextManager, Mind
from PRIMORA.aurora.loop import CognitiveLoop
from TESTS.prime.test_slots import _db, _say
from TESTS.prime.test_m2_inventory_consistency_repair import NoGuessBrain, seed


def _manager_with_contaminating_but_user_authored_preference():
    manager = ContextManager(db_path=_db(), hybrid_retrieval=False)
    seed(manager)
    hist = []
    _say(manager, hist, 'other', 'i love the advanced engineering of regera not c8',
         timestamp=time.time() - 15)
    return manager


def test_attribute_preference_is_not_a_sixth_vehicle():
    manager = _manager_with_contaminating_but_user_authored_preference()
    output = manager.verified_personal_history('what are my fav cars')
    assert output.count('(source message ') == 5, output
    assert 'Advanced Engineering Of Regera Not C8' not in output
    for name in ('Corvette C8', 'Apollo IE', 'ZR1', '1969 Boss Mustang',
                 'Koenigsegg Regera'):
        assert name.casefold() in output.casefold()


def test_dates_followup_uses_previous_user_car_question():
    manager = _manager_with_contaminating_but_user_authored_preference()
    answer = manager.verified_personal_history(
        'when did i tell you about each of them ?',
        context=('what are my fav cars',))
    assert answer and 'earliest stored user-statement dates' in answer
    assert len([line for line in answer.splitlines() if 'UTC' in line and line.startswith('- ')]) == 5
    assert 'advanced engineering of regera not c8' not in answer.casefold()


def test_typo_in_reason_followup_and_repeated_date_followups():
    manager = _manager_with_contaminating_but_user_authored_preference()
    prior = ('what are my fav cars',
             'when did i tell you about each of them ?',
             'when did i tell you about each of them ?',
             'when did i tell you about each of them ?')
    answer = manager.verified_personal_history(
        'can you mentino why i love each of them ?', context=prior)
    assert answer and 'Your stated reason: it has great engineering behind it' in answer
    assert 'visual and design' in answer
    assert answer.count('(source message ') == 5


def test_end_to_end_followups_bypass_brain():
    manager = _manager_with_contaminating_but_user_authored_preference()
    brain = NoGuessBrain()
    loop = CognitiveLoop(Mind(manager), brain)
    messages = [
        {'role': 'user', 'content': 'what are my fav cars'},
        {'role': 'assistant', 'content': 'Here are your cars'},
        {'role': 'user', 'content': 'when did i tell you about each of them ?'},
    ]
    first = loop.run(messages, session_id='fresh-car-repair')
    assert first.success and 'earliest stored user-statement dates' in first.content
    messages += [
        {'role': 'assistant', 'content': first.content},
        {'role': 'user', 'content': 'can you mentino why i love each of them ?'},
    ]
    second = loop.run(messages, session_id='fresh-car-repair')
    assert second.success and 'Your stated reason: it has great engineering behind it' in second.content
    assert brain.calls == 0


def test_unrelated_context_or_explicit_other_category_does_not_borrow_cars():
    manager = _manager_with_contaminating_but_user_authored_preference()
    no_context = manager.verified_personal_history(
        'when did i tell you about each of them?', context=())
    assert 'Apollo IE' not in (no_context or '')
    other = manager.verified_personal_history(
        'when did i tell you about each of those songs?',
        context=('what are my fav cars',))
    assert 'Apollo IE' not in (other or '')


def test_dates_use_earliest_historical_preference_not_later_reason():
    manager = _manager_with_contaminating_but_user_authored_preference()
    answer = manager.verified_personal_history(
        'when did i tell you about each of them?',
        context=('what are my fav cars',))
    regera = next(line for line in answer.splitlines() if 'regera' in line.casefold())
    assert 'recently started liking' in regera.casefold(), regera
    assert "that's why i love" not in regera.casefold()
