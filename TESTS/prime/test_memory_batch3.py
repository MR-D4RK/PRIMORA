"""Batch 3: don't turn ambiguous motives into memories; retain qualifiers."""

from PRIMORA.prime.context import BudgetConfig, ContextManager


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _say(manager, history, text, session="batch3"):
    history.append({"role": "user", "content": text})
    return manager.build(history[:], session, token_counter=_counter,
                         max_new_tokens=12,
                         budget=BudgetConfig(input_target=280,
                                             input_ceiling=450,
                                             output_reserve=12,
                                             total_budget=465))


def _current(manager, concept):
    return manager._get_store().fetch_states(
        concept=concept, status="current", scope="global")


def test_disputed_reason_does_not_reverse_existing_like(tmp_path):
    manager = ContextManager(db_path=str(tmp_path / 'mem.db'))
    hist = []
    _say(manager, hist, "I love yellow.")
    _say(manager, hist, "I don't love yellow because it's warm.")
    assert [s.value for s in _current(manager, 'likes for yellow')] == [
        'loves yellow']
    assert not _current(manager, 'preference for not yellow')
    assert not manager._get_store().fetch_states(
        concept='rationale for likes for yellow', status='current')


def test_assistant_attribution_does_not_become_personal_reason(tmp_path):
    manager = ContextManager(db_path=str(tmp_path / 'mem.db'))
    _say(manager, [], 'I love yellow because you said it means creativity.')
    assert [s.value for s in _current(manager, 'likes for yellow')] == [
        'loves yellow']
    assert not manager._get_store().fetch_states(
        concept='rationale for likes for yellow', status='current')


def test_explicit_first_person_reason_still_stored(tmp_path):
    manager = ContextManager(db_path=str(tmp_path / 'mem.db'))
    _say(manager, [], 'I love yellow because the shade looks beautiful to me.')
    assert _current(manager, 'likes for yellow')
    reasons = manager._get_store().fetch_states(
        concept='rationale for likes for yellow', status='current')
    assert [r.value for r in reasons] == ['the shade looks beautiful to me']


def test_purple_any_shade_is_one_like_and_comparison_independent(tmp_path):
    path = str(tmp_path / 'mem.db')
    manager = ContextManager(db_path=path)
    hist = []
    _say(manager, hist, "I love purple whether its bright or not.")
    _say(manager, hist, "Purple feels like violet to me.")
    _say(manager, hist, "I love purple.")  # weaker repetition must not erase shade qualifier
    assert [s.value for s in _current(manager, 'likes for purple')] == [
        'loves purple in any shade']
    assert [s.value for s in _current(manager, 'color feeling for purple')] == [
        'Purple feels similar to violet to them']
    assert not _current(manager, 'likes for purple whether its bright or not')
    restarted = ContextManager(db_path=path)
    result = _say(restarted, [], 'What do I like about purple?', 'fresh')
    displayed = '\n'.join(m['content'] for m in result.view.messages)
    assert 'loves purple in any shade' in displayed
    assert 'purple feels similar to violet' in displayed.lower()


def test_implicit_feelings_are_not_invented_as_explicit_link(tmp_path):
    manager = ContextManager(db_path=str(tmp_path / 'mem.db'))
    hist = []
    _say(manager, hist, 'I love purple.')
    _say(manager, hist, 'No the feeling is quite like violet color for me')
    assert not _current(manager, 'color feeling for purple')
