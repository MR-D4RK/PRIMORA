"""Domain-neutral verified preference inventory integration tests."""
from PRIMORA.prime.context import ContextManager


def test_broad_preferences_combine_favorites_and_likes(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False)
    store = cm._get_store()
    a, *_ = store.append('a', 0, 'user', 'My favorite instrument is the glass harp')
    b, *_ = store.append('b', 0, 'user', 'I love amber ceramics')
    c, *_ = store.append('c', 0, 'assistant', 'You love paper lanterns')
    store.create_state('a', 'global', 'favorite instrument', 'glass harp', [a])
    store.create_state('b', 'global', 'likes for amber ceramics', 'loves amber ceramics', [b])
    store.create_state('c', 'global', 'likes for paper lanterns', 'loves paper lanterns', [c], authority='assistant', status='unresolved')
    answer = cm.verified_likes_history('What are my preferences?')
    assert 'favorite instrument: glass harp' in answer
    assert 'loves amber ceramics' in answer
    assert 'paper lanterns' not in answer
    assert cm.verified_likes_history('What have I told you I like recently?') is not None


def test_preference_dimensions_not_conflated(tmp_path):
    cm = ContextManager(db_path=tmp_path / 'memory.db', hybrid_retrieval=False)
    store = cm._get_store()
    a, *_ = store.append('a', 0, 'user', 'My favorite book is River Atlas')
    b, *_ = store.append('b', 0, 'user', 'My favorite game is River Atlas')
    store.create_state('a', 'global', 'favorite book', 'River Atlas', [a])
    store.create_state('b', 'global', 'favorite game', 'River Atlas', [b])
    answer = cm.verified_likes_history('List my preferences')
    assert 'favorite book: River Atlas' in answer
    assert 'favorite game: River Atlas' in answer
    assert answer.count('River Atlas') >= 2
