"""Regression tests for the live CLI memory windows.

The production failure was positional: the current fact was present in the
prompt but sat before the assistant's old denial, so the model copied the
denial. These tests rebuild the same shape on temp stores and assert the
final ordered view, not just that a state exists.
"""

import os
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import BudgetConfig, ContextManager

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db(name):
    _SEQ[0] += 1
    return os.path.join(_TEST_TMPDIR.name, f"{name}-{_SEQ[0]}.db")


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget(**kwargs):
    params = dict(
        input_target=1500,
        input_ceiling=1600,
        output_reserve=256,
        total_budget=1856,
        state_reserve=100,
    )
    params.update(kwargs)
    return BudgetConfig(**params)


def _view(mgr, messages, sid="default", budget=None):
    return mgr.build(
        messages, sid, token_counter=_counter, max_new_tokens=256,
        budget=budget or _budget(),
    ).view


def _index(view, needle):
    return next(
        i for i, message in enumerate(view.messages)
        if needle in message.get("content", "")
    )


def test_greeting_does_not_overwrite_color_in_exact_window():
    mgr = ContextManager(db_path=_db("greeting-window"))
    history = [
        {"role": "user", "content": "what is my fav color?"},
        {"role": "assistant",
         "content": "I don't have access to your personal preferences or "
                    "stored information about your favorite color."},
        {"role": "user", "content": "its yellow"},
        {"role": "assistant",
         "content": "Got it! Your favorite color is yellow."},
        {"role": "user", "content": "hi"},
        {"role": "assistant",
         "content": "Hello! How can I assist you today?"},
    ]
    mgr.reconcile(history, "default")
    rows = mgr._get_store().fetch_states(concept="favorite color")
    current = [state for state in rows if state.status == "current"]
    assert [state.value for state in current] == ["yellow"]
    assert all(state.value != "hi" for state in rows)

    view = _view(
        mgr, history + [{"role": "user", "content": "what was my fav color"}]
    )
    joined = "\n".join(m["content"] for m in view.messages)
    assert "The user's favorite color is yellow." in joined
    assert "favorite color is hi" not in joined
    assert view.messages[-1] == {
        "role": "user", "content": "what was my fav color"}


def test_color_fact_follows_old_denials_and_precedes_current_turn():
    mgr = ContextManager(db_path=_db("color-order"))
    history = [
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "assistant",
         "content": "I do not have your favorite color stored."},
        {"role": "user", "content": "It's yellow."},
        {"role": "assistant",
         "content": "Got it! Your favorite color is yellow."},
        {"role": "user", "content": "hi"},
        {"role": "assistant",
         "content": "Hello! How can I assist you today?"},
        {"role": "user", "content": "what is my favorite color?"},
        {"role": "assistant",
         "content": "I do not have your favorite color stored."},
    ]
    view = _view(
        mgr,
        history
        + [{"role": "user", "content": "what was my favorite color"}],
    )
    fact = _index(view, "The user's favorite color is yellow.")
    denial_positions = [
        i for i, message in enumerate(view.messages)
        if "do not have your favorite color" in message.get("content", "")
    ]
    assert denial_positions
    assert fact > max(denial_positions)
    assert fact == len(view.messages) - 2
    assert view.messages[-1]["content"] == "what was my favorite color"
    assert view.input_tokens <= 1600


def test_color_fact_survives_a_full_1600_token_window():
    mgr = ContextManager(db_path=_db("color-budget"))
    filler = [
        {"role": "assistant",
         "content": " ".join(
             f"old note {i} word{j}" for j in range(22))}
        for i in range(48)
    ]
    history = [
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "assistant",
         "content": "I do not have your favorite color stored."},
        {"role": "user", "content": "It's yellow."},
    ] + filler + [
        {"role": "assistant",
         "content": "I do not have your favorite color stored."},
    ]
    view = _view(
        mgr,
        history
        + [{"role": "user", "content": "what was my favorite color"}],
    )
    joined = "\n".join(m["content"] for m in view.messages)
    assert "The user's favorite color is yellow." in joined
    assert view.input_tokens <= 1600
    assert view.messages[-1]["content"] == "what was my favorite color"


def test_fav_cars_query_bridges_likes_states_from_source_context():
    mgr = ContextManager(db_path=_db("cars-bridge"))
    sid = "default"
    mgr.reconcile([
        {"role": "user", "content": "what car is my favorite"},
        {"role": "assistant", "content": "I don't have that stored."},
        {"role": "user", "content": "i like the corvette c8 the most"},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "what are my fav cars"},
        {"role": "user",
         "content": "i love apollo ie the most in terms of visual and design"},
        {"role": "assistant", "content": "Noted."},
    ], sid)
    store = mgr._get_store()
    assert store.fetch_states(concept="favorite car") == []

    view = _view(
        mgr, [{"role": "user", "content": "what was my fav cars"}], sid,
    )
    joined = "\n".join(m["content"] for m in view.messages)
    assert 'The user said about their favorite car: ' in joined
    assert "i like the corvette c8 the most" in joined
    assert "i love apollo ie the most in terms of visual and design" in joined
    assert "The user's facts" in joined
    assert view.messages[-1]["content"] == "what was my fav cars"
    assert store.fetch_states(concept="favorite car") == []


def test_fav_cars_bridge_ignores_assistant_only_context():
    mgr = ContextManager(db_path=_db("cars-false-positive"))
    sid = "default"
    mgr.reconcile([
        {"role": "user", "content": "what were we talking about"},
        {"role": "assistant",
         "content": "We were talking about cars, specifically the Corvette "
                    "C8 and the Apollo IE."},
        {"role": "user", "content": "i love and support bayern munich"},
        {"role": "assistant",
         "content": "Let me know if you'd like to focus on the music and car "
                    "topics you love."},
    ], sid)
    store = mgr._get_store()
    assert store.fetch_states(concept="favorite car") == []
    assert any(
        state.concept.startswith("likes for")
        for state in store.fetch_states(status="current")
    )

    view = _view(
        mgr, [{"role": "user", "content": "what was my fav cars"}], sid,
    )
    joined = "\n".join(m["content"] for m in view.messages)
    assert "bayern" not in joined
    assert "The user said about their favorite car:" not in joined


def test_fav_cars_bridge_yields_to_a_direct_favorite_car_state():
    mgr = ContextManager(db_path=_db("cars-direct"))
    sid = "default"
    mgr.reconcile([
        {"role": "user", "content": "what car is my favorite"},
        {"role": "user", "content": "i like the corvette c8 the most"},
        {"role": "user", "content": "what are my fav cars"},
        {"role": "user",
         "content": "i love apollo ie the most in terms of visual and design"},
    ], sid)
    store = mgr._get_store()
    store.create_state(
        sid, "global", "favorite car", "corvette c8", [1])

    view = _view(
        mgr, [{"role": "user", "content": "what was my fav cars"}], sid,
    )
    joined = "\n".join(m["content"] for m in view.messages)
    assert "The user's favorite car is corvette c8." in joined
    assert "The user said about their favorite car:" not in joined
