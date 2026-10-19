"""History-free recall: stored facts survive, foreign chatter does not.

Conventions follow the repo: stdlib only, temp SQLite files, no live model.
The isolated switch withholds other sessions' verbatim messages; persistent
states and their provenance still flow into a genuinely fresh session.
"""
import os
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context.manager import ContextManager

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _db_path(name):
    return os.path.join(_TEST_TMPDIR.name, name)


def _seed(include_foreign_history=True, name="history_free.db"):
    mgr = ContextManager(
        db_path=_db_path(name),
        include_foreign_history=include_foreign_history,
    )
    mgr.reconcile(
        [{"role": "user", "content": "My favorite color is teal."}],
        "seed",
    )
    mgr.reconcile(
        [{"role": "user",
          "content":
              "The launch codename is Zephyr-9 and the team is Northwind."}],
        "seed",
    )
    return mgr


def _messages(context):
    return getattr(context, "messages", None) or context.view.messages


def _texts(context):
    return [m.get("content", "") for m in _messages(context)]


def _system_text(context):
    return "\n".join(
        m.get("content", "") for m in _messages(context)
        if m.get("role") == "system"
    )


def test_foreign_history_is_withheld():
    mgr = _seed(include_foreign_history=False)
    ctx = mgr.build(
        [{"role": "user", "content": "What is my launch codename?"}],
        "fresh-off",
    )
    joined = "\n".join(_texts(ctx))
    assert "[from another conversation]" not in joined
    assert "Zephyr-9" not in joined
    assert "Northwind" not in joined


def test_foreign_history_is_present_by_default():
    mgr = _seed(include_foreign_history=True)
    ctx = mgr.build(
        [{"role": "user", "content": "What is my launch codename?"}],
        "fresh-on",
    )
    joined = "\n".join(_texts(ctx))
    assert "[from another conversation]" in joined
    assert "Zephyr-9" in joined


def test_stored_global_fact_survives_isolation():
    mgr = _seed(include_foreign_history=False)
    ctx = mgr.build(
        [{"role": "user", "content": "What is my favorite color?"}],
        "fresh-fact",
    )
    messages = _messages(ctx)
    facts = _system_text(ctx)
    assert "favorite color" in facts.lower()
    assert "teal" in facts.lower()
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "What is my favorite color?"


def test_current_turn_is_last_and_history_is_empty():
    mgr = _seed(include_foreign_history=False)
    ctx = mgr.build(
        [{"role": "user", "content": "What is my favorite color?"}],
        "fresh-order",
    )
    messages = _messages(ctx)
    assert messages[-1] == {
        "role": "user", "content": "What is my favorite color?"}
    non_system = [m for m in messages if m.get("role") != "system"]
    assert non_system == [
        {"role": "user", "content": "What is my favorite color?"}]


def test_isolated_setting_is_explicit_and_reversible():
    mgr = _seed(include_foreign_history=True)
    assert mgr.include_foreign_history is True
    mgr.set_include_foreign_history(False)
    assert mgr.include_foreign_history is False
    ctx = mgr.build(
        [{"role": "user", "content": "What is my launch codename?"}],
        "fresh-toggle",
    )
    assert "[from another conversation]" not in "\n".join(_texts(ctx))
    mgr.set_include_foreign_history(True)
    assert mgr.include_foreign_history is True


def test_same_session_turns_are_not_withheld():
    mgr = ContextManager(
        db_path=_db_path("history_same.db"),
        include_foreign_history=False,
    )
    mgr.reconcile(
        [{"role": "user", "content": "My favorite color is teal."}],
        "same",
    )
    ctx = mgr.build(
        [
            {"role": "user", "content": "My favorite color is teal."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "What is my favorite color?"},
        ],
        "same",
    )
    joined = "\n".join(_texts(ctx))
    assert "Noted." in joined
    facts = _system_text(ctx)
    assert "teal" in facts.lower()
