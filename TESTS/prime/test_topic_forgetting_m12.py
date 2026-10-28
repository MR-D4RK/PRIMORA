"""M12 topic forgetting: no model, no production DB, no hardcoded subjects."""
from __future__ import annotations

import sqlite3

import pytest

from PRIMORA.aurora.aurora import Aurora
from PRIMORA.aurora.brain import BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.forgetting import (
    matches_topic, normalize_topic, parse_forget_request,
)
from PRIMORA.prime.context.store import ContextStore


@pytest.mark.parametrize("topic,phrase,other", [
    ("Blender", "I tested Blender for a scenario.", "I like drawing."),
    ("Atlas Kite", "I owned an Atlas-Kite project.", "I own a compass."),
    ("Dawn Protocol", "Dawn Protocol is just a fictional test.", "I like yellow."),
])
def test_forget_persists_and_hides_linked_records(tmp_path, topic, phrase, other):
    path = tmp_path / "isolated.db"
    store = ContextStore(path)
    mid, _, _ = store.append("past", 1, "user", phrase)
    assistant_id, _, _ = store.append(
        "past", 2, "assistant", "Your " + topic + " project is real."
    )
    other_id, _, _ = store.append("past", 3, "user", other)
    state_id = store.create_state(
        "past", "global", "project for " + topic, "fictional", [mid],
    )
    unrelated_state = store.create_state(
        "past", "global", "unrelated favorite", "yellow", [other_id],
    )
    store.save_digest("past", [mid], 1, 2, "extractive", "Notes about the project.")
    assert len(store.fetch_states()) == 2

    receipt = store.forget_topic(topic)
    assert receipt["status"] == "suppressed"
    assert receipt["physical_deletion"] is False
    assert receipt["total_hidden_messages"] >= 2
    assert store.fetch_message(mid) is None
    assert store.fetch_message(assistant_id) is None
    assert store.fetch_message(other_id).content == other
    assert not store.fetch_digests("past")
    assert store.fetch_states(state_ids=[state_id]) == []
    assert store.fetch_states(state_ids=[unrelated_state])
    assert all(not matches_topic(m.content, topic)
               for m in store.search(None, topic))
    assert all(not matches_topic(m.content, topic)
               for m in store.fetch_range("past"))
    assert all(not matches_topic(m.content, topic)
               for m in store.fetch_conversation("past"))

    # Same database, new process/store instance: no restoration.
    restarted = ContextStore(path)
    assert topic.casefold() in restarted.forgotten_topics()
    assert restarted.fetch_message(mid) is None
    assert restarted.fetch_states(state_ids=[state_id]) == []
    assert restarted.fetch_message(other_id).content == other
    assert restarted.forget_topic(topic)["total_hidden_messages"] >= 2

    # Verify this is honest suppression, NOT an unauthorized physical deletion.
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT content FROM messages WHERE id=?", (mid,)).fetchone()[0] == phrase
        assert con.execute("SELECT value FROM states WHERE id=?", (state_id,)).fetchone()[0] == "fictional"


def test_hidden_sources_cannot_reenter_from_caller_cached_history(tmp_path):
    manager = ContextManager(db_path=tmp_path / "cached.db", hybrid_retrieval=False)
    messages = [
        {"role": "user", "content": "I have a Cedar Observatory project."},
        {"role": "assistant", "content": "Tell me about Cedar Observatory."},
        {"role": "user", "content": "I enjoy painting."},
    ]
    manager.reconcile(messages, "session")
    manager.forget_topic("Cedar Observatory")
    safe = manager.filter_conversation_input(messages + [
        {"role": "user", "content": "What did we discuss?"}
    ])
    assert safe == messages[2:] + [{"role": "user", "content": "What did we discuss?"}]
    restarted = ContextManager(db_path=manager._get_store().db_path, hybrid_retrieval=False)
    assert all("Cedar Observatory" not in m["content"]
               for m in restarted.load_transcript("session"))
    assert all("Cedar Observatory" not in m["content"]
               for m in restarted.filter_conversation_input(messages))


def test_parser_requires_explicit_specific_imperative():
    assert parse_forget_request("forget everything about Blender") == "blender"
    assert parse_forget_request("please forget about Cedar Observatory!") == "cedar observatory"
    assert parse_forget_request("Can you forget everything about Blender?") is None
    assert parse_forget_request("Don't forget about Blender!") is None
    assert parse_forget_request("forget everything about everything") is None
    assert matches_topic("ATLAS-KITE sketches", "atlas kite")
    assert not matches_topic("AtlasKite sketches", "atlas kite")
    assert not matches_topic("blenderized product", "blender")
    with pytest.raises(ValueError):
        normalize_topic("everything")


class _Provider:
    def invoke(self, capability, payload):
        raise AssertionError("forgetting must not call Prime provider")


class _NeverGenerate:
    def __init__(self):
        self.calls = 0
    def get_model_name(self):
        return "test-brain"
    def count_tokens(self, messages):
        return sum(len(m["content"]) // 4 + 4 for m in messages)
    def generate(self, *args, **kwargs):
        self.calls += 1
        return BrainResult.success("I don't know.", "test-brain")


def test_aurora_confirms_only_after_verified_sqlite_suppression(tmp_path):
    path = tmp_path / "real-command.db"
    manager = ContextManager(db_path=path, hybrid_retrieval=False)
    manager.reconcile([
        {"role": "user", "content": "My Atlas Kite project was a test."},
        {"role": "assistant", "content": "You work on Atlas Kite."},
    ], "test-session")
    brain = _NeverGenerate()
    prime = Prime(_Provider(), context_manager=manager)
    aurora = Aurora(
        prime, brain=brain,
        config=AuroraConfig(model_path="unused", adapter_path="unused"),
    )
    preview = aurora.respond([
        {"role": "user", "content": "forget everything about Atlas Kite"},
    ], session_id="test-session")
    assert preview.success
    assert "confirm forget atlas kite" in preview.content
    assert manager._get_store().forgotten_topics() == ()
    reply = aurora.respond([
        {"role": "user", "content": "confirm forget Atlas Kite"},
    ], session_id="test-session")
    assert reply.success
    assert "not physically deleted" in reply.content
    assert brain.calls == 0
    assert manager._get_store().forgotten_topics() == ("atlas kite",)
    assert not manager.load_transcript("test-session")

    after_restart = ContextManager(db_path=path, hybrid_retrieval=False)
    assert not after_restart.load_transcript("test-session")
    assert not after_restart.recall("Atlas Kite", "test-session")


def test_forgetting_failure_never_claims_success(tmp_path, monkeypatch):
    manager = ContextManager(db_path=tmp_path / "failed.db")
    prime = Prime(_Provider(), context_manager=manager)
    aurora = Aurora(prime, brain=_NeverGenerate(),
                    config=AuroraConfig(model_path="unused", adapter_path="unused"))
    monkeypatch.setattr(manager, "forget_topic",
                        lambda _: (_ for _ in ()).throw(OSError("disk error")))
    preview = aurora.respond([
        {"role": "user", "content": "forget everything about Atlas Kite"},
    ], session_id="x")
    assert preview.success
    result = aurora.respond([
        {"role": "user", "content": "confirm forget Atlas Kite"},
    ], session_id="x")
    assert not result.success
    assert "disk error" in result.error


def test_confirmation_is_session_scoped_and_not_reused(tmp_path):
    manager = ContextManager(db_path=tmp_path / "confirm.db")
    brain = _NeverGenerate()
    aurora = Aurora(
        Prime(_Provider(), context_manager=manager), brain=brain,
        config=AuroraConfig(model_path="unused", adapter_path="unused"),
    )
    mismatch = aurora.respond([
        {"role": "user", "content": "confirm forget Cedar Observatory"},
    ], session_id="a")
    assert mismatch.success and "nothing changed" in mismatch.content
    request = aurora.respond([
        {"role": "user", "content": "forget everything about Cedar Observatory"},
    ], session_id="a")
    assert request.success
    wrong_session = aurora.respond([
        {"role": "user", "content": "confirm forget Cedar Observatory"},
    ], session_id="b")
    assert wrong_session.success and "nothing changed" in wrong_session.content
    assert manager._get_store().forgotten_topics() == ()
    mismatched = aurora.respond([
        {"role": "user", "content": "confirm forget Laurel Cart"},
    ], session_id="a")
    assert mismatched.success and "nothing changed" in mismatched.content
    assert manager._get_store().forgotten_topics() == ()
    # Authorization is consumed even on a mismatch.
    reattempt = aurora.respond([
        {"role": "user", "content": "confirm forget Cedar Observatory"},
    ], session_id="a")
    assert reattempt.success and "nothing changed" in reattempt.content
    assert brain.calls == 0
