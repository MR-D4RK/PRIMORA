"""M3 repair: read-time integrity, honest recency, and episodic clarification.

Tests use unknown names on purpose: the solution must not whitelist domains.
"""
import time

import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.episode_retrieval import parse_episode_question
from PRIMORA.prime.context.mind import Mind
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.aurora.brain import BrainResult

NOW = 1792800000.0


def report(manager, session, text, timestamp):
    manager.reconcile([{"role": "user", "content": text, "timestamp": timestamp}], session)


@pytest.mark.parametrize("subject", [
    "Big Room", "Indigo Glass", "Northern Trail", "Vesper Watch",
    "novel project Zeta", "coastal gardening",
])
def test_recent_cross_session_filters_old_likes_for_any_subject(tmp_path, monkeypatch, subject):
    monkeypatch.setattr('PRIMORA.prime.context.manager.time.time', lambda: NOW)
    cm = ContextManager(db_path=tmp_path / "memory.db", hybrid_retrieval=False)
    report(cm, "old", "I love Stone Orchard", NOW - 60 * 86400)
    report(cm, "recent", f"I love {subject}", NOW - 3600)
    # Requesting a recent interest is not an unrestricted full-history query.
    fresh = ContextManager(db_path=tmp_path / "memory.db", hybrid_retrieval=False)
    answer = fresh.verified_likes_history("What have I told you I like recently?")
    assert subject.casefold() in answer.casefold()
    assert "Stone Orchard" not in answer
    assert "7 days" in answer
    older = fresh.verified_likes_history("What do I love?")
    assert "Stone Orchard" in older


def test_configurable_recent_window_and_date_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr('PRIMORA.prime.context.manager.time.time', lambda: NOW)
    cm = ContextManager(db_path=tmp_path / "memory.db", hybrid_retrieval=False)
    report(cm, "old", "I love Copper Lantern", NOW - 4 * 86400)
    report(cm, "new", "I like Paper Harbor", NOW - 86400)
    last2 = cm.verified_likes_history("What have I told you I like in the last 2 days?")
    assert "Paper Harbor" in last2 and "Copper Lantern" not in last2
    last5 = cm.verified_likes_history("What have I told you I like in the last 5 days?")
    assert "Paper Harbor" in last5 and "Copper Lantern" in last5


def test_legacy_pronoun_states_hidden_without_deleting(tmp_path, monkeypatch):
    monkeypatch.setattr('PRIMORA.prime.context.manager.time.time', lambda: NOW)
    cm = ContextManager(db_path=tmp_path / "memory.db", hybrid_retrieval=False)
    store = cm._get_store()
    legacy = ["it", "most of the aspect of it", "my feelings about that"]
    for i, label in enumerate(legacy):
        mid, *_ = store.append("old", i, "user", "I love " + label,
                              importance=0.5, timestamp=NOW - (i + 1))
        store.create_state("old", "global", "likes for " + label,
                           "loves " + label, [mid], timestamp=NOW - (i + 1))
    report(cm, "new", "I love Quartz Garden", NOW - 1)
    count_before = len(store.fetch_states(scope="global", status="current"))
    output = cm.verified_likes_history("what have i told you i like recently?")
    assert "Quartz Garden" in output
    assert "loves it" not in output
    assert "aspect of it" not in output
    assert "about that" not in output
    assert len(store.fetch_states(scope="global", status="current")) == count_before


def test_repeated_identical_preference_is_deduplicated(tmp_path, monkeypatch):
    monkeypatch.setattr('PRIMORA.prime.context.manager.time.time', lambda: NOW)
    cm = ContextManager(db_path=tmp_path / "memory.db", hybrid_retrieval=False)
    report(cm, "one", "I love Cloud Kites", NOW - 100)
    report(cm, "two", "I love Cloud Kites", NOW - 10)
    answer = cm.verified_likes_history("What do I love?")
    assert answer.casefold().count("- loves cloud kites") == 1


@pytest.mark.parametrize("question", [
    "aurora , rememeber when we worked on that activity ?",
    "Aurora, remember when we worked on that experience?",
    "Remember when I finished that project?",
    "What happened with that task?",
])
def test_unnamed_episode_must_clarify(tmp_path, question):
    cm = ContextManager(db_path=tmp_path / "memory.db", hybrid_retrieval=False)
    report(cm, "older", "I finished the Glass Harbor dashboard", NOW - 10)
    query = parse_episode_question(question)
    assert query is not None and query.needs_subject
    answer = cm.verified_episode_history(question)
    assert "Which activity" in answer
    assert "Glass Harbor" not in answer


def test_named_task_is_not_misclassified_as_unnamed_episode(tmp_path):
    cm = ContextManager(db_path=tmp_path / "memory.db", hybrid_retrieval=False)
    report(cm, "older", "We tested the task panel", NOW - 10)
    answer = cm.verified_episode_history("What happened with the task panel?")
    assert "tested the task panel" in answer


class NeverAnswerModel:
    def __init__(self):
        self.calls = 0
    def get_model_name(self):
        return "test-model"
    def generate(self, *args, **kwargs):
        self.calls += 1
        return BrainResult.success("invented activity and memories", "test-model")
    def count_tokens(self, msgs):
        return sum(len(m.get("content", "")) // 4 + 1 for m in msgs)


def test_ambiguous_episode_intercepted_in_actual_loop(tmp_path):
    db = tmp_path / "memory.db"
    cm = ContextManager(db_path=db, hybrid_retrieval=False)
    report(cm, "older", "We finished the Sparkle Prism refactor", NOW - 10)
    brain = NeverAnswerModel()
    loop = CognitiveLoop(Mind(ContextManager(db_path=db, hybrid_retrieval=False)), brain)
    result = loop.run([{"role": "system", "content": "Be truthful."},
                       {"role": "user", "content": "aurora , rememeber when we worked on that activity ?"}],
                      session_id="brand-new")
    assert result.success
    assert "Which activity" in result.content
    assert "Sparkle Prism" not in result.content
    assert brain.calls == 0
