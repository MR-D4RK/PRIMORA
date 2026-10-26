"""M9 Batch 2 restoration, duplicate-learning, and natural-recall tests."""

import hashlib
import json
from pathlib import Path

from PRIMORA.aurora.brain import BrainResult
from PRIMORA.aurora.memory_coordinator import interpret
from PRIMORA.learning import Jev, LearningOutcome, StructuredCandidate
from PRIMORA.prime.context import ContextManager, Mind
from PRIMORA.prime.context.memory_restore import MemoryRestoreService
from PRIMORA.prime.context.observation import ObservationProposal, ObservationService
from PRIMORA.prime.context.store import ContextStore
from PRIMORA.prime.contract import PrimeRequest, RequestStatus
from PRIMORA.prime.prime import Prime
from PRIMORA.prime.providers import Provider


class _Provider(Provider):
    def invoke(self, capability, payload):
        raise AssertionError(f"unexpected provider call: {capability}")


class _Jev(Jev):
    def evaluate(self, request):
        return LearningOutcome.confirmed([
            StructuredCandidate(
                "favorite color", "amber", "PREFERENCE", "global"
            )
        ])


class _PlannerBrain:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def generate(self, messages, max_output_tokens=None, **kwargs):
        self.calls.append(messages)
        return BrainResult.success(json.dumps(self.plan), "planner")


class _RecallBrain(_PlannerBrain):
    def generate(self, messages, max_output_tokens=None, **kwargs):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return BrainResult.success(json.dumps(self.plan), "planner")
        return BrainResult.success("Grounded memory answer", "planner")


def _message(store, session, seq, role, content, timestamp):
    return store.append(session, seq, role, content, timestamp=timestamp)[0]


def test_restore_preview_is_read_only_and_import_remaps_sources_idempotently(tmp_path):
    source = tmp_path / "legacy.db"
    target = tmp_path / "isolated-v2.db"
    source_store = ContextStore(source)
    source_store.ensure_session("old", title="old conversation", timestamp=10)
    first = _message(source_store, "old", 0, "user", "I prefer amber ceramics", 11)
    second = _message(source_store, "old", 1, "user", "Yesterday I visited a gallery", 12)
    source_store.create_state(
        "old", "global", "likes for amber ceramics", "loves amber ceramics",
        [first], memory_type="PREFERENCE", observed_at=11,
    )
    source_store.save_experience_event(
        "old", second, "activity", "i", "Yesterday I visited a gallery",
        occurred_at=11, occurred_precision="day", occurred_timezone="UTC",
    )
    before = source_store.fetch_states(scope="global")
    preview = MemoryRestoreService(source, target).preview()
    assert preview.scanned["messages"] == 2
    assert preview.transferable["messages"] == 2
    assert preview.conflicts == []
    assert source_store.fetch_states(scope="global") == before

    imported = MemoryRestoreService(source, target).import_into()
    assert imported.imported["messages"] == 2
    assert imported.imported["states"] == 1
    assert imported.imported["experience_events"] == 1
    assert imported.id_maps["messages"][first] != 0

    repeated = MemoryRestoreService(source, target).import_into()
    assert repeated.changed is False
    assert repeated.imported == {}
    assert repeated.already_present["messages"] == 2
    target_store = ContextStore(target)
    restored = target_store.fetch_states(scope="global")
    assert restored[0].source_ids == [imported.id_maps["messages"][first]]
    events = target_store.fetch_experience_events()
    assert events[0].occurred_at == 11
    assert events[0].reported_at == 12


def test_restore_preview_reports_conflicting_sequence_without_overwriting(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    source_store = ContextStore(source)
    target_store = ContextStore(target)
    _message(source_store, "s", 0, "user", "I like amber", 1)
    _message(target_store, "s", 0, "user", "I like violet", 2)
    preview = MemoryRestoreService(source, target).preview()
    assert any(item.table == "messages" and item.severity == "conflict"
               for item in preview.conflicts)
    result = MemoryRestoreService(source, target).import_into()
    assert result.conflicts
    assert {row.content for row in target_store.fetch_range("s")} == {
        "I like violet", "I like amber",
    }


def test_restore_preserves_temporal_history_and_observation_sources(tmp_path):
    source = tmp_path / "source-observations.db"
    target = tmp_path / "target-observations.db"
    store = ContextStore(source)
    first = _message(store, "s", 0, "user", "I like quiet galleries", 100)
    second = _message(store, "s", 1, "user", "I now prefer busy galleries", 200)
    service = ObservationService(store, enabled=True)
    old = service.create("s", ObservationProposal(
        concept="gallery preference", value="quiet galleries", source_ids=[first],
        scope="global", observation_type="preference", occurred_from=90,
        occurred_until=190,
    ))
    new = service.create("s", ObservationProposal(
        concept="gallery preference", value="busy galleries", source_ids=[second],
        scope="global", observation_type="preference",
    ))
    service.supersede(old.observation_id, replacement_id=new.observation_id,
                      reason="user correction", transition_time=200)
    result = MemoryRestoreService(source, target).import_into()
    target_store = ContextStore(target)
    observations = target_store.fetch_observations(scope="global")
    assert {row.value for row in observations} == {"quiet galleries", "busy galleries"}
    assert target_store.fetch_observation_history(observations[0].observation_id)
    assert result.imported["observation_history"] == 1


def test_restore_remaps_entity_and_relationship_endpoints(tmp_path):
    source = tmp_path / "source-relations.db"
    target = tmp_path / "target-relations.db"
    store = ContextStore(source)
    source_id = _message(store, "s", 0, "user", "The atlas belongs to the archive", 10)
    atlas = store.create_entity(
        "s", "global", "PROJECT", "Atlas", "atlas", [source_id],
        status="active", confidence=1.0,
    )
    archive = store.create_entity(
        "s", "global", "PROJECT", "Archive", "archive", [source_id],
        status="active", confidence=1.0,
    )
    relationship = store.create_relationship(
        "s", "global", "part_of", "entity", atlas, "entity", archive,
        [source_id], status="candidate", source_episode_ids=["episode-a"],
    )
    store.confirm_relationship(relationship, transition_time=11)

    result = MemoryRestoreService(source, target).import_into()
    restored = ContextStore(target)
    entities = restored.fetch_entities(scope="global", status="active")
    relations = restored.fetch_relationships(scope="global", status="current")
    assert {item.normalized_name for item in entities} == {"atlas", "archive"}
    assert len(relations) == 1
    assert relations[0].subject_id == result.id_maps["entities"][atlas]
    assert relations[0].object_id == result.id_maps["entities"][archive]
    assert relations[0].source_ids == [result.id_maps["messages"][source_id]]
    assert relations[0].source_episode_ids == ["episode-a"]


def test_learn_reports_already_known_without_a_redundant_candidate(tmp_path):
    manager = ContextManager(db_path=tmp_path / "learn.db")
    first = manager.learn("I like amber", "user", _Jev(), session_id="one")
    second = manager.learn("Please remember that I like amber", "user", _Jev(), session_id="two")
    assert first["status"] == "learned"
    assert second["status"] == "already_known"
    assert second["already_known"][0]["state_id"] > 0
    assert len(manager._get_store().fetch_candidates()) == 1
    assert len(manager._get_store().fetch_states(scope="global", status="current")) == 1


def test_prime_learn_capability_exposes_actual_already_known_status(tmp_path):
    manager = ContextManager(db_path=tmp_path / "prime-learn.db")
    prime = Prime(_Provider(), context_manager=manager, jev=_Jev())
    first = prime.handle(PrimeRequest("learn", {
        "text": "I like amber", "provenance": "user", "session_id": "one",
    }))
    second = prime.handle(PrimeRequest("learn", {
        "text": "Please remember that I like amber", "provenance": "user",
        "session_id": "two",
    }))
    assert first.status is RequestStatus.SUCCESS
    assert first.output["status"] == "learned"
    assert second.status is RequestStatus.SUCCESS
    assert second.output["status"] == "already_known"
    assert second.output["already_known"][0]["state_id"] > 0


def test_learn_keeps_distinct_values_and_reports_mixed_results(tmp_path):
    manager = ContextManager(db_path=tmp_path / "distinct.db")
    manager.learn("I like amber", "user", _Jev(), session_id="s")

    class Mixed(Jev):
        def evaluate(self, request):
            return LearningOutcome.confirmed([
                StructuredCandidate("favorite color", "amber", "PREFERENCE", "global"),
                StructuredCandidate("favorite color", "violet", "PREFERENCE", "global"),
            ])

    summary = manager.learn("I like amber and violet", "user", Mixed(), session_id="s")
    assert summary["status"] == "learned"
    assert summary["already_known"]
    assert len(summary["learned"]) == 1
    assert {row.value for row in manager._get_store().fetch_states(scope="global", status="current")} == {"violet"}


def test_clear_memory_question_does_not_follow_model_clarification(tmp_path):
    plan = {
        "intent": "memory", "operation": "recall", "sources": ["preferences"],
        "subject": None, "recent_days": None, "first_mention": False,
        "reasons": False, "clarify": True, "result_count": None,
        "order": "relevance", "temporal_scope": "current",
    }
    brain = _PlannerBrain(plan)
    result = interpret(brain, "What is my favorite color?", ())
    assert result.clarify is False


def test_regular_cognitive_loop_uses_verified_memory_for_clear_question(tmp_path):
    manager = ContextManager(db_path=tmp_path / "recall.db", hybrid_retrieval=False)
    store = manager._get_store()
    source = _message(store, "old", 0, "user", "My favorite color is amber", 10)
    store.create_state("old", "global", "favorite color", "amber", [source], memory_type="PREFERENCE")
    plan = {
        "intent": "memory", "operation": "recall", "sources": ["preferences"],
        "subject": None, "recent_days": None, "first_mention": False,
        "reasons": False, "clarify": True, "result_count": None,
        "order": "relevance", "temporal_scope": "current",
    }
    from PRIMORA.aurora.loop import CognitiveLoop

    brain = _RecallBrain(plan)
    result = CognitiveLoop(
        Mind(manager), brain, enable_memory_interpreter=True,
    ).run([{"role": "user", "content": "What is my favorite color?"}], "new")
    assert result.success
    assert result.content == "Grounded memory answer"
    assert len(brain.calls) == 2


def test_unresolved_followup_still_clarifies(tmp_path):
    plan = {
        "intent": "memory", "operation": "recall", "sources": ["preferences"],
        "subject": None, "recent_days": None, "first_mention": False,
        "reasons": False, "clarify": True, "result_count": None,
        "order": "relevance", "temporal_scope": "current",
    }
    result = interpret(_PlannerBrain(plan), "What did I say about it?", ())
    assert result.clarify is True


def test_cross_session_evidence_is_available_without_subject_shortcuts(tmp_path):
    manager = ContextManager(db_path=tmp_path / "cross-session.db", hybrid_retrieval=False)
    store = manager._get_store()
    source = _message(store, "first", 0, "user", "My favorite instrument is a glass harp", 10)
    store.create_state("first", "global", "favorite instrument", "glass harp", [source], memory_type="PREFERENCE")
    evidence = manager.memory_query_evidence({
        "sources": ["preferences"], "query": "What is my favorite instrument?",
        "result_count": 1, "order": "relevance", "temporal_scope": "current",
    }, session_id="second")
    assert "glass harp" in evidence
    assert "message" in evidence
