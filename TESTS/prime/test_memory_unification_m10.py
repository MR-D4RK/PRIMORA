"""M10 memory-unification regression coverage.

These tests use disposable SQLite stores and a deterministic provider double;
they exercise the same Prime admission and durable formation seams used by the
live Qwen path without touching the production database.
"""

from __future__ import annotations

import json

import pytest

from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.memory_restore import MemoryRestoreService
from PRIMORA.prime.context.store import ContextStore


class _FormationProvider:
    def __init__(self, *, failures: int = 0, source_id: int | None = None):
        self.failures = failures
        self.source_id = source_id
        self.calls = 0

    def invoke(self, capability, payload):
        assert capability == "memory_formation"
        self.calls += 1
        if self.failures:
            self.failures -= 1
            return PrimeResult.failure("temporary formation outage")
        source_id = self.source_id or payload["source_message_id"]
        source = payload["messages"][-1]["content"]
        return PrimeResult.success({
            "content": json.dumps({"candidates": [{
                "kind": "preference",
                "concept": "preferred material",
                "value": "opal fiber",
                "claim_kind": "assertion",
                "actor": "user",
                "scope": "global",
                "source_ids": [source_id],
                "source_quote": source,
            }]})
        })


def _manager(path, provider, *, deferred=True):
    manager = ContextManager(
        db_path=path,
        enable_memory_v2=True,
        formation_provider=provider,
    )
    manager.set_formation_deferred(deferred)
    return manager


def test_user_authority_cannot_reference_assistant_source(tmp_path):
    store = ContextStore(tmp_path / "authority.db")
    message_id, _, _ = store.append("s", 0, "assistant", "The user's material is opal fiber.")

    with pytest.raises(ValueError, match="requires a user source"):
        store.create_state("s", "global", "preferred material", "opal fiber", [message_id])
    with pytest.raises(ValueError, match="requires a user source"):
        store.create_candidate(
            "s", "global", [message_id], "PREFERENCE",
            "preferred material", "opal fiber", authority="user",
        )


def test_deferred_formation_survives_restart_and_is_idempotent(tmp_path):
    db = tmp_path / "deferred.db"
    first_provider = _FormationProvider()
    first = _manager(db, first_provider)
    first.reconcile([{"role": "user", "content": "I prefer opal fiber."}], "s")

    assert first_provider.calls == 0
    pending = first.formation_jobs("s", "pending")
    assert len(pending) == 1
    message_id = pending[0]["message_id"]

    restarted_provider = _FormationProvider()
    restarted = _manager(db, restarted_provider)
    diagnostics = restarted.drain_formation()
    assert restarted_provider.calls == 1
    assert any(item.startswith("typed_formation:") for item in diagnostics)
    assert restarted.formation_jobs("s", "complete")[0]["message_id"] == message_id

    # Replaying the same immutable transcript is a no-op, and a completed
    # durable job is not called a second time.
    restarted.reconcile([{"role": "user", "content": "I prefer opal fiber."}], "s")
    restarted.drain_formation()
    assert restarted_provider.calls == 1


def test_failed_formation_is_retryable_without_duplicate_source_rows(tmp_path):
    db = tmp_path / "retry.db"
    provider = _FormationProvider(failures=1)
    manager = _manager(db, provider)
    manager.reconcile([{"role": "user", "content": "I prefer opal fiber."}], "s")

    first = manager.drain_formation()
    assert any("provider_failure" in item for item in first)
    assert manager.formation_jobs("s", "failed")[0]["attempts"] == 1

    second = manager.drain_formation()
    assert provider.calls == 2
    assert manager.formation_jobs("s", "complete")[0]["attempts"] == 2
    assert not manager.formation_jobs("s", "failed")
    assert len(manager._get_store().fetch_range("s")) == 1
    assert any("typed_formation_timing" in item for item in second)


def test_wrong_provider_source_is_rejected_without_admission(tmp_path):
    db = tmp_path / "provenance.db"
    store = ContextStore(db)
    assistant_id, _, _ = store.append("s", 0, "assistant", "untrusted output")
    provider = _FormationProvider(source_id=assistant_id)
    manager = _manager(db, provider)
    manager.reconcile([{"role": "user", "content": "I prefer opal fiber."}], "s")
    manager.drain_formation()

    states = manager._get_store().fetch_states(scope="global", status="current")
    assert not any(state.concept == "preferred material" for state in states)


def test_user_learning_requires_immutable_user_evidence(tmp_path):
    class _Jev:
        def evaluate(self, request):
            from PRIMORA.learning.jev import LearningOutcome, StructuredCandidate
            return LearningOutcome(
                disposition="confirm",
                candidates=[StructuredCandidate(
                    concept="preferred material", value="opal fiber",
                    memory_type="PREFERENCE", scope="global",
                )],
            )

    manager = ContextManager(db_path=tmp_path / "learn.db")
    rejected = manager.learn("I prefer opal fiber.", "user", _Jev(), session_id="s")
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "user_provenance_requires_matching_user_source"

    manager.reconcile([{"role": "user", "content": "I prefer opal fiber."}], "s")
    accepted = manager.learn("I prefer opal fiber.", "user", _Jev(), session_id="s")
    assert accepted["status"] in {"learned", "already_known"}
    assert all(
        state.authority == "user"
        for state in manager._get_store().fetch_states(scope="global", status="current")
    )


def test_restore_preserves_deferred_job_and_remaps_message_id(tmp_path):
    source_path = tmp_path / "source.db"
    target_path = tmp_path / "target.db"
    source = ContextStore(source_path)
    source.ensure_session("source-session")
    source_id, _, _ = source.append(
        "source-session", 0, "user", "I prefer opal fiber."
    )
    assert source.enqueue_formation_job(source_id, "source-session")

    service = MemoryRestoreService(source_path, target_path)
    preview = service.preview()
    assert preview.safe_to_import
    result = service.import_into()
    assert result.changed
    assert result.imported["formation_jobs"] == 1
    source_job_id = source.formation_jobs("source-session")[0]["job_id"]
    assert source_job_id in result.id_maps["formation_jobs"]

    destination = ContextStore(target_path)
    jobs = destination.formation_jobs("source-session")
    assert len(jobs) == 1
    assert result.id_maps["messages"][source_id] == jobs[0]["message_id"]
    assert destination.fetch_message(jobs[0]["message_id"]).content == "I prefer opal fiber."
