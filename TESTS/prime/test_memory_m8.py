"""Disposable-database contracts for Issue #2 M8."""

import pytest

from PRIMORA.prime.context import entities, relationships
from PRIMORA.prime.context.store import ContextStore
from PRIMORA.prime.context.observation import ObservationProposal, ObservationService
from PRIMORA.prime.context.memory_audit import MemoryAuditService


def test_m8_is_opt_in(tmp_path):
    service = MemoryAuditService(ContextStore(tmp_path / "disabled.db"))
    with pytest.raises(RuntimeError):
        service.run()

def test_m8_reports_conflicts_duplicates_insufficient_evidence_and_candidates(tmp_path):
    store = ContextStore(tmp_path / "m8.db")
    first, *_ = store.append("s", 0, "user", "I like one flarnic color.", timestamp=10)
    second, *_ = store.append("s", 1, "user", "I like another flarnic color.", timestamp=11)
    store.create_state("s", "s", "flarnic color", "red", [first], memory_type="PREFERENCE", timestamp=10)
    store.create_state("s", "s", "flarnic color", "blue", [second], memory_type="PREFERENCE", timestamp=11)
    store.create_state("s", "s", "flarnic shape", "round", [first], memory_type="PREFERENCE", timestamp=10)
    store.create_state("s", "s", "flarnic shape", "round", [second], memory_type="PREFERENCE", timestamp=11)
    store.create_candidate("s", "s", [first], "preference", "pending thing", "maybe")
    ObservationService(store, enabled=True).create("s", ObservationProposal(
        "derived thing", "value", [first], actor="derived", status="candidate",
        observation_type="derived", source_quotes={first: "I like one flarnic color"},
    ))
    report = MemoryAuditService(store, enabled=True).run(scope="s")
    kinds = {item.kind for item in report.findings}
    assert {"conflict", "duplicate_fact", "insufficient_evidence", "unresolved_candidate"}.issubset(kinds)
    assert report.proposals and "read_only=true" in report.diagnostics
    assert store.fetch_message(first) is not None


