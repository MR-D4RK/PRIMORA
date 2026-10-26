"""Disposable-database contracts for Issue #2 M6."""
import pytest

from PRIMORA.prime.context.consolidation import ConsolidationService
from PRIMORA.prime.context.observation import ObservationProposal, ObservationService
from PRIMORA.prime.context.store import ContextStore


def test_m6_dry_run_and_replay_are_source_backed_and_idempotent(tmp_path):
    store = ContextStore(tmp_path / "m6.db")
    first, *_ = store.append("one", 0, "user", "I like flarnic ceramics.", timestamp=10)
    second, *_ = store.append("two", 0, "user", "I still like flarnic ceramics.", timestamp=20)
    store.create_state("one", "global", "preferred flarnic ceramics", "flarnic ceramics", [first], memory_type="PREFERENCE", timestamp=10)
    store.create_state("two", "global", "preferred flarnic ceramics", "flarnic ceramics", [second], memory_type="PREFERENCE", timestamp=20)

    service = ConsolidationService(store, enabled=True)
    dry = service.run("one", dry_run=True)
    assert dry.dry_run and dry.applied_ids == []
    assert dry.proposals and dry.proposals[0].proof_count == 2
    assert set(dry.proposals[0].source_ids) == {first, second}
    assert not store.fetch_observations(scope="global")

    applied = service.replay("one", [first, second], dry_run=False)
    assert len(applied.applied_ids) == 1
    row = store.fetch_observations(scope="global", status="candidate")[0]
    assert row.actor == "derived" and row.observation_type == "derived"
    assert row.proof_count == 2
    again = service.run("one", dry_run=False)
    assert again.applied_ids == []
    assert store.fetch_message(first) is not None and store.fetch_message(second) is not None


def test_m6_requires_opt_in_and_merges_support_without_rewriting_value(tmp_path):
    store = ContextStore(tmp_path / "m6-support.db")
    source, *_ = store.append("s", 0, "user", "I like a rare material.", timestamp=10)
    service = ObservationService(store, enabled=True)
    row = service.create("s", ObservationProposal(
        "rare material", "blue", [source], source_quotes={source: "I like a rare material"}
    ))
    second, *_ = store.append("s", 1, "user", "The rare material is still blue.", timestamp=11)
    assert service.add_support(row.observation_id, [second], source_quotes={second: "rare material is still blue"}) == 1
    updated = store.fetch_observations(scope="s")[0]
    assert updated.value == "blue" and updated.source_ids == [source, second]
    with pytest.raises(RuntimeError):
        ConsolidationService(store).run("s")

