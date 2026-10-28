"""Semantic entity projection must preserve the complete store record.

No model server, downloads, or production database. Regression for a
14-column SELECT fed into the 17-column EntityRecord decoder.
"""
from __future__ import annotations

from PRIMORA.prime.context.semantic import SemanticIndex
from PRIMORA.prime.context.store import ContextStore


def test_semantic_index_reads_complete_temporal_entity_record(tmp_path):
    store = ContextStore(tmp_path / "projection.db")
    source_id, inserted, _ = store.append(
        "session-a", 0, "user", "Our project is called Solar Arc."
    )
    assert inserted
    entity_id = store.create_entity(
        "session-a", "global", "PROJECT", "Solar Arc", "solar arc",
        [source_id], status="active",
        source_episode_ids=["episode-one"],
        valid_from=100.0, valid_until=500.0,
    )
    entities = SemanticIndex(store).indexable_entities(scope="global")
    assert len(entities) == 1
    actual = entities[0]
    assert actual.entity_id == entity_id
    assert actual.source_ids == [source_id]
    assert actual.source_episode_ids == ["episode-one"]
    assert actual.valid_from == 100.0
    assert actual.valid_until == 500.0
    assert actual.authority == "user"
    assert actual.status == "active"


def test_semantic_index_excludes_unconfirmed_entities(tmp_path):
    store = ContextStore(tmp_path / "candidate.db")
    source_id, inserted, _ = store.append(
        "session-b", 0, "user", "I might call it the Cloud Ledger."
    )
    assert inserted
    store.create_entity(
        "session-b", "global", "PROJECT", "Cloud Ledger",
        "cloud ledger", [source_id], status="candidate",
    )
    assert SemanticIndex(store).indexable_entities(scope="global") == []
