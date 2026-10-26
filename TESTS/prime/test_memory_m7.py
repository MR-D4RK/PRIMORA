"""Disposable-database contracts for Issue #2 M7."""

import pytest

from PRIMORA.prime.context import entities, relationships
from PRIMORA.prime.context.store import ContextStore
from PRIMORA.prime.context.models import RelationEvidencePacket
from PRIMORA.prime.context.relational_retrieval import RelationalRetriever

def test_m7_resolves_aliases_and_marks_inverse_association(tmp_path):
    store = ContextStore(tmp_path / "m7.db")
    source, *_ = store.append("s", 0, "user", "Flarnic is part of the Aurora suite.", timestamp=100)
    flarnic, _ = entities.create_candidate(store, "s", "Flarnic", "PROJECT", [source])
    suite, _ = entities.create_candidate(store, "s", "Aurora suite", "PROJECT", [source])
    entities.confirm(store, flarnic)
    entities.confirm(store, suite)
    alias, _ = entities.add_alias_candidate(store, flarnic, "Flar", "s", [source])
    entities.confirm_alias(store, alias)
    edge, _ = relationships.create_candidate(store, "s", "part_of", flarnic, suite, [source])
    relationships.confirm(store, edge)

    result = RelationalRetriever(store, enabled=True).retrieve("s", "Flar", hop_limit=2)
    assert result.packets and all(isinstance(item, RelationEvidencePacket) for item in result.packets)
    assert any(item.object == "Aurora suite" and not item.inferred for item in result.packets)
    reverse = RelationalRetriever(store, enabled=True).retrieve("s", "Aurora suite", hop_limit=2)
    assert any(item.subject == "Aurora suite" and item.object == "Flarnic" and item.inferred for item in reverse.packets)
    assert all(item.source_ids == [source] for item in result.packets + reverse.packets)


def test_m7_rejects_ambiguous_alias_and_non_user_relationship(tmp_path):
    store = ContextStore(tmp_path / "m7-collision.db")
    one, *_ = store.append("s", 0, "user", "A is here.", timestamp=1)
    two, *_ = store.append("s", 1, "user", "B is here.", timestamp=2)
    a, _ = entities.create_candidate(store, "s", "Alpha", "PROJECT", [one])
    b, _ = entities.create_candidate(store, "s", "Beta", "PROJECT", [two])
    entities.confirm(store, a)
    entities.confirm(store, b)
    aa, _ = entities.add_alias_candidate(store, a, "shared", "s", [one])
    bb, _ = entities.add_alias_candidate(store, b, "shared", "s", [two])
    entities.confirm_alias(store, aa)
    entities.confirm_alias(store, bb)
    result = RelationalRetriever(store, enabled=True).retrieve("s", "shared")
    assert result.packets == [] and "ambiguous_entity_alias" in result.diagnostics


