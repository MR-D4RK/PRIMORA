"""Migration preview and gated retirement on disposable databases only."""
from __future__ import annotations

import sqlite3

import pytest

from PRIMORA.prime.context.store import ContextStore
from PRIMORA.prime.context.memory_integrity_m13 import preview, apply_reviewed


def test_preview_detects_giant_sentence_subject_without_changing_memory(tmp_path):
    path = tmp_path / "malformed.db"
    store = ContextStore(path)
    bad_id, _, _ = store.append(
        "s", 0, "user",
        "I love Cedar Kite I love every aspect of it, including the design.",
    )
    good_id, _, _ = store.append("s", 1, "user", "I love Birch Lantern.")
    bad_state = store.create_state(
        "s", "global",
        "likes for cedar kite i love every aspect of it including the design",
        "loves cedar kite i love every aspect of it including the design",
        [bad_id], memory_type="PREFERENCE",
    )
    good_state = store.create_state(
        "s", "global", "likes for birch lantern", "loves birch lantern",
        [good_id], memory_type="PREFERENCE",
    )
    flagged = preview(store)
    assert len(flagged) == 1
    assert flagged[0]["state_id"] == bad_state
    assert store.fetch_states(state_ids=[bad_state], status="current")
    with pytest.raises(ValueError):
        apply_reviewed(path, [good_state])
    assert store.fetch_states(state_ids=[bad_state], status="current")
    assert not list(tmp_path.glob("*.before-m13.sqlite"))


def test_explicit_retirement_backups_and_preserves_raw_sources(tmp_path):
    path = tmp_path / "malformed.db"
    store = ContextStore(path)
    source_id, _, _ = store.append(
        "s", 0, "user", "I adore Silver Lanterns I adore every aspect of it."
    )
    invalid = store.create_state(
        "s", "global", "likes for silver lanterns i adore every aspect of it",
        "adores silver lanterns i adore every aspect of it",
        [source_id], memory_type="PREFERENCE",
    )
    receipt = apply_reviewed(path, [invalid])
    assert receipt["retired"] == 1
    assert receipt["backed_up"]
    reopened = ContextStore(path)
    assert reopened.fetch_states(state_ids=[invalid], status="current") == []
    assert reopened.fetch_states(state_ids=[invalid], status="superseded")
    assert reopened.fetch_message(source_id).role == "user"
    with sqlite3.connect(receipt["backed_up"]) as con:
        assert con.execute(
            "SELECT status FROM states WHERE id=?", (invalid,)
        ).fetchone()[0] == "current"
    with pytest.raises(ValueError):
        apply_reviewed(path, [invalid])
