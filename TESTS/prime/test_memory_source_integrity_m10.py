"""Source-integrity regressions for Prime's new user-authority writes.

All records here use a disposable SQLite store. No production memory is opened.
"""
from __future__ import annotations

import pytest

from PRIMORA.prime.context.store import ContextStore


@pytest.fixture
def evidence(tmp_path):
    store = ContextStore(tmp_path / "source-integrity.db")
    user_id, _, _ = store.append("session-a", 0, "user", "I prefer the blue material.")
    other_id, _, _ = store.append("session-b", 0, "user", "I prefer a different material.")
    assistant_id, _, _ = store.append("session-a", 1, "assistant", "I think you prefer green.")
    return store, user_id, other_id, assistant_id


@pytest.mark.parametrize("source_case", ["empty", "missing", "other-session", "assistant"])
def test_state_and_candidate_deny_untrusted_new_sources(evidence, source_case):
    store, user_id, other_id, assistant_id = evidence
    sources = {
        "empty": [],
        "missing": [987654321],
        "other-session": [other_id],
        "assistant": [assistant_id],
    }[source_case]
    with pytest.raises(ValueError):
        store.create_state("session-a", "global", "material preference", "blue", sources)
    with pytest.raises(ValueError):
        store.create_candidate(
            "session-a", "global", sources, "PREFERENCE",
            "material preference", "blue", authority="user",
        )
    assert not store.fetch_states(scope="global", status="current")
    assert not store.fetch_candidates(status="pending")
    # The same message must still be able to back its own user-authority fact.
    assert store.fetch_message(user_id).role == "user"


def test_valid_user_source_is_admitted_for_session_or_global(evidence):
    store, user_id, _, _ = evidence
    session_state = store.create_state(
        "session-a", "session-a", "preferred shade", "blue", [user_id]
    )
    global_state = store.create_state(
        "session-a", "global", "preferred material", "blue", [user_id]
    )
    assert session_state > 0 and global_state > 0


def test_session_scope_cannot_be_forged(evidence):
    store, user_id, _, _ = evidence
    with pytest.raises(ValueError, match="scope"):
        store.create_state("session-a", "session-b", "preference", "blue", [user_id])


def test_successor_and_explanation_reject_fabricated_sources(evidence):
    store, user_id, other_id, _ = evidence
    state_id = store.create_state(
        "session-a", "global", "material preference", "blue", [user_id]
    )
    with pytest.raises(ValueError):
        store.create_state_successor(
            "session-a", "global", "material preference", "green",
            [other_id], [state_id],
        )
    with pytest.raises(ValueError):
        store.create_state_with_explains(
            "session-a", "global", "reason for material preference",
            "a supposed reason", [987654321], state_id,
        )
    state = store.fetch_states(scope="global", status="current")[0]
    assert state.state_id == state_id
    assert state.value == "blue"


def test_entity_alias_and_relationship_reject_missing_sources(evidence):
    store, user_id, _, _ = evidence
    entity = store.create_entity(
        "session-a", "global", "PERSON", "Mira", "mira",
        [user_id], status="active",
    )
    with pytest.raises(ValueError):
        store.create_entity(
            "session-a", "global", "PERSON", "Nora", "nora",
            [987654321], status="active",
        )
    with pytest.raises(ValueError):
        store.create_entity_alias(
            entity, "session-a", "global", "Mir", "mir", [987654321],
        )
    with pytest.raises(ValueError):
        store.create_relationship(
            "session-a", "global", "part_of",
            "entity", entity, "entity", entity, [987654321],
        )


def test_nonuser_authority_records_are_not_relabelled_user(evidence):
    store, _, _, assistant_id = evidence
    candidate_id = store.create_candidate(
        "session-a", "session-a", [assistant_id], "FACT",
        "assistant guess", "green", authority="assistant",
    )
    candidate = store.fetch_candidates(candidate_id=candidate_id)[0]
    assert candidate.authority == "assistant"


def test_current_observation_needs_real_user_message(evidence):
    store, user_id, other_id, _ = evidence
    with pytest.raises(ValueError, match="user message source"):
        store.create_observation(
            "session-a", "global", "likes material", "blue", [],
            source_episode_ids=["unverified-episode"], authority="user",
        )
    with pytest.raises(ValueError, match="crosses session"):
        store.create_observation(
            "session-a", "global", "likes material", "blue", [other_id],
            authority="user",
        )
    valid_id = store.create_observation(
        "session-a", "global", "likes material", "blue",
        [user_id], authority="user",
    )
    assert valid_id > 0


def test_legacy_candidate_cannot_be_confirmed_with_now_invalid_source(evidence):
    import sqlite3

    store, user_id, _, _ = evidence
    entity_id = store.create_entity(
        "session-a", "global", "PERSON", "Mira", "mira",
        [user_id], status="candidate",
    )
    # Simulate an old/imported candidate referencing a missing source row.
    # Existing rows are readable; promotion is the point where they must
    # demonstrate real user evidence.
    with sqlite3.connect(str(store._db_path)) as con:
        con.execute(
            "UPDATE entities SET source_ids=? WHERE id=?",
            ("987654321", entity_id),
        )
    with pytest.raises(ValueError, match="existing user source"):
        store.confirm_entity(entity_id)
