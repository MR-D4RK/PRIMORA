import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import ContextManager, WorkingMemory
from PRIMORA.prime.context import state as state_mod
from PRIMORA.prime.context.store import ContextStore

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db(name=None):
    _SEQ[0] += 1
    filename = name or f"temporal-{_SEQ[0]}.db"
    return os.path.join(_TEST_TMPDIR.name, filename)


def _append(store, session_id, seq, content, timestamp):
    message_id, is_new, conflict = store.append(
        session_id, seq, "user", content, timestamp=timestamp
    )
    assert is_new and conflict is None
    return message_id


def _create_state(
    store,
    session_id,
    scope,
    concept,
    value,
    timestamp,
    observed_at=None,
    valid_from=None,
    valid_until=None,
    superseded_at=None,
    status="current",
    authority="user",
    memory_type="FACT",
):
    return store.create_state(
        session_id,
        scope,
        concept,
        value,
        [],
        status=status,
        authority=authority,
        timestamp=timestamp,
        memory_type=memory_type,
        observed_at=observed_at,
        valid_from=valid_from,
        valid_until=valid_until,
        superseded_at=superseded_at,
    )


def test_migration_adds_fields_and_preserves_legacy_uncertainty():
    path = _db("legacy.db")
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            concept TEXT NOT NULL,
            value TEXT NOT NULL,
            source_ids TEXT NOT NULL,
            created_at REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'current',
            supersedes_id INTEGER,
            authority TEXT NOT NULL DEFAULT 'user',
            memory_type TEXT NOT NULL DEFAULT '',
            confidence REAL
        );
        INSERT INTO states
            (id, session_id, scope, concept, value, source_ids, created_at,
             status, supersedes_id, authority, memory_type, confidence)
        VALUES
            (1, 's', 's', 'current fact', 'yes', '1', 100, 'current',
             NULL, 'user', 'FACT', 1.0),
            (2, 's', 's', 'old fact', 'no', '2', 200, 'superseded',
             1, 'user', 'FACT', 1.0),
            (3, 's', 's', 'conflict', 'unknown', '3', 300, 'unresolved',
             NULL, 'assistant', '', 0.3);
        """
    )
    con.close()

    store = ContextStore(path)
    con = sqlite3.connect(path)
    columns = {row[1] for row in con.execute("PRAGMA table_info(states)")}
    con.close()
    assert {
        "observed_at", "valid_from", "valid_until", "superseded_at"
    } <= columns
    rows = {row.state_id: row for row in store.fetch_states()}
    assert rows[1].observed_at == 100
    assert rows[1].valid_from == 100
    assert rows[1].valid_until is None
    assert rows[1].superseded_at is None
    assert rows[2].observed_at is None
    assert rows[2].valid_from is None
    assert rows[2].valid_until is None
    assert rows[2].superseded_at is None
    assert rows[2].supersedes_id == 1
    assert rows[3].observed_at is None
    assert rows[3].valid_from is None
    assert rows[3].valid_until is None
    assert rows[3].superseded_at is None

    reopened = ContextStore(path)
    assert reopened.fetch_states(scope="s", concept="current fact")[0].valid_from == 100
    con = sqlite3.connect(path)
    index_sql = con.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        ("idx_states_temporal",),
    ).fetchone()[0]
    con.close()
    assert "scope, concept, valid_from, valid_until" in index_sql


def test_creation_copies_source_observation_and_keeps_created_distinct():
    store = ContextStore(_db())
    source_id = _append(store, "s", 0, "The interface is browser UI.", 100)
    state_id = store.create_state(
        "s", "s", "interface", "browser UI", [source_id], timestamp=300
    )
    state = store.fetch_states(scope="s", concept="interface")[0]
    assert state.state_id == state_id
    assert state.created_at == 300
    assert state.observed_at == 100
    assert state.valid_from == 100
    assert state.valid_until is None
    assert state.superseded_at is None


def test_trusted_effective_time_and_interval_validation():
    store = ContextStore(_db())
    source_id = _append(store, "s", 0, "The interface is browser UI.", 300)
    state_id = store.create_state(
        "s", "s", "interface", "browser UI", [source_id],
        timestamp=300, valid_from=100,
    )
    state = store.fetch_states(scope="s", concept="interface")[0]
    assert state.state_id == state_id
    assert state.observed_at == 300
    assert state.valid_from == 100
    try:
        store.create_state(
            "s", "s", "invalid", "value", [], timestamp=300,
            valid_from=200, valid_until=100,
        )
        assert False, "invalid interval must be rejected"
    except ValueError as exc:
        assert "valid_until" in str(exc)
    assert store.fetch_states(scope="s", concept="invalid") == []


def test_ordinary_supersession_stamps_both_states():
    store = ContextStore(_db())
    blue_source = _append(store, "s", 0, "The interface is browser UI.", 100)
    blue_id = store.create_state(
        "s", "s", "interface", "browser UI", [blue_source], timestamp=100
    )
    verse_source = _append(store, "s", 1, "The interface is Verse.", 200)
    verse_id = store.create_state_successor(
        "s", "s", "interface", "Verse", [verse_source], [blue_id],
        timestamp=200, transition_time=200,
    )
    rows = {row.state_id: row for row in store.fetch_states(
        scope="s", concept="interface"
    )}
    blue = rows[blue_id]
    verse = rows[verse_id]
    assert blue.status == "superseded"
    assert blue.valid_until == 200
    assert blue.superseded_at == 200
    assert verse.status == "current"
    assert verse.valid_from == 200
    assert verse.valid_until is None
    assert verse.superseded_at is None
    assert verse.supersedes_id == blue_id


def test_successor_and_predecessor_update_are_atomic():
    store = ContextStore(_db())
    first = _create_state(store, "s", "s", "color", "blue", 100)
    second = _create_state(store, "s", "s", "color", "red", 100)
    con = sqlite3.connect(store.db_path)
    con.executescript(
        f"""
        CREATE TRIGGER fail_supersede
        BEFORE UPDATE OF status ON states
        WHEN NEW.id = {second}
        BEGIN
            SELECT RAISE(ABORT, 'forced predecessor failure');
        END;
        """
    )
    con.close()
    try:
        store.create_state_successor(
            "s", "s", "color", "green", [], [first, second],
            timestamp=200, transition_time=200,
        )
        assert False, "forced update failure must roll back"
    except sqlite3.IntegrityError:
        pass
    rows = store.fetch_states(scope="s", concept="color")
    assert len(rows) == 2
    assert all(row.status == "current" for row in rows)
    assert {row.state_id for row in rows} == {first, second}


def test_as_of_uses_half_open_intervals_and_accepts_superseded_history():
    store = ContextStore(_db())
    mgr = ContextManager(db_path=store.db_path)
    blue_source = _append(store, "s", 0, "The interface is browser UI.", 100)
    blue_id = store.create_state(
        "s", "s", "interface", "browser UI", [blue_source], timestamp=100
    )
    verse_source = _append(store, "s", 1, "The interface is Verse.", 200)
    store.create_state_successor(
        "s", "s", "interface", "Verse", [verse_source], [blue_id],
        timestamp=200, transition_time=200,
    )
    assert mgr.current("s", "interface", at_time=150) is None
    assert mgr.current("s", "interface", at_time=250).value == "Verse"
    assert mgr.as_of("s", "interface", 150).value == "browser UI"
    assert mgr.as_of("s", "interface", 199.999).value == "browser UI"
    assert mgr.as_of("s", "interface", 200).value == "Verse"


def test_history_previous_changes_when_changed_and_replacement():
    store = ContextStore(_db())
    mgr = ContextManager(db_path=store.db_path)
    blue_source = _append(store, "s", 0, "The interface is browser UI.", 100)
    blue_id = store.create_state(
        "s", "s", "interface", "browser UI", [blue_source], timestamp=100
    )
    verse_source = _append(store, "s", 1, "The interface is Verse.", 200)
    verse_id = store.create_state_successor(
        "s", "s", "interface", "Verse", [verse_source], [blue_id],
        timestamp=200, transition_time=200,
    )
    assert [row.value for row in mgr.history("s", "interface")] == [
        "browser UI", "Verse"
    ]
    assert mgr.previous("s", "interface").state_id == blue_id
    assert [row.state_id for row in mgr.changes("s", "interface", 100, 200)] == [
        blue_id
    ]
    assert [row.state_id for row in mgr.changes("s", "interface", 200, 201)] == [
        verse_id
    ]
    assert mgr.when_changed("s", "interface") == 200
    assert mgr.what_replaced(blue_id).state_id == verse_id
    assert mgr.what_replaced(verse_id) is None


def test_previous_falls_back_only_when_nearest_revision_is_unique():
    store = ContextStore(_db())
    mgr = ContextManager(db_path=store.db_path)
    old = _create_state(
        store, "s", "s", "color", "blue", 100, valid_from=100,
        status="superseded", valid_until=200, superseded_at=200,
    )
    _create_state(store, "s", "s", "color", "green", 200, valid_from=200)
    assert mgr.previous("s", "color").state_id == old

    ambiguous = ContextStore(_db("ambiguous.db"))
    manager = ContextManager(db_path=ambiguous.db_path)
    _create_state(
        ambiguous, "s", "s", "color", "blue", 100, valid_from=100,
        status="superseded", valid_until=200, superseded_at=200,
    )
    _create_state(
        ambiguous, "s", "s", "color", "red", 100, valid_from=100,
        status="superseded", valid_until=200, superseded_at=200,
    )
    _create_state(ambiguous, "s", "s", "color", "green", 200, valid_from=200)
    assert manager.previous("s", "color") is None


def test_active_memory_excludes_future_and_expired_and_includes_open_interval():
    store = ContextStore(_db())
    mgr = ContextManager(db_path=store.db_path)
    open_id = _create_state(
        store, "s", "s", "open", "yes", 100, valid_from=100
    )
    _create_state(store, "s", "s", "future", "later", 300, valid_from=300)
    _create_state(
        store, "s", "s", "expired", "old", 100,
        valid_from=0, valid_until=100,
    )
    view = mgr.get_active_memory("s", at_time=200)
    state_items = [item for item in view.items if item.kind == "state"]
    assert [item.concept for item in state_items] == ["open"]
    assert state_items[0].state_id == open_id
    assert state_items[0].valid_from == 100
    assert state_items[0].valid_until is None
    assert mgr.current("s", "open", at_time=200).concept == "open"
    assert mgr.current("s", "future", at_time=200) is None
    assert mgr.current("s", "expired", at_time=200) is None


def test_working_memory_preserves_temporal_metadata_and_evidence_time():
    store = ContextStore(_db())
    mgr = ContextManager(db_path=store.db_path)
    source_id = _append(store, "s", 0, "The interface is browser UI.", 100)
    state_id = store.create_state(
        "s", "s", "interface", "browser UI", [source_id], timestamp=100
    )
    item = next(
        entry for entry in mgr.get_active_memory("s", at_time=150).items
        if entry.kind == "state" and entry.concept == "interface"
    )
    wm = WorkingMemory("s", task="temporal check")
    wm.adopt_active(item)
    wm.add_evidence(
        "user", "original evidence", source="msg:1", timestamp=123
    )
    adopted = wm._memories[0]
    assert adopted.state_id == state_id
    assert adopted.created_at == 100
    assert adopted.observed_at == 100
    assert adopted.valid_from == 100
    assert adopted.valid_until is None
    assert adopted.status == "current"
    assert wm._evidence[0]["timestamp"] == 123
    assert store.count("s") == 1
    assert not hasattr(wm, "_store")


def test_session_scopes_remain_isolated():
    store = ContextStore(_db())
    mgr = ContextManager(db_path=store.db_path)
    _create_state(store, "a", "a", "interface", "browser UI", 100)
    _create_state(store, "b", "b", "interface", "Verse", 100)
    assert mgr.current("a", "interface", at_time=150).value == "browser UI"
    assert mgr.current("b", "interface", at_time=150).value == "Verse"
    assert [row.value for row in mgr.history("a", "interface")] == ["browser UI"]
    assert [row.value for row in mgr.history("b", "interface")] == ["Verse"]


def test_global_chain_can_cross_sessions_without_touching_session_chain():
    store = ContextStore(_db())
    mgr = ContextManager(db_path=store.db_path)
    _create_state(store, "a", "a", "interface", "session value", 100)
    global_source = _append(store, "a", 0, "Global interface.", 100)
    global_id = store.create_state(
        "a", "global", "interface", "browser UI", [global_source], timestamp=100
    )
    correction_source = _append(store, "b", 0, "Global correction.", 200)
    successor_id = store.create_state_successor(
        "b", "global", "interface", "Verse", [correction_source], [global_id],
        timestamp=200, transition_time=200,
    )
    history = mgr.history("global", "interface")
    assert [row.session_id for row in history] == ["a", "b"]
    assert mgr.as_of("global", "interface", 150).value == "browser UI"
    assert mgr.what_replaced(global_id).state_id == successor_id
    assert mgr.current("a", "interface", at_time=250).value == "session value"


def test_trusted_retroactive_boundary_preserves_observation_time():
    store = ContextStore(_db())
    mgr = ContextManager(db_path=store.db_path)
    january_source = _append(
        store, "s", 0, "The interface is browser UI.", 100
    )
    blue_id = store.create_state(
        "s", "s", "interface", "browser UI", [january_source], timestamp=100
    )
    september_source = _append(
        store, "s", 1, "Trusted correction.", 900
    )
    candidate = state_mod.StateCandidate(
        concept="interface",
        value="Verse",
        kind="correct",
        scope="session",
        mtype="DECISION",
        auto_confirm=True,
        confidence=1.0,
        valid_from=200,
        transition_time=900,
    )
    diagnostics = state_mod.apply_user_candidates(
        store, "s", september_source, [candidate]
    )
    assert not any(item.startswith("state_failed") for item in diagnostics)
    rows = {row.state_id: row for row in store.fetch_states(
        scope="s", concept="interface"
    )}
    successor = next(row for row in rows.values() if row.status == "current")
    old = rows[blue_id]
    assert old.observed_at == 100
    assert old.valid_until == 200
    assert old.superseded_at == 900
    assert successor.observed_at == 900
    assert successor.valid_from == 200
    assert successor.created_at != 900
    assert mgr.as_of("s", "interface", 150).value == "browser UI"
    assert mgr.as_of("s", "interface", 250).value == "Verse"


def test_trusted_invalidation_boundary_does_not_create_successor():
    store = ContextStore(_db())
    state_id = _create_state(
        store, "s", "s", "interface", "browser UI", 100, valid_from=100
    )
    september_source = _append(
        store, "s", 0, "Trusted invalidation.", 900
    )
    candidate = state_mod.StateCandidate(
        concept="interface",
        kind="invalidate",
        scope="session",
        auto_confirm=True,
        valid_until=200,
        transition_time=900,
    )
    diagnostics = state_mod.apply_user_candidates(
        store, "s", september_source, [candidate]
    )
    assert not any(item.startswith("state_failed") for item in diagnostics)
    rows = store.fetch_states(scope="s", concept="interface")
    assert len(rows) == 1
    assert rows[0].state_id == state_id
    assert rows[0].status == "superseded"
    assert rows[0].valid_until == 200
    assert rows[0].superseded_at == 900


def test_natural_language_temporal_parsing_remains_deferred():
    mgr = ContextManager(db_path=_db())
    diagnostics = mgr.reconcile(
        [{"role": "user", "content": "I stopped using X in March."}], "s"
    )
    assert not any(item.startswith("state_failed") for item in diagnostics)
    assert mgr._get_store().count("s") == 1
    assert mgr._get_store().fetch_states() == []


def test_pending_unresolved_and_assistant_content_never_become_valid():
    store = ContextStore(_db())
    mgr = ContextManager(db_path=store.db_path)
    user_id = _create_state(
        store, "s", "s", "color", "blue", 100, valid_from=100
    )
    _create_state(
        store, "s", "s", "color", "green", 200,
        status="unresolved", authority="assistant", valid_from=100,
    )
    store.create_candidate(
        "s", "s", [], "PREFERENCE", "preference", "red",
        authority="user", status="pending",
    )
    assert mgr.current("s", "color", at_time=150).state_id == user_id
    assert mgr.as_of("s", "color", 150).state_id == user_id
    assert [row.state_id for row in mgr.history("s", "color")] == [user_id]
    active_states = [
        item for item in mgr.get_active_memory("s", at_time=150).items
        if item.kind == "state"
    ]
    assert [item.state_id for item in active_states] == [user_id]


def test_reconciliation_is_idempotent_for_temporal_state():
    mgr = ContextManager(db_path=_db())
    snapshot = [{
        "role": "user",
        "content": "The interface is browser UI.",
        "timestamp": 100,
    }]
    mgr.reconcile(snapshot, "s")
    first = mgr._get_store().fetch_states(
        scope="s", concept="interface", status="current"
    )[0]
    mgr.reconcile(snapshot, "s")
    rows = mgr._get_store().fetch_states(scope="s", concept="interface")
    assert len(rows) == 1
    assert rows[0] == first
    assert rows[0].observed_at == 100
    assert rows[0].valid_from == 100
    assert mgr._get_store().count("s") == 1


def test_runtime_interface_temporal_chain():
    store = ContextStore(_db())
    mgr = ContextManager(db_path=store.db_path)
    january_source = _append(
        store, "runtime", 0, "I use the browser UI.", 100
    )
    browser_id = store.create_state(
        "runtime", "runtime", "interface", "browser UI",
        [january_source], timestamp=100,
    )
    march_source = _append(
        store, "runtime", 1, "I switched to Verse.", 200
    )
    verse_id = store.create_state_successor(
        "runtime", "runtime", "interface", "Verse",
        [march_source], [browser_id], timestamp=200, transition_time=200,
    )
    assert mgr.current("runtime", "interface").state_id == verse_id
    assert mgr.as_of("runtime", "interface", 150).state_id == browser_id
    assert [row.state_id for row in mgr.history("runtime", "interface")] == [
        browser_id, verse_id
    ]
    assert mgr.previous("runtime", "interface").state_id == browser_id
    assert mgr.when_changed("runtime", "interface") == 200
    assert mgr.what_replaced(browser_id).state_id == verse_id
    stored = store.fetch_states(scope="runtime", concept="interface")
    assert [row.observed_at for row in stored] == [100, 200]
    assert [row.valid_from for row in stored] == [100, 200]
    assert stored[0].valid_until == 200
    assert stored[0].superseded_at == 200


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("All temporal memory tests passed!")
