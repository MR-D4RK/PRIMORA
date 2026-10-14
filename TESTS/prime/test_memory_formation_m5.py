"""M5 automatic formation from conversation messages.

Covers the live path `reconcile -> _extract_states -> _extract_entities`:
explicit entity formation, exact-name reuse, scope isolation, alias candidacy,
part_of confirmation, memory-to-entity concerns links, assistant silence,
provenance, idempotency, and the guarantee that M4 state formation is
unchanged.

Every test uses a temporary database. The production database is never
opened.
"""

import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context import entity_formation as ef
from PRIMORA.prime.context import state as state_mod
from PRIMORA.prime.context.store import ContextStore

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db(name=None):
    _SEQ[0] += 1
    return os.path.join(_TEST_TMPDIR.name, f"{name or 'm5form'}-{_SEQ[0]}.db")


def _manager(name=None):
    path = _db(name)
    store = ContextStore(path)
    return ContextManager(db_path=path), store


def _say(manager, text, session="s", role="user", timestamp=1000.0):
    """Append one message the way a real caller does: a growing snapshot."""
    history = getattr(manager, "_hist", None)
    if history is None or getattr(manager, "_hist_session", None) != session:
        history = []
        manager._hist_session = session
    history.append({"role": role, "content": text, "timestamp": timestamp})
    manager._hist = history
    return manager.reconcile(list(history), session)


def _rows(store, table, where="", params=()):
    with store._connect() as con:
        cur = con.execute(f"SELECT * FROM {table} {where} ORDER BY id", params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _by_name(store, session, name):
    for row in _rows(store, "entities", "WHERE scope = ?", (session,)):
        if row["normalized_name"] == name:
            return row
    return None


# ---------------------------------------------------------------- 1..5 types
class TestExplicitEntityFormation:
    def test_explicit_person_creation(self):
        manager, store = _manager()
        _say(manager, "My name is Amir.")
        row = _by_name(store, "s", "amir")
        assert row is not None
        assert row["entity_type"] == "PERSON"
        assert row["display_name"] == "Amir"
        assert row["status"] == "active", "explicit assertion takes create->confirm"
        assert row["authority"] == "user"

    def test_explicit_project_creation(self):
        manager, store = _manager()
        _say(manager, "My project is PRIMORA.")
        row = _by_name(store, "s", "primora")
        assert row is not None and row["entity_type"] == "PROJECT"

    def test_explicit_agent_creation(self):
        manager, store = _manager()
        _say(manager, "Aurora is an agent.")
        row = _by_name(store, "s", "aurora")
        assert row is not None and row["entity_type"] == "AGENT"

    def test_explicit_component_creation(self):
        manager, store = _manager()
        _say(manager, "Verse is a component.")
        row = _by_name(store, "s", "verse")
        assert row is not None and row["entity_type"] == "COMPONENT"

    def test_file_extraction_from_path(self):
        manager, store = _manager()
        _say(manager, "look at C:\\PRIMORA PROJECT\\PRIMORA\\prime\\context\\manager.py")
        row = _by_name(store, "s", "manager.py")
        assert row is not None, "a path basename is an explicit file reference"
        assert row["entity_type"] == "FILE"

    def test_file_extraction_collects_every_path(self):
        manager, store = _manager()
        _say(manager, "check ./a/b/hybrid.py and ./c/store.py")
        assert _by_name(store, "s", "hybrid.py") is not None
        assert _by_name(store, "s", "store.py") is not None

    def test_no_entity_without_explicit_type_wording(self):
        manager, store = _manager()
        for text in ("the Verse interface", "Aurora's world",
                     "the world of Verse", "the project is nice",
                     "he is an agent of some kind", "manager.py",
                     "Sounds good, thanks!"):
            _say(manager, text, timestamp=1000.0 + len(text))
        assert _rows(store, "entities") == [], (
            "no explicit type assertion means no entity"
        )

    def test_demonstrative_is_not_an_entity_name(self):
        """Regression: 'This is an agent.' created an entity 'This'."""
        assert ef._extract_entity_assertions("This is an agent.") == []
        assert ef._extract_entity_assertions("It is a file.") == []
        manager, store = _manager()
        _say(manager, "This is an agent.")
        assert _rows(store, "entities") == []

    def test_sentence_terminator_does_not_block_formation(self):
        """Regression: the segmenter kept '.', so \\s*$ patterns never matched."""
        assert ef._extract_entity_assertions("Aurora is an agent.") == [
            ("Aurora", "AGENT")]


# ---------------------------------------------------------------- 6 reuse
class TestEntityReuse:
    def test_exact_duplicate_entity_is_reused_not_duplicated(self):
        manager, store = _manager()
        _say(manager, "My name is Amir.")
        first = _by_name(store, "s", "amir")
        diags = _say(manager, "My name is Amir.", timestamp=1100.0)
        assert any("entity_reused" in d for d in diags)
        rows = _rows(store, "entities")
        assert len(rows) == 1
        assert rows[0]["id"] == first["id"]

    def test_case_and_punctuation_variants_reuse_the_same_entity(self):
        manager, store = _manager()
        _say(manager, "My name is Amir.")
        _say(manager, "My name is amir!", timestamp=1100.0)
        _say(manager, "My name is 'Amir'.", timestamp=1200.0)
        assert len(_rows(store, "entities")) == 1

    def test_similar_names_are_not_merged(self):
        manager, store = _manager()
        _say(manager, "My name is Amir.")
        _say(manager, "My name is Amira.", timestamp=1100.0)
        names = {r["normalized_name"] for r in _rows(store, "entities")}
        assert names == {"amir", "amira"}
        assert all(r["merged_into_entity_id"] is None
                   for r in _rows(store, "entities"))

    def test_formation_never_calls_merge(self):
        """Structural guarantee: no merge entry point is reachable."""
        source = open(
            os.path.join("PRIMORA", "prime", "context", "entity_formation.py"),
            encoding="utf-8").read()
        assert ".merge(" not in source
        assert "entities_mod.merge" not in source
        assert "merge_entities" not in source


# ---------------------------------------------------------------- 7..8 scope
class TestScope:
    def test_default_formation_is_session_scoped(self):
        manager, store = _manager()
        _say(manager, "My name is Amir.")
        assert _by_name(store, "s", "amir") is not None
        assert _by_name(store, "global", "amir") is None

    def test_formation_isolated_between_sessions(self):
        manager, store = _manager()
        _say(manager, "My name is Amir.", session="s1")
        assert _by_name(store, "s1", "amir") is not None
        assert _by_name(store, "s2", "amir") is None

    def test_no_automatic_cross_session_merge(self):
        manager, store = _manager()
        _say(manager, "My name is Amir.", session="s1")
        _say(manager, "My name is Amir.", session="s2", timestamp=2000.0)
        a = _by_name(store, "s1", "amir")
        b = _by_name(store, "s2", "amir")
        assert a is not None and b is not None
        assert a["id"] != b["id"], "same name in two scopes is two entities"
        assert a["merged_into_entity_id"] is None
        assert b["merged_into_entity_id"] is None

    def test_no_automatic_promotion_to_global(self):
        manager, store = _manager()
        _say(manager, "My name is Amir.")
        assert _rows(store, "entities", "WHERE scope = 'global'") == []

    def test_cross_session_part_of_is_refused(self):
        """Endpoints are resolved inside one scope, so a foreign scope fails."""
        manager, store = _manager()
        _say(manager, "Verse is a component.", session="s1")
        diags = _say(manager, "Verse is part of PRIMORA.", session="s2",
                     timestamp=2000.0)
        assert any("relationship_skipped_part_of" in d for d in diags)
        assert _rows(store, "relationships") == []


# ---------------------------------------------------------------- 9..10 alias
class TestAliasFormation:
    def test_explicit_alias_becomes_a_candidate(self):
        manager, store = _manager()
        _say(manager, "My project is PRIMORA.")
        diags = _say(manager, "PRIMORA is also called Prime.", timestamp=1100.0)
        assert any("alias_candidate_created" in d for d in diags)
        rows = _rows(store, "entity_aliases")
        assert len(rows) == 1
        assert rows[0]["alias_text"] == "Prime"
        assert rows[0]["status"] == "candidate", (
            "an alias stays a candidate; there is no defined safe "
            "deterministic auto-confirmation"
        )
        assert rows[0]["authority"] == "user"

    def test_i_call_alias_splits_on_the_last_determiner(self):
        """Regression: name/alias split produced name='the'."""
        assert ef._extract_alias_assertions(
            "I call the Verse interface the Verse world."
        ) == [("the Verse interface", "the Verse world")]

    def test_non_explicit_phrases_never_become_aliases(self):
        for text in ("the Verse interface", "Aurora's world",
                     "the world of Verse"):
            assert ef._extract_alias_assertions(text) == []
        manager, store = _manager()
        _say(manager, "Verse is a component.")
        for i, text in enumerate(("the Verse interface", "Aurora's world",
                                  "the world of Verse"), start=2):
            _say(manager, text, timestamp=1000.0 + i)
        assert _rows(store, "entity_aliases") == []

    def test_alias_without_an_established_entity_is_skipped(self):
        manager, store = _manager()
        diags = _say(manager, "Nonexistent is also called Something.")
        assert any("alias_skipped" in d for d in diags)
        assert _rows(store, "entity_aliases") == []

    def test_ambiguous_i_call_is_refused(self):
        assert ef._extract_alias_assertions(
            "I call something something else entirely here now") == []


# ---------------------------------------------------------------- 11..13 rel
class TestRelationshipFormation:
    def test_part_of_is_formed_and_confirmed(self):
        manager, store = _manager()
        _say(manager, "Verse is a component. My project is PRIMORA.")
        _say(manager, "Verse is part of PRIMORA.", timestamp=1100.0)
        rows = [r for r in _rows(store, "relationships")
                if r["relationship_type"] == "part_of"]
        assert len(rows) == 1
        rel = rows[0]
        assert rel["status"] == "current", "an explicit assertion is confirmed"
        assert rel["authority"] == "user"
        assert rel["subject_kind"] == "entity" and rel["object_kind"] == "entity"
        verse = _by_name(store, "s", "verse")
        primora = _by_name(store, "s", "primora")
        assert rel["subject_id"] == verse["id"]
        assert rel["object_id"] == primora["id"]

    def test_part_of_is_found_inside_a_longer_message(self):
        manager, store = _manager()
        _say(manager, "Verse is a component. My project is PRIMORA.")
        diags = _say(manager, "Sure! Verse is part of PRIMORA. Thanks.",
                     timestamp=1100.0)
        assert any("part_of" in d and "created" in d for d in diags)

    def test_part_of_needs_both_endpoints_resolvable(self):
        manager, store = _manager()
        diags = _say(manager, "Verse is part of Nothing.")
        assert any("relationship_skipped_part_of" in d for d in diags)
        assert _rows(store, "relationships") == []

    def test_concerns_links_state_to_entity_from_the_same_message(self):
        manager, store = _manager()
        _say(manager, "My project is PRIMORA. The deployment window is Friday.")
        rows = [r for r in _rows(store, "relationships")
                if r["relationship_type"] == "concerns"]
        assert rows, "a state from the same message naming the entity links"
        for rel in rows:
            assert rel["subject_kind"] == "memory_state"
            assert rel["object_kind"] == "entity"
            assert rel["status"] == "candidate", (
                "containment is an inference, so it stays a candidate"
            )
            assert rel["authority"] == "user"

    def test_concerns_does_not_duplicate_memory_text(self):
        manager, store = _manager()
        _say(manager, "My project is PRIMORA. The deployment window is Friday.")
        for rel in _rows(store, "relationships"):
            if rel["relationship_type"] == "concerns":
                blob = " ".join(str(v) for v in rel.values())
                assert "deployment window" not in blob

    def test_explains_mechanism_is_untouched(self):
        """The M4 rationale path must still create its own explains edge."""
        manager, store = _manager()
        _say(manager, "The vault code is BLUE because it rotates monthly.")
        rows = [r for r in _rows(store, "relationships")
                if r["relationship_type"] == "explains"]
        assert rows, "M4 rationale still creates explains"
        assert all(r["subject_kind"] == "memory_state" for r in rows)
        assert all(r["object_kind"] == "memory_state" for r in rows)
        # And formation did not add a second rationale system.
        source = open(
            os.path.join("PRIMORA", "prime", "context", "entity_formation.py"),
            encoding="utf-8").read()
        assert "explains" not in source

    def test_no_generic_related_to_relationship(self):
        manager, store = _manager()
        _say(manager, "My name is Amir. My project is PRIMORA.")
        _say(manager, "Amir and PRIMORA are friends.", timestamp=1100.0)
        types = {r["relationship_type"] for r in _rows(store, "relationships")}
        assert types <= {"part_of", "concerns", "explains"}
        assert "related_to" not in types


# ---------------------------------------------------------------- 14 assistant
class TestAssistantSilence:
    def test_assistant_message_establishes_no_entity(self):
        manager, store = _manager()
        diags = _say(manager, "My name is Amir.")
        diags = _say(manager, "My name is Bob. Bob is an agent. "
                              "Widget is part of Gadget.",
                     role="assistant", timestamp=1100.0)
        assert not any("entity_candidate_created" in d for d in diags)
        assert _by_name(store, "s", "bob") is None
        assert _by_name(store, "s", "widget") is None
        assert _by_name(store, "s", "gadget") is None

    def test_assistant_message_establishes_no_relationship(self):
        manager, store = _manager()
        _say(manager, "Verse is a component. My project is PRIMORA.")
        _say(manager, "Verse is part of PRIMORA.", role="assistant",
             timestamp=1100.0)
        assert [r for r in _rows(store, "relationships")
                if r["relationship_type"] == "part_of"] == []

    def test_assistant_message_establishes_no_alias(self):
        manager, store = _manager()
        _say(manager, "My project is PRIMORA.")
        _say(manager, "PRIMORA is also called Prime.", role="assistant",
             timestamp=1100.0)
        assert _rows(store, "entity_aliases") == []

    def test_system_and_tool_roles_are_ignored(self):
        manager, store = _manager()
        for i, role in enumerate(("system", "tool"), start=2):
            _say(manager, "My name is Zed.", role=role, timestamp=1000.0 + i)
        assert _by_name(store, "s", "zed") is None


# ---------------------------------------------------------------- 15 provenance
class TestProvenance:
    def test_entity_provenance_points_at_the_source_user_message(self):
        manager, store = _manager()
        _say(manager, "My name is Amir.")
        row = _by_name(store, "s", "amir")
        ids = [int(v) for v in row["source_ids"].split(",") if v.strip()]
        assert len(ids) == 1
        with store._connect() as con:
            msg = con.execute(
                "SELECT session_id, role, content FROM messages WHERE id=?",
                (ids[0],)).fetchone()
        assert msg is not None
        assert msg[0] == "s" and msg[1] == "user"
        assert "Amir" in msg[2]

    def test_relationship_provenance_points_at_the_source_user_message(self):
        manager, store = _manager()
        _say(manager, "Verse is a component. My project is PRIMORA.")
        _say(manager, "Verse is part of PRIMORA.", timestamp=1100.0)
        rel = [r for r in _rows(store, "relationships")
               if r["relationship_type"] == "part_of"][0]
        ids = [int(v) for v in rel["source_ids"].split(",") if v.strip()]
        assert ids
        with store._connect() as con:
            roles = [r[0] for r in con.execute(
                "SELECT role FROM messages WHERE id IN (%s)"
                % ",".join("?" * len(ids)), ids).fetchall()]
        assert roles == ["user"]

    def test_confirmed_rows_carry_full_confidence(self):
        manager, store = _manager()
        _say(manager, "My name is Amir.")
        assert _by_name(store, "s", "amir")["confidence"] == 1.0
        _say(manager, "Verse is a component. My project is PRIMORA.")
        _say(manager, "Verse is part of PRIMORA.", timestamp=1100.0)
        part = [r for r in _rows(store, "relationships")
                if r["relationship_type"] == "part_of"][0]
        assert part["confidence"] == 1.0

    def test_candidate_rows_keep_the_reduced_confidence(self):
        manager, store = _manager()
        _say(manager, "My project is PRIMORA.")
        _say(manager, "PRIMORA is also called Prime.", timestamp=1100.0)
        assert _rows(store, "entity_aliases")[0]["confidence"] == 0.3


# ---------------------------------------------------------------- 16 idempot
class TestIdempotency:
    def test_replaying_the_same_snapshot_changes_nothing(self):
        manager, store = _manager()
        for text in ("My name is Amir. My project is PRIMORA.",
                     "Aurora is an agent. Verse is a component.",
                     "Verse is part of PRIMORA.",
                     "PRIMORA is also called Prime."):
            _say(manager, text, timestamp=1000.0 + len(text))
        tables = ("messages", "states", "entities", "entity_aliases",
                  "relationships")
        with store._connect() as con:
            before = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                      for t in tables}
        manager.reconcile(list(manager._hist), "s")
        with store._connect() as con:
            after = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                     for t in tables}
        assert before == after

    def test_repeating_the_assertion_creates_no_second_entity(self):
        manager, store = _manager()
        for i in range(4):
            _say(manager, "My name is Amir.", timestamp=1000.0 + i)
        assert len(_rows(store, "entities")) == 1

    def test_repeating_the_assertion_creates_no_second_relationship(self):
        manager, store = _manager()
        _say(manager, "Verse is a component. My project is PRIMORA.")
        for i in range(3):
            _say(manager, "Verse is part of PRIMORA.", timestamp=1100.0 + i)
        part = [r for r in _rows(store, "relationships")
                if r["relationship_type"] == "part_of"]
        assert len(part) == 1, "an identical edge must not be re-created"

    def test_same_message_twice_in_one_snapshot_forms_once(self):
        manager, store = _manager()
        snapshot = [{"role": "user", "content": "My name is Amir.",
                     "timestamp": 1000.0}]
        manager.reconcile(list(snapshot), "s")
        manager.reconcile(list(snapshot), "s")
        assert len(_rows(store, "entities")) == 1


# ---------------------------------------------------------------- 17 M4
class TestM4Unchanged:
    def test_state_extraction_still_works_alongside_formation(self):
        manager, store = _manager()
        diags = _say(manager, "My favorite color is blue.")
        assert any(d.startswith("state_updated:") for d in diags)
        with store._connect() as con:
            row = con.execute(
                "SELECT concept, value, status, authority FROM states"
                " ORDER BY id DESC LIMIT 1").fetchone()
        assert row[0] == "favorite color" and row[1] == "blue"
        assert row[2] == "current" and row[3] == "user"

    def test_supersession_still_works(self):
        manager, store = _manager()
        _say(manager, "The vault code is BLUE-FALCON-7.")
        _say(manager, "The new vault code is RED-9.", timestamp=1100.0)
        with store._connect() as con:
            statuses = [r[0] for r in con.execute(
                "SELECT status FROM states WHERE concept='vault code'"
                " ORDER BY id").fetchall()]
            current = con.execute(
                "SELECT value FROM states WHERE concept='vault code'"
                " AND status='current'").fetchone()
        assert "superseded" in statuses
        assert current[0] == "RED-9"

    def test_formation_does_not_change_state_counts_for_a_plain_message(self):
        manager, store = _manager()
        _say(manager, "Sounds good, thanks!")
        with store._connect() as con:
            assert con.execute("SELECT count(*) FROM states").fetchone()[0] == 0
        _say(manager, "The vault code is BLUE-FALCON-7.", timestamp=1100.0)
        with store._connect() as con:
            assert con.execute("SELECT count(*) FROM states").fetchone()[0] == 1

    def test_assistant_conflict_detection_still_works(self):
        manager, store = _manager()
        _say(manager, "The vault code is BLUE-FALCON-7.")
        diags = _say(manager, "The vault code is RED-9.",
                     role="assistant", timestamp=1100.0)
        assert any(d.startswith("state_conflict:") for d in diags)

    def test_extraction_module_does_not_import_forbidden_modules(self):
        source = open(
            os.path.join("PRIMORA", "prime", "context", "entity_formation.py"),
            encoding="utf-8").read()
        for banned in ("import torch", "transformers", "sentence_transformers",
                       "import requests", "urllib", "ollama", "openai",
                       "subprocess", "decision"):
            assert banned not in source, banned


# ---------------------------------------------------------------- isolation
class TestDatabaseIsolation:
    def test_tests_never_open_the_production_database(self):
        from PRIMORA.prime.context.store import default_db_path
        production = os.path.abspath(str(default_db_path()))
        for name in os.listdir(_TEST_TMPDIR.name):
            assert os.path.abspath(
                os.path.join(_TEST_TMPDIR.name, name)) != production

    def test_form_module_has_no_default_path_fallback(self):
        source = open(
            os.path.join("PRIMORA", "prime", "context", "entity_formation.py"),
            encoding="utf-8").read()
        assert "ContextStore(" not in source, (
            "formation receives a store; it must never construct one"
        )
        assert "default_db_path" not in source


class TestDiagnostics:
    def test_formation_diagnostics_are_returned_by_reconcile(self):
        manager, store = _manager()
        diags = _say(manager, "My name is Amir.")
        assert any(d.startswith("entity_") for d in diags)

    def test_empty_message_is_a_no_op(self):
        manager, store = _manager()
        result = ef.form_from_user_message(store, "s", 1, "   ")
        assert result.diagnostics == ["m5_skipped: empty message"]
        assert result.entity_ids == {}

    def test_extraction_is_pure(self):
        """form_from_user_message must not write when nothing is asserted."""
        manager, store = _manager()
        with store._connect() as con:
            before = con.execute("SELECT count(*) FROM entities").fetchone()[0]
        ef.form_from_user_message(store, "s", 1, "just some chit-chat here")
        with store._connect() as con:
            after = con.execute("SELECT count(*) FROM entities").fetchone()[0]
        assert before == after
