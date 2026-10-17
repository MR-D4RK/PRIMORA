"""One logical memory, one current row, whichever write path it arrives by.

PRIMORA's memory system has two write paths into structured state:

* **fresh extraction** — ``apply_user_candidates``, from the user's own words
* **confirmation**       — ``state.confirm_pending``, from a held candidate

Both funnel into ``state._apply_confirmed``, which short-circuits when the
concept already has a current row with the same value. This file proves that
sharing is real, and pins the boundary that keeps it true: the two paths only
deduplicate *within one session scope*, which is why a capability must record
into the session the conversation is actually in.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.learning import Jev, LearningOutcome, StructuredCandidate
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeRequest
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.providers import Provider
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context import state as state_mod

FAVOURITE = "favorite movie"
DUNE = "Dune Part Two"


def _db(name="u.db"):
    return os.path.join(tempfile.mkdtemp(prefix="uniq_"), name)


def _say(manager, text, session="s", role="user"):
    """Fresh extraction: the normal formation path."""
    return manager.reconcile([{"role": role, "content": text}], session)


class _Jev(Jev):
    def __init__(self, concept=FAVOURITE, value=DUNE, memory_type="PREFERENCE"):
        self._concept, self._value, self._type = concept, value, memory_type

    def evaluate(self, request):
        return LearningOutcome.confirmed([
            StructuredCandidate(concept=self._concept, value=self._value,
                                memory_type=self._type, scope="session")])


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"value": 4})


_SEQ = [0]


def _prime(jev=None):
    """A Prime on a throwaway database, with every capability bound."""
    _SEQ[0] += 1
    prime = Prime(_Provider(), jev=jev or _Jev())
    prime._context = ContextManager(db_path=_db(f"p{_SEQ[0]}.db"))
    prime._mind = prime._mind.__class__(prime._context)
    for name in ("recall", "learn"):
        cap = prime.registry.get_specialist(name)
        if cap is not None:
            cap.bind(prime._mind)
    return prime


def _current(manager, session="s", concept=FAVOURITE):
    # Slot-routed personal facts live in the global scope; the dedup
    # invariant holds there regardless of which session wrote first.
    return manager._get_store().fetch_states(
        concept=concept, status="current")


def _learn(prime, text, provenance="user", **kw):
    payload = {"text": text, "provenance": provenance}
    payload.update(kw)
    return prime.handle(PrimeRequest("learn", payload)).output


# ======================================================================
# Both write paths deduplicate
# ======================================================================


class TestWritePathsDeduplicate:
    def test_fresh_extraction_twice_is_one_memory(self):
        mgr = ContextManager(db_path=_db("a.db"))
        _say(mgr, f"my {FAVOURITE} is {DUNE}")
        _say(mgr, f"my {FAVOURITE} is {DUNE}")
        assert len(_current(mgr)) == 1

    def test_confirmation_twice_is_one_memory(self):
        mgr = ContextManager(db_path=_db("b.db"))
        store = mgr._get_store()
        for _ in range(2):
            message_id, _new, _d = store.append(
                "s", store.count("s") + 1, "assistant", f"my {FAVOURITE} is {DUNE}")
            candidate_id = store.create_candidate(
                "s", "s", [message_id], proposed_type="PREFERENCE",
                proposed_concept=FAVOURITE, proposed_value=DUNE,
                authority="user", status="pending")
            state_mod.confirm_pending(store, "s", candidate_id)
        assert len(_current(mgr)) == 1

    def test_learn_then_fresh_extraction_is_one_memory(self):
        prime = _prime()
        _learn(prime, f"Amir's {FAVOURITE} is {DUNE}", session_id="s")
        _say(prime._context, f"my {FAVOURITE} is {DUNE}", session="s")
        assert len(_current(prime._context)) == 1

    def test_fresh_extraction_then_learn_is_one_memory(self):
        prime = _prime()
        _say(prime._context, f"my {FAVOURITE} is {DUNE}", session="s")
        _learn(prime, f"Amir's {FAVOURITE} is {DUNE}", session_id="s")
        assert len(_current(prime._context)) == 1

    def test_the_two_paths_share_one_implementation(self):
        """The invariant has a single enforcement point, not two."""
        import ast
        tree = ast.parse(open(
            os.path.join("PRIMORA", "prime", "context", "state.py"),
            encoding="utf-8").read())
        callers = {n.name for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)
                   and any(isinstance(x, ast.Call)
                           and getattr(x.func, "id", "") == "_apply_confirmed"
                           for x in ast.walk(n))}
        assert {"apply_user_candidates", "confirm_pending"} <= callers, callers

    def test_dedup_is_case_insensitive(self):
        mgr = ContextManager(db_path=_db("c.db"))
        _say(mgr, f"my {FAVOURITE} is {DUNE}")
        _say(mgr, f"my {FAVOURITE} is dune part two")
        assert len(_current(mgr)) == 1


# ======================================================================
# The boundary that keeps it true
# ======================================================================


class TestSessionBoundary:
    def test_a_capability_records_into_the_conversation_session(self):
        """The bug this pins: a wrong default looks like a duplicate.

        learn() used to fall back to "default" while the conversation lived
        elsewhere, so formation of the user's own words and the learned
        statement produced two current rows for one memory.
        """
        prime = _prime()
        prime._mind.begin_task("real-session", "remember")
        _learn(prime, f"Amir's {FAVOURITE} is {DUNE}")
        _say(prime._context, f"my {FAVOURITE} is {DUNE}", session="real-session")
        rows = prime._context._get_store().fetch_states(status="current")
        assert len(rows) == 1, "one memory must not become two rows"
        assert rows[0].session_id == "real-session"

    def test_recall_uses_the_conversation_session_too(self):
        prime = _prime()
        prime._mind.begin_task("real-session", "t")
        assert prime.mind.current_session_id == "real-session"

    def test_distinct_sessions_are_scoped_not_merged(self):
        """Different sessions are genuinely different memories, not dupes.

        Uses a free-form concept: personal-fact slots are global by
        design, so the session-scoped boundary is proven on a concept
        the registry does not own.
        """
        mgr = ContextManager(db_path=_db("d.db"))
        _say(mgr, "My deploy key is BLUE-7", session="one")
        _say(mgr, "My deploy key is BLUE-7", session="two")
        store = mgr._get_store()
        assert len(store.fetch_states(scope="one", concept="deploy key",
                                      status="current")) == 1
        assert len(store.fetch_states(scope="two", concept="deploy key",
                                      status="current")) == 1

    def test_current_session_id_falls_back_to_default(self):
        prime = _prime()
        assert prime.mind.current_session_id == "default"


# ======================================================================
# Existing semantics are preserved
# ======================================================================


class TestExistingSemanticsPreserved:
    def test_distinct_memories_coexist(self):
        prime = _prime()
        prime.registry.get_specialist("learn").set_jev(
            _Jev(concept=FAVOURITE, value=DUNE, memory_type="PREFERENCE"))
        _learn(prime, f"Amir's {FAVOURITE} is {DUNE}", session_id="s")
        prime.registry.get_specialist("learn").set_jev(
            _Jev(concept="deploy key", value="BLUE-7", memory_type="FACT"))
        _learn(prime, "Amir's deploy key is BLUE-7", session_id="s")
        _say(prime._context, "Amir runs a team of six", session="s")
        concepts = {s.concept for s in
                    prime._context._get_store().fetch_states(status="current")}
        assert {FAVOURITE, "deploy key"}.issubset(concepts)

    def test_a_conflicting_value_supersedes_and_keeps_history(self):
        mgr = ContextManager(db_path=_db("e.db"))
        _say(mgr, f"my {FAVOURITE} is {DUNE}")
        _say(mgr, f"my {FAVOURITE} is Arrival")
        store = mgr._get_store()
        current = _current(mgr)
        assert len(current) == 1
        assert current[0].value == "Arrival"
        assert len(store.fetch_states(concept=FAVOURITE)) == 2, \
            "the superseded value must be kept as history"

    def test_a_conflicting_learn_supersedes_rather_than_duplicating(self):
        prime = _prime()
        _learn(prime, f"Amir's {FAVOURITE} is {DUNE}", session_id="s")
        prime.registry.get_specialist("learn").set_jev(
            _Jev(concept=FAVOURITE, value="Arrival", memory_type="PREFERENCE"))
        _learn(prime, "Amir's favorite movie is Arrival", session_id="s")
        store = prime._context._get_store()
        current = _current(prime._context)
        assert len(current) == 1 and current[0].value == "Arrival"
        assert len(store.fetch_states(concept=FAVOURITE)) == 2

    def test_authority_is_preserved_through_dedup(self):
        mgr = ContextManager(db_path=_db("f.db"))
        _say(mgr, f"my {FAVOURITE} is {DUNE}")
        _say(mgr, f"my {FAVOURITE} is {DUNE}")
        assert _current(mgr)[0].authority == "user"

    def test_provenance_anchor_is_kept_on_the_surviving_row(self):
        mgr = ContextManager(db_path=_db("g.db"))
        _say(mgr, f"my {FAVOURITE} is {DUNE}")
        first = _current(mgr)[0]
        _say(mgr, f"my {FAVOURITE} is {DUNE}")
        assert _current(mgr)[0].state_id == first.state_id, (
            "dedup must keep the original row, not replace it")
        assert _current(mgr)[0].source_ids == first.source_ids

    def test_authority_user_invariant_is_untouched(self):
        mgr = ContextManager(db_path=_db("h.db"))
        with pytest.raises(ValueError, match="user-authored"):
            mgr._get_store().create_state(
                "s", "s", "x", "y", [1], status="current", authority="research")

    def test_assistant_conflict_still_records_rather_than_resolving(self):
        mgr = ContextManager(db_path=_db("i.db"))
        _say(mgr, f"my {FAVOURITE} is {DUNE}")
        store = mgr._get_store()
        message_id, _n, _d = store.append(
            "s", store.count("s") + 1, "assistant",
            f"my {FAVOURITE} is Arrival")
        diags = state_mod.check_assistant_conflict(
            store, "s", message_id, f"my {FAVOURITE} is Arrival")
        assert any(d.startswith("state_conflict") for d in diags), diags
        assert _current(mgr)[0].value == DUNE, "conflict must not resolve itself"

    def test_m6_retrieval_is_untouched(self):
        from PRIMORA.prime.context import hybrid
        assert hybrid.SEMANTIC_TAU == 0.45
        assert hybrid.RRF_K == 60.0
        assert hybrid.EVIDENCE_FLOOR_RATIO == 0.725

    def test_jev_is_undisturbed(self):
        from PRIMORA.learning import HOLDS, PROVENANCE
        assert set(PROVENANCE) == {"user", "research", "specialist", "tool",
                                   "observation", "inference", "unknown"}
        assert HOLDS == {"research", "specialist", "tool", "observation",
                         "inference", "unknown"}

    def test_loop_and_verse_undisturbed(self):
        from PRIMORA.aurora import loop as loopmod
        src = open(loopmod.__file__, encoding="utf-8").read()
        assert "state.py" not in src and "_apply_confirmed" not in src
        assert os.path.isdir(os.path.join("PRIMORA", "verse", "qml"))

    def test_production_database_untouched(self):
        """Production memory must be unchanged by this test run.

        Content equality via production_digest(), not emptiness: a used
        install legitimately holds real memories (MEMORY_AUDIT C2), so
        the invariant is "unchanged", never "empty". Isolated work runs
        under harness_session, which refuses production opens and
        re-checks the digest on exit.
        """
        from PRIMORA.prime.context.isolation import (
            harness_session,
            production_db_path,
            production_digest,
        )
        if not production_db_path().exists():
            pytest.skip("no production database")
        before = production_digest()
        with harness_session("c2_untouched_") as session:
            store = session.context_manager()._get_store()
            # An empty starting state belongs to the private store, never
            # to production.
            assert store.fetch_states(limit=5) == []
        assert production_digest() == before, (
            "production memory changed during this test")
