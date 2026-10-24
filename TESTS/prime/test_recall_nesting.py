"""Regression tests for the recall tool-response nesting bug.

The failure this file exists for:

    ContextManager.recall returned stored messages verbatim, including the
    loop's own ``role="tool"`` turns. The loop appends every capability result
    to its live message list as a tool turn, that list is reconciled into the
    session store, so a capability result became a stored message, became
    retrievable evidence, and the next recall handed the loop its own previous
    output. Each round wrapped the last.

    Measured stored tool-turn growth in a real 166-example evaluation:

        6,789 -> 19,819 -> 36,352 -> 79,463 -> 173,225 -> 313,685 chars

    and the wrapper is what got tokenized for the next segment, reaching
    42,414 tokens mid-run and 164,382 by the end. That is far past the 1,800
    token input ceiling and it OOMs the brain.

The fix filters internal roles out of recall evidence and caps each passage,
without changing what gets stored, what retrieval ranks, or the cross-session
policy. These tests pin all of that.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.manager import (
    INTERNAL_ROLES,
    RECALL_MAX_CONTENT_CHARS,
    RECALL_TRUNCATION_MARKER,
)
from PRIMORA.prime.contract import PrimeRequest, RequestStatus
from PRIMORA.prime.providers import Provider


def _db(name: str) -> str:
    directory = tempfile.mkdtemp(prefix="recall_nest_")
    return os.path.join(directory, name + ".db")


class _Provider(Provider):
    def __init__(self):
        self.calls = []

    def invoke(self, capability, payload):
        self.calls.append(capability)
        from PRIMORA.prime.contract import PrimeResult
        return PrimeResult.success({"content": "ok"})


def _prime(db_name: str = "n.db") -> Prime:
    """A Prime on a throwaway isolated database."""
    prime = Prime(_Provider())
    prime._context = ContextManager(db_path=_db(db_name))
    prime._mind = prime._mind.__class__(prime._context)
    prime._registry.get_specialist("recall").bind(prime._mind)
    return prime


def _recalled(prime: Prime, query: str, session_id: str, limit: int = 5):
    out = prime.handle(
        PrimeRequest("recall", {"query": query, "session_id": session_id,
                                "limit": limit}))
    assert out.status is RequestStatus.SUCCESS
    return out.output["results"]


def _stored_tool_sizes(prime: Prime, session_id: str) -> list[int]:
    store = prime.context._get_store()
    with sqlite3.connect(str(store._db_path)) as con:
        return [
            int(row[0]) for row in con.execute(
                "SELECT LENGTH(content) FROM messages "
                "WHERE session_id = ? AND role = 'tool' ORDER BY id",
                (session_id,))
        ]


# ======================================================================
# 1. A session that already contains tool turns
# ======================================================================


class TestSessionWithPriorToolTurns:
    def test_recall_from_a_session_containing_tool_turns(self):
        """The exact shape that broke: conversation plus stored plumbing."""
        prime = _prime("s1.db")
        prime.context.reconcile([
            {"role": "user", "content": "the deployment key rotates friday"},
            {"role": "assistant", "content": "friday rotation confirmed"},
            {"role": "tool", "content": json.dumps({
                "status": "success",
                "output": {"query": "key", "results": [
                    {"role": "assistant",
                     "content": "key rotation is friday"}]}})},
        ], "s1")
        results = _recalled(prime, "key rotation", "s1")
        assert results, "real memory must still be retrievable"
        roles = {item["role"] for item in results}
        assert "tool" not in roles

    def test_recall_excludes_tool_turns(self):
        prime = _prime("s2.db")
        prime.context.reconcile([
            {"role": "user", "content": "budget review notes"},
            {"role": "tool", "content": json.dumps({
                "status": "success",
                "output": {"query": "budget",
                           "results": [{"role": "tool",
                                        "content": "budget"}]}})},
        ], "s2")
        results = _recalled(prime, "budget review", "s2")
        assert all(item["role"] != "tool" for item in results)

    def test_recall_excludes_system_turns(self):
        prime = _prime("s3.db")
        prime.context.reconcile([
            {"role": "system", "content": "you are Aurora, a quiet assistant"},
            {"role": "user", "content": "the quiet assistant question"},
            {"role": "assistant", "content": "answering quietly"},
        ], "s3")
        results = _recalled(prime, "quiet assistant", "s3")
        assert all(item["role"] != "system" for item in results)

    def test_internal_roles_constant_covers_both(self):
        """The filter is not an ad hoc one-off."""
        assert INTERNAL_ROLES == frozenset({"tool", "system"})

    def test_legitimate_roles_survive_the_filter(self):
        """The filter removes plumbing, not conversation."""
        prime = _prime("s4.db")
        prime.context.reconcile([
            {"role": "user", "content": "shipping label question"},
            {"role": "assistant", "content": "label shipped yesterday"},
        ], "s4")
        roles = {item["role"]
                 for item in _recalled(prime, "shipping label", "s4")}
        assert roles <= {"user", "assistant"}
        assert roles


# ======================================================================
# 2. Normal conversational memory is preserved
# ======================================================================


class TestLegitimateMemoryPreserved:
    def test_normal_user_and_assistant_memories_are_retrievable(self):
        prime = _prime("p1.db")
        prime.context.reconcile([
            {"role": "user", "content": "the deployment key rotates friday"},
            {"role": "assistant", "content": "friday rotation confirmed"},
            {"role": "user", "content": "unrelated: office plants need water"},
            {"role": "assistant", "content": "noted, watering is tuesday"},
        ], "s1")
        results = _recalled(prime, "key rotation", "s1")
        contents = [item["content"] for item in results]
        assert any("deployment key" in c for c in contents)
        assert any("friday rotation confirmed" in c for c in contents)

    def test_verbatim_text_is_still_returned_for_short_passages(self):
        """Truncation must not alter passages that fit."""
        prime = _prime("p2.db")
        prime.context.reconcile([
            {"role": "user", "content": "the deployment key rotates every friday"},
        ], "s1")
        results = _recalled(prime, "deployment key", "s1")
        assert "the deployment key rotates every friday" in [
            item["content"] for item in results]

    def test_empty_store_still_returns_empty(self):
        prime = _prime("p3.db")
        assert _recalled(prime, "anything", "s1") == []


# ======================================================================
# 3. The exponential growth pattern is gone
# ======================================================================


class TestNoExponentialGrowth:
    def test_repeated_recall_does_not_nest_prior_recall_output(self):
        """Recall output must never reappear inside later recall output."""
        prime = _prime("g1.db")
        prime.context.reconcile([
            {"role": "user", "content": "the deployment key rotates friday"},
            {"role": "assistant", "content": "friday rotation confirmed"},
        ], "s1")

        sizes = []
        for _ in range(6):
            results = _recalled(prime, "key rotation", "s1")
            blob = json.dumps(results)
            assert '"role": "tool"' not in blob, (
                "recall handed back a tool turn, so its own output can be "
                "recalled and wrapped again")
            sizes.append(len(blob))
            # Exactly what the loop does: the answer becomes a tool turn, and
            # that list is reconciled into the same session.
            prime.context.reconcile(
                [{"role": "user", "content": "key rotation"},
                 {"role": "assistant", "content": "checking"},
                 {"role": "tool", "content": blob}], "s1")

        # The first round can only see the two seed messages, so it may be
        # smaller. What matters is that it settles and stays there. The bug
        # grew without bound (129 -> 174 -> ... -> 313,685 chars); stability
        # is the property being pinned.
        tail = sizes[1:]
        assert len(set(tail)) == 1, (
            f"recall output kept growing across rounds: {sizes}")
        assert max(sizes) < 2_000, (
            f"recall output is not proportionate to the question: {sizes}")

    def test_stored_tool_turns_do_not_compound(self):
        """The measured failure: 6,789 -> 313,685 chars across rounds."""
        prime = _prime("g2.db")
        prime.context.reconcile([
            {"role": "user", "content": "the deployment key rotates friday"},
            {"role": "assistant", "content": "friday rotation confirmed"},
        ], "s1")

        previous_max = 0
        for _ in range(8):
            blob = json.dumps(_recalled(prime, "key rotation", "s1"))
            prime.context.reconcile(
                [{"role": "user", "content": "key rotation again"},
                 {"role": "tool", "content": blob}], "s1")
            sizes = _stored_tool_sizes(prime, "s1")
            current_max = max(sizes) if sizes else 0
            assert current_max <= previous_max + RECALL_MAX_CONTENT_CHARS, (
                f"stored tool turns are compounding: {sizes}")
            previous_max = current_max

    def test_multiple_consecutive_recalls_stay_bounded(self):
        prime = _prime("g3.db")
        prime.context.reconcile([
            {"role": "user", "content": "quarterly revenue forecast notes"},
        ], "s1")
        lengths = []
        for _ in range(10):
            blob = json.dumps(_recalled(prime, "quarterly revenue", "s1"))
            lengths.append(len(blob))
            prime.context.reconcile(
                [{"role": "tool", "content": blob}], "s1")
        assert max(lengths) <= min(lengths) + RECALL_MAX_CONTENT_CHARS
        assert max(lengths) < 10_000

    def test_recall_output_is_bounded_by_construction(self):
        """Worst case is limit x cap, far below any tokenizer's limit."""
        prime = _prime("g4.db")
        prime.context.reconcile([
            {"role": "user", "content": "the quarterly " + "x" * 400_000},
            {"role": "assistant",
             "content": "quarterly summary " + "y" * 400_000},
        ], "s1")
        results = _recalled(prime, "quarterly summary", "s1", limit=12)
        assert results
        for item in results:
            assert len(item["content"]) <= (
                RECALL_MAX_CONTENT_CHARS + len(RECALL_TRUNCATION_MARKER))

    def test_truncation_is_marked_not_silent(self):
        prime = _prime("g5.db")
        prime.context.reconcile([
            {"role": "user", "content": "verbose " + "z" * 5_000},
        ], "s1")
        results = _recalled(prime, "verbose", "s1")
        assert results
        assert any(RECALL_TRUNCATION_MARKER in item["content"]
                   for item in results)


# ======================================================================
# 4. Ranking, scoping, and provenance are unchanged
# ======================================================================


class TestBehaviourPreserved:
    def test_active_session_scoping_still_works(self):
        """The active session ranks first; foreign passages stay tagged."""
        prime = _prime("b1.db")
        prime.context.reconcile([
            {"role": "user", "content": "in the active session about kayaking"},
        ], "s1")
        prime.context.reconcile([
            {"role": "user", "content": "in another session about kayaking"},
        ], "s2")
        results = _recalled(prime, "kayaking", "s1")
        joined = " ".join(item["content"] for item in results)
        assert "kayaking" in joined
        # Active-session evidence comes back unprefixed, exactly as stored.
        assert results[0]["content"].startswith("in the active session")
        # Anything from another session is still visibly marked, never merged.
        for item in results:
            if item["content"].startswith("[from another conversation]"):
                continue
            assert not item["content"].startswith("in another session")

    def test_cross_session_prefix_still_applied(self):
        prime = _prime("b2.db")
        prime.context.reconcile([
            {"role": "user", "content": "elsewhere we discussed kayaking"},
        ], "s2")
        results = _recalled(prime, "kayaking", "s1")
        assert results
        assert all("[from another conversation]" in item["content"]
                   for item in results)

    def test_recall_remains_read_only(self):
        """A search must still never add memory."""
        prime = _prime("b3.db")
        prime.context.reconcile([
            {"role": "user", "content": "the deployment key rotates friday"},
        ], "s1")
        store = prime.context._get_store()
        path = str(store._db_path)

        def counts():
            with sqlite3.connect(path) as con:
                return tuple(
                    con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("messages", "states", "entities",
                              "relationships", "candidates"))

        before = counts()
        _recalled(prime, "key rotation", "s1")
        assert counts() == before

    def test_result_shape_is_unchanged(self):
        """Only role and content, as the contract already specified."""
        prime = _prime("b4.db")
        prime.context.reconcile([
            {"role": "user", "content": "shape check on recall results"},
        ], "s1")
        for item in _recalled(prime, "shape check", "s1"):
            assert {"role", "content", "source_type", "source_id",
                    "authority", "status", "session_id", "timestamp",
                    "confidence"} <= set(item)

    def test_capability_budget_and_repeat_guard_untouched(self):
        """The fix must not have been made by loosening loop limits."""
        import inspect

        from PRIMORA.aurora.loop import (
            DEFAULT_CAPABILITY_BUDGET,
            DEFAULT_MAX_REPEATS,
            CapabilityExecutor,
            CognitiveLoop,
        )
        assert DEFAULT_CAPABILITY_BUDGET == 4
        assert DEFAULT_MAX_REPEATS == 2
        # Both limits are still the enforced defaults, not merely declared.
        # max_repeats is keyword-only, so it is read from __kwdefaults__.
        assert CapabilityExecutor.__init__.__defaults__[0] == (
            DEFAULT_CAPABILITY_BUDGET)
        assert CognitiveLoop.__init__.__kwdefaults__["max_repeats"] == (
            DEFAULT_MAX_REPEATS)


# ======================================================================
# 5. Isolation
# ======================================================================


class TestNoProductionPollution:
    def test_production_database_is_untouched(self):
        production = os.path.join(
            r"C:\PRIMORA PROJECT", "PRIMORA", "data", "context",
            "aurora_context.db")
        if not os.path.exists(production):
            pytest.skip("production database not present")

        def row_counts():
            with sqlite3.connect(f"file:{production}?mode=ro", uri=True) as con:
                return {
                    table: con.execute(
                        f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in ("messages", "states", "entities")
                }

        before = row_counts()
        prime = _prime("iso.db")
        prime.context.reconcile([
            {"role": "user", "content": "isolated probe about deployment keys"},
            {"role": "tool", "content": json.dumps({"status": "success"})},
        ], "s1")
        for _ in range(5):
            _recalled(prime, "deployment keys", "s1")
        assert row_counts() == before

    def test_recall_works_inside_a_harness_session(self):
        """The real isolated path, not just a hand-built Prime."""
        from PRIMORA.prime.context.isolation import (
            build_prime_with_isolation_guard,
            harness_session,
        )

        with harness_session("recall_nest_probe") as session:
            context = session.context_manager()
            prime = build_prime_with_isolation_guard(_Provider())
            prime._context = context
            prime._mind = prime._mind.__class__(context)
            prime._registry.get_specialist("recall").bind(prime._mind)

            context.reconcile([
                {"role": "user", "content": "isolated session about kayaking"},
                {"role": "tool", "content": json.dumps({
                    "status": "success", "output": {"query": "kayaking"}})},
            ], "s1")
            results = _recalled(prime, "kayaking", "s1")
            assert all(item["role"] != "tool" for item in results)
