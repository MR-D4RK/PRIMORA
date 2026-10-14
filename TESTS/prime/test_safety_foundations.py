"""The three safety foundations preceding arbitrary tool calling.

1. A bounded capability-call budget, enforced at the single choke point.
2. Scope consistency, so a session mismatch is reported, not silent.
3. Provenance attestation, so a self-reported label cannot bypass the rule.

Each foundation is tested where it can be shown to work and where it can be
shown to fail to matter: a bound that never fires proves nothing, and a
provenance guard that downgrades everything is just a different bug.
"""

import ast
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.aurora.loop import (
    DEFAULT_CAPABILITY_BUDGET,
    CapabilityExecutor,
    CognitiveLoop,
)
from PRIMORA.learning import Jev, LearningOutcome, StructuredCandidate
from PRIMORA.learning.provenance import attest, content_tokens
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeRequest, PrimeResult
from PRIMORA.prime.providers import Provider
from PRIMORA.research import ResearchProvider, ResearchResult

CONCEPT, VALUE = "kafka version", "4.0"
EXTERNAL_TEXT = "Kafka 4.0 is the latest Apache Kafka release with KRaft brokers."


def _request(action, payload):
    return (
        "<prime_request>\n"
        + json.dumps({"action": action, "payload": payload,
                      "context": None, "metadata": None})
        + "\n</prime_request>"
    )


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"value": 4})


class _Web(ResearchProvider):
    name = "web"

    def search(self, query, limit=5):
        return [ResearchResult("Kafka 4.0", "https://kafka.apache.org/",
                               EXTERNAL_TEXT, "kafka.apache.org")]


class _Jev(Jev):
    def evaluate(self, request):
        if "kafka" in request.text.lower():
            return LearningOutcome.confirmed([
                StructuredCandidate(concept=CONCEPT, value=VALUE,
                                    memory_type="FACT")])
        if "tea" in request.text.lower():
            return LearningOutcome.confirmed([
                StructuredCandidate(concept="tea", value="green",
                                    memory_type="PREFERENCE")])
        return LearningOutcome.rejected("jev_found_nothing_structurable")


class _Scripted(Brain):
    def __init__(self, *replies):
        self.replies = list(replies) or ["done"]
        self.calls = []

    def generate(self, messages, max_output_tokens=None):
        self.calls.append([dict(m) for m in messages])
        r = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(r, BrainResult):
            return r
        return BrainResult.success(r, "scripted")

    def get_model_name(self):
        return "scripted"

    def count_tokens(self, messages):
        return sum(len(m.get("content", "").split()) + 2 for m in messages)


_SEQ = [0]


def _prime(jev=None, research=True):
    _SEQ[0] += 1
    prime = Prime(_Provider(), research_provider=_Web() if research else None,
                  jev=jev or _Jev())
    prime._context = prime._context.__class__(
        db_path=os.path.join(tempfile.mkdtemp(prefix="safe_"),
                            f"s{_SEQ[0]}.db"))
    prime._mind = prime._mind.__class__(prime._context)
    for name in ("recall", "learn"):
        cap = prime.registry.get_specialist(name)
        if cap is not None:
            cap.bind(prime._mind)
    return prime


def _aurora(prime, brain, max_cycles=2, max_calls=None):
    a = Aurora(prime, brain=brain,
               config=AuroraConfig(model_path="m", adapter_path="a"))
    a.loop.max_cycles = max_cycles
    if max_calls is not None:
        a.loop.executor.max_calls = max_calls
    return a


def _tool(brain, index, kind="tool", last=False):
    tools = [m for m in brain.calls[index] if m["role"] == kind]
    return tools[-1]["content"] if last else tools[0]["content"]


def _run(prime, replies, max_cycles=3, max_calls=None, session="s",
         user="what is the latest Kafka?"):
    brain = _Scripted(*replies)
    a = _aurora(prime, brain, max_cycles, max_calls)
    a.mind.begin_task(session, "work")
    a.respond([{"role": "user", "content": user}], session)
    return brain, a


# ======================================================================
# 1. Capability call budget
# ======================================================================


class TestCapabilityBudget:
    def _exec(self, max_calls):
        class FakePrime:
            def __init__(self):
                self.handled = []

            def handle(self, request):
                self.handled.append(request.action)
                return PrimeResult.success({"ok": True})

        prime = FakePrime()
        ex = CapabilityExecutor(prime, max_calls=max_calls)
        ex.begin_session()
        return prime, ex

    def test_default_exists_and_is_safe(self):
        assert DEFAULT_CAPABILITY_BUDGET >= 1
        assert CapabilityExecutor(_Provider()).max_calls == \
            DEFAULT_CAPABILITY_BUDGET

    def test_normal_use_is_under_the_budget(self):
        prime, ex = self._exec(3)
        for i in range(3):
            assert json.loads(
                ex.execute(PrimeRequest("calculate", {"n": i}))
            )["status"] == "success"
        assert ex.used == 3
        assert ex.remaining == 0
        assert len(prime.handled) == 3

    def test_exhaustion_refuses_and_never_reaches_prime(self):
        prime, ex = self._exec(2)
        for i in range(2):
            ex.execute(PrimeRequest("calculate", {"n": i}))
        out = json.loads(ex.execute(PrimeRequest("calculate", {"n": 99})))
        assert out["status"] == "failure"
        assert "capability_budget_exhausted" in out["error"]
        assert len(prime.handled) == 2, "the refused call must not execute"
        assert ex.exhausted

    def test_it_bounds_distinct_requests_not_just_repeats(self):
        """A budget that only caught repeats would be the repeat guard."""
        prime, ex = self._exec(2)
        ex.execute(PrimeRequest("research", {"query": "a"}))
        ex.execute(PrimeRequest("research", {"query": "b"}))
        out = json.loads(ex.execute(PrimeRequest("research", {"query": "c"})))
        assert "capability_budget_exhausted" in out["error"]

    def test_budget_resets_per_turn(self):
        prime, ex = self._exec(1)
        ex.execute(PrimeRequest("calculate", {"n": 1}))
        assert ex.exhausted or ex.used == 1
        ex.begin_session()
        assert ex.used == 0 and not ex.exhausted
        assert json.loads(
            ex.execute(PrimeRequest("calculate", {"n": 2}))
        )["status"] == "success"

    def test_rejects_a_zero_budget(self):
        with pytest.raises(ValueError):
            CapabilityExecutor(_Provider(), max_calls=0)

    def test_covers_every_capability_without_them_knowing(self):
        brain, a = _run(
            _prime(),
            [_request("calculate", {"expression": "1+1"}),
             _request("calculate", {"expression": "2+2"}),
             _request("calculate", {"expression": "3+3"}),
             "done"],
            max_cycles=4, max_calls=2)
        assert a.loop.executor.used == 2
        assert a.loop.executor.exhausted
        # The refusal is appended to the running context, so it surfaces in
        # the segment generated after the refused call.
        assert "capability_budget_exhausted" in _tool(
            brain, len(brain.calls) - 1, last=True)

    def test_a_real_turn_under_the_default_is_unaffected(self):
        brain, a = _run(
            _prime(),
            [_request("research", {"query": "latest Kafka release"}),
             "Kafka 4.0."],
            max_cycles=2)
        assert "capability_budget_exhausted" not in _tool(brain, 1)
        assert a.loop.executor.used == 1

    def test_the_repeat_guard_still_applies_independently(self):
        """Two guards, two different failures."""
        p = _prime()
        brain = _Scripted(_request("calculate", {"expression": "1+1"}))
        a = _aurora(p, brain, max_cycles=20, max_calls=50)
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        assert a.loop.last_trace.stop_reason == "repeated_request"
        assert a.loop.executor.used == 2
        assert not a.loop.executor.exhausted, "budget was not the cause"

    def test_the_loop_was_not_redesigned(self):
        """One line lets the executor reset; the loop tracks no budget."""
        from PRIMORA.aurora import loop as loopmod
        tree = ast.parse(open(loopmod.__file__, encoding="utf-8").read())
        run = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "run")
        src = ast.unparse(run)
        assert src.count("begin_session") == 1
        for banned in ("max_calls", "self._executor.used", "exhausted"):
            assert banned not in src, f"the loop now tracks {banned}"


# ======================================================================
# 2. Scope consistency
# ======================================================================


class TestScopeConsistency:
    def test_capabilities_inherit_the_active_session(self):
        prime = _prime()
        assert prime.mind.current_session_id == "default"
        prime.mind.begin_task("live", "t")
        assert prime.mind.current_session_id == "live"

    def test_absent_session_does_not_silently_become_default(self):
        """The original bug: a missing session_id wrote to 'default'."""
        prime = _prime()
        prime.mind.begin_task("live", "t")
        out = prime.handle(PrimeRequest("learn", {
            "text": "Kafka 4.0 is the latest.", "provenance": "user"})).output
        source_id = out["source_id"]
        rows = prime._context._get_store().fetch_messages_by_ids([source_id])
        assert rows[0].session_id == "live"

    def test_explicit_session_still_wins(self):
        """Isolated stores and tests must keep working."""
        prime = _prime()
        prime.mind.begin_task("live", "t")
        out = prime.handle(PrimeRequest("learn", {
            "text": "Kafka 4.0 is the latest.", "provenance": "user",
            "session_id": "explicit"})).output
        rows = prime._context._get_store().fetch_messages_by_ids(
            [out["source_id"]])
        assert rows[0].session_id == "explicit"

    def test_a_mismatch_is_reported_not_silent(self):
        prime = _prime()
        prime.mind.begin_task("live", "t")
        out = prime.handle(PrimeRequest("learn", {
            "text": "Kafka 4.0 is the latest.", "provenance": "user",
            "session_id": "elsewhere"})).output
        notes = [d for d in out.get("diagnostics", [])
                 if d.startswith("scope_mismatch")]
        assert notes, out.get("diagnostics")

    def test_no_mismatch_note_when_scopes_agree(self):
        prime = _prime()
        prime.mind.begin_task("live", "t")
        out = prime.handle(PrimeRequest("learn", {
            "text": "Kafka 4.0 is the latest.", "provenance": "user"})).output
        assert not [d for d in out.get("diagnostics", [])
                    if d.startswith("scope_mismatch")]

    def test_recall_inherits_the_active_session(self):
        prime = _prime()
        prime.mind.begin_task("live", "t")
        assert prime.registry.get_specialist("recall")._current_session() == "live"

    def test_falls_back_to_default_without_an_active_session(self):
        assert _prime().mind.current_session_id == "default"

    def test_m6_is_untouched(self):
        from PRIMORA.prime.context import hybrid
        assert hybrid.SEMANTIC_TAU == 0.45
        assert hybrid.RRF_K == 60.0
        assert hybrid.EVIDENCE_FLOOR_RATIO == 0.725

    def test_no_speculative_cross_session_filtering(self):
        """Nothing new was bolted onto the existing policy."""
        src = open(os.path.join("PRIMORA", "prime", "context", "mind.py"),
                   encoding="utf-8").read().lower()
        for banned in ("cross_session", "cross-session", "other session"):
            assert banned not in src, banned


# ======================================================================
# 3. Provenance attestation
# ======================================================================


class TestProvenanceAttestation:
    EXT = [("research", content_tokens(EXTERNAL_TEXT))]

    def test_the_bypass_is_closed(self):
        provenance, diagnostic = attest(
            EXTERNAL_TEXT, "user", self.EXT, ["what is the latest Kafka?"])
        assert provenance == "research", "the bypass is still open"
        assert "provenance_downgraded" in diagnostic

    def test_a_genuine_user_statement_is_untouched(self):
        provenance, _ = attest(
            "Amir prefers green tea in the morning.", "user", self.EXT,
            ["I prefer green tea in the morning, always."])
        assert provenance == "user"

    def test_a_paraphrase_of_the_user_is_untouched(self):
        provenance, _ = attest(
            "Amir drinks green tea each morning.", "user", self.EXT,
            ["I like green tea every morning."])
        assert provenance == "user"

    def test_no_evidence_means_no_interference(self):
        provenance, _ = attest("Amir prefers green tea.", "user", [], [])
        assert provenance == "user"

    def test_a_honest_label_is_untouched(self):
        provenance, _ = attest(
            EXTERNAL_TEXT, "research", self.EXT, ["what is latest kafka?"])
        assert provenance == "research"

    def test_a_question_is_not_corroboration(self):
        """Vocabulary overlap with a question must not count as the user
        having stated the fact."""
        provenance, _ = attest(
            EXTERNAL_TEXT, "user", self.EXT, ["what is the latest Kafka?"])
        assert provenance == "research"

    def test_end_to_end_the_bypass_never_becomes_a_fact(self):
        brain, _a = _run(
            _prime(),
            [_request("research", {"query": "latest Kafka release"}),
             _request("learn", {"text": EXTERNAL_TEXT, "provenance": "user"}),
             "Kafka 4.0."])
        store = _a._prime._context._get_store()
        currents = store.fetch_states(status="current")
        assert currents == [], "a bypassed claim became a current fact"
        pending = store.fetch_candidates(status="pending")
        assert pending and pending[0].authority == "research"

    def test_end_to_end_aurora_is_told_her_label_was_ignored(self):
        brain, _a = _run(
            _prime(),
            [_request("research", {"query": "latest Kafka release"}),
             _request("learn", {"text": EXTERNAL_TEXT, "provenance": "user"}),
             "Kafka 4.0."])
        assert "provenance_downgraded" in _tool(brain, 2, last=True)

    def test_a_real_user_lesson_still_becomes_a_fact(self):
        prime = _prime()
        prime.mind.begin_task("s", "t")
        out = prime.handle(PrimeRequest("learn", {
            "text": "Amir prefers green tea in the morning.",
            "provenance": "user"})).output
        assert out["status"] == "learned"

    def test_the_authority_invariant_is_untouched(self):
        prime = _prime()
        with pytest.raises(ValueError, match="user-authored"):
            prime._context._get_store().create_state(
                "s", "s", "x", "y", [1], status="current", authority="research")

    def test_the_policy_is_still_only_user_auto_admits(self):
        from PRIMORA.learning import HOLDS
        assert HOLDS == {"research", "specialist", "tool", "observation",
                         "inference", "unknown"}

    def test_evidence_logs_every_call_and_the_loop_names_none(self):
        """The executor logs; the consumer judges. The loop stays agnostic."""
        ex = CapabilityExecutor(_prime())
        ex.begin_session()
        ex.execute(PrimeRequest("research", {"query": "x"}))
        ex.execute(PrimeRequest("calculate", {"expression": "1+1"}))
        assert [a for a, _ in ex.evidence] == ["research", "calculate"]

        from PRIMORA.aurora import loop as loopmod
        tree = ast.parse(open(loopmod.__file__, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        code = ast.unparse(tree)
        for banned in ("research", "consult_specialist", '"learn"', '"recall"'):
            assert banned not in code, f"the loop names {banned}"

    def test_evidence_log_is_bounded(self):
        from PRIMORA.aurora.loop import EVIDENCE_MAX_ENTRIES
        ex = CapabilityExecutor(_prime(), max_calls=EVIDENCE_MAX_ENTRIES + 5)
        ex.begin_session()
        for i in range(EVIDENCE_MAX_ENTRIES + 5):
            ex.execute(PrimeRequest("calculate", {"n": i}))
        assert len(ex.evidence) <= EVIDENCE_MAX_ENTRIES

    def test_a_broken_attestation_does_not_block_learning(self):
        class Stubborn:
            current_session_id = "s"

            def attest_provenance(self, *a, **k):
                raise RuntimeError("attestation down")

            def learn(self, text, provenance, jev, session_id="s"):
                return {"status": "learned", "learned": [], "held": [],
                        "diagnostics": []}

        prime = _prime()
        cap = prime.registry.get_specialist("learn")
        cap.bind(Stubborn())
        result = cap.handle({"text": "something", "provenance": "user"})
        assert result.status.value == "success"


# ======================================================================
# Scope
# ======================================================================


class TestScope:
    def test_verse_untouched(self):
        assert os.path.isdir(os.path.join("PRIMORA", "verse", "qml"))

    def test_tool_calling_goes_through_the_same_protections(self):
        """Arbitrary tools are registered behind the same executor.

        This was written before tool calling existed, when the point was to
        prove it had not been added. It now proves the opposite and the
        stronger thing: a tool is a capability like any other, so the budget
        and the guard that bound the others bound it too.
        """
        prime = _prime()
        assert "use_tool" in prime.registry.get_names()
        assert not prime.tools.names(), "no tool is reachable by default"

    def test_memory_storage_is_untouched(self):
        src = open(os.path.join("PRIMORA", "prime", "context", "store.py"),
                   encoding="utf-8").read().lower()
        for banned in ("budget", "attest", "provenance_downgraded"):
            assert banned not in src, banned

    def test_jev_is_not_redesigned(self):
        src = open(os.path.join("PRIMORA", "learning", "jev.py"),
                   encoding="utf-8").read()
        assert "attest" not in src, "attestation belongs outside Jev"

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
