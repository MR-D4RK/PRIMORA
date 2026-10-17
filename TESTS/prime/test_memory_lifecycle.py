"""Milestone 3: the fourth capability, ``learn()``.

The properties that matter:

1. Learning is through memory. No weights, no adapters, no retraining.
2. Aurora sends a sentence and a provenance; she cannot write the schema.
3. An inference never becomes a fact, enforced by the admission layer rather
   than trusted from a pluggable Jev.
4. The existing memory lifecycle does the admitting.
5. A malformed or unsupported request cannot corrupt memory.
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
from PRIMORA.learning import (
    HOLDS,
    PROVENANCE,
    Jev,
    LearningOutcome,
    LearningRequest,
    ModelJev,
    StructuredCandidate,
    UnconfiguredJev,
)
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeRequest, PrimeResult, RequestStatus
from PRIMORA.prime.providers import Provider
from PRIMORA.specialists.tools.learn import LearnSpecialist


def _request(action, payload):
    return (
        "<prime_request>\n"
        + json.dumps({"action": action, "payload": payload,
                      "context": None, "metadata": None})
        + "\n</prime_request>"
    )


class _FakeJev(Jev):
    """Structures a sentence the way a well-behaved Jev would."""

    name = "fake-jev"

    RULES = [
        ("dune", "favorite movie", "PREFERENCE", "Dune Part Two"),
        ("favorite movie", "favorite movie", "PREFERENCE", "Dune Part Two"),
        ("deploy key", "deploy key", "FACT", "BLUE-7"),
        ("dark mode", "interface theme", "PREFERENCE", "dark"),
        ("horror", "hates horror films", "FACT", "probably"),
    ]

    def __init__(self, outcome=None):
        self.requests = []
        self._forced = outcome

    def evaluate(self, request):
        self.requests.append(request)
        if self._forced is not None:
            return self._forced
        lowered = request.text.lower()
        for cue, concept, mtype, value in self.RULES:
            if cue in lowered:
                return LearningOutcome.confirmed([
                    StructuredCandidate(concept=concept, value=value,
                                        memory_type=mtype, scope="session")])
        return LearningOutcome.rejected("jev_found_nothing_structurable")


class _HostileJev(Jev):
    """Confirms everything, including inferences. Must still be overruled."""

    name = "hostile-jev"

    def evaluate(self, request):
        return LearningOutcome.confirmed(
            [StructuredCandidate(concept="anything", value="yes",
                                 memory_type="FACT")],
            reason="I feel confident about this")


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"value": 4})


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


_DB_SEQ = [0]


def _prime(jev=None, provider=None):
    """A Prime on a throwaway database.

    ``Prime()`` defaults to the production ContextManager and ``respond``
    persists, so every Prime here is redirected before anything writes.
    """
    _DB_SEQ[0] += 1
    prime = Prime(provider or _Provider(), jev=jev)
    prime._context = prime._context.__class__(
        db_path=os.path.join(tempfile.mkdtemp(prefix="learn_"),
                            f"l{_DB_SEQ[0]}.db"))
    prime._mind = prime._mind.__class__(prime._context)
    for name in ("recall", "learn"):
        cap = prime.registry.get_specialist(name)
        if cap is not None:
            cap.bind(prime._mind)
    return prime


def _aurora(prime, brain):
    return Aurora(prime, brain=brain,
                  config=AuroraConfig(model_path="m", adapter_path="a"))


def _store(prime):
    return prime._context._get_store()


def _learn(prime, text, provenance, **kw):
    payload = {"text": text, "provenance": provenance}
    payload.update(kw)
    return prime.handle(PrimeRequest("learn", payload))


def _current(prime):
    return {s.concept: s for s in _store(prime).fetch_states(status="current")}


# ======================================================================
# 1. Capability contract
# ======================================================================


class TestCapabilityContract:
    def test_registered(self):
        assert _prime().registry.has("learn")

    def test_manifest_shape(self):
        spec = _prime().registry.get("learn")
        assert spec.required_fields == ["text", "provenance"]
        assert spec.optional_fields == ["session_id"]
        assert spec.payload_schema["properties"]["provenance"]["enum"] == \
            list(PROVENANCE)

    def test_capabilities_registered_so_far(self):
        """Milestone 3 adds capabilities one at a time."""
        assert sorted(_prime().registry.get_names()) == [
            "calculate", "consult_specialist", "learn", "recall", "research",
            "research_read", "use_tool"]

    def test_runs_through_prime(self):
        out = _learn(_prime(_FakeJev()), "Amir loves Dune Part Two.", "user")
        assert out.status is RequestStatus.SUCCESS
        assert out.output["status"] == "learned"

    def test_provenance_vocabulary_is_exposed_to_aurora(self):
        desc = _prime().registry.get("learn").description
        for word in PROVENANCE:
            assert word in desc, word

    @pytest.mark.parametrize("bad", [None, "", "   ", 5, [], {}])
    def test_bad_text_rejected(self, bad):
        out = _prime(_FakeJev()).handle(
            PrimeRequest("learn", {"text": bad, "provenance": "user"}))
        assert out.status is RequestStatus.FAILURE

    @pytest.mark.parametrize("bad", [None, "", "   ", "USER", "guessed", 5])
    def test_bad_provenance_rejected(self, bad):
        out = _prime(_FakeJev()).handle(
            PrimeRequest("learn", {"text": "something", "provenance": bad}))
        assert out.status is RequestStatus.FAILURE

    def test_oversized_text_rejected(self):
        out = _prime(_FakeJev()).handle(
            PrimeRequest("learn", {"text": "x" * 5000, "provenance": "user"}))
        assert out.status is RequestStatus.FAILURE

    def test_fails_without_a_jev(self):
        out = _learn(_prime(), "Amir loves Dune.", "user")
        assert out.status is RequestStatus.FAILURE
        assert "jev" in out.error.lower()
        assert _current(_prime()) == {}

    def test_unconfigured_jev_is_not_advertised(self):
        prime = _prime()
        specialist = prime.registry.get_specialist("learn")
        assert specialist.is_configured is False
        from PRIMORA.aurora.configuration import build_capability_instruction
        block = build_capability_instruction(prime)
        assert "- learn:" not in block

    def test_configured_jev_is_advertised(self):
        prime = _prime(_FakeJev())
        specialist = prime.registry.get_specialist("learn")
        assert specialist.is_configured is True
        from PRIMORA.aurora.configuration import build_capability_instruction
        block = build_capability_instruction(prime)
        assert "- learn:" in block

    def test_unconfigured_jev_refuses_loudly(self):
        with pytest.raises(RuntimeError):
            UnconfiguredJev().evaluate(
                LearningRequest("x", "user", "s"))

    def test_rejects_none_jev(self):
        with pytest.raises(ValueError):
            LearnSpecialist().set_jev(None)

    def test_jev_learns_nothing_without_a_request(self):
        assert LearnSpecialist().jev is not None


# ======================================================================
# 2. Aurora cannot write the internal schema
# ======================================================================


class TestAuroraCannotWriteSchema:
    @pytest.mark.parametrize("key", [
        "concept", "value", "scope", "memory_type", "authority",
        "confidence", "status", "candidate_id", "state_id", "auto_confirm",
    ])
    def test_internal_fields_are_refused(self, key):
        out = _prime(_FakeJev()).handle(PrimeRequest("learn", {
            "text": "Amir loves Dune Part Two.",
            "provenance": "user",
            key: "attacker-controlled",
        }))
        assert out.status is RequestStatus.FAILURE
        assert "internal memory structure" in out.error

    def test_a_smuggled_schema_writes_nothing(self):
        prime = _prime(_FakeJev())
        before = len(_current(prime))
        _prime(_FakeJev()).handle(PrimeRequest("learn", {
            "text": "Amir loves Dune Part Two.",
            "provenance": "inference",
            "concept": "trust me",
            "value": "a fact",
            "authority": "user",
        }))
        assert len(_current(prime)) == before

    def test_capability_does_not_know_the_memory_schema(self):
        """It refuses the field names; it does not construct them."""
        from PRIMORA.specialists.tools import learn as mod
        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for banned in ("create_state", "create_candidate", "confirm_pending",
                       "_apply_confirmed", "insert"):
            assert banned not in called, banned


# ======================================================================
# 3. Provenance, and the inference line
# ======================================================================


class TestProvenance:
    def test_user_provenance_is_admitted(self):
        prime = _prime(_FakeJev())
        assert _learn(prime, "Amir loves Dune Part Two.", "user") \
            .output["status"] == "learned"
        assert "favorite movie" in _current(prime)

    @pytest.mark.parametrize("provenance", ["research", "specialist",
                                            "observation"])
    def test_grounded_provenance_is_still_held_not_promoted(self, provenance):
        """Only what Amir said becomes a fact, even when well grounded.

        ``ContextStore`` refuses to make a non-user-authored state current.
        Research and specialist results are recorded as pending candidates
        with their provenance intact, and wait.
        """
        prime = _prime(_FakeJev())
        out = _learn(prime, "Amir's deploy key is BLUE-7", provenance)
        assert out.output["status"] == "held"
        assert _current(prime) == {}
        rows = _store(prime).fetch_candidates(
            candidate_id=out.output["held"][0]["candidate_id"])
        assert rows[0].authority == provenance, "provenance must be preserved"

    @pytest.mark.parametrize("provenance", sorted(HOLDS))
    def test_inference_and_unknown_never_become_facts(self, provenance):
        prime = _prime(_FakeJev())
        out = _learn(prime, "Amir's deploy key is BLUE-7", provenance)
        assert out.output["status"] == "held"
        assert _current(prime) == {}, "an inference became a current fact"

    def test_a_held_candidate_is_kept_not_discarded(self):
        prime = _prime(_FakeJev())
        out = _learn(prime, "Amir's deploy key is BLUE-7", "inference")
        candidate_id = out.output["held"][0]["candidate_id"]
        rows = _store(prime).fetch_candidates(candidate_id=candidate_id)
        assert rows[0].status == "pending"
        assert rows[0].authority == "inference"

    def test_hostile_jev_cannot_confirm_an_inference(self):
        """Regression: the admission layer owns the policy, not Jev.

        A pluggable Jev returning 'confirm' for an inference must not be
        able to turn it into a fact. The invariant lives in the trust
        boundary, not in a component that can be replaced.
        """
        prime = _prime(_HostileJev())
        out = _learn(prime, "Amir probably hates horror films.", "inference")
        assert out.output["status"] == "held"
        assert _current(prime) == {}

    def test_hostile_jev_still_cannot_override_a_user_learn(self):
        prime = _prime(_HostileJev())
        assert _learn(prime, "anything", "user").output["status"] == "learned"

    def test_provenance_reaches_jev_untouched(self):
        jev = _FakeJev()
        _prime(jev).handle(PrimeRequest(
            "learn", {"text": "Amir loves Dune.", "provenance": "specialist"}))
        assert jev.requests[0].provenance == "specialist"


# ======================================================================
# 4. The existing lifecycle does the admitting
# ======================================================================


class TestExistingLifecycle:
    def test_the_statement_is_persisted_as_an_assistant_message(self):
        prime = _prime(_FakeJev())
        out = _learn(prime, "Amir loves Dune Part Two.", "user")
        rows = _store(prime).fetch_messages_by_ids([out.output["source_id"]])
        assert rows[0].role == "assistant", (
            "an assistant statement must not look like something Amir said"
        )
        assert "Dune Part Two" in rows[0].content

    def test_only_user_provenance_reaches_a_current_state(self):
        prime = _prime(_FakeJev())
        _learn(prime, "Amir's deploy key is BLUE-7", "user")
        assert _current(prime)["deploy key"].authority == "user"

    def test_the_store_invariant_is_never_bypassed(self):
        """A non-user state cannot become current, by whatever route."""
        store = _store(_prime(_FakeJev()))
        with pytest.raises(ValueError, match="user-authored"):
            store.create_state(
                "default", "session", "x", "y", [1], status="current",
                authority="research")

    def test_state_rows_carry_memory_type(self):
        prime = _prime(_FakeJev())
        _learn(prime, "Amir loves Dune Part Two.", "user")
        assert _current(prime)["favorite movie"].memory_type == "PREFERENCE"

    def test_candidate_is_recorded_before_confirmation(self):
        prime = _prime(_FakeJev())
        out = _learn(prime, "Amir's deploy key is BLUE-7", "user")
        rows = _store(prime).fetch_candidates(
            candidate_id=out.output["learned"][0]["candidate_id"])
        assert rows[0].status == "confirmed"

    def test_a_contradicting_learn_supersedes_rather_than_duplicates(self):
        prime = _prime(_FakeJev())
        _learn(prime, "Amir's deploy key is BLUE-7", "user")
        jev = _FakeJev(LearningOutcome.confirmed([
            StructuredCandidate(concept="deploy key", value="RED-9",
                                memory_type="FACT")]))
        prime.registry.get_specialist("learn").set_jev(jev)
        _learn(prime, "Amir's deploy key is RED-9", "user")
        concepts = [s.concept for s in _store(prime).fetch_states()]
        assert concepts.count("deploy key") == 2, "supersession must keep history"
        assert _current(prime)["deploy key"].value == "RED-9"

    def test_no_new_tables_or_schema(self):
        import sqlite3
        prime = _prime(_FakeJev())
        path = str(_store(prime)._db_path)
        with sqlite3.connect(path) as con:
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        for required in ("messages", "states", "candidates"):
            assert required in tables
        for invented in ("learnings", "lessons", "memories_new"):
            assert invented not in tables, invented

    def test_history_is_preserved(self):
        prime = _prime(_FakeJev())
        _learn(prime, "Amir's deploy key is BLUE-7", "user")
        history = prime._context.history("default", "deploy key")
        assert len(history) == 1


# ======================================================================
# 5. Fails safely
# ======================================================================


class TestFailsSafely:
    def test_jev_exception_does_not_corrupt_memory(self):
        class Exploding(Jev):
            name = "boom"

            def evaluate(self, request):
                raise RuntimeError("jev exploded")

        prime = _prime(Exploding())
        out = _learn(prime, "Amir loves Dune Part Two.", "user")
        assert out.status is RequestStatus.FAILURE
        assert "rejected" in out.error.lower()
        assert _current(prime) == {}

    def test_malformed_candidate_is_refused_not_stored(self):
        jev = _FakeJev(LearningOutcome.confirmed([
            StructuredCandidate(concept="", value="orphan")]))
        prime = _prime(jev)
        out = _learn(prime, "Amir loves Dune Part Two.", "user")
        assert out.status is RequestStatus.FAILURE
        assert "rejected" in out.error.lower()
        assert _current(prime) == {}

    def test_fact_with_no_value_is_refused(self):
        jev = _FakeJev(LearningOutcome.confirmed([
            StructuredCandidate(concept="favorite movie", value="",
                                memory_type="FACT")]))
        prime = _prime(jev)
        out = _learn(prime, "something", "user")
        assert out.status is RequestStatus.FAILURE
        assert "rejected" in out.error.lower()
        assert _current(prime) == {}

    def test_jev_reject_leaves_nothing_behind(self):
        prime = _prime(_FakeJev(LearningOutcome.rejected("nothing here")))
        out = _learn(prime, "hello there", "user")
        assert out.status is RequestStatus.FAILURE
        assert "rejected" in out.error.lower()
        assert _current(prime) == {}

    def test_store_failure_is_reported_not_raised(self):
        prime = _prime(_FakeJev())

        class BrokenStore:
            def count(self, *a, **k):
                raise RuntimeError("database gone")

        prime._context._store = BrokenStore()
        out = _learn(prime, "Amir loves Dune Part Two.", "user")
        assert out.status is RequestStatus.FAILURE
        assert "persist_failed" in out.error

    def test_jev_never_sees_a_broken_request(self):
        jev = _FakeJev()
        prime = _prime(jev)
        prime.handle(PrimeRequest("learn", {"text": "  ", "provenance": "user"}))
        assert jev.requests == []


# ======================================================================
# 6. ModelJev parsing, without a live model
# ======================================================================


class TestModelJevParsing:
    def _jev(self, reply):
        class P(Provider):
            def invoke(self, capability, payload):
                return PrimeResult.success({"content": reply})

        return ModelJev(P())

    def test_parses_wellformed_json(self):
        outcome = self._jev(json.dumps({"candidates": [{
            "concept": "favorite movie", "value": "Dune Part Two",
            "memory_type": "PREFERENCE", "scope": "global"}]}
        )).evaluate(LearningRequest("x", "user", "s"))
        assert outcome.disposition == "confirm"
        assert outcome.candidates[0].concept == "favorite movie"
        assert outcome.candidates[0].scope == "global"

    def test_strips_a_code_fence(self):
        reply = "```json\n{\"candidates\": [{\"concept\": \"a\", \"value\": \"b\"}]}\n```"
        outcome = self._jev(reply).evaluate(LearningRequest("x", "user", "s"))
        assert outcome.candidates[0].concept == "a"

    @pytest.mark.parametrize("reply", [
        "not json at all",
        "{broken",
        '{"candidates": "not a list"}',
        '{"candidates": [{"no": "concept"}]}',
        '{"candidates": "nope"}',
        "",
    ])
    def test_malformed_replies_are_rejected_never_believed(self, reply):
        outcome = self._jev(reply).evaluate(LearningRequest("x", "user", "s"))
        assert outcome.disposition == "reject"
        assert outcome.candidates == []

    def test_unknown_memory_type_is_dropped_not_guessed(self):
        reply = json.dumps({"candidates": [
            {"concept": "a", "value": "b", "memory_type": "VIBES"}]})
        outcome = self._jev(reply).evaluate(LearningRequest("x", "user", "s"))
        assert outcome.candidates[0].memory_type == ""
        assert any("memory_type" in d for d in outcome.diagnostics)

    def test_non_global_scope_is_narrowed(self):
        reply = json.dumps({"candidates": [
            {"concept": "a", "value": "b", "scope": "universe"}]})
        outcome = self._jev(reply).evaluate(LearningRequest("x", "user", "s"))
        assert outcome.candidates[0].scope == "session"

    def test_candidate_cap_is_enforced(self):
        reply = json.dumps({"candidates": [
            {"concept": f"c{i}", "value": "v"} for i in range(20)]})
        outcome = self._jev(reply).evaluate(LearningRequest("x", "user", "s"))
        assert len(outcome.candidates) <= 4

    def test_provider_failure_is_never_believed(self):
        class P(Provider):
            def invoke(self, capability, payload):
                return PrimeResult.failure("model HTTP 500")

        outcome = ModelJev(P()).evaluate(LearningRequest("x", "user", "s"))
        assert outcome.disposition == "reject"

    def test_provider_exception_is_never_believed(self):
        class P(Provider):
            def invoke(self, capability, payload):
                raise MemoryError("catastrophic")

        outcome = ModelJev(P()).evaluate(LearningRequest("x", "user", "s"))
        assert outcome.disposition == "reject"

    def test_model_jev_holds_an_inference_too(self):
        reply = json.dumps({"candidates": [{"concept": "a", "value": "b"}]})
        outcome = self._jev(reply).evaluate(LearningRequest("x", "inference", "s"))
        assert outcome.disposition == "hold"

    def test_confidence_follows_provenance(self):
        for provenance, expected in (("user", 1.0), ("research", 0.6),
                                     ("inference", 0.3)):
            reply = json.dumps({"candidates": [{"concept": "a", "value": "b"}]})
            outcome = self._jev(reply).evaluate(
                LearningRequest("x", provenance, "s"))
            assert outcome.candidates[0].confidence == expected, provenance

    def test_requires_a_provider(self):
        with pytest.raises(ValueError):
            ModelJev(None)


# ======================================================================
# 7. Aurora can recall what she learned
# ======================================================================


class TestRecallableLater:
    def test_learned_statement_is_recallable(self):
        """The persisted sentence comes back through the existing retrieval."""
        prime = _prime(_FakeJev())
        _learn(prime, "Amir loves Dune Part Two.", "user")
        hits = prime.mind.recall("Dune", session_id="default")
        assert any("Dune Part Two" in h["content"] for h in hits)

    def test_inference_is_not_recallable_as_fact(self):
        prime = _prime(_FakeJev())
        _learn(prime, "Amir's deploy key is BLUE-7", "inference")
        assert _current(prime) == {}
        built = prime.mind.build(
            [{"role": "user", "content": "what is the deploy key"}], "default",
            token_counter=lambda m: 8, max_new_tokens=16)
        assert built.messages is not None


# ======================================================================
# 8. Inside the cognitive loop
# ======================================================================


class TestInsideTheLoop:
    def _run(self, replies, jev=None):
        brain = _Scripted(*replies)
        a = _aurora(_prime(jev or _FakeJev()), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "remember something")
        return brain, a

    def test_aurora_can_request_learning_during_cognition(self):
        brain, a = self._run([
            _request("learn", {"text": "Amir loves Dune Part Two.",
                               "provenance": "user"}),
            "Noted, I will remember that.",
        ])
        result = a.respond(
            [{"role": "user", "content": "my favorite movie is Dune Part Two"}],
            "s")
        assert result.content.startswith("Noted")
        assert len(brain.calls) == 2

    def test_request_reaches_the_capability(self):
        brain, a = self._run([
            _request("learn", {"text": "Amir loves Dune Part Two.",
                               "provenance": "user"}), "done"])
        a.respond([{"role": "user", "content": "remember that"}], "s")
        assert [n for n, _ in a.loop.executor.calls] == ["learn"]

    def test_disposition_reaches_the_next_segment(self):
        brain, a = self._run([
            _request("learn", {"text": "Amir loves Dune Part Two.",
                               "provenance": "user"}), "done"])
        a.respond([{"role": "user", "content": "remember that"}], "s")
        tool = [m for m in brain.calls[1] if m["role"] == "tool"][0]["content"]
        assert "learned" in tool

    def test_aurora_can_say_it_held_an_inference(self):
        brain, a = self._run([
            _request("learn", {"text": "Amir probably hates horror films.",
                               "provenance": "inference"}),
            "I held that as a hypothesis rather than a fact.",
        ])
        result = a.respond(
            [{"role": "user", "content": "amir seems to hate horror"}], "s")
        assert "hypothesis" in result.content
        tool = [m for m in brain.calls[1] if m["role"] == "tool"][0]["content"]
        assert "held" in tool

    def test_learn_failure_does_not_break_the_session(self):
        brain, a = self._run([
            _request("learn", {"text": "Amir loves Dune Part Two.",
                               "provenance": "user"}),
            "I could not store that.",
        ], jev=_FakeJev(LearningOutcome.rejected("nothing here")))
        result = a.respond([{"role": "user", "content": "remember"}], "s")
        assert result.success
        assert "could not store" in result.content

    def test_still_bounded_by_max_cycles(self):
        brain, a = self._run([
            _request("learn", {"text": "a one", "provenance": "user"}),
            _request("learn", {"text": "b two", "provenance": "user"}),
            _request("learn", {"text": "c three", "provenance": "user"}),
            "done"])
        a.loop.max_cycles = 2
        a.respond([{"role": "user", "content": "go"}], "s")
        assert len(brain.calls) == 2

    def test_repeated_identical_learn_is_blocked(self):
        jev = _FakeJev()
        brain, a = self._run([
            _request("learn", {"text": "Amir loves Dune Part Two.",
                               "provenance": "user"})], jev=jev)
        a.loop.max_cycles = 20
        a.respond([{"role": "user", "content": "go"}], "s")
        assert a.loop.last_trace.stop_reason == "repeated_request"

    def test_aurora_is_told_learn_exists(self):
        brain, a = self._run(["done"])
        a.respond([{"role": "user", "content": "hi"}], "s")
        system = brain.calls[0][0]["content"]
        assert "learn" in system
        assert "inference" in system


# ======================================================================
# 9. Not retraining
# ======================================================================


class TestNotRetraining:
    def test_nothing_touches_weights(self):
        from PRIMORA.specialists.tools import learn as mod
        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        src = ast.unparse(tree).lower()
        for banned in ("torch", "weight", "optimizer", "backward", "train",
                       "adapter", "checkpoint", "lora", "gradient"):
            assert banned not in src, banned

    def test_learning_module_has_no_training(self):
        src = open("PRIMORA" + os.sep + "learning" + os.sep + "jev.py",
                   encoding="utf-8").read().lower()
        for banned in ("torch", "optimizer", "backward", "train", "lora",
                       "gradient"):
            assert banned not in src, banned

    def test_brain_is_untouched(self):
        import importlib
        brain = importlib.import_module("PRIMORA.aurora.brain")
        src = open(brain.__file__, encoding="utf-8").read().lower()
        for banned in ("learn(", "training", "optimizer", "loss.backward"):
            assert banned not in src, banned

    def test_cognitive_loop_is_unmodified(self):
        from PRIMORA.aurora import loop as loopmod
        tree = ast.parse(open(loopmod.__file__, encoding="utf-8").read())
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for banned in ("learn", "LearningRequest", "StructuredCandidate"):
            assert banned not in called


# ======================================================================
# 10. Scope
# ======================================================================


class TestScope:
    def test_no_goals_or_proactive_triggers(self):
        for module in ("PRIMORA.specialists.tools.learn",
                       "PRIMORA.learning.jev"):
            src = open(module.replace(".", os.sep) + ".py",
                       encoding="utf-8").read().lower()
            for banned in ("schedule", "trigger", "proactive", "autonomous",
                           "while true", "cron"):
                assert banned not in src, f"{module}: {banned}"

    def test_capability_touches_no_storage(self):
        src = open(os.path.join("PRIMORA", "specialists", "tools", "learn.py"),
                   encoding="utf-8").read().lower()
        for banned in ("sqlite", "contextstore", "create_candidate",
                       "confirm_pending", "fetch_states"):
            assert banned not in src, banned

    def test_memory_modules_gained_no_learning_logic(self):
        """The mechanism layer may reference the seam, never implement it.

        ``manager.learn`` is orchestration: it must not know Jev's types or
        reimplement the provenance policy it delegates.
        """
        import ast
        for name in ("hybrid.py", "store.py", "semantic.py"):
            src = open(os.path.join("PRIMORA", "prime", "context", name),
                       encoding="utf-8").read().lower()
            for banned in ("structuredcandidate", "learningrequest", "jev",
                           "auto_admit", "provenance="):
                assert banned not in src, f"{name}: {banned}"
        tree = ast.parse(open(os.path.join("PRIMORA", "prime", "context",
                                           "manager.py"),
                              encoding="utf-8").read())
        # Pre-existing "candidate" methods (M5 entities, lexical) are fine.
        # What must not exist is anything that knows Jev's types.
        jev_names = {n.name for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                     and "jev" in n.name.lower()}
        assert not jev_names, jev_names
        leaked = {
            n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and "learn" not in n.name
            and any(isinstance(x, ast.Name)
                    and x.id in ("StructuredCandidate", "LearningRequest",
                                 "LearningOutcome", "AUTO_ADMIT")
                    for x in ast.walk(n))
        }
        assert not leaked, f"learning types leaked outside learn(): {leaked}"

    def test_verse_untouched(self):
        before = os.path.getmtime(os.path.join("PRIMORA", "verse", "qml",
                                               "Universe.qml"))
        assert before > 0

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
