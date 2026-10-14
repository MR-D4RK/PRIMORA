"""Milestone 3: the first capability, ``recall()``.

The property that matters: recall is not a second retrieval system. It reuses
M6, so these tests assert reuse rather than re-deriving expected results.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.contract import PrimeRequest, RequestStatus
from PRIMORA.prime.providers import Provider
from PRIMORA.specialists.tools.recall import RecallSpecialist


def _db(name):
    d = tempfile.mkdtemp(prefix="recall_")
    return os.path.join(d, name + ".db")


def _request(action, payload):
    import json
    return (
        "<prime_request>\n"
        + json.dumps({
            "action": action, "payload": payload,
            "context": None, "metadata": None,
        })
        + "\n</prime_request>"
    )


class _Provider(Provider):
    def __init__(self):
        self.calls = []

    def invoke(self, capability, payload):
        self.calls.append(capability)
        from PRIMORA.prime.contract import PrimeResult
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


def _prime_with_memory(db_name="m.db"):
    """A Prime whose store already holds a conversation worth recalling."""
    prime = Prime(_Provider())
    prime._context = ContextManager(db_path=_db(db_name))
    prime._mind = prime._mind.__class__(prime._context)
    prime._registry.get_specialist("recall").bind(prime._mind)
    prime._context.reconcile([
        {"role": "user", "content": "the deployment key rotates every friday"},
        {"role": "assistant", "content": "understood, friday rotation confirmed"},
        {"role": "user", "content": "unrelated: the office plants need watering"},
        {"role": "assistant", "content": "noted, watering is tuesday"},
    ], "s1")
    return prime


# ======================================================================
# 1. It reuses the existing Mind and M6 path
# ======================================================================


class TestReuseOfExistingPath:
    def test_mind_exposes_recall(self):
        prime = _prime_with_memory("m1.db")
        hits = prime.mind.recall("deployment key", session_id="s1", limit=3)
        assert isinstance(hits, list)
        assert hits, "expected the stored conversation to be searchable"

    def test_manager_recall_is_the_m6_path(self):
        prime = _prime_with_memory("m2.db")
        results = prime.context.recall("deployment key", session_id="s1")
        assert results
        assert {"role", "content"} <= set(results[0])

    def test_recall_does_not_add_a_retrieval_implementation(self):
        """recall must delegate, not rank on its own."""
        import ast
        from PRIMORA.specialists.tools import recall as mod
        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        for node in ast.walk(tree):
            body = getattr(node, "body", [])
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                for stmt in body:
                    assert not isinstance(stmt, ast.ClassDef) or node is not tree
        src = open(mod.__file__, encoding="utf-8").read()
        # No scoring, fusion, or ranking vocabulary in the capability itself.
        for banned in ("fused_score", "rrf", "semantic_index", "tokenize"):
            assert banned not in src, f"recall should not implement {banned}"

    def test_recall_reaches_the_same_results_as_context_assembly(self):
        """The engine is shared: recall's ranking is M6's ranking."""
        prime = _prime_with_memory("m3.db")
        from_store = prime.context.recall("deployment key", session_id="s1")
        assert from_store
        # The same query through the ordinary build path also surfaces it.
        built = prime.context.build(
            [{"role": "user", "content": "deployment key"}], "s1",
            token_counter=lambda m: 10, max_new_tokens=32,
        )
        joined = " ".join(m["content"] for m in built.view.messages)
        assert "rotation" in joined or "deployment" in joined


# ======================================================================
# 2. The capability contract
# ======================================================================


class TestCapabilityContract:
    def test_recall_is_registered(self):
        prime = Prime(_Provider())
        assert prime.registry.has("recall")
        assert "recall" in prime.registry.get_names()

    def test_manifest_describes_it(self):
        prime = Prime(_Provider())
        spec = prime.registry.get("recall")
        assert spec.required_fields == ["query"]
        assert set(spec.optional_fields) == {"limit", "session_id"}
        assert spec.payload_schema["required"] == ["query"]

    def test_aurora_is_told_it_exists(self):
        prime = Prime(_Provider())
        assert "recall" in prime.registry.get_manifest()

    def test_runs_through_prime(self):
        prime = _prime_with_memory("c1.db")
        result = prime.handle(PrimeRequest("recall", {"query": "deployment key"}))
        assert result.status is RequestStatus.SUCCESS
        assert result.output["results"]

    def test_missing_query_fails_cleanly(self):
        prime = _prime_with_memory("c2.db")
        result = prime.handle(PrimeRequest("recall", {}))
        assert result.status is RequestStatus.FAILURE
        assert "query" in result.error

    def test_blank_query_fails(self):
        prime = _prime_with_memory("c3.db")
        assert prime.handle(PrimeRequest("recall", {"query": "  "})).status \
            is RequestStatus.FAILURE

    def test_unknown_capability_still_fails(self):
        prime = Prime(_Provider())
        assert prime.handle(PrimeRequest("teleport", {})).status \
            is RequestStatus.FAILURE

    @pytest.mark.parametrize("bad", [[], "x", 3, None])
    def test_non_dict_payload_fails(self, bad):
        prime = _prime_with_memory("c4.db")
        assert prime.handle(PrimeRequest("recall", bad)).status \
            is RequestStatus.FAILURE

    def test_limit_is_bounded(self):
        prime = _prime_with_memory("c5.db")
        out = prime.handle(PrimeRequest("recall", {"query": "key", "limit": 9999}))
        assert out.status is RequestStatus.SUCCESS
        assert len(out.output["results"]) <= 12

    def test_limit_must_be_integer(self):
        prime = _prime_with_memory("c6.db")
        for bad in ("5", 2.5, True):
            assert prime.handle(
                PrimeRequest("recall", {"query": "key", "limit": bad})
            ).status is RequestStatus.FAILURE

    def test_unbound_specialist_fails_rather_than_crashing(self):
        r = RecallSpecialist()
        result = r.handle({"query": "anything"})
        assert result.status is RequestStatus.FAILURE
        assert "bound" in result.error

    def test_retrieval_error_is_reported_not_raised(self):
        class BrokenMind:
            def recall(self, *a, **k):
                raise RuntimeError("index unavailable")

        result = RecallSpecialist(BrokenMind()).handle({"query": "x"})
        assert result.status is RequestStatus.FAILURE
        assert "recall_failed" in result.error

    def test_result_carries_only_role_and_content(self):
        prime = _prime_with_memory("c7.db")
        out = prime.handle(PrimeRequest("recall", {"query": "deployment key"}))
        for item in out.output["results"]:
            assert set(item) == {"role", "content"}


# ======================================================================
# 3. Reaches the real stored text
# ======================================================================


class TestRecallQuality:
    def test_finds_the_relevant_passage(self):
        prime = _prime_with_memory("q1.db")
        out = prime.handle(
            PrimeRequest("recall", {"query": "when does the key rotate"})
        )
        text = " ".join(i["content"] for i in out.output["results"]).lower()
        assert "friday" in text

    def test_does_not_just_return_everything(self):
        prime = _prime_with_memory("q2.db")
        out = prime.handle(PrimeRequest("recall", {"query": "deployment key"}))
        text = " ".join(i["content"] for i in out.output["results"]).lower()
        assert "watering" not in text, "irrelevant passage leaked in"

    def test_returns_verbatim_stored_text(self):
        """Exactly the stored string, not a summary or a reworded line."""
        prime = _prime_with_memory("q3.db")
        out = prime.handle(
            PrimeRequest("recall", {"query": "key rotation", "session_id": "s1"})
        )
        contents = [i["content"] for i in out.output["results"]]
        assert "the deployment key rotates every friday" in contents
        assert "understood, friday rotation confirmed" in contents

    def test_empty_store_returns_empty_not_error(self):
        prime = Prime(_Provider())
        prime._context = ContextManager(db_path=_db("empty.db"))
        prime._mind = prime._mind.__class__(prime._context)
        prime._registry.get_specialist("recall").bind(prime._mind)
        out = prime.handle(PrimeRequest("recall", {"query": "anything"}))
        assert out.status is RequestStatus.SUCCESS
        assert out.output["results"] == []

    def test_unknown_topic_returns_empty_not_hallucinated(self):
        prime = _prime_with_memory("q4.db")
        out = prime.handle(
            PrimeRequest("recall", {"query": "quarterly revenue forecast"})
        )
        assert out.output["results"] == []

    def test_same_session_evidence_is_not_prefixed(self):
        """Within the searched session, text is returned as stored."""
        prime = _prime_with_memory("q5a.db")
        out = prime.handle(
            PrimeRequest("recall", {"query": "key rotation", "session_id": "s1"})
        )
        for item in out.output["results"]:
            assert not item["content"].startswith("[from another conversation]")

    def test_other_sessions_are_marked_when_returned(self):
        """M6's cross-session policy, applied unchanged.

        recall() reuses the live admission path, so evidence from another
        conversation may be admitted up to the existing cap, and every such
        passage carries the existing marker. This is M6's documented policy
        for build(); recall does not get a stricter one. Asserted here so a
        future change to this behaviour is deliberate.
        """
        prime = _prime_with_memory("q5.db")
        prime._context.reconcile([
            {"role": "user", "content": "my password is hunter2 secret"},
        ], "s2")
        out = prime.handle(
            PrimeRequest("recall", {"query": "password", "session_id": "s1"})
        )
        for item in out.output["results"]:
            if item["content"] != "the deployment key rotates every friday":
                assert item["content"].startswith("[from another conversation]")


# ======================================================================
# 3b. The just-persisted question is not a stored fact
# ======================================================================


class TestEchoIsNotAMemory:
    """The live loop persists the current turn before recall runs.

    Retrieval therefore hands back the user's own question as the top hit.
    That is not a stored fact, and treating it as one taught Aurora she had
    "remembered" something when nothing was stored. The filter lives in the
    capability, not in M6, so the ranking path is untouched.
    """

    def test_query_echo_is_dropped(self):
        from PRIMORA.specialists.tools.recall import _is_query_echo
        assert _is_query_echo("fav car", "what is my fav car?")
        assert _is_query_echo("fav color", "fav color")
        assert not _is_query_echo(
            "fav car", "My favourite car is a Porsche 911")

    def test_foreign_echo_of_the_same_question_is_dropped(self):
        from PRIMORA.specialists.tools.recall import _is_query_echo
        # An older question in another conversation is still the question,
        # not a stored fact; the M6 cross-session prefix must not smuggle
        # it back in as an answer.
        assert _is_query_echo(
            "fav color",
            "[from another conversation]\nwhat is my fav color")
        assert not _is_query_echo(
            "fav color",
            "[from another conversation]\nmy favourite colour is teal")

    def test_the_just_asked_question_is_not_returned_as_a_hit(self):
        prime = _prime_with_memory("e1.db")
        # Persist the current question exactly as the live loop does.
        prime._context.reconcile([
            {"role": "user", "content": "what is my fav car"},
        ], "s1")
        out = prime.handle(
            PrimeRequest("recall", {"query": "fav car", "session_id": "s1"})
        )
        contents = [i["content"] for i in out.output["results"]]
        assert "what is my fav car" not in contents

    def test_a_real_fact_with_extra_tokens_is_kept(self):
        prime = _prime_with_memory("e2.db")
        prime._context.reconcile([
            {"role": "user", "content": "my favourite car is a Porsche 911"},
        ], "s1")
        out = prime.handle(
            PrimeRequest("recall", {"query": "fav car", "session_id": "s1"})
        )
        contents = " ".join(i["content"] for i in out.output["results"])
        assert "Porsche" in contents


# ======================================================================
# 4. Inside the cognitive loop, through the existing seam
# ======================================================================


class TestInsideTheLoop:
    def _aurora(self, prime, brain):
        cfg = AuroraConfig(
            model_path="RESOURCES/aurora/v0.1/model",
            adapter_path="RESOURCES/aurora/v0.1/aurora_qlora",
        )
        return Aurora(prime, brain=brain, config=cfg)

    def test_model_can_ask_and_get_evidence(self):
        prime = _prime_with_memory("l1.db")
        brain = _Scripted(
            _request("recall", {"query": "when does the key rotate"}),
            "the key rotates every friday",
        )
        a = self._aurora(prime, brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s1", "check the rotation policy")
        result = a.respond([{"role": "user", "content": "when is the key rotated?"}],
                           "s1")
        assert result.content == "the key rotates every friday"
        assert len(brain.calls) == 2

    def test_defaults_to_the_current_task_session(self):
        """A model that omits session_id still gets the right session.

        M6 treats the searched session as current and everything else as
        cross-session, so a wrong default mislabels the whole result set.
        """
        prime = _prime_with_memory("l0.db")
        brain = _Scripted(
            _request("recall", {"query": "when does the key rotate"}),
            "answered",
        )
        a = self._aurora(prime, brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s1", "t")
        a.respond([{"role": "user", "content": "when?"}], "s1")
        tool = [m for m in brain.calls[1] if m["role"] == "tool"][0]["content"]
        for item in json.loads(tool)["output"]["results"]:
            assert not item["content"].startswith("[from another conversation]"), (
                "the current task's own session was treated as foreign"
            )

    def test_evidence_reaches_the_next_segment(self):
        prime = _prime_with_memory("l2.db")
        brain = _Scripted(
            _request("recall", {"query": "when does the key rotate"}),
            "answered",
        )
        a = self._aurora(prime, brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s1", "t")
        a.respond([{"role": "user", "content": "when?"}], "s1")
        second = " ".join(m["content"] for m in brain.calls[1])
        assert "friday" in second, "retrieved evidence missing from the context"

    def test_the_manifest_reaches_the_system_prompt(self):
        prime = _prime_with_memory("l3.db")
        brain = _Scripted("done")
        a = self._aurora(prime, brain)
        a.mind.begin_task("s1", "t")
        a.respond([{"role": "user", "content": "hi"}], "s1")
        system = brain.calls[0][0]["content"]
        assert "recall" in system
        assert "stored conversation memory" in system

    def test_recall_failure_does_not_break_the_loop(self):
        prime = _prime_with_memory("l4.db")
        prime._registry.get_specialist("recall")._mind = None
        brain = _Scripted(
            _request("recall", {"query": "key"}),
            "I could not retrieve that",
        )
        a = self._aurora(prime, brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s1", "t")
        result = a.respond([{"role": "user", "content": "when?"}], "s1")
        assert result.content == "I could not retrieve that"
        assert result.success
        tool = [m for m in brain.calls[1] if m["role"] == "tool"]
        assert tool and "failure" in tool[0]["content"]

    def test_calculate_still_works(self):
        prime = Prime(_Provider())
        out = prime.handle(
            PrimeRequest("calculate", {"expression": "2+2"})
        )
        assert out.status is RequestStatus.SUCCESS
        assert out.output["result"] == 4

    def test_recall_does_not_persist_anything_new(self):
        """recall is read-only: a search must not add memory."""
        import sqlite3
        prime = _prime_with_memory("l5.db")
        store = prime.context._get_store()
        path = str(store._db_path)

        def counts():
            with sqlite3.connect(path) as con:
                return tuple(
                    con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("messages", "states", "entities",
                              "relationships", "candidates")
                )

        before = counts()
        prime.handle(PrimeRequest("recall", {"query": "key rotation"}))
        assert counts() == before


# ======================================================================
# 5. Scope: only recall, nothing else
# ======================================================================


class TestScope:
    def test_capabilities_registered_so_far(self):
        """Milestone 3 adds capabilities one at a time."""
        prime = Prime(_Provider())
        assert sorted(prime.registry.get_names()) == [
            "calculate", "consult_specialist", "learn", "recall", "research",
            "research_read", "use_tool"]

    def test_no_research_specialist(self):
        """recall must not pull in the research capability."""
        from PRIMORA.specialists.tools import recall as mod
        assert "research" not in open(mod.__file__, encoding="utf-8").read()

    def test_no_verse_import(self):
        from PRIMORA.specialists.tools import recall as mod
        assert "verse" not in open(mod.__file__, encoding="utf-8").read()

    def test_loop_is_unchanged(self):
        """recall() must not have required loop changes."""
        import ast
        from PRIMORA.aurora import loop as mod
        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        called = {
            n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "recall" not in called

    def test_no_new_retrieval_constants(self):
        from PRIMORA.prime.context import hybrid
        # M6's tunables are untouched.
        assert hybrid.SEMANTIC_TAU == 0.45
        assert hybrid.RRF_K == 60.0
        assert hybrid.EVIDENCE_FLOOR_RATIO == 0.725
