"""M9 Batch 1: live runtime wiring over disposable Memory V2 stores."""

import json

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult, OpenAIBrain
from PRIMORA.main import build_prime
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.providers import Provider


class FormationProvider(Provider):
    def __init__(self, *, failure=False):
        self.calls = []
        self.failure = failure

    def invoke(self, capability, payload):
        self.calls.append((capability, payload))
        if self.failure:
            return PrimeResult.failure("test formation provider unavailable")
        if capability != "memory_formation":
            return PrimeResult.failure("unexpected capability")
        content = payload["messages"][-1]["content"]
        source_id = payload["source_message_id"]
        return PrimeResult.success({"content": json.dumps({"candidates": [{
            "kind": "preference",
            "concept": "preferred quiet libraries",
            "value": "quiet libraries",
            "claim_kind": "assertion",
            "actor": "user",
            "scope": "global",
            "source_ids": [source_id],
            "source_quote": content,
        }]})})


class CausalFormationProvider(Provider):
    """Typed formation fixture that omits rationale, as Qwen sometimes does."""

    def __init__(self):
        self.calls = []

    def invoke(self, capability, payload):
        self.calls.append((capability, payload))
        source = payload["messages"][-1]["content"]
        return PrimeResult.success({"content": json.dumps({"candidates": [{
            "kind": "preference",
            "concept": "preferred techno music when working",
            "value": "enjoys techno music when working",
            "claim_kind": "assertion",
            "actor": "user",
            "scope": "global",
            "source_ids": [payload["source_message_id"]],
            "source_quote": source,
        }]})})


class ScriptedBrain(Brain):
    def __init__(self, answer="I heard you."):
        self.answer = answer
        self.calls = []

    def generate(self, messages, max_output_tokens=None):
        self.calls.append(list(messages))
        return BrainResult.success(self.answer, "test-brain")

    def get_model_name(self):
        return "test-brain"

    def count_tokens(self, messages):
        return sum(max(1, len(str(item.get("content", ""))) // 4) for item in messages)


class CausalRecallBrain(OpenAIBrain):
    """OpenAI-shaped fixture for an end-to-end follow-up and abstention."""

    def __init__(self):
        self._config = type("Config", (), {})()
        self._model = "test-qwen"
        self.calls = []
        self.next_outputs = []

    def queue_recall(self, query, subject):
        self.next_outputs.append(
            "<prime_request>\n" + json.dumps({
                "action": "recall",
                "payload": {
                    "query": query, "sources": ["preferences"],
                    "subject": subject, "reasons": True,
                }, "context": None, "metadata": None,
            }) + "\n</prime_request>"
        )

    def count_tokens(self, messages):
        return sum(max(1, len(str(item.get("content", ""))) // 4)
                   for item in messages)

    def generate(self, messages, max_output_tokens=None, on_delta=None,
                 cancelled=None):
        self.calls.append(messages)
        tool_messages = [item for item in messages if item.get("role") == "tool"]
        if tool_messages:
            evidence = str(tool_messages[-1].get("content", ""))
            if "focused and calm" in evidence:
                return BrainResult.success(
                    "Because it helps you stay focused and calm.",
                    self.get_model_name(),
                )
            return BrainResult.success(
                "I couldn't verify a user-stated reason.", self.get_model_name())
        if self.next_outputs:
            return BrainResult.success(self.next_outputs.pop(0), self.get_model_name())
        return BrainResult.success("I heard you.", self.get_model_name())


def _aurora(db_path, provider, *, enabled=True, brain=None):
    manager = ContextManager(
        db_path=db_path,
        enable_memory_v2=enabled,
        formation_provider=None,
        hybrid_retrieval=False,
    )
    prime = Prime(provider, context_manager=manager)
    return Aurora(prime, brain=brain or ScriptedBrain())


def test_aurora_respond_wires_one_provider_and_forms_once(tmp_path):
    provider = FormationProvider()
    aurora = _aurora(tmp_path / "runtime.db", provider)
    messages = [{"role": "user", "content": "I like quiet libraries"}]

    first = aurora.respond(messages, session_id="alpha")
    second = aurora.respond(messages, session_id="alpha")

    assert first.success and second.success
    assert [capability for capability, _payload in provider.calls] == [
        "memory_formation"
    ]
    context = aurora._prime.context
    states = context._get_store().fetch_states(scope="global", status="current")
    assert any(state.memory_type == "PREFERENCE"
               and "quiet libraries" in state.value for state in states)
    assert context.formation_status("alpha")[0]["status"] == "complete"


def test_disabled_memory_v2_preserves_aurora_and_makes_no_formation_call(tmp_path):
    provider = FormationProvider()
    aurora = _aurora(tmp_path / "disabled.db", provider, enabled=False)

    result = aurora.respond(
        [{"role": "user", "content": "I like quiet libraries"}],
        session_id="alpha",
    )

    assert result.success
    assert provider.calls == []
    assert aurora._prime.context.memory_v2_enabled is False


def test_formation_failure_does_not_break_conversation(tmp_path):
    provider = FormationProvider(failure=True)
    aurora = _aurora(tmp_path / "failure.db", provider)

    result = aurora.respond(
        [{"role": "user", "content": "I like quiet libraries"}],
        session_id="alpha",
    )

    assert result.success
    assert len(provider.calls) == 1
    status = aurora._prime.context.formation_status("alpha")
    assert status and all(item["status"] == "complete" for item in status)
    assert any("provider_failure" in diagnostic
               for item in status for diagnostic in item["diagnostics"])


def test_memory_v2_survives_fresh_runtime_and_cross_session(tmp_path):
    db_path = tmp_path / "restart.db"
    first_provider = FormationProvider()
    first = _aurora(db_path, first_provider)
    assert first.respond(
        [{"role": "user", "content": "I like quiet libraries"}],
        session_id="alpha",
    ).success

    second_provider = FormationProvider()
    second = _aurora(db_path, second_provider)
    assert second.respond(
        [{"role": "user", "content": "Hello again"}],
        session_id="beta",
    ).success
    evidence = second._prime.context.memory_query_evidence(
        {"sources": ["preferences"],
         "query": "What are the last things I told you I liked?",
         "result_count": 3, "order": "recent", "temporal_scope": "current"},
        session_id="beta",
    )
    assert "quiet libraries" in evidence
    # The new beta user turn is formed once; alpha's already-persisted
    # preference is never replayed by the fresh runtime.
    assert len(second_provider.calls) == 1
    assert second_provider.calls[0][1]["messages"][-1]["content"] == "Hello again"


def test_runtime_env_selects_isolated_db_and_shared_provider(monkeypatch, tmp_path):
    db_path = tmp_path / "acceptance.db"
    provider = FormationProvider()
    monkeypatch.setenv("PRIMORA_MEMORY_V2", "1")
    monkeypatch.setenv("PRIMORA_MEMORY_V2_DB", str(db_path))

    prime = build_prime(provider)

    assert prime.context.memory_v2_enabled is True
    assert str(prime.context._db_path) == str(db_path)
    assert prime.context._formation_provider is provider
    assert prime.mind.mechanisms is prime.context


def test_live_causal_sentence_forms_verified_reason_and_why_followup_is_fast(
    tmp_path,
):
    provider = CausalFormationProvider()
    brain = CausalRecallBrain()
    aurora = _aurora(tmp_path / "causal.db", provider, brain=brain)
    source = "I enjoy techno music when working. It helps me stay focused and calm."

    first = aurora.respond([{"role": "user", "content": source}], session_id="one")
    assert first.success
    formation_timings = dict(aurora.loop.last_trace.timings_ms)
    assert formation_timings["formation_ms"] >= 0
    assert formation_timings["formation_provider_ms"] >= 0
    before = len(provider.calls)
    brain_before = len(brain.calls)
    brain.queue_recall("Why did I listen to techno music when working?",
                       "techno music when working")
    followup = aurora.respond([
        {"role": "user", "content": source},
        {"role": "assistant", "content": first.content},
        {"role": "user", "content": "Why?"},
    ], session_id="two")

    assert followup.success
    assert "focused and calm" in followup.content
    # The replayed source is not re-formed; only the new follow-up is sent
    # through Prime's typed formation path.
    assert len(provider.calls) == before + 1
    states = aurora.mind.mechanisms._get_store().fetch_states(
        scope="global", status="current")
    rationale = [row for row in states if row.memory_type == "RATIONALE"]
    assert rationale and "focused and calm" in rationale[0].value
    assert rationale[0].source_ids
    timings = aurora.loop.last_trace.timings_ms
    assert not any(
        "interpreting the LATEST USER MESSAGE" in str(call[0].get("content", ""))
        for call in brain.calls[brain_before:]
    )
    assert timings["planning_ms"] < 5.0
    assert "total_ms" in timings and timings["response_generation_ms"] >= 0

    brain.queue_recall("Why do I listen to techno?", "techno")
    explicit = aurora.respond([
        {"role": "user", "content": source},
        {"role": "assistant", "content": first.content},
        {"role": "user", "content": "Why do I listen to techno?"},
    ], session_id="two")
    assert explicit.success and "focused and calm" in explicit.content


def test_live_causal_reason_abstains_without_explicit_reason(tmp_path):
    class NoReasonProvider(CausalFormationProvider):
        def invoke(self, capability, payload):
            self.calls.append((capability, payload))
            source = payload["messages"][-1]["content"]
            return PrimeResult.success({"content": json.dumps({"candidates": [{
                "kind": "preference", "concept": "preferred watercolor",
                "value": "enjoys watercolor", "claim_kind": "assertion",
                "actor": "user", "scope": "global",
                "source_ids": [payload["source_message_id"]],
                "source_quote": source,
            }]})})

    provider = NoReasonProvider()
    brain = CausalRecallBrain()
    aurora = _aurora(tmp_path / "no-reason.db", provider, brain=brain)
    source = "I enjoy watercolor painting on weekends."
    first = aurora.respond([{"role": "user", "content": source}], session_id="one")
    assert first.success
    brain.queue_recall("Why do I listen to watercolor?", "watercolor")
    result = aurora.respond([
        {"role": "user", "content": source},
        {"role": "assistant", "content": "Watercolor."},
        {"role": "user", "content": "Why?"},
    ], session_id="two")
    assert result.success
    assert "couldn't verify a user-stated reason" in result.content
    assert not any(row.memory_type == "RATIONALE"
                   for row in aurora.mind.mechanisms._get_store().fetch_states(
                       scope="global", status="current"))
