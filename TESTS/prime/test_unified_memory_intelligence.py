"""Integration contracts: Aurora interprets, Prime validates, evidence proves.

No real model is available in CI, so scripted outputs stand in for a semantic
planner without teaching routing any subject- or phrase-specific patterns.
"""
import json

from PRIMORA.aurora.brain import BrainResult
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.aurora.memory_coordinator import validate_plan, interpret, MemoryPlan
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.mind import Mind


def plan(**kw):
    return dict(intent="memory", sources=["preferences"], subject=None,
                recent_days=None, first_mention=False, reasons=False,
                clarify=False, **kw)


class SemanticBrain:
    def __init__(self, plan_json, output=None):
        self.plan_json, self.output = plan_json, output
        self.calls = []
    def get_model_name(self): return "semantic-test"
    def generate(self, messages, max_output_tokens=None, on_delta=None):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return BrainResult.success(json.dumps(self.plan_json), "semantic-test")
        if self.output is not None:
            return BrainResult.success(self.output, "semantic-test")
        return BrainResult.success("Grounded answer: " + messages[-1]["content"][-400:], "semantic-test")
    def count_tokens(self, messages): return 30


def setup_memory(tmp_path):
    cm = ContextManager(db_path=tmp_path / "mem.db", hybrid_retrieval=False)
    store = cm._get_store()
    a, *_ = store.append("earlier", 0, "user", "My favorite instrument is the glass harp")
    b, *_ = store.append("other", 0, "user", "I love amber ceramics")
    c, *_ = store.append("earlier", 1, "assistant", "Yellow means creativity to you")
    store.create_state("earlier", "global", "favorite instrument", "glass harp", [a])
    store.create_state("other", "global", "likes for amber ceramics", "loves amber ceramics", [b])
    store.create_state("earlier", "global", "likes for yellow", "loves yellow because creativity", [c],
                       authority="assistant", status="unresolved")
    return cm


def test_plan_validation_rejects_invalid_schema_and_injection():
    original = plan()
    assert isinstance(validate_plan(original), MemoryPlan)
    assert validate_plan({**original, "sources": ["delete_database"]}) is None
    assert validate_plan({**original, "recent_days": 100000}) is None
    assert validate_plan({**original, "subject": "x"*101}) is None
    assert validate_plan({**original, "first_mention": "yes"}) is None
    assert validate_plan({"intent": "other"}) == "other"


def test_semantic_paraphrases_and_typos_route_through_model(tmp_path):
    cm = setup_memory(tmp_path)
    for question in ("what are my prefernces", "What have I been into lately?", "Remind me about my tastes"):
        brain = SemanticBrain(plan())
        loop = CognitiveLoop(Mind(cm), brain, enable_memory_interpreter=True)
        answer = loop.run([{"role": "user", "content": question}], session_id="fresh")
        assert answer.success and len(brain.calls) == 2
        presented = brain.calls[1][-1]["content"]
        assert "glass harp" in presented and "amber ceramics" in presented
        assert "creativity" not in presented


def test_date_and_named_subject_are_source_backed(tmp_path):
    cm = setup_memory(tmp_path)
    p = {**plan(), "subject": "glass harp", "first_mention": True}
    evidence = cm.memory_query_evidence(p)
    assert "glass harp" in evidence and "UTC" in evidence
    assert "amber ceramics" not in evidence
    assert "creativity" not in evidence


def test_recent_window_filters_evidence(tmp_path):
    cm = setup_memory(tmp_path)
    p = {**plan(), "recent_days": 1}
    # Default stored timestamps are current; simulate an archived source using
    # a finite past timestamp in its original immutable message record.
    old = cm._get_store()
    import sqlite3, time
    with sqlite3.connect(str(cm._db_path)) as con:
        con.execute("UPDATE messages SET timestamp=? WHERE content=?",
                    (time.time() - 10*86400, "I love amber ceramics"))
    evidence = cm.memory_query_evidence(p)
    assert "glass harp" in evidence and "amber ceramics" not in evidence


def test_episodes_and_possible_followup_are_source_backed(tmp_path):
    cm = setup_memory(tmp_path)
    store = cm._get_store()
    a, *_ = store.append("watch", 0, "user", "I recently watched the Atlas documentary")
    b, *_ = store.append("watch", 1, "user", "I loved it")
    p = {**plan(), "sources": ["experiences"], "subject": "Atlas documentary"}
    evidence = cm.memory_query_evidence(p)
    assert f"message {a}" in evidence
    assert f"message {b}" in evidence
    assert f"possible_followup_to_{a}" in evidence
    assert "Yellow means creativity" not in evidence


def test_prior_assistant_is_context_not_evidence(tmp_path):
    cm = setup_memory(tmp_path)
    brain = SemanticBrain({**plan(), "subject": "glass harp"})
    loop = CognitiveLoop(Mind(cm), brain, enable_memory_interpreter=True)
    loop.run([{"role": "user", "content": "Which instrument did I mention?"},
              {"role": "assistant", "content": "You mentioned glass harp"},
              {"role": "user", "content": "When did I tell you about it?"}],
             session_id="fresh")
    planning_payload = brain.calls[0][-1]["content"]
    assert "You mentioned glass harp" in planning_payload
    evidence = brain.calls[1][-1]["content"]
    assert "My favorite instrument" in evidence
    assert "You mentioned glass harp" not in evidence


def test_other_queries_remain_in_cognitive_loop(tmp_path):
    brain = SemanticBrain({"intent": "other"}, "Ordinary response")
    loop = CognitiveLoop(Mind(setup_memory(tmp_path)), brain, enable_memory_interpreter=True)
    result = loop.run([{"role": "user", "content": "Explain addition"}], session_id="fresh")
    assert result.success and result.content == "Ordinary response"
    assert len(brain.calls) == 2


def test_unknown_evidence_never_causes_unsupported_model_summary(tmp_path):
    brain = SemanticBrain({**plan(), "subject": "nonexistent evidence"})
    loop = CognitiveLoop(Mind(setup_memory(tmp_path)), brain, enable_memory_interpreter=True)
    response = loop.run([{"role": "user", "content": "What did I say about nonexistent evidence?"}],
                        session_id="fresh")
    assert response.success and "couldn't find" in response.content
    assert len(brain.calls) == 1


def test_ambiguous_reference_asks_for_clarification_without_retrieval(tmp_path):
    brain = SemanticBrain({**plan(), "clarify": True})
    loop = CognitiveLoop(Mind(setup_memory(tmp_path)), brain, enable_memory_interpreter=True)
    response = loop.run([{"role": "user", "content": "When did it happen?"}], session_id="fresh")
    assert "Which specific" in response.content and len(brain.calls) == 1


def test_unsupported_generated_date_falls_back_to_attributable_evidence(tmp_path):
    brain = SemanticBrain(plan(), "You told me on 2099-01-01.")
    loop = CognitiveLoop(Mind(setup_memory(tmp_path)), brain,
                         enable_memory_interpreter=True)
    result = loop.run([{"role": "user", "content": "When did I tell you about my interests?"}],
                      session_id="fresh")
    assert result.success
    assert "2099-01-01" not in result.content
    assert "glass harp" in result.content
