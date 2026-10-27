"""Do not mistake user context for a user-stated cause.

These tests use model doubles and disposable stores; no local Qwen calls.
"""
import json

from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context.formation import apply_proposals
from PRIMORA.prime.context.store import ContextStore


class _Proposals:
    def __init__(self, rows):
        self.rows = rows

    def invoke(self, capability, payload):
        assert capability == "memory_formation"
        source = payload["messages"][-1]["content"]
        rows = []
        for item in self.rows:
            item = dict(item)
            item.setdefault("source_ids", [payload["source_message_id"]])
            item.setdefault("source_quote", source)
            rows.append(item)
        return PrimeResult.success({"content": json.dumps({"candidates": rows})})


def _run(tmp_path, source, rows):
    store = ContextStore(tmp_path / "reasons.db")
    source_id, _, _ = store.append("s", 0, "user", source)
    result = apply_proposals(
        store, "s", store.fetch_message(source_id), _Proposals(rows),
    )
    return store, result


def test_noncausal_source_cannot_back_standalone_rationale(tmp_path):
    statement = "I also enjoy ambient music while editing photos."
    store, result = _run(tmp_path, statement, [{
        "kind": "state",
        "concept": "rationale for ambient music",
        "value": statement,
        "scope": "global",
    }])
    assert not result.admitted_ids
    assert "proposal_rejected:unsupported_rationale_value" in result.diagnostics
    assert not store.fetch_states(scope="global", status="current")


def test_bad_model_reason_dropped_but_supported_preference_remains(tmp_path):
    statement = "I enjoy ambient music while editing photos."
    store, result = _run(tmp_path, statement, [{
        "kind": "preference",
        "concept": "ambient music",
        "value": "enjoy ambient music",
        "scope": "global",
        "rationale": statement,
    }])
    assert "rationale_dropped:no_explicit_user_reason" in result.diagnostics
    assert any(row.memory_type == "PREFERENCE"
               for row in store.fetch_states(scope="global", status="current"))
    assert not any(row.memory_type == "RATIONALE"
                   for row in store.fetch_states(scope="global", status="current"))


def test_explicit_because_reason_is_preserved(tmp_path):
    store, result = _run(
        tmp_path,
        "I enjoy rainfall while reading because it helps me concentrate.",
        [{
            "kind": "preference",
            "concept": "rainfall",
            "value": "enjoy rainfall",
            "scope": "global",
            "rationale": "it helps me concentrate",
        }],
    )
    assert result.admitted_ids
    states = store.fetch_states(scope="global", status="current")
    assert any(row.memory_type == "RATIONALE" and
               row.value == "it helps me concentrate" for row in states)
    assert any(row.memory_type == "PREFERENCE" for row in states)


def test_attribute_is_an_explicit_user_reason(tmp_path):
    store, result = _run(
        tmp_path,
        "I admire the Cedar Compass for its unusual shape and design.",
        [{
            "kind": "state",
            "concept": "rationale for the Cedar Compass",
            "value": "its unusual shape and design",
            "scope": "global",
        }],
    )
    assert result.admitted_ids
    assert any(row.value == "its unusual shape and design"
               for row in store.fetch_states(scope="global", status="current"))
