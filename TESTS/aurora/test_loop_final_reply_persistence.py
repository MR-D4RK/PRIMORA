"""End-of-turn persistence: the final assistant reply is stored before
``run`` returns.

Covers the M1+ gap where ``loop.run`` broke out of the cycle on a plain
answer without ever reconciling it, so a process that exited right after
the turn lost the last reply. The tests drive a real ``ContextManager``
over a temp SQLite file through a Mind-shaped double whose ``build``
mimics production's reconcile-then-build contract, so sequence rules,
idempotency, and the immediate-restart reload are asserted against real
storage, not mocks.
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from types import SimpleNamespace

from PRIMORA.aurora.brain import BrainResult
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.prime.context.manager import ContextManager

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _db_path(name):
    return os.path.join(_TEST_TMPDIR.name, name)


def _tokens(messages):
    return sum(len(str(m.get("content", ""))) // 4 + 1 for m in messages)


class StubBrain:
    """Deterministic brain: pops scripted replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.generate_calls = 0

    def generate(self, messages, max_output_tokens=None):
        self.generate_calls += 1
        if len(self.replies) > 1:
            content = self.replies.pop(0)
        else:
            content = self.replies[0]
        return BrainResult.success(content, "stub-brain")

    def count_tokens(self, messages):
        return _tokens(messages)

    def get_model_name(self):
        return "stub-brain"


class RecordingMind:
    """Mind-shaped double over a real store.

    ``build`` mirrors production (reconcile the snapshot first, then
    return the context), and every ``reconcile`` call is recorded with
    its diagnostics so tests can assert exact payloads and the absence
    of ``seq_conflict`` chatter.
    """

    def __init__(self, db_path, build_delay_s=0.0):
        self.mgr = ContextManager(db_path=db_path)
        self.build_delay_s = build_delay_s
        self.builds = []
        self.reconciles = []
        self.all_diags = []

    def build(self, messages, session_id="default", token_counter=None,
              max_new_tokens=256, budget=None, **kwargs):
        if self.build_delay_s:
            time.sleep(self.build_delay_s)
        snapshot = [dict(m) for m in messages]
        self.builds.append((snapshot, session_id))
        diags = self.mgr.reconcile(snapshot, session_id)
        self.all_diags.extend(diags)
        return SimpleNamespace(
            messages=snapshot, allowed_output=64, diagnostics=list(diags))

    def reconcile(self, messages, session_id="default"):
        snapshot = [dict(m) for m in messages]
        diags = self.mgr.reconcile(snapshot, session_id)
        self.all_diags.extend(diags)
        self.reconciles.append((snapshot, session_id, list(diags)))
        return diags


def _loop(mind, brain, **kwargs):
    kwargs.setdefault("token_counter", _tokens)
    kwargs.setdefault("max_new_tokens", 64)
    kwargs.setdefault("request_extractor", lambda text: None)
    return CognitiveLoop(mind, brain, **kwargs)


def _no_seq_conflicts(mind):
    return not any(
        "seq_conflict" in d for d in mind.all_diags
    )


# 1. The answer is stored before run() returns; a restart sees it ----------
def test_final_reply_persisted_before_turn_completes():
    db = _db_path("final_reply.db")
    mind = RecordingMind(db)
    brain = StubBrain(["The final answer."])
    loop = _loop(mind, brain)

    result = loop.run([{"role": "user", "content": "question one"}], "s1")

    assert result.success
    assert result.content == "The final answer."
    assert loop.last_trace.stop_reason == "finished"
    # Exactly one end-of-turn persist, and it ends with the reply.
    assert len(mind.reconciles) == 1
    snapshot, session_id, diags = mind.reconciles[0]
    assert session_id == "s1"
    assert snapshot[-1] == {
        "role": "assistant", "content": "The final answer."}
    assert diags == []
    assert _no_seq_conflicts(mind)

    # IMMEDIATE RESTART: a brand-new manager over the same file must see
    # the reply as the last transcript row - no further turn happened.
    fresh = ContextManager(db_path=db)
    transcript = fresh.load_transcript("s1")
    assert [(m["role"], m["content"]) for m in transcript] == [
        ("user", "question one"),
        ("assistant", "The final answer."),
    ]
    assert [m["seq"] for m in transcript] == [0, 1]


# 2. Restart, then a second turn: appended without conflicts ---------------
def test_restart_then_second_turn_appends_cleanly():
    db = _db_path("second_turn.db")
    mind = RecordingMind(db)
    brain = StubBrain(["answer one", "answer two"])
    loop = _loop(mind, brain)

    loop.run([{"role": "user", "content": "q1"}], "s1")
    # Restart: fresh manager, transcript travels with its stored seqs.
    fresh = ContextManager(db_path=db)
    restored = [
        {"role": m["role"], "content": m["content"], "seq": m["seq"]}
        for m in fresh.load_transcript("s1")
    ]
    assert [m["content"] for m in restored] == ["q1", "answer one"]

    loop.run(restored + [{"role": "user", "content": "q2"}], "s1")

    transcript = ContextManager(db_path=db).load_transcript("s1")
    assert [(m["role"], m["content"]) for m in transcript] == [
        ("user", "q1"),
        ("assistant", "answer one"),
        ("user", "q2"),
        ("assistant", "answer two"),
    ]
    assert [m["seq"] for m in transcript] == [0, 1, 2, 3]
    assert _no_seq_conflicts(mind)


# 3. Re-running the identical turn adds no rows ----------------------------
def test_rerunning_identical_turn_adds_no_rows():
    db = _db_path("idempotent.db")
    payload = [{"role": "user", "content": "same question"}]

    mind1 = RecordingMind(db)
    _loop(mind1, StubBrain(["same answer"])).run(list(payload), "s1")
    rows_after_first = ContextManager(db_path=db)._get_store().count("s1")
    assert rows_after_first == 2

    # "Second process": new mind/brain over the same file, identical turn.
    mind2 = RecordingMind(db)
    _loop(mind2, StubBrain(["same answer"])).run(list(payload), "s1")

    store = ContextManager(db_path=db)._get_store()
    assert store.count("s1") == rows_after_first
    assert _no_seq_conflicts(mind2)


# 4. A failed generation persists nothing beyond the build's own row -------
def test_failed_generation_appends_nothing():
    db = _db_path("failed_generation.db")
    mind = RecordingMind(db)

    class FailingBrain(StubBrain):
        def generate(self, messages, max_output_tokens=None):
            self.generate_calls += 1
            return BrainResult.failure("generation boom", "stub-brain")

    loop = _loop(mind, FailingBrain(["unused"]))
    result = loop.run([{"role": "user", "content": "q"}], "s1")

    assert not result.success
    assert loop.last_trace.stop_reason == "generation_failed"
    # Build persisted the user row; no end-of-turn persist fired.
    assert mind.reconciles == []
    transcript = ContextManager(db_path=db).load_transcript("s1")
    assert [(m["role"], m["content"]) for m in transcript] == [("user", "q")]


# 5. Deadline after a correction cycle: appended rows are written ----------
def test_deadline_after_correction_persists_appended_rows():
    db = _db_path("deadline_heal.db")
    mind = RecordingMind(db, build_delay_s=0.2)
    request_block = (
        '<prime_request>\n'
        '{"action": "probe", "payload": {}}\n'
        "</prime_request>"
    )
    brain = StubBrain([request_block, "late answer"])
    executor = SimpleNamespace(
        begin_session=lambda sid: None,
        execute=lambda request: json.dumps(
            {"status": "success", "output": "ok"}),
    )
    loop = CognitiveLoop(
        mind,
        brain,
        executor=executor,
        token_counter=_tokens,
        max_new_tokens=64,
        request_extractor=(
            lambda text: SimpleNamespace(action="probe", payload={})
            if "<prime_request>" in (text or "") else None
        ),
        deadline_s=0.05,
    )

    loop.run([{"role": "user", "content": "q"}], "s1")

    # The loop stopped on the deadline at the cycle *after* the append, so
    # the assistant-request and tool rows were written end-of-turn.
    assert loop.last_trace.stop_reason == "deadline"
    assert loop.last_trace.count == 1  # only the request cycle generated
    assert mind.reconciles, "appended rows were never persisted"
    final_snapshot, _, final_diags = mind.reconciles[-1]
    assert final_snapshot[-2]["role"] == "assistant"
    assert "<prime_request>" in final_snapshot[-2]["content"]
    assert final_snapshot[-1]["role"] == "tool"
    assert final_diags == []

    # Reload stays clean: tool and protocol rows never reach the
    # transcript, so the restart view is exactly the user's turn.
    transcript = ContextManager(db_path=db).load_transcript("s1")
    assert [(m["role"], m["content"]) for m in transcript] == [("user", "q")]


if __name__ == "__main__":
    test_final_reply_persisted_before_turn_completes()
    test_restart_then_second_turn_appends_cleanly()
    test_rerunning_identical_turn_adds_no_rows()
    test_failed_generation_appends_nothing()
    test_deadline_after_correction_persists_appended_rows()
    print("All loop persistence tests passed!")
