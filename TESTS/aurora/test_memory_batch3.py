"""Batch 3: acknowledge a challenged assumption, don't silently deny it."""

from PRIMORA.aurora.conversation_guard import denies_assumption_after_correction
from PRIMORA.aurora.configuration import DEFAULT_SYSTEM_INSTRUCTION
from TESTS.aurora.test_conversation_guard_batch2 import _run


def test_denial_of_challenged_assumption_gets_retry(tmp_path):
    messages = [
        {'role': 'assistant', 'content': 'You love yellow because of warmth.'},
        {'role': 'user', 'content': 'Those were your assumptions. I never said that.'},
    ]
    result, _, brain, _ = _run(
        tmp_path, messages,
        ["You're right — I didn't assume that.",
         "You're right. I inferred warmth without evidence. You never said why."],
    )
    assert result.success
    assert 'without evidence' in result.content
    assert len(brain.calls) == 2
    assert 'accountability' in str(brain.calls[1])


def test_continued_denial_fails_closed(tmp_path):
    messages = [{'role': 'user', 'content': "You assumed that about me."}]
    result, _, _, loop = _run(
        tmp_path, messages,
        ["I didn't assume anything.", "I didn't assume anything."],
    )
    assert not result.success
    assert loop.last_trace.stop_reason == 'unacknowledged_assumption'


def test_one_cycle_never_emits_denial(tmp_path):
    messages = [{'role': 'user', 'content': "Those were your assumptions."}]
    result, _, _, loop = _run(
        tmp_path, messages, ["I didn't assume anything."], max_cycles=1)
    assert not result.success
    assert loop.last_trace.stop_reason == 'unacknowledged_assumption'


def test_normal_disagreement_does_not_trigger_guard():
    messages = [{'role': 'user', 'content': 'I assumed that was correct.'}]
    assert not denies_assumption_after_correction(
        messages, "I didn't assume that at all.")
    assert not denies_assumption_after_correction(
        [{'role': 'user', 'content': 'Those were your assumptions.'}],
        "You're right, I should not have guessed.")
    assert 'acknowledge the mistake instead of denying it' in DEFAULT_SYSTEM_INSTRUCTION
