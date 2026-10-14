"""CPU tests for the masking layer. The 8B base model is never loaded.

The tokenizer is the real one, because the masking is a property of how the real
Qwen3 ChatML template renders - a mock would pass while the real thing failed.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))

from aurora.masking import (  # noqa: E402
    CHATML_MARKERS,
    IGNORE_INDEX,
    MaskingError,
    build_mask,
    verify_mask,
)

BASE_MODEL = os.path.join(_ROOT, "RESOURCES", "aurora", "v0.1", "model")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


def _messages(system="SYSTEM"):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "How many words in this note?"},
        {"role": "assistant", "content": "WRONG_REQUEST"},
        {"role": "tool", "content": '{"status": "failure", "output": null, "error": "no such capability"}'},
        {"role": "assistant", "content": "RIGHT_REQUEST"},
        {"role": "assistant", "content": "THE ANSWER"},
    ]


# -- train_on subset semantics -------------------------------------------



def _simple_messages(answer="THE ANSWER"):
    """One system turn, one user turn, one assistant turn.

    `build_mask` requires loss to reach the final assistant turn, so a
    single-turn record is the smallest thing that can be masked at all.
    """
    return [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "How many words in this note?"},
        {"role": "assistant", "content": answer},
    ]



def test_failed_request_is_masked_and_later_turns_trained(tokenizer):
    result = build_mask(tokenizer, _messages(), [1, 2], 1792)
    text = result.trained_text(tokenizer)
    assert "WRONG_REQUEST" not in text
    assert "RIGHT_REQUEST" in text
    assert "THE ANSWER" in text
    assert result.trained_turns == (1, 2)


def test_system_user_and_tool_are_masked(tokenizer):
    result = build_mask(tokenizer, _messages(), [1, 2], 1792)
    text = result.trained_text(tokenizer)
    assert "SYSTEM" not in text
    assert "How many words in this note?" not in text
    assert "no such capability" not in text


def test_no_chatml_marker_carries_loss(tokenizer):
    """No ChatML control token is supervised except a selected turn eos.

    Qwen3 ends an assistant turn with <|im_end|>, which is also the tokenizer's
    eos_token_id, and that one token is now deliberately supervised. Every other
    marker must still be masked.
    """
    result = build_mask(tokenizer, _messages(), [1, 2], 1792)
    text = result.trained_text(tokenizer)
    eos_text = tokenizer.decode([tokenizer.eos_token_id])
    for marker in CHATML_MARKERS:
        if marker == eos_text:
            continue
        assert marker not in text, f"{marker!r} carries loss"
    # And the eos is present, supervised, exactly twice for two turns.
    sup = [result.input_ids[i] for i, lab in enumerate(result.labels)
           if lab != IGNORE_INDEX]
    assert sup.count(tokenizer.eos_token_id) == 2


def test_single_turn_record_trains_only_that_turn(tokenizer):
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "the answer"},
    ]
    result = build_mask(tokenizer, msgs, [0], 1792)
    assert "the answer" in result.trained_text(tokenizer)
    verify_mask(result, msgs, tokenizer, "probe")


def test_mask_reaches_the_final_assistant_turn(tokenizer):
    """A record whose loss stops short of the last turn is rejected."""
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    with pytest.raises(MaskingError, match="loss must reach it"):
        build_mask(tokenizer, msgs, [0], 1792)


def test_empty_train_on_is_rejected(tokenizer):
    with pytest.raises(MaskingError, match="selected no assistant turn"):
        build_mask(tokenizer, _messages(), [], 1792)


def test_out_of_range_train_on_is_rejected(tokenizer):
    with pytest.raises(MaskingError, match="out of range"):
        build_mask(tokenizer, _messages(), [1, 9], 1792)


def test_every_label_position_is_supervised_or_ignored(tokenizer):
    result = build_mask(tokenizer, _messages(), [1, 2], 1792)
    assert len(result.input_ids) == len(result.labels)
    for label in result.labels:
        assert label == IGNORE_INDEX or isinstance(label, int)


# -- verify_mask catches a broken mask ----------------------------------


def test_verify_mask_rejects_loss_on_a_failed_request(tokenizer):
    msgs = _messages()
    good = build_mask(tokenizer, msgs, [1, 2], 1792)
    verify_mask(good, msgs, tokenizer, "probe")

    broken = build_mask(tokenizer, msgs, [0, 1, 2], 1792)
    with pytest.raises(MaskingError):
        verify_mask(broken, msgs, tokenizer, "probe")


def test_verify_mask_rejects_a_declared_turn_with_no_labels(tokenizer):
    """A turn train_on names must actually carry labels."""
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "ALPHA"},
        {"role": "assistant", "content": "BETA"},
    ]
    result = build_mask(tokenizer, msgs, [0, 1], 1792)
    result.labels = [-100] * len(result.labels)   # simulate a silent mask bug
    with pytest.raises(MaskingError, match="marked for loss"):
        verify_mask(result, msgs, tokenizer, "probe")


def test_verify_mask_rejects_a_mask_covering_an_unselected_turn(tokenizer):
    """The reverse: labels covering a turn train_on never named."""
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "ALPHA"},
        {"role": "assistant", "content": "BETA"},
    ]
    result = build_mask(tokenizer, msgs, [0, 1], 1792)
    result.trained_turns = (1,)
    with pytest.raises(MaskingError, match="NOT in train_on"):
        verify_mask(result, msgs, tokenizer, "probe")


# -- truncation ---------------------------------------------------------


def test_truncation_that_clips_supervision_is_rejected(tokenizer):
    """A max_length that removes every supervised turn must fail, not ship an
    unsupervised example."""
    with pytest.raises(MaskingError, match="no trainable token"):
        build_mask(tokenizer, _messages(), [1, 2], 20)


def test_truncation_reports_and_keeps_labels_aligned(tokenizer):
    """A long middle turn makes the record exceed max_length while the first
    assistant turn still fits, so truncation survives and is reported."""
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "EARLY_ANSWER"},
        {"role": "user", "content": "second question " * 40},
        {"role": "assistant", "content": "LATE_ANSWER"},
    ]
    full = build_mask(tokenizer, msgs, [0, 1], 4096)
    assert full.truncated is False

    cut = build_mask(tokenizer, msgs, [0, 1], 40)
    assert cut.truncated is True
    assert len(cut.input_ids) == 40 and len(cut.labels) == 40
    # The early turn survived the cut; the late one did not.
    assert cut.trained_tokens > 0
    assert "EARLY_ANSWER" in cut.trained_text(tokenizer)


# -- the real corpus ----------------------------------------------------


def test_real_shipped_records_mask_and_verify():
    from aurora.dataset_v2 import load_split
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    records = load_split("train")[:60]
    for example in records:
        result = build_mask(tok, example.messages, example.train_on, 1792)
        verify_mask(result, example.messages, tok, example.id)


def test_real_recovery_record_masks_its_failed_request():
    from aurora.dataset_v2 import load_split
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    example = next(e for e in load_split("train") if e.seed_id == "C15-R01")
    result = build_mask(tok, example.messages, example.train_on, 1792)
    text = result.trained_text(tok)
    assert "count_words" not in text, "the failed capability must not be trained"
    assert "use_tool" in text, "the corrected request must be trained"


# ---------------------------------------------------------------------------
# Assistant-turn EOS supervision.
#
# Qwen3 terminates an assistant turn with <|im_end|>, which is the tokenizer's
# eos_token_id. If the mask covers the content but not that terminator, the
# model learns to begin a reply and never to end one, and at inference it runs
# to max_new_tokens. The span is therefore extended by exactly one token.
#
# These tests assert on token IDs, not decoded text, because decoded special
# tokens are invisible in a diff and easy to pass by accident.
# ---------------------------------------------------------------------------


def _supervised_ids(result):
    return [i for i, l in enumerate(result.labels) if l != IGNORE_INDEX]


def _supervised_tokens(result):
    return [result.input_ids[i] for i in _supervised_ids(result)]


def test_selected_assistant_content_is_supervised(tokenizer):
    msgs = _simple_messages()
    result = build_mask(tokenizer, msgs, [0], 4096)
    text = result.trained_text(tokenizer)
    assert msgs[2]["content"] in text


def test_following_eos_token_is_supervised(tokenizer):
    """The regression test for the repetition bug."""
    eos = tokenizer.eos_token_id
    assert eos is not None
    msgs = _messages()
    result = build_mask(tokenizer, msgs, [1, 2], 1792)
    sup = [result.input_ids[i] for i, lab in enumerate(result.labels)
           if lab != IGNORE_INDEX]
    assert eos in sup, "the assistant turn's eos_token_id carries no loss"
    verify_mask(result, msgs, tokenizer)


def test_exactly_one_eos_is_added_per_selected_turn(tokenizer):
    """One extra token per selected turn - no more."""
    msgs = _simple_messages()
    eos = tokenizer.eos_token_id
    result = build_mask(tokenizer, msgs, [0], 4096)
    sup = _supervised_tokens(result)
    assert sup.count(eos) == 1
    assert sup[-1] == eos, "the eos must be the final supervised token"


def test_eos_is_the_token_immediately_after_the_content(tokenizer):
    msgs = _simple_messages()
    result = build_mask(tokenizer, msgs, [0], 4096)
    ids = _supervised_ids(result)
    assert result.input_ids[ids[-1]] == tokenizer.eos_token_id
    # ...and the position before it is content, not another control token.
    prev = result.input_ids[ids[-2]]
    assert prev != tokenizer.eos_token_id


def test_im_start_is_not_supervised(tokenizer):
    result = build_mask(tokenizer, _simple_messages(), [0], 4096)
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert im_start is not None and im_start > 0
    assert im_start not in _supervised_tokens(result)


def test_think_markers_are_not_supervised(tokenizer):
    result = build_mask(tokenizer, _simple_messages(), [0], 4096)
    sup = _supervised_tokens(result)
    for marker in ("<think>", "</think>"):
        tid = tokenizer.convert_tokens_to_ids(marker)
        assert tid is not None and tid > 0
        assert tid not in sup, f"{marker} carries loss"
    for marker in CHATML_MARKERS:
        if marker == tokenizer.decode([tokenizer.eos_token_id]):
            continue
        assert marker not in result.trained_text(tokenizer)


def test_unselected_assistant_eos_is_not_supervised(tokenizer):
    """Only a *selected* turn's terminator may be trained."""
    msgs = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "q one"},
        {"role": "assistant", "content": "UNSELECTED_ANSWER"},
        {"role": "user", "content": "q two"},
        {"role": "assistant", "content": "SELECTED_ANSWER"},
    ]
    result = build_mask(tokenizer, msgs, [1], 4096)
    sup = _supervised_ids(result)
    eos = tokenizer.eos_token_id
    supervised_eos = [i for i in sup if result.input_ids[i] == eos]
    # Exactly one supervised eos: the selected (final) turn's.
    assert len(supervised_eos) == 1
    # It must be the last supervised position, i.e. the final turn's.
    assert supervised_eos[0] == sup[-1]
    text = result.trained_text(tokenizer)
    assert "UNSELECTED_ANSWER" not in text
    verify_mask(result, msgs, tokenizer)


def test_multiple_selected_turns_each_get_their_own_eos(tokenizer):
    msgs = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "q one"},
        {"role": "assistant", "content": "FIRST_ANSWER"},
        {"role": "user", "content": "q two"},
        {"role": "assistant", "content": "SECOND_ANSWER"},
    ]
    result = build_mask(tokenizer, msgs, [0, 1], 4096)
    sup = _supervised_tokens(result)
    assert sup.count(tokenizer.eos_token_id) == 2
    assert "FIRST_ANSWER" in result.trained_text(tokenizer)
    assert "SECOND_ANSWER" in result.trained_text(tokenizer)
    verify_mask(result, msgs, tokenizer)


def test_real_record_supervises_eos(tokenizer):
    """End-to-end on shipped data, not a synthetic fixture."""
    from aurora.dataset_v2 import load_split
    ex = load_split("train")[0]
    result = build_mask(tokenizer, ex.messages, ex.train_on, 1792)
    assert tokenizer.eos_token_id in _supervised_tokens(result)
    verify_mask(result, ex.messages, tokenizer)


def test_verify_mask_rejects_a_supervised_eos_detached_from_content(tokenizer):
    """The invariant must actually bite, not just pass on good input."""
    from aurora.masking import MaskResult
    msgs = _simple_messages()
    good = build_mask(tokenizer, msgs, [0], 4096)
    # Forge a lone supervised eos at the very start of the sequence.
    bad_ids = list(good.input_ids)
    bad_labels = [IGNORE_INDEX] * len(bad_ids)
    bad_ids[0] = tokenizer.eos_token_id
    bad_labels[0] = bad_ids[0]
    forged = MaskResult(
        input_ids=bad_ids, labels=bad_labels,
        assistant_turns=good.assistant_turns,
        trained_turns=good.trained_turns,
        trained_tokens=1, truncated=False,
    )
    with pytest.raises(MaskingError):
        verify_mask(forged, msgs, tokenizer)
