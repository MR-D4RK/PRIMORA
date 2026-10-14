"""Loss masking for Aurora Brain v0.1 QLoRA training.

Only the assistant turns a record names in ``train_on`` may carry loss. System,
user and tool turns are context, and so is a deliberately failed capability
request - that turn has to stay visible in the context or the model never sees
what it did wrong, but training on it would reward reproducing it.

Why the mask is computed this way
---------------------------------
Two obvious approaches are both unusable here, and both were verified rather
than assumed:

*   ``apply_chat_template(..., return_assistant_tokens_mask=True)`` returns all
    zeros with a warning, because the shipped Qwen3 template has no
    ``{% generation %}`` block. Training on that result would mean training on
    nothing.
*   Rendering message *prefixes* and diffing token counts is unsound, because
    the template's assistant branch keys off ``loop.last``. The same assistant
    turn renders with the ``<think>`` block when it is last and without it
    otherwise, so 3 of 126 measured prefixes were not exact prefixes. Scanning
    the rendered string for ``<|im_start|>assistant`` headers also fails on
    334 of 967 records, because the template merges consecutive ``tool`` turns
    into a single ``<|im_start|>user`` block.

So the span is found by substitution: re-render the *whole* conversation with
one assistant turn's content replaced by a sentinel, then take the region
between the longest common prefix and the longest common suffix. Substituting
rather than appending matters - appending re-tokenises the preceding words and
collapses the span to zero width. Replacing leaves the surrounding ChatML and
the ``loop.last`` branch untouched, so the span is exact.

The tokenizer and template are the runtime's own. No template is defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

IGNORE_INDEX = -100

# A private-use codepoint run. It never appears in Aurora's data, and being a
# single unusual token keeps it from merging with the text around it.
SENTINEL = "\u2588AURORA_MASK\u2588"

# ChatML control tokens. None of these may ever carry loss: they delimit turns
# rather than being part of a turn's content.
CHATML_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|im_start|>\n",
    "<think>",
    "</think>",
)


class MaskingError(RuntimeError):
    """Raised when a mask cannot be produced or violates an invariant."""


@dataclass
class MaskResult:
    input_ids: list[int]
    labels: list[int]
    assistant_turns: int
    trained_turns: tuple[int, ...]
    trained_tokens: int
    truncated: bool = False

    def trained_text(self, tokenizer) -> str:
        ids = [i for i, l in zip(self.input_ids, self.labels) if l != IGNORE_INDEX]
        return tokenizer.decode(ids)


def render(tokenizer, messages: Sequence[dict[str, str]]) -> str:
    """Render exactly as the runtime does, minus the generation prompt.

    ``add_generation_prompt=True`` is used for inference; for a training record
    the final assistant turn supplies its own continuation, so the record is
    rendered as a complete conversation.
    """
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def encode(tokenizer, text: str) -> list[int]:
    # add_special_tokens=False because the ChatML template already emits the
    # control tokens. brain.py does the same at inference.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _span_by_substitution(
    tokenizer,
    messages: Sequence[dict[str, str]],
    base_ids: list[int],
    index: int,
) -> tuple[int, int]:
    """Token span of assistant turn ``index`` inside the full conversation."""
    modified = [dict(m) for m in messages]
    modified[index]["content"] = SENTINEL
    probe_ids = encode(tokenizer, render(tokenizer, modified))

    head = 0
    limit = min(len(base_ids), len(probe_ids))
    while head < limit and base_ids[head] == probe_ids[head]:
        head += 1

    tail = 0
    while (tail < limit - head
           and base_ids[len(base_ids) - 1 - tail] == probe_ids[len(probe_ids) - 1 - tail]):
        tail += 1

    start, end = head, len(base_ids) - tail
    if end <= start:
        raise MaskingError(
            f"assistant turn {index} produced an empty span; the substitution "
            f"probe did not differ from the base render")
    return start, end


def build_mask(
    tokenizer,
    messages: Sequence[dict[str, str]],
    train_on: Sequence[int],
    max_length: int,
) -> MaskResult:
    """Compute input_ids and labels for one record.

    Raises MaskingError rather than returning a partially masked record, so a
    bad mask cannot reach the optimiser.
    """
    if not messages:
        raise MaskingError("record has no messages")

    base_ids = encode(tokenizer, render(tokenizer, messages))
    if not base_ids:
        raise MaskingError("record rendered to zero tokens")

    assistant_positions = [i for i, m in enumerate(messages)
                           if m["role"] == "assistant"]
    wanted = tuple(sorted(set(int(i) for i in train_on)))

    if not wanted:
        raise MaskingError("train_on selected no assistant turn")
    for i in wanted:
        if i < 0 or i >= len(assistant_positions):
            raise MaskingError(
                f"train_on index {i} is out of range for "
                f"{len(assistant_positions)} assistant turn(s)")
    if wanted[-1] != len(assistant_positions) - 1:
        raise MaskingError(
            f"train_on ends at assistant turn {wanted[-1]} but the record's "
            f"final turn is {len(assistant_positions) - 1}; loss must reach it")

    truncated = len(base_ids) > max_length

    labels = [IGNORE_INDEX] * len(base_ids)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for assistant_index in wanted:
        start, end = _span_by_substitution(
            tokenizer, messages, base_ids, assistant_positions[assistant_index])
        # Extend over the <|im_end|> that closes this turn.
        #
        # Qwen3 uses <|im_end|> (tokenizer.eos_token_id) to end an assistant
        # turn. Training on the content but not on the token that terminates it
        # teaches the model to start a reply and never finish one, so at
        # inference it runs to max_new_tokens. One extra token per selected turn
        # is the entire fix.
        #
        # Only ever the EOS immediately after the content span, and only when it
        # really is the EOS: <|im_start|>, the <think> pair and unrelated ChatML
        # control tokens stay masked.
        if (eos_id is not None
                and end < len(base_ids)
                and base_ids[end] == eos_id):
            end += 1
        labels[start:end] = base_ids[start:end]

    # Truncation is applied after masking, not before. Clipping first would
    # compare a truncated sequence against a probe rendered from the full
    # conversation, and the span would look empty for every turn past the cut.
    if truncated:
        base_ids = base_ids[:max_length]
        labels = labels[:max_length]

    trained = sum(1 for l in labels if l != IGNORE_INDEX)
    if trained == 0:
        raise MaskingError(
            "mask produced no trainable token; every selected assistant turn "
            "was truncated away by max_length "
            f"{max_length} (record renders to "
            f"{len(encode(tokenizer, render(tokenizer, messages)))} tokens)")

    return MaskResult(
        input_ids=list(base_ids),
        labels=labels,
        assistant_turns=len(assistant_positions),
        trained_turns=wanted,
        trained_tokens=trained,
        truncated=truncated,
    )


# --------------------------------------------------------------------------
# Invariants. These run before training on every record, not as a test.
# --------------------------------------------------------------------------


def verify_mask(
    result: MaskResult,
    messages: Sequence[dict[str, str]],
    tokenizer,
    record_id: str = "<record>",
) -> None:
    """Fail loudly if the mask is not exactly the selected assistant turns."""
    if len(result.input_ids) != len(result.labels):
        raise MaskingError(
            f"{record_id}: input_ids {len(result.input_ids)} and labels "
            f"{len(result.labels)} differ in length")

    trained_text = result.trained_text(tokenizer)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_text = tokenizer.decode([eos_id]) if eos_id is not None else None

    # 1. No ChatML control token may carry loss, with one deliberate exception:
    #    the <|im_end|> that terminates a selected assistant turn. Qwen3 uses
    #    it as eos_token_id, so the model must learn to emit it or it never
    #    finishes a reply. Everything else - <|im_start|>, the newline, the
    #    <think> pair - stays masked.
    for marker in CHATML_MARKERS:
        if marker == eos_text:
            continue
        if marker in trained_text:
            raise MaskingError(
                f"{record_id}: ChatML marker {marker!r} carries loss")

    # A supervised EOS must sit inside a trained span, immediately after
    # supervised assistant content. That is what stops an unselected turn's
    # terminator from being trained.
    for pos, (tid, lab) in enumerate(zip(result.input_ids, result.labels)):
        if lab == IGNORE_INDEX or eos_id is None or tid != eos_id:
            continue
        prev_supervised = pos > 0 and result.labels[pos - 1] != IGNORE_INDEX
        if not prev_supervised:
            raise MaskingError(
                f"{record_id}: a supervised {eos_text!r} does not directly "
                f"follow supervised assistant content")

    # 2. Every selected turn must be reproduced verbatim.
    assistant_positions = [i for i, m in enumerate(messages)
                           if m["role"] == "assistant"]
    for i in result.trained_turns:
        content = messages[assistant_positions[i]]["content"]
        if content not in trained_text:
            raise MaskingError(
                f"{record_id}: assistant turn {i} is marked for loss but its "
                f"content is not present in the trained text")

    # 3. No unselected assistant turn may carry loss.
    for i, pos in enumerate(assistant_positions):
        if i in result.trained_turns:
            continue
        content = messages[pos]["content"]
        if content and content in trained_text:
            raise MaskingError(
                f"{record_id}: assistant turn {i} is NOT in train_on but its "
                f"content appears in the trained text")

    # 4. A failed capability request must stay masked. The turn before a
    #    failure tool turn is context by construction, so catch it if the mask
    #    ever starts covering it.
    assistant_seen = 0
    for m in messages:
        if m["role"] == "assistant":
            assistant_seen += 1
        elif m["role"] == "tool":
            status = _tool_status(m.get("content", ""))
            if status == "failure":
                failed_turn = assistant_seen - 1
                if failed_turn >= 0 and failed_turn in result.trained_turns:
                    raise MaskingError(
                        f"{record_id}: the request before a failed tool turn "
                        f"(assistant turn {failed_turn}) carries loss")


def _tool_status(content: str) -> Optional[str]:
    import json
    try:
        parsed = json.loads(content)
    except Exception:
        return None
    if isinstance(parsed, dict):
        status = parsed.get("status")
        if isinstance(status, str):
            return status
    return None


__all__ = [
    "IGNORE_INDEX",
    "SENTINEL",
    "CHATML_MARKERS",
    "MaskingError",
    "MaskResult",
    "build_mask",
    "encode",
    "render",
    "verify_mask",
]
