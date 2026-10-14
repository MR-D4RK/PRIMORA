"""Compute the causal-LM loss only at supervised positions.

The problem
-----------
Qwen3's vocabulary is 151,936, so materialising logits for every position costs
1.449 MiB per token (0.30 bf16 logits + 0.60 for the float() copy + 0.60 for the
saved log_softmax). A 1,440-token record needs ~2,087 MiB of logits to compute a
loss over the ~46 positions whose labels are not -100. On an 8 GiB card with
5,836 MiB of 4-bit weights already resident, that does not fit.

Why selecting positions is exactly equivalent
--------------------------------------------
The installed transformers path (``Qwen3ForCausalLM.forward`` then
``ForCausalLMLoss``) does, in full:

    h[t]        = last_hidden_state[t]        # post final norm, so already normed
    logits[t]   = lm_head(h[t])
    shift[t]    = labels[t + 1]              # from pad(labels, (0,1), -100)[..., 1:]
    loss        = sum over t of CE(logits[t], shift[t])   where shift[t] != -100

``nn.functional.cross_entropy(..., ignore_index=-100, reduction="sum")`` discards
exactly the -100 targets, so

    sum over ALL t of CE(logits[t], shift[t])  ==  sum over t in P of CE(logits[t], shift[t])

where ``P = {t : shift[t] != -100}``. ``lm_head`` is a per-position linear map, so
computing it only on ``h[t] for t in P`` yields exactly those same terms and
nothing else. The reduction and the ``num_items_in_batch`` denominator are
untouched, so the value is identical.

Gradients are identical too, and this is the part that is easy to get wrong. The
backbone still runs over the whole sequence and is *not* sliced, so ``h[t]`` for
an unsupervised ``t`` still reaches a supervised ``t'`` through causal attention
and still receives gradient that way. Only the ``lm_head`` matmul is skipped. A
correct implementation must not truncate the hidden states before the decoder
layers.

The causal shift is the other easy mistake. Position ``t`` predicts token
``t + 1``, so selection happens on ``shift_labels``, never on ``labels``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

IGNORE_INDEX = -100


class SupervisedLossError(RuntimeError):
    """Raised instead of silently returning a loss that trains nothing."""


def shift_causal_labels(labels: torch.Tensor,
                        ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Reproduce transformers' shift exactly.

    ``ForCausalLMLoss`` does ``pad(labels, (0, 1), -100)[..., 1:]``, so index
    ``t`` of the result is ``labels[t + 1]``, and the final position is always
    -100. That trailing -100 is what guarantees the last token of a sequence
    never supervises anything.
    """
    padded = nn.functional.pad(labels, (0, 1), value=ignore_index)
    return padded[..., 1:].contiguous()


def unwrap_causal_lm(model: torch.nn.Module) -> nn.Module:
    """Return the ``Qwen3ForCausalLM`` (or Qwen2 equivalent) inside any wrapper.

    The wrapper must keep the LoRA layers injected, since gradients flow
    through them, so this walks down to the real CausalLM rather than
    unwrapping PEFT.

    The name test excludes PeftModelForCausalLM explicitly: it also ends in
    "ForCausalLM", and returning it makes ``.model`` resolve to a LoraModel
    whose output is a CausalLMOutputWithPast rather than hidden states. A
    duck-typed check cannot tell the two apart here, because LoraModel proxies
    attribute access to the inner model.

    ``.model`` is preferred over ``.base_model`` when walking down. LoraModel's
    ``base_model`` proxies through to the *backbone*, so following it skips the
    CausalLM entirely; its ``model`` is the CausalLM, as is PeftModel's.
    """
    seen = 0
    current = model
    while seen < 6:
        name = type(current).__name__
        if name.endswith("ForCausalLM") and not name.startswith("Peft"):
            return current
        nxt = getattr(current, "model", None)
        if not isinstance(nxt, torch.nn.Module) or nxt is current:
            nxt = getattr(current, "base_model", None)
        if not isinstance(nxt, torch.nn.Module) or nxt is current:
            break
        current = nxt
        seen += 1
    raise SupervisedLossError(
        f"could not find a CausalLM inside {type(model).__name__}")


def compute_supervised_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    shift_labels: Optional[torch.Tensor] = None,
    num_items_in_batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Causal-LM loss computed only at supervised positions.

    Same value as the stock path; the only difference is that ``lm_head`` is not
    evaluated at the ~97% of positions whose next token is ``-100``.
    """
    if labels is None and shift_labels is None:
        raise SupervisedLossError("either labels or shift_labels is required")

    causal_lm = unwrap_causal_lm(model)
    backbone = causal_lm.model

    if shift_labels is None:
        shift_labels = shift_causal_labels(labels)
    else:
        shift_labels = shift_labels.contiguous()

    if shift_labels.shape[-1] != input_ids.shape[-1]:
        raise SupervisedLossError(
            f"shift_labels length {shift_labels.shape[-1]} does not match "
            f"input length {input_ids.shape[-1]}")

    # The backbone runs over the full sequence, unmasked and untruncated. This
    # is what preserves gradients for hidden states whose own position is
    # unsupervised but which feed a supervised later position.
    outputs = backbone(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False)
    hidden = outputs.last_hidden_state

    supervised = shift_labels != IGNORE_INDEX
    count = int(supervised.sum())
    if count == 0:
        raise SupervisedLossError(
            "no supervised positions in this batch: every target is "
            f"{IGNORE_INDEX}. Refusing to return a loss that trains nothing.")

    selected_hidden = hidden[supervised]
    selected_targets = shift_labels[supervised]

    # The one matmul we actually need, at ~1/38th the positions.
    selected_logits = causal_lm.lm_head(selected_hidden).float()

    if num_items_in_batch is None:
        # Matches cross_entropy(reduction="mean") over the non-ignored terms.
        return nn.functional.cross_entropy(
            selected_logits, selected_targets, ignore_index=IGNORE_INDEX,
            reduction="mean")

    if not torch.is_tensor(num_items_in_batch):
        num_items_in_batch = torch.tensor(num_items_in_batch)
    return nn.functional.cross_entropy(
        selected_logits, selected_targets, ignore_index=IGNORE_INDEX,
        reduction="sum") / num_items_in_batch.to(selected_logits.device)


__all__ = [
    "IGNORE_INDEX",
    "SupervisedLossError",
    "compute_supervised_loss",
    "shift_causal_labels",
    "unwrap_causal_lm",
]
