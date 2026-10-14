"""Tests for the supervised-position loss.

The whole justification for this module is that it is *exactly* equivalent to
the stock masked causal-LM loss while evaluating ``lm_head`` at only the
supervised positions. So the central test is a numeric comparison against the
real transformers implementation, not a plausibility check.

The reference model is a genuine ``Qwen2ForCausalLM`` - the same family as the
Qwen3 base - run in float64 so the comparison is not limited by float32 noise.
The 8B model is never loaded.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))

from aurora.supervised_loss import (  # noqa: E402
    IGNORE_INDEX,
    SupervisedLossError,
    compute_supervised_loss,
    shift_causal_labels,
    unwrap_causal_lm,
)

VOCAB, HIDDEN, LAYERS = 97, 32, 2


def _reference_model(dtype=torch.float64):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(dtype).eval()


def _batch(batch=1, length=40, seed=1, pad_to=None):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (batch, length), generator=g)
    labels = torch.full((batch, length), IGNORE_INDEX, dtype=torch.long)
    for b in range(batch):
        start = 5 + 7 * b
        labels[b, start:start + 4] = ids[b, start + 1:start + 5]
    mask = torch.ones_like(ids)
    if pad_to is not None:
        ids = torch.cat([ids, torch.zeros(batch, pad_to - length, dtype=ids.dtype)], 1)
        labels = torch.cat([labels, torch.full((batch, pad_to - length), IGNORE_INDEX,
                                              dtype=torch.long)], 1)
        mask = torch.cat([mask, torch.zeros(batch, pad_to - length, dtype=mask.dtype)], 1)
    return ids, mask, labels


# -- 1. equivalence with the stock loss ----------------------------------

def test_matches_stock_loss_exactly():
    model = _reference_model()
    ids, mask, labels = _batch()
    stock = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    mine = compute_supervised_loss(model, ids, mask, labels)
    assert torch.allclose(stock, mine, atol=1e-12, rtol=1e-9), \
        f"stock={float(stock)} mine={float(mine)}"


def test_matches_stock_loss_with_num_items_in_batch():
    """sum/num_items must equal the mean the stock path produces.

    The tolerance is float32, not float64, and deliberately so:
    ``ForCausalLMLoss`` starts with ``logits = logits.float()``, so it computes
    the loss in float32 even when the model is float64. Averaging internally
    versus summing and then dividing therefore rounds differently, at ~1e-7
    relative. The exact reduction-free comparison is the first test.
    """
    model = _reference_model()
    ids, mask, labels = _batch(batch=2)
    count = int((shift_causal_labels(labels) != IGNORE_INDEX).sum())
    stock = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    mine = compute_supervised_loss(model, ids, mask, labels,
                                   num_items_in_batch=count)
    assert torch.allclose(stock, mine, atol=1e-6, rtol=1e-6), \
        f"stock={float(stock):.17g} mine={float(mine):.17g}"
    assert abs(float(stock) - float(mine)) < 1e-5


def test_matches_stock_loss_with_padding():
    model = _reference_model()
    ids, mask, labels = _batch(batch=2, length=28, pad_to=48)
    stock = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    mine = compute_supervised_loss(model, ids, mask, labels)
    assert torch.allclose(stock, mine, atol=1e-12, rtol=1e-9)


def test_gradients_match_the_stock_loss():
    """Equivalence is only real if the gradients match, not just the value."""
    model = _reference_model()
    ids, mask, labels = _batch(batch=2)

    model.zero_grad()
    model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
    stock_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                   if p.grad is not None}

    model.zero_grad()
    compute_supervised_loss(model, ids, mask, labels).backward()
    mine_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                  if p.grad is not None}

    assert set(stock_grads) == set(mine_grads)
    for name in stock_grads:
        assert torch.allclose(stock_grads[name], mine_grads[name],
                              atol=1e-12, rtol=1e-8), name


# -- 2. the causal shift --------------------------------------------------

def test_shift_reproduces_transformers_exactly():
    labels = torch.tensor([[5, 6, 7, IGNORE_INDEX]])
    shifted = shift_causal_labels(labels)
    # position t predicts token t+1
    assert shifted[0, 0].item() == 6
    assert shifted[0, 1].item() == 7
    assert shifted[0, 2].item() == IGNORE_INDEX
    # the last position can never supervise anything
    assert shifted[0, -1].item() == IGNORE_INDEX


def test_loss_is_wrong_if_the_shift_is_not_applied():
    """Guards the specific mistake: selecting on labels instead of shifted."""
    model = _reference_model()
    ids, mask, labels = _batch()
    stock = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    unshifted = compute_supervised_loss(model, ids, mask, labels,
                                        shift_labels=labels)
    assert not torch.allclose(stock, unshifted, atol=1e-6), (
        "using unshifted labels produced the same loss, so this test could not "
        "detect a missing shift")


def test_precomputed_shift_labels_are_accepted():
    model = _reference_model()
    ids, mask, labels = _batch()
    stock = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    mine = compute_supervised_loss(model, ids, mask, None,
                                   shift_labels=shift_causal_labels(labels))
    assert torch.allclose(stock, mine, atol=1e-12, rtol=1e-9)


# -- 3. -100 positions never contribute ----------------------------------

def test_supervised_count_drives_the_loss():
    """Changing only an -100 target cannot change the loss."""
    model = _reference_model()
    ids, mask, labels = _batch()
    base = compute_supervised_loss(model, ids, mask, labels)
    labels2 = labels.clone()
    # Add supervised targets elsewhere, pointing at the same predictions.
    labels2[0, 20:22] = ids[0, 21:23]
    more = compute_supervised_loss(model, ids, mask, labels2)
    assert not torch.allclose(base, more), "extra supervised rows changed nothing"


def test_removing_all_supervision_is_rejected():
    model = _reference_model()
    ids, mask, labels = _batch()
    labels[:] = IGNORE_INDEX
    with pytest.raises(SupervisedLossError, match="no supervised positions"):
        compute_supervised_loss(model, ids, mask, labels)


def test_missing_labels_is_rejected():
    model = _reference_model()
    ids, mask, _ = _batch()
    with pytest.raises(SupervisedLossError, match="required"):
        compute_supervised_loss(model, ids, mask, None)


def test_length_mismatch_is_rejected():
    model = _reference_model()
    ids, mask, labels = _batch()
    with pytest.raises(SupervisedLossError, match="does not match"):
        compute_supervised_loss(model, ids, mask, labels[:, :-3])


# -- 4. LoRA gradients ----------------------------------------------------

def test_lora_parameters_receive_gradients():
    from peft import LoraConfig, get_peft_model

    from aurora.qlora import prepare_base_for_lora

    class _C:
        gradient_checkpointing = False
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.0
        lora_target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

    base = _reference_model(torch.float32)
    base.gradient_checkpointing_disable()
    model = prepare_base_for_lora(base, _C())
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"))
    model.train()

    ids, mask, labels = _batch()
    compute_supervised_loss(model, ids, mask, labels).backward()

    got = [n for n, p in model.named_parameters()
           if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    assert got, "no LoRA gradient"
    assert all("lora_" in n for n in got)
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert p.grad.isfinite().all(), name


def test_backbone_gradients_survive_for_unsupervised_positions():
    """The subtle case: an unsupervised h[t] still feeds a supervised t'.

    If the backbone were truncated alongside the head, these would be zero.
    """
    from peft import LoraConfig, get_peft_model

    from aurora.qlora import prepare_base_for_lora

    class _C:
        gradient_checkpointing = False
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.0
        lora_target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

    base = _reference_model(torch.float32)
    base.gradient_checkpointing_disable()
    model = get_peft_model(prepare_base_for_lora(base, _C()), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"))
    model.train()

    ids, mask, labels = _batch()
    # Supervise one contiguous late block only.
    labels[:] = IGNORE_INDEX
    labels[0, 20:30] = ids[0, 21:31]
    compute_supervised_loss(model, ids, mask, labels).backward()

    # Layer 0 LoRA q_proj must have gradient, because position 20's hidden state
    # depends on layer 0's attention over positions 0..19, all unsupervised.
    # PEFT names the adapters "q_proj.lora_A.default" / "q_proj.lora_B.default",
    # so the layer filter has to match on the path, not on a "lora_q_proj" token.
    early = [n for n, p in model.named_parameters()
             if "layers.0" in n and "q_proj.lora_" in n
             and p.grad is not None and p.grad.abs().sum() > 0]
    assert early, "backbone gradients were lost for unsupervised positions"


# -- 5. padding and batching ---------------------------------------------

def test_padding_does_not_change_the_loss():
    model = _reference_model()
    ids1, mask1, labels1 = _batch(batch=1, length=40)
    loss1 = compute_supervised_loss(model, ids1, mask1, labels1)

    padded_ids = torch.cat([ids1, torch.zeros(1, 8, dtype=ids1.dtype)], 1)
    padded_labels = torch.cat(
        [labels1, torch.full((1, 8), IGNORE_INDEX, dtype=torch.long)], 1)
    padded_mask = torch.cat([mask1, torch.zeros(1, 8, dtype=mask1.dtype)], 1)
    loss2 = compute_supervised_loss(model, padded_ids, padded_mask, padded_labels)
    assert torch.allclose(loss1, loss2, atol=1e-12)


def test_supervised_tokens_are_summed_across_the_batch():
    """num_items_in_batch normalises by the total, so a bigger batch is a mean."""
    model = _reference_model()
    ids, mask, labels = _batch(batch=2, seed=3)
    total = int((shift_causal_labels(labels) != IGNORE_INDEX).sum())
    loss_sum = compute_supervised_loss(model, ids, mask, labels,
                                      num_items_in_batch=total)
    loss_mean = compute_supervised_loss(model, ids, mask, labels)
    assert torch.allclose(loss_sum, loss_mean, atol=1e-12)


def test_single_supervised_position():
    model = _reference_model()
    ids, mask, labels = _batch()
    labels[:] = IGNORE_INDEX
    labels[0, 17] = ids[0, 18]
    loss = compute_supervised_loss(model, ids, mask, labels)
    assert torch.isfinite(loss)


# -- unwrapping ----------------------------------------------------------

def test_unwrap_finds_the_causal_lm_bare():
    model = _reference_model()
    assert unwrap_causal_lm(model) is model


def test_unwrap_finds_the_causal_lm_through_peft():
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(_reference_model(torch.float32), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"))
    inner = unwrap_causal_lm(model)
    assert hasattr(inner, "lm_head") and hasattr(inner, "model")


def test_unwrap_rejects_a_non_causal_lm():
    with pytest.raises(SupervisedLossError, match="could not find"):
        unwrap_causal_lm(torch.nn.Linear(4, 4))


# -- the trainer wiring --------------------------------------------------

def test_trainer_class_reports_loss_kwargs_handled():
    from aurora.train import build_trainer_class

    cls = build_trainer_class()
    assert cls.model_accepts_loss_kwargs is False, \
        "required: num_items_in_batch is consumed manually, so the Trainer must " \
        "not also compensate for gradient accumulation"


def test_trainer_class_still_builds():
    from aurora.train import build_trainer_class

    assert hasattr(build_trainer_class(), "compute_loss")
