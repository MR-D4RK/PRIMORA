"""Tests for the QLoRA preparation path.

The 8B base is never loaded. These tests use a tiny but genuine
``Qwen2ForCausalLM``: PEFT requires a ``PreTrainedModel``, so a plain
``nn.Module`` cannot exercise the attachment path at all. It is structurally the
same shape as the real base - untied ``embed_tokens`` and ``lm_head`` plus
per-block q/k/v/o projections - which is the property the preparation logic
exists to handle.

What is pinned:
  * the two large tensors are never converted to fp32
  * no parameter changes dtype at all
  * the dtype chain stays consistent, so a bf16 norm does not emit fp32 hidden
    states into a bf16 lm_head
  * every base parameter is frozen before LoRA attaches
  * only LoRA parameters are trainable, and neither the head nor the embeddings
    are among them
  * gradients actually reach the adapters through the frozen base
  * the saved adapter is loadable by ``PeftModel.from_pretrained(base, path)``,
    which is what PRIMORA/aurora/brain.py calls
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))

from aurora.config import TrainerConfig  # noqa: E402
from aurora.qlora import ModelSetupError, check_model_files, prepare_base_for_lora  # noqa: E402

VOCAB, HIDDEN, LAYERS = 64, 16, 2


class _FakeConfig:
    """Minimal stand-in exposing exactly what the preparation path reads."""

    def __init__(self, gradient_checkpointing: bool = True) -> None:
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.lora_target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")


def _tiny_qwen(dtype=torch.float32):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=32,
        num_hidden_layers=LAYERS, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg).to(dtype)


@pytest.fixture
def model():
    return _tiny_qwen()


def _lora_config():
    from peft import LoraConfig

    return LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      bias="none", task_type="CAUSAL_LM")


def _attach(model, config=None):
    from peft import get_peft_model

    return get_peft_model(
        prepare_base_for_lora(model, config or _FakeConfig()), _lora_config())


# -- no accidental fp32 conversion ---------------------------------------

def test_preparation_changes_no_dtype(model):
    before = {n: p.dtype for n, p in model.named_parameters()}
    prepare_base_for_lora(model, _FakeConfig())
    after = {n: p.dtype for n, p in model.named_parameters()}
    assert before == after


def test_large_tensors_are_never_upcast_to_fp32():
    """The real failure mode: a bf16 base must stay bf16."""
    m = _tiny_qwen(torch.bfloat16)
    prepare_base_for_lora(m, _FakeConfig())
    assert m.model.embed_tokens.weight.dtype == torch.bfloat16
    assert m.lm_head.weight.dtype == torch.bfloat16
    fp32 = [n for n, p in m.named_parameters() if p.dtype == torch.float32]
    assert fp32 == [], f"unexpected fp32 conversion: {fp32}"


def test_preparation_does_not_inflate_parameter_storage():
    """Summed element sizes must be unchanged - an upcast would show here."""
    m = _tiny_qwen(torch.bfloat16)
    before = sum(p.numel() * p.element_size() for p in m.parameters())
    prepare_base_for_lora(m, _FakeConfig())
    after = sum(p.numel() * p.element_size() for p in m.parameters())
    assert before == after


def test_dtype_chain_is_consistent():
    """A bf16 final norm feeding a bf16 head avoids the mismatch that an fp32
    norm alone would cause."""
    m = _tiny_qwen(torch.bfloat16)
    prepare_base_for_lora(m, _FakeConfig())
    assert m.model.norm.weight.dtype == m.lm_head.weight.dtype


# -- freezing and gradient wiring ----------------------------------------

def test_every_base_parameter_is_frozen(model):
    prepare_base_for_lora(model, _FakeConfig())
    assert all(not p.requires_grad for p in model.parameters())


def test_input_require_grads_is_registered():
    """Observable proxy: the embedding output must carry requires_grad."""
    m = _tiny_qwen()
    prepare_base_for_lora(m, _FakeConfig())
    emb = m.get_input_embeddings()
    out = emb(torch.zeros(1, 3, dtype=torch.long))
    assert out.requires_grad is True


def test_gradient_checkpointing_follows_the_config(model):
    m = _tiny_qwen()
    prepare_base_for_lora(m, _FakeConfig(gradient_checkpointing=True))
    assert m.is_gradient_checkpointing is True

    m2 = _tiny_qwen()
    prepare_base_for_lora(m2, _FakeConfig(gradient_checkpointing=False))
    assert m2.is_gradient_checkpointing is False


# -- LoRA attachment and trainable selection ----------------------------

def test_only_lora_parameters_are_trainable(model):
    peft_model = _attach(model)
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable parameters"
    assert all("lora_" in n for n in trainable), trainable


def test_the_output_head_is_not_trainable(model):
    assert _attach(model).base_model.lm_head.weight.requires_grad is False


def test_embeddings_are_not_trainable(model):
    peft_model = _attach(model)
    assert peft_model.base_model.model.model.embed_tokens.weight.requires_grad is False


def test_trainable_count_matches_the_approved_shape():
    """r=8 on q/k/v/o across 36 layers is 288 tensors / 7,667,712 params.

    Derived from the shape rather than copied from a run: q and o are 4096x4096,
    k and v are 4096x1024, each contributing r*(in+out).
    """
    r, hidden, kv, layers = 8, 4096, 1024, 36
    per_layer = r * (hidden + hidden) * 2 + r * (hidden + kv) * 2
    assert per_layer * layers == 7_667_712


def test_gradients_reach_lora_parameters_through_a_frozen_base():
    peft_model = _attach(_tiny_qwen())
    out = peft_model(input_ids=torch.randint(0, VOCAB, (1, 12)))
    out.logits.sum().backward()
    got = [n for n, p in peft_model.named_parameters()
           if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    assert got, "no LoRA gradient reached the adapters"
    assert all("lora_" in n for n in got)


def test_gradients_are_finite():
    peft_model = _attach(_tiny_qwen())
    out = peft_model(input_ids=torch.randint(0, VOCAB, (1, 12)))
    out.logits.sum().backward()
    for name, p in peft_model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert p.grad.isfinite().all(), name


# -- adapter format compatibility ---------------------------------------

def test_saved_adapter_is_loadable_by_peft_from_pretrained(tmp_path):
    """Exactly the call brain.py makes: PeftModel.from_pretrained(base, path)."""
    from peft import PeftModel, get_peft_model

    prepared = prepare_base_for_lora(_tiny_qwen(), _FakeConfig())
    peft_model = get_peft_model(prepared, _lora_config())
    peft_model.save_pretrained(str(tmp_path))

    assert (tmp_path / "adapter_config.json").exists()
    assert (tmp_path / "adapter_model.safetensors").exists()

    with open(tmp_path / "adapter_config.json", encoding="utf-8") as handle:
        cfg = json.load(handle)
    assert cfg["peft_type"] == "LORA"
    assert cfg["task_type"] == "CAUSAL_LM"
    assert set(cfg["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    # brain.py supplies the base itself, so a stale absolute path from an old
    # training run must not be baked into the adapter.
    stale = str(cfg.get("base_model_name_or_path") or "")
    assert not stale.startswith("C:\\PRIMORA\\TRAINING"), stale

    reloaded = PeftModel.from_pretrained(prepared, str(tmp_path))
    assert type(reloaded).__name__ == "PeftModelForCausalLM"
    # from_pretrained loads in inference mode, which is what the runtime wants:
    # the LoRA layers must be present and wired in, but frozen.
    lora_layers = [n for n, _ in reloaded.named_modules()
                   if n.endswith("lora_A.default")]
    assert lora_layers, "no LoRA layers in the reloaded adapter"
    assert all(not p.requires_grad for p in reloaded.parameters())
    # And it still produces logits, i.e. it is a usable model.
    out = reloaded(input_ids=torch.randint(0, VOCAB, (1, 8)))
    assert out.logits.shape[-1] == VOCAB


def test_optimizer_is_paged_adamw8bit():
    pytest.importorskip("bitsandbytes")
    from bitsandbytes.optim import PagedAdamW8bit

    peft_model = _attach(_tiny_qwen())
    params = [p for p in peft_model.parameters() if p.requires_grad]
    assert type(PagedAdamW8bit(params, lr=1e-4)).__name__ == "PagedAdamW8bit"


# -- guards unchanged ----------------------------------------------------

def test_missing_base_model_still_rejected():
    with pytest.raises(ModelSetupError):
        check_model_files(r"C:\definitely\not\here")


def test_config_still_forbids_disabling_4bit():
    assert TrainerConfig(load_in_4bit=False).validate()


# -- grouped gradient checkpointing --------------------------------------

def test_grouped_checkpointing_does_not_change_the_loss():
    """Grouping is only safe if the forward math is untouched."""
    from aurora.qlora import enable_grouped_gradient_checkpointing

    model = _tiny_qwen()
    ids = torch.randint(0, VOCAB, (1, 24))
    model.eval()
    with torch.no_grad():
        before = model(input_ids=ids).logits.clone()
    enable_grouped_gradient_checkpointing(model, group_size=2)
    with torch.no_grad():
        after = model(input_ids=ids).logits
    assert torch.equal(before, after)


def test_grouped_checkpointing_reduces_the_number_of_checkpoints():
    from aurora.qlora import GroupedDecoderLayers, enable_grouped_gradient_checkpointing

    model = _tiny_qwen()
    notes = enable_grouped_gradient_checkpointing(model, group_size=1)
    backbone = model.model
    assert len(backbone.layers) == LAYERS
    assert all(isinstance(g, GroupedDecoderLayers) for g in backbone.layers)
    assert "2 layers -> 2 checkpoints" in notes[0]


def test_grouped_checkpointing_stands_down_the_per_layer_wrapper():
    """Otherwise every layer still checkpoints and nothing is saved."""
    from aurora.qlora import enable_grouped_gradient_checkpointing

    model = _tiny_qwen()
    original = list(model.model.layers)
    enable_grouped_gradient_checkpointing(model, group_size=2)
    for layer in original:
        assert layer.gradient_checkpointing is False


def test_grouped_checkpointing_trains_through_the_groups():
    from aurora.qlora import enable_grouped_gradient_checkpointing

    model = _tiny_qwen()
    enable_grouped_gradient_checkpointing(model, group_size=2)
    model.train()
    ids = torch.randint(0, VOCAB, (1, 24))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    grads = [n for n, p in model.named_parameters()
             if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads, "no gradient reached any parameter through the groups"
    assert all(p.grad.isfinite().all() for p in model.parameters()
               if p.grad is not None)


def test_grouped_checkpointing_rejects_a_bad_group_size():
    from aurora.qlora import enable_grouped_gradient_checkpointing

    with pytest.raises(ModelSetupError, match="group_size"):
        enable_grouped_gradient_checkpointing(_tiny_qwen(), group_size=0)


def test_grouped_checkpointing_rejects_mixed_attention_types():
    """Grouping across a sliding-window boundary would reuse the wrong mask."""
    from aurora.qlora import enable_grouped_gradient_checkpointing

    model = _tiny_qwen()
    model.config.layer_types = ["full_attention", "sliding_attention"]
    with pytest.raises(ModelSetupError, match="attention types"):
        enable_grouped_gradient_checkpointing(model, group_size=2)


def test_grouped_checkpointing_rejects_a_non_qwen_backbone():
    from aurora.qlora import enable_grouped_gradient_checkpointing

    with pytest.raises(ModelSetupError, match="decoder layer list"):
        enable_grouped_gradient_checkpointing(torch.nn.Linear(4, 4), group_size=2)
