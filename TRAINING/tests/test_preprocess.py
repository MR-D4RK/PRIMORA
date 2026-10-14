"""CPU tests for preprocessing, configuration and model-file checks.

No model weights are loaded anywhere here: the tokenizer is the only artifact
read, and the QLoRA tests only exercise the guards that run *before* a load.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))

from aurora.config import DEVIATIONS, REQUIRED_MODEL_FILES, TrainerConfig  # noqa: E402
from aurora.dataset_v2 import DEFAULT_OUTDIR  # noqa: E402
from aurora.preprocess import (  # noqa: E402
    CausalCollator,
    SupervisedDataset,
    build_split,
)
from aurora.qlora import ModelSetupError, check_device, check_model_files  # noqa: E402

BASE_MODEL = os.path.join(_ROOT, "RESOURCES", "aurora", "v0.1", "model")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


# -- collator ------------------------------------------------------------


def test_collator_pads_and_keeps_pad_out_of_the_loss():
    collator = CausalCollator(pad_token_id=0)
    batch = collator([
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7],
         "attention_mask": [1, 1, 1]},
        {"input_ids": [8, 9], "labels": [-100, 9], "attention_mask": [1, 1]},
    ])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][1].tolist() == [8, 9, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    # The padded position must not carry loss.
    assert batch["labels"][1].tolist() == [-100, 9, -100]
    assert batch["labels"][0].tolist() == [-100, 6, 7]


def test_collator_keeps_the_last_supervised_token(tokenizer):
    collator = CausalCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator([{"input_ids": [1, 2, 3], "labels": [-100, -100, 3],
                       "attention_mask": [1, 1, 1]}])
    assert batch["labels"].tolist() == [[-100, -100, 3]]


# -- dataset -------------------------------------------------------------


def test_build_split_produces_a_torch_dataset(tokenizer):
    dataset, stats = build_split("train", tokenizer, 1792, limit=12)
    assert isinstance(dataset, SupervisedDataset)
    assert len(dataset) == 12
    assert stats["records"] == 12
    assert stats["trained_tokens"] > 0
    assert stats["length_max"] <= 1792


def test_build_split_refuses_holdout_by_default(tokenizer):
    with pytest.raises(ValueError, match="protected"):
        build_split("holdout", tokenizer, 1792, data_dir=DEFAULT_OUTDIR)


def test_build_split_can_read_holdout_explicitly(tokenizer):
    dataset, stats = build_split(
        "holdout", tokenizer, 1792, data_dir=DEFAULT_OUTDIR,
        include_holdout=True, limit=5)
    assert len(dataset) == 5
    assert stats["split"] == "holdout"


def test_build_split_refuses_a_length_that_clips_all_supervision(tokenizer):
    """The real guarantee: an unusable max_length fails loudly.

    The system prompt alone is 1,375 tokens, so a small max_length removes every
    supervised turn. That must be an error, not a silently unsupervised record.
    """
    with pytest.raises(Exception, match="no trainable token"):
        build_split("train", tokenizer, 600, limit=4)


def test_configured_length_truncates_nothing_in_the_real_data(tokenizer):
    """max_seq_length 1792 sits above the longest real record (1717).

    This matters more than it looks: most records have a single assistant turn at
    the very end, so any truncation would remove all supervision and the strict
    build would fail. 1792 leaves headroom for that.
    """
    _, stats = build_split("train", tokenizer, 1792, limit=120)
    assert stats["length_max"] <= 1792
    assert stats["truncated"] == 0


def test_length_below_the_longest_record_is_refused(tokenizer):
    """A length that clips a single-assistant-turn record must fail loudly."""
    with pytest.raises(Exception, match="no trainable token"):
        build_split("train", tokenizer, 1000, limit=3)


def test_validation_split_loads_and_masks(tokenizer):
    dataset, stats = build_split("validation", tokenizer, 1792, limit=10)
    assert len(dataset) == 10
    assert stats["records"] == 10


# -- configuration -------------------------------------------------------


def test_default_config_matches_the_approved_values():
    c = TrainerConfig()
    assert c.load_in_4bit and c.bnb_4bit_quant_type == "nf4"
    assert c.bnb_4bit_use_double_quant is True
    assert (c.lora_r, c.lora_alpha, c.lora_dropout) == (8, 16, 0.05)
    assert c.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert c.max_seq_length == 1792
    assert c.per_device_train_batch_size == 1
    assert c.gradient_accumulation_steps == 16
    assert c.learning_rate == 1e-4
    assert 1e-4 <= c.learning_rate <= 2e-4
    assert c.lr_scheduler_type == "cosine"
    assert c.warmup_ratio == 0.03
    assert c.num_train_epochs == 2.0
    assert c.optim == "paged_adamw_8bit"
    assert c.gradient_checkpointing is True
    assert c.eval_steps == 100 and c.save_steps == 100
    assert c.metric_for_best_model == "eval_loss"
    assert c.load_best_model_at_end is True
    assert c.validate() == []


def test_config_rejects_disabling_4bit():
    c = TrainerConfig(load_in_4bit=False)
    assert any("QLoRA is required" in p for p in c.validate())


def test_config_rejects_non_unit_batch_size():
    assert any("per_device_train_batch_size must be 1" in p
               for p in TrainerConfig(per_device_train_batch_size=4).validate())


def test_config_rejects_lr_outside_the_approved_range():
    assert any("outside the approved" in p
               for p in TrainerConfig(learning_rate=5e-4).validate())
    assert any("outside the approved" in p
               for p in TrainerConfig(learning_rate=1e-6).validate())


def test_config_rejects_alpha_that_breaks_comparability():
    assert any("comparability" in p
               for p in TrainerConfig(lora_alpha=32).validate())


def test_config_rejects_disabling_checkpointing():
    assert any("gradient_checkpointing is required" in p
               for p in TrainerConfig(gradient_checkpointing=False).validate())


def test_step_arithmetic_matches_trainer():
    c = TrainerConfig()
    # 967 records / (1 * 16) = 61 steps per epoch, x2 epochs = 122.
    assert c.total_steps(967) == 122
    assert c.effective_batch_size() == 16
    # 3% of 122 -> 4.
    assert c.warmup_steps(967) == 4


def test_max_steps_caps_the_run():
    c = TrainerConfig(max_steps=8)
    assert c.total_steps(967) == 8
    assert c.warmup_steps(967) == 1


def test_training_arguments_accept_the_configured_values():
    c = TrainerConfig(output_dir=os.path.join(
        os.environ.get("TEMP", "."), "aurora_args_probe"))
    args = c.to_training_arguments(967)
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 16
    assert args.warmup_steps == 4
    assert args.optim == "paged_adamw_8bit"
    assert args.remove_unused_columns is False
    assert args.bf16 is True


def test_deviations_are_documented():
    assert len(DEVIATIONS) >= 1
    joined = " ".join(DEVIATIONS)
    assert "warmup_ratio" in joined
    assert "save_safetensors" in joined


# -- pre-GPU guards ------------------------------------------------------


def test_missing_base_model_is_rejected():
    with pytest.raises(ModelSetupError, match="not found"):
        check_model_files(os.path.join(BASE_MODEL, "does_not_exist"))


def test_incomplete_base_model_is_rejected(tmp_path):
    with pytest.raises(ModelSetupError, match="missing"):
        check_model_files(str(tmp_path))


def test_real_base_model_has_every_required_file():
    check_model_files(BASE_MODEL)
    for name in REQUIRED_MODEL_FILES:
        assert os.path.exists(os.path.join(BASE_MODEL, name)), name


def test_cpu_device_request_is_allowed():
    check_device("cpu")


def test_cuda_request_fails_when_unavailable(monkeypatch):
    """A CPU-only environment must refuse --device cuda before loading 16 GB."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ModelSetupError, match="CUDA is not available"):
        check_device("cuda")
