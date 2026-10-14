"""Training configuration for Aurora Brain v0.1 QLoRA.

One place for every value that was decided during the architecture pass, so the
run is auditable from a single object and nothing is set implicitly inside the
Trainer.

Two values differ from the approved configuration because transformers 5.15.1
does not accept the original forms. Both are recorded in DEVIATIONS rather than
silently substituted:

*   ``warmup_ratio`` is gone. ``TrainingArguments.__init__`` rejects it; only
    ``warmup_steps`` (an integer) is accepted. The approved ~3% is therefore
    computed into an integer at construction time from the real step count,
    which is what a ratio would have produced anyway.
*   ``save_safetensors`` is gone. safetensors is the default and only output
    format in this version.

``Learning rate 1e-4`` is the chosen end of the approved 1e-4..2e-4 range. With
batch 1, accumulation 16 and 4 epochs over 1080 records the model sees 272
optimiser steps, so the lower end of the range remains the defensible pick: at
that step count a higher LR is more likely to skip the narrow behaviour changes
the seeds describe than to converge faster.

v0.4 data
---------
The dataset moved to the v0.4 splits (``data/v4``: 1080 train, 180 validation,
180 holdout) and the output moved to ``RESOURCES/aurora/v0.4``. The LoRA shape,
quantisation, optimiser, precision, sequence length and learning rate are
unchanged from v0.3: v0.4 trains a *fresh* adapter on the same Qwen3-8B base and
never loads the v0.3 adapter. The v0.4 records are shorter per record and carry
about half the supervised tokens of v0.2, so the approved run uses 4 epochs
instead of 2 (``--num-train-epochs 4``): 68 steps per epoch, 272 optimizer steps
overall, 8 warmup steps at the approved ~3%.

Where each approved value lives: ``data_dir``, ``output_dir`` and
``save_total_limit`` are defaults here (``save_total_limit`` has no CLI flag at
all); ``eval_steps`` (50) and ``num_train_epochs`` (4) are passed explicitly on
the command line, because this module's defaults for those two are still pinned
at the v0.3 values by ``TRAINING/tests/test_preprocess.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Base model and adapter locations. Both are repository-relative by convention
# (see PRIMORA/aurora/configuration.py).
#
# The base model is unchanged for v0.4: v0.4 trains a *fresh* LoRA adapter on the
# same Qwen3-8B base that v0.1, v0.2 and v0.3 used. It is not a continuation of
# the v0.3 adapter, so nothing from v0.3 is loaded into this run.
#
# DEFAULT_OUTPUT_DIR is the parent, not the adapter directory, because
# ``train.save_adapter`` appends "aurora_qlora" itself. That puts the v0.4 adapter
# at RESOURCES/aurora/v0.4/aurora_qlora - the same shape as v0.1, which is where
# PRIMORA_AURORA_ADAPTER_PATH points by default.
DEFAULT_BASE_MODEL = r"C:\PRIMORA PROJECT\RESOURCES\aurora\v0.1\model"
DEFAULT_OUTPUT_DIR = r"C:\PRIMORA PROJECT\RESOURCES\aurora\v0.4"

# Files the base model must have before any GPU work. Checked by qlora.py so a
# missing download fails in seconds instead of after a 16 GB read.
REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)

DEVIATIONS = (
    "warmup_ratio is not a parameter of transformers 5.15.1 "
    "TrainingArguments; the approved ~3% is computed into an integer "
    "warmup_steps from the real total step count instead.",
    "save_safetensors is not a parameter in this version; safetensors is the "
    "only output format, which is what was required.",
)


@dataclass
class TrainerConfig:
    """Every decided value, with no hidden defaults."""

    # -- paths ----------------------------------------------------------
    base_model: str = DEFAULT_BASE_MODEL
    output_dir: str = DEFAULT_OUTPUT_DIR
    # v0.4 splits. The holdout is never selectable here: load_split refuses it
    # without include_holdout=True and build_split refuses it outright.
    data_dir: str = r"C:\PRIMORA PROJECT\TESTS\training\data\v4"
    # Which split to train on. The holdout is deliberately not selectable here.
    split: str = "train"

    # -- QLoRA ----------------------------------------------------------
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"   # "float16" when unsupported
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Verified against the safetensors key list of the local base: every one of
    # these exists at all 36 layers. q_norm/k_norm also exist but are left
    # untouched, matching the v0.1 adapter.
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # -- sequences and batching ----------------------------------------
    max_seq_length: int = 1792
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16

    # -- optimisation ---------------------------------------------------
    learning_rate: float = 1e-4
    # The approved v0.4 run is 4 epochs; that is passed as --num-train-epochs
    # because this default is pinned at 2.0 by test_preprocess.
    num_train_epochs: float = 2.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    bf16: bool = True
    gradient_checkpointing: bool = True

    # -- memory plumbing (not a hyperparameter) ---------------------------
    # embed_tokens and lm_head are frozen, unquantised bf16 and together are
    # 2.31 GiB - 41% of VRAM on an 8 GiB card. They are moved to CPU and
    # streamed per call. Chunking is over the vocab (output) axis only, so each
    # logit is still one full-length dot product and the forward is exact.
    cpu_offload_frozen_head: bool = True
    offload_chunk_rows: int = 16384            # 128 MiB per streamed chunk

    # -- evaluation and checkpointing -----------------------------------
    # The approved v0.4 run evaluates every 50 steps; that value is passed as
    # --eval-steps because this default is pinned at 100 by test_preprocess.
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    # 3 rather than 2: the approved run spans 272 steps with saves at 100/200
    # plus the end, and a limit of 2 can evict the checkpoint holding the best
    # eval_loss before load_best_model_at_end asks for it.
    save_total_limit: int = 3
    logging_steps: int = 10
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # -- determinism ----------------------------------------------------
    seed: int = 42
    data_seed: int = 42

    # -- runtime switches ------------------------------------------------
    max_steps: Optional[int] = None          # smoke mode
    limit_train_records: Optional[int] = None
    limit_eval_records: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    device: str = "cuda"

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.offload_chunk_rows < 1:
            problems.append(
                f"offload_chunk_rows is {self.offload_chunk_rows}; must be >= 1")
        if not self.load_in_4bit:
            problems.append(
                "load_in_4bit is False; a bf16 8B base is 15.26 GiB and cannot "
                "fit the 8 GiB card, so QLoRA is required, not optional")
        if self.bnb_4bit_quant_type != "nf4":
            problems.append("bnb_4bit_quant_type must be 'nf4'")
        if not self.bnb_4bit_use_double_quant:
            problems.append("double quantisation is required for 8 GiB")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            problems.append("LoRA rank and alpha must be positive")
        if self.lora_alpha != 2 * self.lora_r:
            problems.append(
                f"alpha {self.lora_alpha} is not 2*r ({2 * self.lora_r}); the "
                f"v0.1 adapter used r=8/alpha=16 and comparability needs the same")
        if self.per_device_train_batch_size != 1:
            problems.append(
                "per_device_train_batch_size must be 1 at this sequence length "
                "on 8 GiB")
        if self.gradient_accumulation_steps < 1:
            problems.append("gradient_accumulation_steps must be >= 1")
        if not 1e-4 <= self.learning_rate <= 2e-4:
            problems.append(
                f"learning_rate {self.learning_rate} is outside the approved "
                f"1e-4..2e-4 range")
        if not self.gradient_checkpointing:
            problems.append(
                "gradient_checkpointing is required at seq "
                f"{self.max_seq_length} on 8 GiB")
        if self.warmup_ratio <= 0 or self.warmup_ratio >= 1:
            problems.append("warmup_ratio must be in (0, 1)")
        if self.device not in ("cuda", "cpu"):
            problems.append("device must be 'cuda' or 'cpu'")
        return problems

    # -- derived ---------------------------------------------------------

    def total_steps(self, train_records: int) -> int:
        """Optimiser steps for the run, matching Trainer's arithmetic."""
        per_step = max(1, self.per_device_train_batch_size) * \
            max(1, self.gradient_accumulation_steps)
        if self.max_steps is not None:
            return min(self.max_steps, math.ceil(train_records / per_step))
        return int(math.ceil(train_records / per_step) * self.num_train_epochs)

    def warmup_steps(self, train_records: int) -> int:
        """The approved ratio, resolved to the integer this version accepts."""
        return max(1, int(round(self.total_steps(train_records) * self.warmup_ratio)))

    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lora_target_modules"] = list(self.lora_target_modules)
        return data

    def save(self, path: str, train_records: int) -> None:
        """Write the run record, with step counts derived from the real corpus.

        ``train_records`` is the number of examples the Trainer will actually
        see - ``len(train_ds)`` from preflight - which is the same figure passed
        to ``to_training_arguments``, so the metadata here and the schedule the
        run uses cannot disagree.

        It is required rather than defaulted because the previous fallback was
        the literal ``967`` (the v0.2 corpus). Every later dataset inherited it:
        the v0.4 record would have been stamped ``total_steps 122`` /
        ``warmup_steps 4`` instead of the approved ``272`` / ``8``. A caller that
        knows its own record count cannot silently produce a stale value.
        ``limit_train_records`` still wins when set, since that is what the
        Trainer is limited to.
        """
        records = self.limit_train_records or train_records
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "config": self.to_dict(),
                "total_steps": self.total_steps(records),
                "warmup_steps": self.warmup_steps(records),
                "effective_batch_size": self.effective_batch_size(),
                "deviations_from_approved_config": list(DEVIATIONS),
            }, handle, indent=2)

    def to_training_arguments(self, train_records: int):
        """Build TrainingArguments with only fields this version accepts."""
        from transformers import TrainingArguments

        kwargs: dict[str, Any] = {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            # warmup_ratio is unavailable in 5.15.1; this is the integer it
            # would have produced.
            "warmup_steps": self.warmup_steps(train_records),
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "optim": self.optim,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "seed": self.seed,
            "data_seed": self.data_seed,
            "report_to": [],
            # The dataset is a plain torch Dataset of dicts; the Trainer must
            # not try to coerce its columns.
            "remove_unused_columns": False,
        }
        if self.max_steps is not None:
            kwargs["max_steps"] = self.max_steps
            # A limited-step run stops long before eval_steps, so no eval ever
            # happens and load_best_model_at_end has nothing to select on. The
            # Trainer raises rather than quietly keeping the last checkpoint.
            # Smoke mode only; the full run keeps best-model selection.
            kwargs["load_best_model_at_end"] = False
        return TrainingArguments(**kwargs)


__all__ = [
    "DEVIATIONS",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_OUTPUT_DIR",
    "REQUIRED_MODEL_FILES",
    "TrainerConfig",
]
