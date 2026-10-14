"""CLI entry point for Aurora Brain v0.1 QLoRA training.

Not run by default. Everything before the model is loaded is a preflight that
can fail on CPU, so a broken dataset, a stale system fingerprint or a bad mask is
caught in seconds rather than after a 16 GB read.

    python -m aurora.train --preflight-only            # CPU, no model load
    python -m aurora.train --max-steps 8 --output-dir ... # later smoke run

The trainer loads the base model in the same 4-bit configuration
PRIMORA/aurora/brain.py uses at inference, so the adapter this produces is
loadable by the unchanged runtime through PRIMORA_AURORA_ADAPTER_PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Running this file directly puts TRAINING/aurora on sys.path, not TRAINING, so
# the package root is added explicitly. Both forms then work: `python -m
# aurora.train` from TRAINING, and `python TRAINING/aurora/train.py` from the
# repository root.
sys.path.insert(0, r"C:\PRIMORA PROJECT")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurora.config import DEVIATIONS, DEFAULT_BASE_MODEL, DEFAULT_OUTPUT_DIR, TrainerConfig
from aurora.masking import MaskingError
from aurora.preprocess import CausalCollator, build_split


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aurora.train",
        description="Aurora Brain v0.1 QLoRA trainer (does not run unless invoked).")
    p.add_argument("--split", default="train", choices=["train", "validation"],
                   help="split to train on")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    # Defaults to the config's data_dir (v0.4) rather than the loader module's
    # DEFAULT_OUTDIR (data/v2). The old default silently pointed three dataset
    # generations back, and preflight does not print the data path - only record
    # counts - so a wrong data_dir was easy to miss.
    p.add_argument("--data-dir", default=TrainerConfig.data_dir)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-seq-length", type=int, default=1792)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=10,
                   help="loss logging cadence; the default 10 exceeds a short "
                        "smoke run, which then reports no loss at all")
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop after N optimiser steps (smoke mode)")
    p.add_argument("--limit-train-records", type=int, default=None)
    p.add_argument("--limit-eval-records", type=int, default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--compute-dtype", default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--preflight-only", action="store_true",
                   help="validate config, model files, dataset and masks on CPU "
                        "and exit without loading the model")
    p.add_argument("--save-adapter-only", action="store_true", default=True,
                   help="save the LoRA adapter rather than a merged model")
    return p


def config_from_args(args: argparse.Namespace) -> TrainerConfig:
    return TrainerConfig(
        split=args.split,
        base_model=args.base_model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        limit_train_records=args.limit_train_records,
        limit_eval_records=args.limit_eval_records,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
        data_seed=args.seed,
        device=args.device,
        bnb_4bit_compute_dtype=args.compute_dtype,
        bf16=args.compute_dtype == "bfloat16",
    )


def preflight(config: TrainerConfig, verbose: bool = True) -> dict:
    """Everything that must hold before a GPU is touched."""
    from aurora import qlora

    problems = list(config.validate())
    if problems:
        raise SystemExit("config invalid:\n  " + "\n  ".join(problems))

    # Base model files are checked even on the CPU preflight: a missing file is
    # a missing file regardless of which device is doing the work.
    qlora.check_model_files(config.base_model)

    tokenizer = qlora.load_tokenizer(config.base_model)

    train_ds, train_stats = build_split(
        config.split, tokenizer, config.max_seq_length,
        data_dir=config.data_dir, limit=config.limit_train_records)
    eval_ds, eval_stats = build_split(
        "validation", tokenizer, config.max_seq_length,
        data_dir=config.data_dir, limit=config.limit_eval_records)

    if not len(train_ds):
        raise SystemExit(f"split {config.split!r} produced no usable records")

    if verbose:
        print("=" * 72)
        print("AURORA BRAIN v0.1 - QLoRA PREFLIGHT")
        print("=" * 72)
        print(f"  base model          : {config.base_model}")
        print(f"  output dir          : {config.output_dir}")
        print(f"  compute dtype       : {config.bnb_4bit_compute_dtype}")
        print(f"  LoRA r/alpha/dropout: {config.lora_r}/{config.lora_alpha}/"
              f"{config.lora_dropout}")
        print(f"  LoRA targets        : {', '.join(config.lora_target_modules)}")
        print(f"  seq length          : {config.max_seq_length}")
        print(f"  batch x accumulate  : {config.per_device_train_batch_size} x "
              f"{config.gradient_accumulation_steps} = "
              f"{config.effective_batch_size()}")
        print(f"  learning rate       : {config.learning_rate}")
        print(f"  scheduler           : {config.lr_scheduler_type}, "
              f"warmup {config.warmup_steps(len(train_ds))} steps "
              f"(~{config.warmup_ratio:.0%} of {config.total_steps(len(train_ds))})")
        print(f"  optimiser           : {config.optim}")
        print(f"  seed                : {config.seed}")
        if config.max_steps is not None:
            print(f"  max steps (smoke)   : {config.max_steps}")
        print()
        for name, stats in (("train", train_stats), ("validation", eval_stats)):
            print(f"  [{name}] {stats['records']} records, "
                  f"{stats['seeds']} seeds, {stats['categories']} categories")
            print(f"      length min/mean/max: {stats['length_min']}/"
                  f"{stats['length_mean']:.0f}/{stats['length_max']}")
            print(f"      truncated at {config.max_seq_length}: {stats['truncated']}")
            print(f"      supervised tokens: {stats['trained_tokens']:,} of "
                  f"{stats['total_tokens']:,} ({stats['trained_fraction']:.1%})")
        print()
        print("  deviations from the approved configuration:")
        for d in DEVIATIONS:
            print(f"      - {d}")
        print()
        print("  PREFLIGHT PASSED (no model loaded)")
        print("=" * 72)

    return {
        "config": config,
        "tokenizer": tokenizer,
        "train": train_ds,
        "eval": eval_ds,
        "train_stats": train_stats,
        "eval_stats": eval_stats,
    }


def save_adapter(model, tokenizer, output_dir: str) -> str:
    """Write the LoRA adapter the runtime loads.

    Only the adapter is saved. A merged model would double the on-disk size and
    would no longer be loadable by brain.py's PeftModel.from_pretrained call.
    The tokenizer and chat template are deliberately NOT copied: brain.py loads
    both from the base model path.

    ``base_model_name_or_path`` is verified after the write. A stale or
    absolute-from-another-machine value there is what makes an adapter refuse to
    load later, so it is checked against the base model this run was configured
    with. The value is read from the *PEFT config* - ``model.peft_config[...].``
    - because that is where PEFT itself populates it during ``save_pretrained``,
    deriving it from the loaded backbone's ``name_or_path``. It is not an
    attribute of PeftModelForCausalLM, LoraModel or Qwen3ForCausalLM, and reading
    it off the model object raises AttributeError - which is what aborted the
    first v0.3 run after its final step. The file is otherwise untouched.
    """
    final_dir = os.path.join(output_dir, "aurora_qlora")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir, safe_serialization=True)

    peft_configs = getattr(model, "peft_config", None) or {}
    if not peft_configs:
        raise AttributeError(
            "model exposes no peft_config, so the adapter's base model path "
            "cannot be verified; refusing to report success")
    adapter_name = getattr(model, "active_adapter", None) or sorted(peft_configs)[0]
    base_model_path = peft_configs[adapter_name].base_model_name_or_path

    adapter_config_path = os.path.join(final_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(
            f"save_pretrained did not write {adapter_config_path}; refusing to "
            "report success")
    with open(adapter_config_path, encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    recorded = adapter_config.get("base_model_name_or_path")
    if recorded != base_model_path:
        adapter_config["base_model_name_or_path"] = base_model_path
        with open(adapter_config_path, "w", encoding="utf-8") as handle:
            json.dump(adapter_config, handle, indent=2)
        print(f"  adapter base_model : repointed {recorded!r} -> "
              f"{base_model_path!r}")
    print(f"  adapter base_model : {adapter_config['base_model_name_or_path']}")

    print(f"  adapter written to : {final_dir}")
    print("  files              : "
          + ", ".join(sorted(os.listdir(final_dir))))
    return final_dir


class FeasibilityProbe:
    """Records what a feasibility run has to report: VRAM, step time, loss.

    Added for the GPU smoke test. It only observes - it never changes training
    behaviour - and it exists because the questions that decide whether QLoRA
    fits on this card (peak allocated vs reserved, and whether the 8-bit paged
    optimiser was actually used) are otherwise invisible.
    """

    def __init__(self) -> None:
        import torch
        self.torch = torch
        self.losses: list[float] = []
        self.step_seconds: list[float] = []
        self.grad_nonfinite = 0
        self.grad_zero = 0
        self.optimizer_class: str = "not created"
        self._last = None

    def attach(self, trainer) -> None:
        from transformers import TrainerCallback

        probe = self

        class _Callback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kw):
                probe.reset_peaks()

            def on_step_end(self, args, state, control, **kw):
                import time
                if probe._last is None:
                    probe._last = time.perf_counter()
                    return
                step_seconds = time.perf_counter() - probe._last
                probe._last = time.perf_counter()
                # Accumulation steps are collapsed by the Trainer into one
                # optimiser step, so this is wall time per optimiser step.
                probe.step_seconds.append(step_seconds)
                log = kw.get("logs") or {}
                if "loss" in log:
                    probe.losses.append(float(log["loss"]))

            def on_optimizer_step(self, args, state, control, model=None, **kw):
                # `model` arrives as a named argument, not inside kw. Reading
                # kwargs.get("model") here raised NameError, which aborted the
                # first 8-step smoke run before any step completed.
                if model is None:
                    return
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    total = float(p.grad.detach().abs().sum())
                    if total != total or total == float("inf"):
                        probe.grad_nonfinite += 1
                    elif total == 0.0:
                        probe.grad_zero += 1

        trainer.add_callback(_Callback())

    def note_optimizer(self, trainer) -> None:
        try:
            self.optimizer_class = type(trainer.create_optimizer()).__name__
        except Exception as exc:
            self.optimizer_class = f"could not create: {type(exc).__name__}"

    def reset_peaks(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

    def report(self) -> dict:
        cuda = self.torch.cuda
        stats = {}
        if cuda.is_available():
            stats = {
                "peak_allocated_mib": round(cuda.max_memory_allocated() / 1024 ** 2, 1),
                "peak_reserved_mib": round(cuda.max_memory_reserved() / 1024 ** 2, 1),
                "total_mib": round(
                    cuda.get_device_properties(0).total_memory / 1024 ** 2, 1),
            }
        return {
            "optimizer_class": self.optimizer_class,
            "losses": [round(x, 5) for x in self.losses],
            "steps_timed": len(self.step_seconds),
            "step_seconds_mean": (round(sum(self.step_seconds) / len(self.step_seconds), 2)
                                  if self.step_seconds else None),
            "step_seconds_each": [round(x, 2) for x in self.step_seconds],
            "nonfinite_grad_tensors": self.grad_nonfinite,
            "zero_grad_tensors": self.grad_zero,
            **stats,
        }


def build_trainer_class():
    """A Trainer whose loss is evaluated only at supervised positions.

    The stock path materialises full-vocabulary logits for all ~1,440 positions
    to train on ~46 of them. See ``aurora/supervised_loss.py`` for why selecting
    positions is exactly equivalent rather than merely close.

    ``model_accepts_loss_kwargs = False`` is required by the transformers docs
    whenever ``num_items_in_batch`` is consumed manually; otherwise the Trainer
    compensates for gradient accumulation a second time and the per-step loss
    average is wrong.
    """
    from transformers import Trainer

    from aurora.supervised_loss import compute_supervised_loss

    class AuroraTrainer(Trainer):
        model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            loss = compute_supervised_loss(
                model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=inputs.get("labels"),
                shift_labels=inputs.get("shift_labels"),
                num_items_in_batch=num_items_in_batch,
            )
            if not return_outputs:
                return loss
            # Trainer's prediction_step does `outputs[1:]` whenever
            # return_outputs is True, so returning None here crashes eval with
            # "TypeError: 'NoneType' object is not subscriptable" - which is
            # how the first 8-step run died after finishing all 8 steps.
            #
            # Labels are returned rather than logits on purpose: eval only needs
            # the loss, and materialising 151,936 logits per validation record
            # would reintroduce the exact memory blowup the supervised loss
            # exists to avoid.
            return loss, {"labels": inputs.get("labels")}

    return AuroraTrainer


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    try:
        state = preflight(config)
    except MaskingError as exc:
        print(f"MASKING ERROR: {exc}", file=sys.stderr)
        return 2
    except SystemExit:
        raise
    except Exception as exc:
        print(f"PREFLIGHT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.preflight_only:
        return 0

    # From here on the 8B base is loaded. Nothing above this line needs a GPU.
    from aurora import qlora

    try:
        model, tokenizer, notes = qlora.load_for_training(config)
    except qlora.ModelSetupError as exc:
        print(f"MODEL SETUP FAILED: {exc}", file=sys.stderr)
        return 3
    for note in notes:
        print(f"  {note}")

    os.makedirs(config.output_dir, exist_ok=True)
    # len(state["train"]) is the record count the Trainer is about to be built
    # with, so the saved total_steps/warmup_steps describe this run's dataset
    # rather than whatever corpus the config was first written against.
    config.save(os.path.join(config.output_dir, "aurora_train_config.json"),
                train_records=len(state["train"]))

    args_obj = config.to_training_arguments(len(state["train"]))
    trainer = build_trainer_class()(
        model=model,
        args=args_obj,
        train_dataset=state["train"],
        eval_dataset=state["eval"],
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )

    probe = FeasibilityProbe()
    probe.attach(trainer)
    probe.note_optimizer(trainer)
    print(f"  optimizer in use   : {probe.optimizer_class}")

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    metrics = trainer.evaluate()
    print("  eval metrics       : "
          + json.dumps({k: round(v, 5) for k, v in metrics.items()}))

    report = probe.report()
    print()
    print("  FEASIBILITY REPORT")
    print(f"    optimizer class          : {report['optimizer_class']}")
    print(f"    peak VRAM allocated      : {report.get('peak_allocated_mib')} MiB")
    print(f"    peak VRAM reserved       : {report.get('peak_reserved_mib')} MiB")
    print(f"    total VRAM               : {report.get('total_mib')} MiB")
    print(f"    timed optimizer steps    : {report['steps_timed']}")
    print(f"    mean step seconds        : {report['step_seconds_mean']}")
    print(f"    per-step seconds         : {report['step_seconds_each']}")
    print(f"    losses                   : {report['losses']}")
    print(f"    non-finite grad tensors  : {report['nonfinite_grad_tensors']}")
    print(f"    zero grad tensors        : {report['zero_grad_tensors']}")

    adapter_dir = save_adapter(trainer.model, tokenizer, config.output_dir)

    with open(os.path.join(config.output_dir, "final_metrics.json"), "w",
              encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(config.output_dir, "feasibility_report.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"report": report, "metrics": metrics,
                   "adapter_dir": adapter_dir,
                   "config": config.to_dict()}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
