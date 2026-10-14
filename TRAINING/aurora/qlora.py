"""QLoRA model setup for Aurora Brain v0.1.

Mirrors how PRIMORA/aurora/brain.py loads the model at inference, so the thing
trained here is loaded the same way afterwards:

    BitsAndBytesConfig(load_in_4bit=True, nf4, double quant, fp16/bf16 compute)
    AutoModelForCausalLM.from_pretrained(..., quantization_config=..., device_map={"": 0})
    PeftModel.from_pretrained(base, adapter_path, device_map={"": 0})

Nothing in PRIMORA is imported for the model itself - only the tokenizer path
and chat template, which must be the runtime's own. The 8B weights are never
touched by the tests; these functions are only called from train.py.
"""

from __future__ import annotations

import inspect
import os
from typing import Optional

import torch

from aurora.config import REQUIRED_MODEL_FILES, TrainerConfig


class ModelSetupError(RuntimeError):
    """Raised before any GPU work if the model or the device cannot serve it."""


def check_model_files(base_model: str) -> None:
    """Fail in seconds if the base model is incomplete.

    A missing shard otherwise surfaces after a long read, or as an opaque error
    from bitsandbytes once the GPU is already allocated.
    """
    if not os.path.isdir(base_model):
        raise ModelSetupError(f"base model directory not found: {base_model}")
    missing = [f for f in REQUIRED_MODEL_FILES
               if not os.path.exists(os.path.join(base_model, f))]
    if missing:
        raise ModelSetupError(
            f"base model {base_model} is missing {missing}; every file listed "
            f"in REQUIRED_MODEL_FILES must be present before training")


def check_device(device: str) -> None:
    """QLoRA on an 8 GiB card needs CUDA; do not fail after loading."""
    if device != "cuda":
        return
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ModelSetupError(f"PyTorch is not importable: {exc}") from exc
    if not torch.cuda.is_available():
        raise ModelSetupError(
            "device='cuda' was requested but CUDA is not available; pass "
            "--device cpu for a preprocessing-only check, which cannot train a "
            "4-bit model on CPU in any practical way")
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if total < 7.0:
        raise ModelSetupError(
            f"GPU reports {total:.1f} GiB; the measured 4-bit Qwen3-8B "
            f"footprint is about 7.1 GiB before LoRA state, so there is no "
            f"headroom on this card")


def load_tokenizer(base_model: str):
    """The runtime's tokenizer, with the runtime's chat template."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # brain.py relies on the pad token existing.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_quantization_config(config: TrainerConfig):
    import torch
    from transformers import BitsAndBytesConfig

    dtype = torch.bfloat16 if config.bnb_4bit_compute_dtype == "bfloat16" \
        else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def load_base_model(config: TrainerConfig, tokenizer):
    """Load the 4-bit base exactly as brain.py does."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=build_quantization_config(config),
        device_map={"": 0},
        trust_remote_code=True,
        dtype=torch.bfloat16 if config.bf16 else torch.float16,
        attn_implementation="sdpa",
    )
    return model


# Tensors PEFT's helper would upcast to fp32 that must NOT be: Qwen3-8B has
# untied embed_tokens and lm_head at 622M parameters each, 1,187 MiB apiece in
# bf16. Upcasting either to fp32 costs ~2,374 MiB, and doing both costs more
# than the whole 8 GiB card. PEFT's helper upcasts them unconditionally.
#
# The second of the two is not merely wasteful: it is load-bearing for dtype
# consistency *if* the norms are upcast, because an fp32 final norm emits fp32
# hidden states that the bf16 lm_head cannot consume. So the upcast is dropped
# as a whole rather than selectively - keeping it only for lm_head would still
# cost 1,187 MiB, and leaving the norms bf16 keeps the hidden states bf16 and
# the dtype chain consistent without it.
def prepare_base_for_lora(model, config: TrainerConfig):
    """Freeze the base and wire up the two things PEFT's helper does that matter.

    PEFT's ``prepare_model_for_kbit_training`` does four things. Three are
    required here; one does not fit.

      1. freeze every base parameter          -> kept
      2. cast non-quantised params to fp32    -> dropped, see note above
      3. make the input embedding output
         require grad so checkpointed blocks
         carry a gradient                     -> kept
      4. enable gradient checkpointing        -> kept

    Ordering note: this runs on the base model *before* ``get_peft_model``.
    Calling it on the wrapped PeftModel raises AttributeError.
    """
    for param in model.parameters():
        param.requires_grad = False

    # Required with a frozen base: the first checkpointed block would otherwise
    # see inputs that require no grad and produce no gradient path.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, args, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(
            _make_inputs_require_grad)

    if config.gradient_checkpointing:
        kwargs = {}
        if "gradient_checkpointing_kwargs" in inspect.signature(
                model.gradient_checkpointing_enable).parameters:
            # use_reentrant=False is required: it composes correctly with
            # frozen base weights and LoRA adapters.
            kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        model.gradient_checkpointing_enable(**kwargs)
    return model


def attach_lora(model, config: TrainerConfig):
    """Wrap the quantised base in the v0.1-shaped LoRA adapter.

    ``prepare_base_for_lora`` must run on the base model *before*
    ``get_peft_model``; calling it on the already-wrapped PeftModel raises
    AttributeError, which is what the first smoke run did.
    """
    from peft import LoraConfig, get_peft_model

    model = prepare_base_for_lora(model, config)

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


class GroupedDecoderLayers(torch.nn.Module):
    """Runs several decoder layers inside a single gradient checkpoint.

    transformers checkpoints each ``GradientCheckpointingLayer`` on its own
    (see ``GradientCheckpointingLayer.__call__``), so a 36-layer model stores
    36 hidden states. At 1,440 tokens x 4,096 hidden x bf16 that is ~425 MiB of
    activations that stay resident until the backward pass. Measured on an
    8 GiB card, the backward pass needs ~1,270 MiB above the 4-bit weights but
    only ~1,073 MiB is obtainable, so the ~425 MiB is the binding constraint.

    Wrapping N consecutive layers in one checkpoint cuts the stored inputs to
    ``36 / N`` tensors (~142 MiB at N=3). The trade is recompute: N layers are
    recomputed together during backward instead of one at a time.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.gradient_checkpointing = True

    def _run(self, hidden_states, **kwargs):
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states

    def forward(self, hidden_states, **kwargs):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._run, hidden_states, **kwargs, use_reentrant=False)
        return self._run(hidden_states, **kwargs)


def resolve_causal_lm(model):
    """Walk PEFT/Trainer wrappers down to the real ``*ForCausalLM``.

    ``PeftModelForCausalLM`` also ends in "ForCausalLM" but its ``.model`` is a
    LoraModel, so the name test has to exclude Peft explicitly. ``.model`` is
    preferred over ``.base_model`` because LoraModel's ``base_model`` proxies
    through to the backbone and would skip the CausalLM entirely.
    """
    current = model
    for _ in range(6):
        name = type(current).__name__
        if name.endswith("ForCausalLM") and not name.startswith("Peft"):
            return current
        nxt = getattr(current, "model", None)
        if not isinstance(nxt, torch.nn.Module) or nxt is current:
            nxt = getattr(current, "base_model", None)
        if not isinstance(nxt, torch.nn.Module) or nxt is current:
            break
        current = nxt
    return None


def enable_grouped_gradient_checkpointing(model, group_size: int = 3) -> list[str]:
    """Replace per-layer checkpointing with one checkpoint per ``group_size`` layers.

    Must run after LoRA is attached, because it re-parents the decoder layers.

    Note this changes the adapter's parameter names from
    ``...layers.7.self_attn.q_proj.lora_A`` to
    ``...layers.3.layers.1.self_attn.q_proj.lora_A``. PEFT matches target
    modules by suffix, so adapters still load, but the keys in
    adapter_model.safetensors are no longer index-aligned with the base model.

    Measured on an 8 GiB card with a 1,440-token record, this does *not* make
    the backward pass fit: stored inputs shrink roughly 1:1 against the larger
    simultaneous recompute, so live memory stays ~7.0 GiB at every group size
    from 1 to 12. It is kept as an available utility, not enabled by default.
    """
    if group_size < 1:
        raise ModelSetupError(f"group_size must be >= 1, got {group_size}")

    causal_lm = resolve_causal_lm(model)
    backbone = getattr(causal_lm, "model", None) if causal_lm is not None else None
    if backbone is None or not hasattr(backbone, "layers"):
        raise ModelSetupError(
            "could not locate the decoder layer list; grouped checkpointing "
            "expects a Qwen3ForCausalLM or PeftModelForCausalLM")
    # A mask is built per layer type and reused across layers, so layers may
    # only be grouped while they share one type. Qwen3-8B has no sliding-window
    # layers, so this holds; refuse rather than silently mix mask semantics.
    layer_types = list(getattr(backbone.config, "layer_types", []) or [])
    layers = list(backbone.layers)
    if layer_types and len(layer_types) == len(layers):
        for start in range(0, len(layers), group_size):
            chunk = layer_types[start:start + group_size]
            if len(set(chunk)) > 1:
                raise ModelSetupError(
                    f"layers {start}-{start + len(chunk) - 1} mix attention "
                    f"types {sorted(set(chunk))}; grouped checkpointing is "
                    "unsupported for sliding-window models")

    # The per-layer wrapper must stand down or every layer would still
    # checkpoint individually and nothing would be saved.
    for layer in layers:
        layer.gradient_checkpointing = False
        layer._gradient_checkpointing_func = None

    groups = [
        GroupedDecoderLayers(layers[i:i + group_size])
        for i in range(0, len(layers), group_size)
    ]
    backbone.layers = torch.nn.ModuleList(groups)
    for group in backbone.layers:
        group.gradient_checkpointing = True

    return [
        f"grouped gradient checkpointing: {len(layers)} layers -> "
        f"{len(groups)} checkpoints of up to {group_size}"
    ]


def verify_adapter_targets(model, config: TrainerConfig) -> list[str]:
    """Confirm every requested target module actually received an adapter."""
    import torch

    names = {n for n, _ in model.named_modules()}
    wanted = list(config.lora_target_modules)
    # names look like base_model.model.model.layers.0.self_attn.q_proj
    found = {w: any(n.endswith(w) for n in names) for w in wanted}
    missing = [w for w, ok in found.items() if not ok]
    if missing:
        raise ModelSetupError(
            f"LoRA target modules not found in the loaded model: {missing}. "
            f"The base architecture may not be the one this config assumes.")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise ModelSetupError("no trainable parameters after attaching LoRA")
    return [
        f"trainable {trainable:,} / {total:,} "
        f"({100 * trainable / total:.3f}%)"
    ]


def load_for_training(config: TrainerConfig):
    """Full setup path: files, device, tokenizer, quantised base, LoRA."""
    check_model_files(config.base_model)
    check_device(config.device)
    tokenizer = load_tokenizer(config.base_model)
    model = load_base_model(config, tokenizer)
    notes = []
    if config.cpu_offload_frozen_head:
        # Must happen before prepare_base_for_lora: enable_input_require_grads
        # registers a hook on the input embedding, and that has to land on the
        # CpuEmbedding that replaces it.
        from aurora.offload import offload_frozen_head
        notes.extend(offload_frozen_head(model, chunk_rows=config.offload_chunk_rows))
    model = attach_lora(model, config)
    notes.extend(verify_adapter_targets(model, config))
    return model, tokenizer, notes


__all__ = [
    "GroupedDecoderLayers",
    "ModelSetupError",
    "attach_lora",
    "enable_grouped_gradient_checkpointing",
    "prepare_base_for_lora",
    "build_quantization_config",
    "check_device",
    "check_model_files",
    "resolve_causal_lm",
    "load_base_model",
    "load_for_training",
    "load_tokenizer",
    "verify_adapter_targets",
]
