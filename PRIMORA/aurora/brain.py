from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import os
import time

from PRIMORA.logutil import emit_block


@dataclass(frozen=True)
class BrainResult:
    content: str
    model: str
    success: bool
    error: str | None = None

    @classmethod
    def success(cls, content: str, model: str) -> "BrainResult":
        return cls(content=content, model=model, success=True)

    @classmethod
    def failure(cls, error: str, model: str = "") -> "BrainResult":
        return cls(content="", model=model, success=False, error=error)


class Brain(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        max_output_tokens: Optional[int] = None,
    ) -> BrainResult:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    def warmup(self) -> Optional[str]:
        return None

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        raise NotImplementedError(
            "This Brain does not provide token counting."
        )


class AuroraBrain(Brain):
    def __init__(self, config: Optional["AuroraConfig"] = None):
        from PRIMORA.aurora.configuration import AuroraConfig

        self._config = config or AuroraConfig.from_env()
        self._model = None
        self._tokenizer = None
        self._initialized = False

        # Hard safety ceiling for the entire generation sequence.
        # This includes input + generated tokens.
        self._max_context_tokens = int(
            os.environ.get(
                "PRIMORA_AURORA_MAX_CONTEXT_TOKENS",
                "1200",
            )
        )

        # Keep some room for an actual answer after context trimming.
        self._minimum_response_tokens = int(
            os.environ.get(
                "PRIMORA_AURORA_MIN_RESPONSE_TOKENS",
                "64",
            )
        )

    def get_model_name(self) -> str:
        return os.path.basename(self._config.model_path)

    def warmup(self) -> Optional[str]:
        return self._initialize()

    def _initialize(self) -> Optional[str]:
        if self._initialized:
            return None

        try:
            import torch
            import transformers
            import peft

            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )
            from peft import PeftModel

            try:
                import bitsandbytes

                bnb_version = getattr(
                    bitsandbytes,
                    "__version__",
                    "unknown",
                )
            except Exception as e:
                bnb_version = f"unavailable: {e}"

        except ImportError as e:
            return f"Missing dependencies: {e}"

        try:
            emit_block([
                "",
                "=" * 70,
                "AURORA INFERENCE",
                "=" * 70,
                "",
                "[PACKAGE VERSIONS]",
                f"PyTorch:       {torch.__version__}",
                f"CUDA runtime:  {torch.version.cuda}",
                f"Transformers:  {transformers.__version__}",
                f"PEFT:          {peft.__version__}",
                f"bitsandbytes:  {bnb_version}",
                "",
                "[CUDA]",
                f"CUDA available: {torch.cuda.is_available()}",
            ] + (
                [
                    f"GPU:            {torch.cuda.get_device_name(0)}",
                    f"VRAM total:     "
                    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.3f} GB",
                ]
                if torch.cuda.is_available()
                else []
            ) + [
                "",
                "[MODEL]",
                f"Model path:     {self._config.model_path}",
                f"Adapter path:   {self._config.adapter_path}",
            ])

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_path,
                trust_remote_code=True,
            )

            # Deterministic all-GPU placement. device_map="auto" lets
            # Accelerate spill modules to CPU/disk based on transient free
            # VRAM, which bitsandbytes 4-bit then rejects. {"": 0} keeps
            # every module on GPU 0; genuine shortage surfaces as CUDA OOM.
            base_model = AutoModelForCausalLM.from_pretrained(
                self._config.model_path,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True,
                dtype=torch.float16,
                attn_implementation="sdpa",
            )

            emit_block([
                f"Attention:      "
                f"{getattr(base_model.config, '_attn_implementation', 'N/A')}"
            ])

            self._model = PeftModel.from_pretrained(
                base_model,
                self._config.adapter_path,
                device_map={"": 0},
            )

            self._model.eval()
            self._initialized = True

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

                allocated = (
                    torch.cuda.memory_allocated() / 1024**3
                )
                reserved = (
                    torch.cuda.memory_reserved() / 1024**3
                )

                emit_block([
                    f"VRAM after load: "
                    f"allocated={allocated:.3f} GB, "
                    f"reserved={reserved:.3f} GB"
                ])

            emit_block([
                f"Context safety ceiling: "
                f"{self._max_context_tokens} total tokens",
                "[AURORA] Model ready.",
                "",
            ])

            return None

        except Exception as e:
            return f"Model initialization failed: {e}"

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Reusable token counter for the Context Manager.

        Requires a warmed-up Brain (loaded tokenizer). Shared as an
        injected callable so Prime never loads a second tokenizer.
        """
        if self._tokenizer is None:
            raise RuntimeError("Brain is not warmed up; tokenizer unavailable.")
        return self._token_count(messages)

    def _token_count(
        self,
        messages: list[dict[str, str]],
    ) -> int:
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        return encoded["input_ids"].shape[1]

    def _prepare_messages(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], int]:
        if not messages:
            return messages, 0

        system_message = None
        history = []

        if messages[0].get("role") == "system":
            system_message = messages[0]
            history = list(messages[1:])
        else:
            history = list(messages)

        # Leave room for a real response.
        max_input_tokens = max(
            1,
            self._max_context_tokens
            - self._minimum_response_tokens,
        )

        if system_message is not None:
            base = [system_message]
        else:
            base = []

        # First check whether the complete context already fits.
        complete = base + history
        complete_tokens = self._token_count(complete)

        if complete_tokens <= max_input_tokens:
            return complete, complete_tokens

        # Keep the newest messages and discard oldest history first.
        selected_reversed = []

        for message in reversed(history):
            candidate = (
                base
                + list(reversed(selected_reversed + [message]))
            )

            candidate_tokens = self._token_count(candidate)

            if candidate_tokens > max_input_tokens:
                break

            selected_reversed.append(message)

        selected = base + list(reversed(selected_reversed))
        selected_tokens = self._token_count(selected)

        print(
            f"[CONTEXT] trimmed: "
            f"{complete_tokens} -> {selected_tokens} tokens"
        )

        return selected, selected_tokens

    def generate(
        self,
        messages: list[dict[str, str]],
        max_output_tokens: Optional[int] = None,
    ) -> BrainResult:
        if not isinstance(messages, list) or not messages:
            return BrainResult.failure(
                "Missing or invalid 'messages'",
                self.get_model_name(),
            )

        init_error = self._initialize()

        if init_error:
            return BrainResult.failure(
                init_error,
                self.get_model_name(),
            )

        try:
            import torch

            prepared_messages, input_tokens = (
                self._prepare_messages(messages)
            )

            if input_tokens >= self._max_context_tokens:
                return BrainResult.failure(
                    "Input context is too large for the current "
                    "Aurora VRAM safety limit.",
                    self.get_model_name(),
                )

            prompt_start = time.perf_counter()

            prompt = self._tokenizer.apply_chat_template(
                prepared_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            prompt_time = (
                time.perf_counter() - prompt_start
            )

            token_start = time.perf_counter()

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self._model.device)

            tokenization_time = (
                time.perf_counter() - token_start
            )

            input_tokens = inputs["input_ids"].shape[1]

            # Never allow total sequence length to exceed the safety limit.
            available_output_tokens = (
                self._max_context_tokens - input_tokens
            )

            generation_tokens = min(
                self._config.max_new_tokens,
                available_output_tokens,
            )

            if max_output_tokens is not None:
                generation_tokens = min(
                    generation_tokens,
                    max(0, max_output_tokens),
                )

            if generation_tokens < 1:
                return BrainResult.failure(
                    "No generation capacity remains inside "
                    "Aurora's current context safety limit.",
                    self.get_model_name(),
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                allocated_before = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )
                reserved_before = (
                    torch.cuda.memory_reserved()
                    / 1024**3
                )

            generation_start = time.perf_counter()

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=generation_tokens,
                    do_sample=True,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    repetition_penalty=self._config.repetition_penalty,
                    use_cache=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            generation_time = (
                time.perf_counter() - generation_start
            )

            if torch.cuda.is_available():
                peak_allocated = (
                    torch.cuda.max_memory_allocated()
                    / 1024**3
                )
                peak_reserved = (
                    torch.cuda.max_memory_reserved()
                    / 1024**3
                )

                allocated_after = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )
                reserved_after = (
                    torch.cuda.memory_reserved()
                    / 1024**3
                )

            generated_ids = outputs[0][
                inputs["input_ids"].shape[1]:
            ]

            output_tokens = generated_ids.shape[0]

            response = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            if not response.strip():
                return BrainResult.failure(
                    "Model returned empty response",
                    self.get_model_name(),
                )

            total_time = (
                prompt_time
                + tokenization_time
                + generation_time
            )

            tok_per_sec = (
                output_tokens / generation_time
                if generation_time > 0
                else 0
            )

            block = [
                "",
                "-" * 70,
                "AURORA GENERATION",
                "-" * 70,
                f"Input tokens:       {input_tokens}",
                f"Output tokens:      {output_tokens}",
                f"Generation budget:  {generation_tokens}",
                f"Prompt build:       {prompt_time:.3f}s",
                f"Tokenization:       {tokenization_time:.3f}s",
                f"Generation:         {generation_time:.3f}s",
                f"Total:              {total_time:.3f}s",
                f"Generation speed:   {tok_per_sec:.2f} tok/s",
            ]

            if torch.cuda.is_available():
                block += [
                    "",
                    "[VRAM]",
                    f"Allocated before:   {allocated_before:.3f} GB",
                    f"Reserved before:    {reserved_before:.3f} GB",
                    f"Peak allocated:     {peak_allocated:.3f} GB",
                    f"Peak reserved:      {peak_reserved:.3f} GB",
                    f"Allocated after:    {allocated_after:.3f} GB",
                    f"Reserved after:     {reserved_after:.3f} GB",
                ]

            block += ["-" * 70, ""]
            emit_block(block)

            return BrainResult.success(
                response.strip(),
                self.get_model_name(),
            )

        except Exception as e:
            return BrainResult.failure(
                f"Generation error: {e}",
                self.get_model_name(),
            )
