from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import json
import os
import re
import time
import urllib.error
import urllib.request

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


def aurora_backend_from_env(env=None) -> str:
    """Select Aurora's generation backend from PRIMORA_AURORA_BACKEND.

    ``transformers`` (the default, for an absent or empty value) is the
    local HuggingFace + bitsandbytes path that loads the configured model
    and QLoRA adapter. ``openai`` (alias ``llamacpp``) generates through
    the OpenAI-compatible ``/chat/completions`` endpoint served by
    llama.cpp llama-server and never loads the HF model, adapter, or
    tokenizer. An explicit but unsupported value fails closed
    (ValueError) rather than silently substituting a backend.
    """
    source = os.environ if env is None else env
    choice = str(source.get("PRIMORA_AURORA_BACKEND") or "").strip().lower()
    if choice in ("", "transformers"):
        return "transformers"
    if choice in ("openai", "llamacpp"):
        return "openai"
    raise ValueError(
        f"unsupported PRIMORA_AURORA_BACKEND={choice!r}; expected "
        "'transformers' or 'openai'"
    )


class AuroraBrain(Brain):
    def __init__(self, config: Optional["AuroraConfig"] = None):
        from PRIMORA.aurora.configuration import AuroraConfig

        self._config = config or AuroraConfig.from_env()
        self._model = None
        self._tokenizer = None
        self._initialized = False

        # Hard safety ceiling for the entire generation sequence: input plus
        # generated tokens.
        #
        # Derived from AuroraConfig.context_total so the two can never
        # disagree: a ceiling below it silently trims a context the budget
        # believed would fit, and a ceiling above it asks this 8 GB card for
        # more than it can hold, which is how a long turn turned into
        # "CUDA out of memory, 0 bytes free" instead of a slow answer.
        # PRIMORA_AURORA_MAX_CONTEXT_TOKENS still wins when an operator sets
        # it deliberately.
        self._max_context_tokens = int(
            os.environ.get(
                "PRIMORA_AURORA_MAX_CONTEXT_TOKENS",
                str(self._config.context_total),
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
            # Reduce allocator fragmentation. With the model occupying 88% of
            # VRAM the allocator is the difference between a long context
            # running and failing with "0 bytes free", and PyTorch reads this
            # when the CUDA allocator initialises, so it must be set before
            # the model is loaded. setdefault, so an operator can override it.
            os.environ.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            )

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


# Qwen3-style thinking fences. The transformers path disables thinking via
# the chat template (enable_thinking=False); an external server applies its
# own template, so any thinking trace it emits has to be removed here before
# the text reaches the loop.
_THINKING_FENCE_RE = re.compile(r"```think.*?```", re.DOTALL | re.IGNORECASE)


class OpenAIBrain(Brain):
    """Aurora's generation brain over an OpenAI-compatible HTTP endpoint.

    Selected with ``PRIMORA_AURORA_BACKEND=openai`` (alias ``llamacpp``).
    Generates through ``{base_url}/chat/completions`` - llama.cpp
    llama-server by default - and never loads the HF model, adapter, or
    tokenizer. Generation settings map 1:1 from ``AuroraConfig``
    (``PRIMORA_AURORA_MAX_TOKENS`` / ``_TEMPERATURE`` / ``_TOP_P`` /
    ``_REPETITION_PENALTY``); the endpoint comes from the same variables
    the Prime OpenAI provider reads (``PRIMORA_OPENAI_URL`` /
    ``PRIMORA_OPENAI_MODEL`` / ``PRIMORA_OPENAI_TIMEOUT``). Messages
    arrive fully assembled - system prompt included - from Aurora's
    instruction builder, identical to the transformers path.
    Construction performs no network I/O.
    """

    def __init__(
        self,
        config: Optional["AuroraConfig"] = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        from PRIMORA.aurora.configuration import AuroraConfig

        self._config = config or AuroraConfig.from_env()
        self._base_url = base_url or os.environ.get(
            "PRIMORA_OPENAI_URL", "http://localhost:8080/v1")
        self._model = model or os.environ.get(
            "PRIMORA_OPENAI_MODEL", "llama-server")
        self._timeout = timeout or int(
            os.environ.get("PRIMORA_OPENAI_TIMEOUT", "120"))

    def get_model_name(self) -> str:
        return self._model

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Heuristic count; this backend has no local tokenizer.

        Exact counting needs the HF tokenizer, which this backend
        deliberately does not load. Uses the same character estimate the
        ContextManager falls back to, so budgeting stays consistent and
        the selection path never degrades with a counter failure.
        """
        from PRIMORA.prime.context.manager import estimate_tokens

        return sum(
            estimate_tokens(m.get("content", "")) + 4 for m in messages
        )

    @staticmethod
    def _strip_thinking(content: str) -> str:
        content = _THINKING_FENCE_RE.sub("", content)
        return content.replace("```", "").strip()

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

        generation_tokens = self._config.max_new_tokens
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

        # Qwen3-class chat templates think by default. On llama-server with
        # --jinja the reasoning lands in reasoning_content, and a thinking
        # budget that consumes max_tokens leaves content empty - which the
        # loop reports as "Model returned empty response". Disable thinking
        # by default, matching the transformers path's enable_thinking=False;
        # an operator can turn it back on with PRIMORA_OPENAI_ENABLE_THINKING=1.
        _thinking = str(
            os.environ.get("PRIMORA_OPENAI_ENABLE_THINKING", "0")
        ).strip().lower()
        enable_thinking = _thinking in ("1", "true", "yes", "on")

        request_body: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "max_tokens": generation_tokens,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            # llama-server accepts repeat_penalty as an extension on its
            # OpenAI-compatible endpoint; strict OpenAI servers ignore
            # the extra key rather than rejecting the request.
            "repeat_penalty": self._config.repetition_penalty,
            # Passed through to the Jinja chat template. OpenAI-compatible
            # servers that do not understand the key ignore it.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }

        url = f"{self._base_url.rstrip('/')}/chat/completions"

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status != 200:
                    return BrainResult.failure(
                        f"OpenAI HTTP {resp.status}: "
                        f"{resp.read().decode('utf-8')}",
                        self.get_model_name(),
                    )
                response_data = json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            return BrainResult.failure(
                f"OpenAI HTTP error {e.code}: {e.read().decode('utf-8')}",
                self.get_model_name(),
            )
        except urllib.error.URLError as e:
            return BrainResult.failure(
                f"OpenAI connection error: {e.reason}",
                self.get_model_name(),
            )
        except json.JSONDecodeError as e:
            return BrainResult.failure(
                f"OpenAI malformed response: {e}",
                self.get_model_name(),
            )
        except TimeoutError:
            return BrainResult.failure(
                f"OpenAI timeout after {self._timeout}s",
                self.get_model_name(),
            )
        except Exception as e:
            return BrainResult.failure(
                f"OpenAI unexpected error: {e}",
                self.get_model_name(),
            )

        try:
            choices = response_data.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content") or ""
        except Exception as e:
            return BrainResult.failure(
                f"OpenAI response parse error: {e}",
                self.get_model_name(),
            )

        content = self._strip_thinking(content)

        if not content:
            # An empty body is almost always a thinking budget that ate the
            # whole max_tokens. Report enough of the response shape for the
            # failure to be diagnosable rather than a bare "empty response".
            choice0 = choices[0] if choices else {}
            finish = choice0.get("finish_reason")
            reasoning = (
                message.get("reasoning_content")
                or message.get("reasoning")
                or ""
            )
            detail = (
                f"finish_reason={finish!r}, "
                f"reasoning_chars={len(reasoning)}, "
                f"enable_thinking={enable_thinking}"
            )
            return BrainResult.failure(
                f"Model returned empty response ({detail})",
                self.get_model_name(),
            )

        return BrainResult.success(content, self.get_model_name())
