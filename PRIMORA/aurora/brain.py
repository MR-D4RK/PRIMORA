from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import os


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
    def generate(self, messages: list[dict[str, str]]) -> BrainResult:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    def warmup(self) -> Optional[str]:
        return None


class AuroraBrain(Brain):
    def __init__(self, config: Optional["AuroraConfig"] = None):
        from PRIMORA.aurora.configuration import AuroraConfig
        self._config = config or AuroraConfig.from_env()
        self._model = None
        self._tokenizer = None
        self._initialized = False

    def get_model_name(self) -> str:
        return os.path.basename(self._config.model_path)

    def warmup(self) -> Optional[str]:
        return self._initialize()

    def _initialize(self) -> Optional[str]:
        if self._initialized:
            return None
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel
        except ImportError as e:
            return f"Missing dependencies: {e}"

        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_path,
                trust_remote_code=True
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                self._config.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

            self._model = PeftModel.from_pretrained(
                base_model,
                self._config.adapter_path,
                device_map="auto"
            )

            self._model.eval()
            self._initialized = True
            return None

        except Exception as e:
            return f"Model initialization failed: {e}"

    def generate(self, messages: list[dict[str, str]]) -> BrainResult:
        if not isinstance(messages, list):
            return BrainResult.failure("Missing or invalid 'messages'", self.get_model_name())

        init_error = self._initialize()
        if init_error:
            return BrainResult.failure(init_error, self.get_model_name())

        try:
            import torch

            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_new_tokens,
                    do_sample=True,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    repetition_penalty=self._config.repetition_penalty,
                    use_cache=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            if not response.strip():
                return BrainResult.failure("Model returned empty response", self.get_model_name())

            return BrainResult.success(response.strip(), self.get_model_name())

        except Exception as e:
            return BrainResult.failure(f"Generation error: {e}", self.get_model_name())