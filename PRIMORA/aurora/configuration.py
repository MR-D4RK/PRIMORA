import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_SYSTEM_INSTRUCTION = (
    "You are Aurora. Prime is your technical execution system: when a task requires "
    "real-world action, computation, or live information beyond conversation, you decide "
    "WHAT should happen and request it from Prime; Prime decides HOW to technically "
    "execute it, including planning and breaking the request into technical steps. "
    "When technical execution is required, emit exactly ONE request block:\n"
    "<prime_request>\n"
    '{"action":"...", "payload": {...}, "context": null, "metadata": null}\n'
    "</prime_request>\n"
    "The request may describe a single goal or multiple related goals in one payload. "
    "Never perform Prime's work yourself and never claim that you did. "
    "If no technical execution is required, answer normally and emit no request block. "
    "Never emit more than one <prime_request> block in a single response."
)


def build_capability_instruction(prime) -> str:
    """Build the capability awareness section for Aurora's system instruction."""
    if not hasattr(prime, 'registry'):
        return ""
    
    manifest = prime.registry.get_manifest()
    if not manifest:
        return ""
    
    lines = ["\nAvailable Prime Capabilities:"]
    for name, spec in manifest.items():
        lines.append(f"- {name}: {spec['description']}")
        if spec.get('required_fields'):
            lines.append(f"  Required payload fields: {', '.join(spec['required_fields'])}")
        if spec.get('optional_fields'):
            lines.append(f"  Optional payload fields: {', '.join(spec['optional_fields'])}")
    
    lines.append("\nWhen technical execution is needed, select the appropriate capability from the list above.")
    lines.append("Use the exact canonical name (e.g., 'calculate') in your prime_request action field.")
    lines.append("The payload must match the required fields for that capability.")
    
    return "\n".join(lines)


@dataclass(frozen=True)
class AuroraConfig:
    model_path: str
    adapter_path: str
    max_new_tokens: int = 256
    temperature: float = 0.75
    top_p: float = 0.9
    repetition_penalty: float = 1.08
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION
    context_target: int = 1000
    context_ceiling: int = 1100
    context_output_reserve: int = 128
    context_total: int = 1228

    @classmethod
    def from_env(cls) -> "AuroraConfig":
        root = get_project_root()
        default_model = str(root / "RESOURCES" / "aurora" / "v0.1" / "model")
        default_adapter = str(root / "RESOURCES" / "aurora" / "v0.1" / "aurora_qlora")

        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except (TypeError, ValueError):
                return default

        return cls(
            model_path=os.environ.get("PRIMORA_AURORA_MODEL_PATH", default_model),
            adapter_path=os.environ.get("PRIMORA_AURORA_ADAPTER_PATH", default_adapter),
            max_new_tokens=int(os.environ.get("PRIMORA_AURORA_MAX_TOKENS", "256")),
            temperature=float(os.environ.get("PRIMORA_AURORA_TEMPERATURE", "0.75")),
            top_p=float(os.environ.get("PRIMORA_AURORA_TOP_P", "0.9")),
            repetition_penalty=float(os.environ.get("PRIMORA_AURORA_REPETITION_PENALTY", "1.08")),
            system_instruction=os.environ.get("PRIMORA_AURORA_SYSTEM_INSTRUCTION", DEFAULT_SYSTEM_INSTRUCTION),
            context_target=_int("PRIMORA_AURORA_CONTEXT_TARGET", 1000),
            context_ceiling=_int("PRIMORA_AURORA_CONTEXT_CEILING", 1100),
            context_output_reserve=_int("PRIMORA_AURORA_CONTEXT_OUTPUT_RESERVE", 128),
            context_total=_int("PRIMORA_AURORA_CONTEXT_TOTAL", 1228)
        )

    def validate(self) -> Optional[str]:
        if not os.path.isdir(self.model_path):
            return f"Model path not found: {self.model_path}"
        if not os.path.isdir(self.adapter_path):
            return f"Adapter path not found: {self.adapter_path}"
        return None