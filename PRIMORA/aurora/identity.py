"""Aurora's runtime identity: who she is, and what she runs on.

The system instruction is a static string, but the identity it has to carry is
not: the base model and the adapter are resolved per run from ``AuroraConfig``
(``PRIMORA_AURORA_MODEL_PATH`` / ``PRIMORA_AURORA_ADAPTER_PATH``), so a
version hardcoded anywhere would advertise a release the runtime may not be
using. This module derives the identity facts from the live configuration and
renders them as an instruction block, so the prompt always describes the
configuration that actually produced it.

Only what the configuration establishes is stated: the project, the two
configured paths, and any version marker those paths carry. Nothing is
inferred from conversational memory, no architecture or capability is
described, and missing configuration is reported as missing rather than
filled in.
"""

import os
import re
from typing import Any, Optional

# Header of the block. Doubles as the idempotence marker: if a caller passes a
# system message that already carries it, it is not appended a second time.
IDENTITY_HEADER = "IDENTITY AND VERSION"

# A version marker as it appears in a directory name, e.g. ``v1`` or ``vX.Y``.
_VERSION_COMPONENT = re.compile(r"^v\d+(?:\.\d+)*$", re.IGNORECASE)


def _components(path: Optional[str]) -> list[str]:
    """Split a configured path into non-empty components, robustly."""
    if not path:
        return []
    parts = re.split(r"[\\/]+", str(path))
    return [p for p in parts if p and not p.endswith(":")]


def path_version(path: Optional[str]) -> Optional[str]:
    """Rightmost version-marker component of a configured path, if any.

    ``.../aurora/<release>/aurora_qlora`` -> ``<release>`` when the release
    directory carries a version marker; a path with no version marker returns
    None, and the caller reports that honestly rather than guessing a release.
    """
    for part in reversed(_components(path)):
        if _VERSION_COMPONENT.match(part):
            return part
    return None


def path_label(path: Optional[str]) -> str:
    """Last two components of a path, so the marker stays visible.

    ``.../aurora/<release>/aurora_qlora`` -> ``<release>/aurora_qlora``. An
    absolute path is deliberately not shown: the prompt needs the identity,
    not the filesystem layout.
    """
    parts = _components(path)
    if not parts:
        return ""
    return "/".join(parts[-2:])


def resolve_identity(config: Any) -> dict:
    """The identity facts this runtime's configuration establishes."""
    model_path = getattr(config, "model_path", "") or ""
    adapter_path = getattr(config, "adapter_path", "") or ""
    model_version = path_version(model_path)
    return {
        "model": {
            "path": str(model_path),
            "label": path_label(model_path),
            "version": model_version,
            "exists": bool(model_path) and os.path.isdir(str(model_path)),
        },
        "adapter": {
            "configured": bool(str(adapter_path).strip()),
            "path": str(adapter_path),
            "label": path_label(adapter_path),
            "version": path_version(adapter_path),
            "exists": bool(adapter_path) and os.path.isdir(str(adapter_path)),
        },
    }


def _model_sentence(model: dict) -> str:
    """No trailing period: the adapter clause joins the same sentence."""
    sentence = f"Your brain runs on the configured base model '{model['label']}'"
    if model["version"]:
        sentence += f", version {model['version']}"
    else:
        sentence += ", which carries no version marker"
    if not model["exists"]:
        sentence += "; its path is not present on disk"
    return sentence


def _adapter_sentence(adapter: dict) -> str:
    if not adapter["configured"]:
        return ("; no adapter is configured for this run, so you run on the "
                "base model alone")
    sentence = f", with a separately configured adapter '{adapter['label']}'"
    if adapter["version"]:
        sentence += f", version {adapter['version']}"
    else:
        sentence += ", which carries no version marker"
    if not adapter["exists"]:
        sentence += ", whose path is not present on disk"
    return sentence


# Every word here is load-bearing: what to state (configuration), what to
# attribute (Amir's claims), what never to invent, and how to answer. It stays
# this short because every token competes with conversation history for the
# measured input ceiling. The middle clause exists because retrieval injects
# this session's own stored replies beside this block: pre-identity turns
# answered version questions with "no version recorded", and a prompt that
# only says "state these facts" lost to the model's consistency with its own
# prior replies. Configuration has to outrank conversation explicitly.
_CLAIMS_PARAGRAPH = (
    "These are configuration facts, not memories: state them, and never deny "
    "your identity, PRIMORA, or a configured model or adapter version - "
    "neither when absent from conversation nor when retrieved history or "
    "your own earlier replies say you have none; those predate this "
    "configuration. A project or version fact Amir "
    "establishes is his claim - keep it distinct from configuration and "
    "unsupported claims; no statement is automatically true. Do not invent "
    "architecture, capabilities, memories you do not hold, or actions you "
    "have not performed. Answer identity questions directly, in the first "
    "person."
)


def build_identity_instruction(config: Any) -> str:
    """The identity block for the system prompt, built from live config.

    Always renders the block; the caller decides placement, and skips the
    append when the system content already carries the header (a caller that
    echoes its history back must not get a second copy).
    """
    facts = resolve_identity(config)
    body = [
        "You are Aurora, the conversational identity in PRIMORA. PRIMORA is "
        "the project this runtime belongs to; Prime is the orchestration and "
        "tool-execution layer you use.",
        _model_sentence(facts["model"]) + _adapter_sentence(facts["adapter"]) + ".",
        _CLAIMS_PARAGRAPH,
    ]
    return "\n" + IDENTITY_HEADER + "\n" + "\n".join(body) + "\n"
