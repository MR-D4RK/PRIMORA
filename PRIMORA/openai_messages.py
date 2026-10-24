"""Provider-neutral OpenAI chat message wire-format validation.

Some compatible chat templates allow a system turn only at the beginning.
The conversation builders may add later evidence/context system turns; keep
those instructions but send them as one leading system turn on the wire.
This is transport formatting, not natural-language or memory routing.
"""

from typing import Any


def leading_system_message(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Consolidate system messages without changing non-system turn ordering.

    Never mutate input messages or guess how to flatten structured system
    content. Unsupported system content raises a clear validation error.
    """
    system = [message for message in messages if message.get("role") == "system"]
    if not system or (len(system) == 1 and messages[0] is system[0]):
        return messages
    if any(not isinstance(message.get("content"), str) for message in system):
        raise ValueError("System message content must be text for this chat template")

    content = "\n\n".join(message["content"] for message in system if message["content"].strip())
    history = [message for message in messages if message.get("role") != "system"]
    return ([{"role": "system", "content": content}] if content else []) + history
