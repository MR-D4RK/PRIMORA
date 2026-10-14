"""Registered tools: the things Aurora may ask for by name.

Milestone 3, arbitrary tool calling.

Naming
------
``PRIMORA.specialists.tools`` holds *capability handlers*; this package holds
the *tools those capabilities may invoke*. They are different things, so they
deliberately do not share a package name.

The shape
---------
A tool is a registered, named operation with a declared input schema, an
availability flag, and a handler. Aurora may only ask for a tool that is
registered and available, and only with arguments that validate against its
schema. That is the entire security model:

* there is no ``execute_anything``, no code path from a model to Python
* there is no shell, and no tool here reaches the filesystem or the network
* a tool that is not registered does not exist as far as Aurora is concerned

The division of responsibility is unchanged from every other capability:

    Aurora chooses WHICH available tool she wants.
    Prime decides WHETHER it may run.
    The tool decides HOW the operation is performed.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

# Nothing a tool returns is unbounded. A tool that wants more must ask.
MAX_OUTPUT_CHARS = 2000
MAX_ARGUMENT_CHARS = 4000
MAX_TOOLS = 64


@dataclass(frozen=True)
class ToolResult:
    """What a tool returns. Truncated on the way out, never on the way in."""

    ok: bool
    output: Any = None
    error: Optional[str] = None

    @classmethod
    def success(cls, output: Any = None) -> "ToolResult":
        return cls(True, output, None)

    @classmethod
    def failure(cls, error: str) -> "ToolResult":
        return cls(False, None, error)

    def rendered(self, limit: int = MAX_OUTPUT_CHARS) -> dict[str, Any]:
        """Bounded, JSON-safe form handed back to Aurora."""
        body = self.output
        if isinstance(body, str) and len(body) > limit:
            body = body[:limit] + "...[truncated]"
        elif body is not None:
            try:
                text = json.dumps(body, default=str)
            except Exception:
                text = str(body)
            if len(text) > limit:
                body = text[:limit] + "...[truncated]"
        return {
            "ok": self.ok,
            "output": body,
            "error": self.error,
        }


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool.

    Frozen, and field-limited on purpose. There is nowhere to put a shell
    command, a module path, or a callable that would be invoked as code, so
    a tool cannot grow into a general execution escape hatch.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Optional[Callable[[dict[str, Any]], ToolResult]] = None
    available: bool = True
    max_output_chars: int = MAX_OUTPUT_CHARS

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("tool description must be a non-empty string")
        if not isinstance(self.input_schema, dict):
            raise ValueError("input_schema must be an object")

    @property
    def type(self) -> str:
        return self.input_schema.get("type", "object")

    @property
    def required(self) -> list[str]:
        return list(self.input_schema.get("required", []))

    def properties(self) -> dict[str, Any]:
        return dict(self.input_schema.get("properties", {}))

    def to_dict(self) -> dict[str, Any]:
        """What Aurora is told. No handler, no paths, no Python."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """The allowlist. Only what is here can be executed.

    Empty by default: no tool is reachable until one is registered, so a
    missing configuration is an obvious failure rather than a surprise.
    """

    def __init__(self, specs: Optional[list[ToolSpec]] = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if not isinstance(spec, ToolSpec):
            raise TypeError("register expects a ToolSpec")
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        if len(self._tools) >= MAX_TOOLS:
            raise ValueError(f"Tool registry is full (max {MAX_TOOLS})")
        self._tools[spec.name] = spec

    def get(self, name: Any) -> Optional[ToolSpec]:
        if not isinstance(name, str):
            return None
        return self._tools.get(name.strip())

    def has(self, name: Any) -> bool:
        return self.get(name) is not None

    def names(self) -> list[str]:
        return list(self._tools)

    def available_names(self) -> list[str]:
        return [n for n, s in self._tools.items() if s.available]

    def describe(self) -> list[dict[str, Any]]:
        """What Aurora is offered: available tools only."""
        return [s.to_dict() for s in self._tools.values() if s.available]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._tools.values())

    @classmethod
    def with_builtins(cls) -> "ToolRegistry":
        """A registry seeded with the small, safe built-ins."""
        from PRIMORA.toolbox.builtin import default_tools

        return cls(default_tools())
