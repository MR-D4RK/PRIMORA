"""The specialist catalogue: which instruments Aurora may consult.

Milestone 3.

Naming, because it matters here
-------------------------------
``PRIMORA.specialists.specialist.Specialist`` is a *Prime capability handler*
--- the object that answers for an action like ``calculate``. A *consultable
specialist* is something else entirely: a model Aurora can hand a hard
sub-problem to. They are different concepts that share a word, so this module
uses ``SpecialistSpec`` / ``SpecialistRegistry`` and never imports the other
one.

What a specialist is
--------------------
An instrument. Not an agent.

* no identity, no name of its own, no goals
* no memory, no session, no conversation with Amir
* no ability to act, call tools, or continue anything
* no lifecycle: Aurora sends a task, receives a string, and that is the end of
  the interaction

The spec carries no field that could hold any of that, and that is enforced by
test rather than by convention.

A specialist is selected by name from a replaceable catalogue and executed
through the existing :class:`PRIMORA.prime.providers.Provider`, so both the
catalogue and the model backend can change without touching Aurora.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class SpecialistSpec:
    """One consultable specialist.

    Frozen and field-limited on purpose. There is nowhere to put a goal, a
    memory handle, or a session, so an instrument cannot quietly grow into an
    agent.
    """

    name: str
    description: str
    model: str = ""
    system_prompt: str = ""
    available: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("specialist name must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("specialist description must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


class SpecialistRegistry:
    """The replaceable catalogue of consultable specialists.

    Empty by default. A capability that can reach a model nobody configured
    would be a surprise at runtime and an unpredictable cost, so the honest
    default is an empty catalogue and a clear failure.
    """

    def __init__(self, specs: Optional[list[SpecialistSpec]] = None) -> None:
        self._specs: dict[str, SpecialistSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: SpecialistSpec) -> None:
        if not isinstance(spec, SpecialistSpec):
            raise TypeError("register expects a SpecialistSpec")
        if spec.name in self._specs:
            raise ValueError(f"Specialist already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> Optional[SpecialistSpec]:
        if not isinstance(name, str):
            return None
        return self._specs.get(name.strip())

    def has(self, name: str) -> bool:
        return self.get(name) is not None

    def names(self) -> list[str]:
        return list(self._specs)

    def available_names(self) -> list[str]:
        return [n for n, s in self._specs.items() if s.available]

    def describe(self) -> list[dict[str, str]]:
        """What Aurora is told exists. No models, no paths, no transport."""
        return [s.to_dict() for s in self._specs.values() if s.available]

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._specs

    def __iter__(self) -> Iterator[SpecialistSpec]:
        return iter(self._specs.values())

    @classmethod
    def from_env(cls, prefix: str = "PRIMORA_SPECIALIST_") -> "SpecialistRegistry":
        """Build a catalogue from environment variables.

        ``PRIMORA_SPECIALIST_REASONING_MODEL`` registers a specialist named
        ``reasoning`` on that model. An optional
        ``PRIMORA_SPECIALIST_REASONING_DESCRIPTION`` supplies its description.
        Enabling an instrument is an operator decision, so it is configuration
        rather than code.
        """
        registry = cls()
        seen: set[str] = set()
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix):]
            for suffix in ("_MODEL", "_DESCRIPTION"):
                if remainder.endswith(suffix):
                    seen.add(remainder[: -len(suffix)].lower())
                    break
        for name in sorted(seen):
            model = os.environ.get(f"{prefix}{name.upper()}_MODEL", "").strip()
            description = os.environ.get(
                f"{prefix}{name.upper()}_DESCRIPTION", ""
            ).strip()
            registry.register(
                SpecialistSpec(
                    name=name,
                    description=description or f"{name} specialist",
                    model=model,
                )
            )
        return registry
