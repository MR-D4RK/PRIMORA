"""The built-in tools.

Small, local, deterministic, and deliberately boring. They exist to prove the
registry and the call path work end to end, and to be safe enough to ship.

Every one of these:

* runs entirely in-process, with no I/O
* reads no file, opens no socket, and spawns nothing
* cannot be turned into code execution by any argument
* is pure, so the same arguments always give the same result

A tool that genuinely needs the network or the filesystem would be a different
kind of object with its own authorization, and is deliberately not here.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any

from .base import ToolResult, ToolSpec

# Unit conversion factors, relative to a base unit. A fixed table, not a
# formula interpreter, so there is nothing to inject into.
_UNITS = {
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
    "g": 1.0, "kg": 1000.0,
    "s": 1.0, "min": 60.0, "h": 3600.0, "day": 86400.0,
    "b": 1.0, "kb": 1024.0, "mb": 1048576.0, "gb": 1073741824.0,
}

_LENGTH = {"mm", "cm", "m", "km"}
_MASS = {"g", "kg"}
_TIME = {"s", "min", "h", "day"}
_DATA = {"b", "kb", "mb", "gb"}

_FAMILIES = (_LENGTH, _MASS, _TIME, _DATA)


def _convert(arguments: dict[str, Any]) -> ToolResult:
    value = arguments.get("value")
    source = str(arguments.get("from_unit", "")).strip().lower()
    target = str(arguments.get("to_unit", "")).strip().lower()
    if source not in _UNITS or target not in _UNITS:
        return ToolResult.failure(
            f"unknown unit: {source!r} or {target!r}. "
            f"Known: {', '.join(sorted(_UNITS))}"
        )
    family = next(
        (f for f in _FAMILIES if source in f and target in f), None
    )
    if family is None:
        return ToolResult.failure(
            f"cannot convert {source!r} to {target!r}: different quantities"
        )
    return ToolResult.success(
        {"value": round(value * _UNITS[source] / _UNITS[target], 10),
         "from": source, "to": target}
    )


def _describe_text(arguments: dict[str, Any]) -> ToolResult:
    text = arguments.get("text", "")
    words = re.findall(r"\S+", text)
    lines = text.splitlines() or [""]
    return ToolResult.success({
        "characters": len(text),
        "words": len(words),
        "lines": len(lines),
        "unique_words": len({w.lower() for w in words}),
    })


def _now(arguments: dict[str, Any]) -> ToolResult:
    moment = time.time()
    return ToolResult.success({
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(moment)),
        "epoch": round(moment, 3),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(moment)),
    })


def _word_count(arguments: dict[str, Any]) -> ToolResult:
    text = unicodedata.normalize("NFKC", arguments.get("text", ""))
    return ToolResult.success({"words": len(re.findall(r"\S+", text))})


def default_tools() -> list[ToolSpec]:
    """The shipped set. Add tools here, or register your own."""
    return [
        ToolSpec(
            name="convert_units",
            description=(
                "Convert a number between units of length, mass, time, or data. "
                "Use when a measurement needs restating in other units."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "The number to convert"},
                    "from_unit": {"type": "string",
                                  "description": "Unit to convert from, e.g. kg"},
                    "to_unit": {"type": "string",
                                "description": "Unit to convert to, e.g. g"},
                },
                "required": ["value", "from_unit", "to_unit"],
                "additionalProperties": False,
            },
            handler=_convert,
        ),
        ToolSpec(
            name="describe_text",
            description=(
                "Report size statistics for a piece of text: characters, "
                "words, lines, and distinct words."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to measure"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=_describe_text,
        ),
        ToolSpec(
            name="current_time",
            description="Report the current date and time.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler=_now,
        ),
        ToolSpec(
            name="count_words",
            description="Count the words in a piece of text.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to count"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=_word_count,
        ),
    ]
