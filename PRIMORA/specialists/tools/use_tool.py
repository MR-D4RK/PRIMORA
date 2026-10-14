"""The ``use_tool`` capability: Aurora choosing a registered tool.

Milestone 3, arbitrary tool calling.

    Aurora chooses WHICH available tool she wants.
    Prime decides WHETHER it may run.
    The tool decides HOW the operation is performed.

What Aurora sends is a name and an object of arguments. Nothing else reaches
the tool: no Python, no class, no module path, no shell, no filesystem
location. The registry is an allowlist, so a name that is not registered does
not exist, and arguments are validated against the tool's own declared schema
*before* the handler is touched.

There is deliberately no "execute anything" tool here. Every operation is
something someone registered, with a schema, on purpose.
"""

from typing import Any, Optional

from PRIMORA.prime.contract import PrimeResult
from PRIMORA.specialists.specialist import Specialist
from PRIMORA.toolbox.base import MAX_ARGUMENT_CHARS, ToolRegistry, ToolResult
from PRIMORA.toolbox.validation import validate


class UseToolSpecialist(Specialist):
    """Run one registered tool with validated arguments."""

    name = "use_tool"

    def __init__(self, tools: Optional[ToolRegistry] = None) -> None:
        self._tools = tools or ToolRegistry()

    def set_tools(self, tools: ToolRegistry) -> None:
        if tools is None:
            raise ValueError("tools cannot be None")
        self._tools = tools

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def is_configured(self) -> bool:
        """Whether a real allowlist of tools is wired.

        An empty registry fails closed on every request (``Unknown
        tool... Available: none registered``), so advertising the
        capability would only invite a call that is guaranteed to fail.
        The capability instruction reads this and omits the entry,
        exactly as ``research`` and ``learn`` do when unconfigured.
        """
        return bool(self._tools.names())

    def handle(self, payload: dict[str, Any]) -> PrimeResult:
        if not isinstance(payload, dict):
            return PrimeResult.failure("Invalid payload type")

        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            return PrimeResult.failure("Missing or invalid 'name' in payload")

        spec = self._tools.get(name)
        if spec is None:
            known = ", ".join(self._tools.names()) or "none registered"
            return PrimeResult.failure(
                f"Unknown tool: {name.strip()}. Available: {known}"
            )
        if not spec.available:
            return PrimeResult.failure(f"Tool is unavailable: {spec.name}")

        raw_arguments = payload.get("arguments", {})
        if isinstance(raw_arguments, str):
            return PrimeResult.failure(
                "'arguments' must be an object, not a string"
            )
        try:
            if len(json_size(raw_arguments)) > MAX_ARGUMENT_CHARS:
                return PrimeResult.failure("'arguments' are too large")
        except Exception:
            return PrimeResult.failure("'arguments' could not be read")

        # Validation happens before the handler exists to the caller, so an
        # invalid request never reaches the operation.
        cleaned, errors = validate(raw_arguments, spec.input_schema)
        if cleaned is None:
            return PrimeResult.failure(
                f"Invalid arguments for tool '{spec.name}': " + "; ".join(errors)
            )

        if spec.handler is None:
            return PrimeResult.failure(
                f"Tool has no handler available: {spec.name}"
            )

        try:
            result = spec.handler(cleaned)
        except Exception as exc:
            return PrimeResult.failure(f"tool_failed: {spec.name}: {exc}")

        if not isinstance(result, ToolResult):
            return PrimeResult.failure(
                f"tool_returned_nothing: {spec.name}"
            )
        if not result.ok:
            return PrimeResult.failure(
                f"tool_failed: {spec.name}: {result.error or 'unknown error'}"
            )

        return PrimeResult.success(
            {"tool": spec.name, "arguments": cleaned,
             "result": result.rendered(spec.max_output_chars)}
        )


def json_size(value: Any) -> str:
    import json

    return json.dumps(value, default=str)
