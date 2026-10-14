"""Argument validation for registered tools.

A deliberately small JSON-Schema subset, because there is no schema library in
this project and adding one is not worth a dependency:

    type, required, properties, additionalProperties, enum,
    minimum, maximum, minLength, maxLength, items

Enough to reject the mistakes a model actually makes - a missing field, a
string where a number is wanted, an unknown argument - and no more. Anything
not understood is treated as *rejected* rather than ignored, so an unknown
constraint cannot become an accidental permission.
"""

from typing import Any, Optional

_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate(
    arguments: Any, schema: dict[str, Any]
) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Return ``(cleaned, errors)``. ``cleaned`` is None when invalid."""
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None, ["'arguments' must be an object"]

    errors: list[str] = []
    schema = schema or {}
    properties = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []
    additional = schema.get("additionalProperties", False)

    for field in required:
        if field not in arguments:
            errors.append(f"missing required argument: {field}")

    unknown = sorted(set(arguments) - set(properties))
    if unknown and additional is False:
        errors.append(f"unknown argument(s): {', '.join(unknown)}")

    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        spec = properties.get(key)
        if spec is None:
            if additional is not False:
                cleaned[key] = value
            continue
        problem = _check(key, value, spec)
        if problem:
            errors.append(problem)
        else:
            cleaned[key] = value

    declared = schema.get("type")
    if declared and declared != "object":
        errors.append(f"arguments must be of type {declared}")

    if errors:
        return None, errors
    return cleaned, []


def _check(name: str, value: Any, spec: dict[str, Any]) -> Optional[str]:
    if not isinstance(spec, dict):
        return f"invalid schema for argument: {name}"

    declared = spec.get("type")
    if declared:
        expected = _TYPES.get(declared)
        if expected is None:
            return f"unknown type '{declared}' for argument: {name}"
        if declared == "integer" and isinstance(value, bool):
            return f"argument '{name}' must be an integer"
        if declared == "number" and isinstance(value, bool):
            return f"argument '{name}' must be a number"
        if not isinstance(value, expected):
            return f"argument '{name}' must be of type {declared}"

    if "enum" in spec and value not in spec["enum"]:
        allowed = ", ".join(str(v) for v in spec["enum"])
        return f"argument '{name}' must be one of: {allowed}"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            return f"argument '{name}' must be >= {spec['minimum']}"
        if "maximum" in spec and value > spec["maximum"]:
            return f"argument '{name}' must be <= {spec['maximum']}"

    if isinstance(value, str):
        if "minLength" in spec and len(value) < spec["minLength"]:
            return f"argument '{name}' is too short"
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            return f"argument '{name}' is too long"

    if isinstance(value, list) and isinstance(spec.get("items"), dict):
        for index, item in enumerate(value):
            problem = _check(f"{name}[{index}]", item, spec["items"])
            if problem:
                return problem

    if isinstance(value, dict) and spec.get("type") == "object":
        return None

    return None
