from dataclasses import dataclass, field
from typing import Any, Optional
from PRIMORA.prime.contract import PrimeResult


@dataclass(frozen=True)
class CapabilitySpec:
    """Canonical description of a Prime capability."""
    name: str
    description: str
    payload_schema: dict[str, Any] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    optional_fields: list[str] = field(default_factory=list)


class CapabilityRegistry:
    """
    Prime's capability registry.
    
    Owned by Prime. Aurora has read-only access to the capability manifest.
    """
    
    def __init__(self):
        self._capabilities: dict[str, CapabilitySpec] = {}
        self._specialists: dict[str, Any] = {}
        
        # Register verified capabilities
        self._register_verified_capabilities()
    
    def _register_verified_capabilities(self) -> None:
        """Register only capabilities verified by current implementation and tests."""
        from PRIMORA.specialists.tools.calculate import CalculateSpecialist
        
        self.register(
            CapabilitySpec(
                name="calculate",
                description="Perform deterministic arithmetic calculations. Supports +, -, *, /, //, %, **, parentheses, unary +/-, and negative numbers.",
                payload_schema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Arithmetic expression to evaluate"}
                    },
                    "required": ["expression"],
                    "additionalProperties": False
                },
                required_fields=["expression"],
                optional_fields=[]
            ),
            CalculateSpecialist()
        )
        
    def register(self, spec: CapabilitySpec, specialist: object) -> None:
        if spec.name in self._capabilities:
            raise ValueError(f"Capability already registered: {spec.name}")
        self._capabilities[spec.name] = spec
        self._specialists[spec.name] = specialist

    def get_specialist(self, name: str) -> Optional[object]:
        return self._specialists.get(name)
    
    def get(self, name: str) -> Optional[CapabilitySpec]:
        """Get a capability specification by name."""
        return self._capabilities.get(name)
    
    def get_manifest(self) -> dict[str, dict[str, Any]]:
        """
        Get read-only capability manifest for Aurora's awareness.
        
        Returns a JSON-serializable manifest containing only what Aurora needs:
        - name, description, required/optional payload fields
        """
        return {
            name: {
                "name": spec.name,
                "description": spec.description,
                "required_fields": spec.required_fields,
                "optional_fields": spec.optional_fields,
                "payload_schema": spec.payload_schema
            }
            for name, spec in self._capabilities.items()
        }
    
    def get_names(self) -> list[str]:
        """Get list of registered capability names."""
        return list(self._capabilities.keys())
    
    def has(self, name: str) -> bool:
        """Check if a capability is registered."""
        return name in self._capabilities
    
    def validate_action(self, action: str) -> Optional[str]:
        """
        Validate and resolve an action to a registered capability.
        
        Returns the canonical capability name if valid, None if unknown.
        """
        if action in self._capabilities:
            return action
        return None
