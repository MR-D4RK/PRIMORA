from typing import Optional
from .contract import PrimeResult
from .providers import Provider
from .capability import CapabilityRegistry


class Execution:
    def __init__(
        self,
        provider: Provider,
        registry: CapabilityRegistry,
    ):
        if provider is None:
            raise ValueError("Execution requires a Provider")
        if registry is None:
            raise ValueError("Execution requires a CapabilityRegistry")

        self._provider = provider
        self._registry = registry

    def execute(self, capability: str, payload: dict) -> PrimeResult:
        print(f"[Prime] Executing Capability: {capability}")

        specialist = self._registry.get_specialist(capability)

        if specialist is not None:
            return specialist.handle(payload)

        # Provider-backed capability: a capability registered without a
        # specialist (or reached through a direct Execution/Orchestrator
        # call) is handled by the provider itself. This is a supported,
        # tested contract (see test_execution_uses_registered_provider_capability
        # and test_orchestrator_passes_payload_to_provider), not dead code.
        # On the live path Prime.handle rejects unknown actions first, so
        # this branch only fires for deliberately provider-backed
        # capabilities.
        return self._provider.invoke(capability, payload)
