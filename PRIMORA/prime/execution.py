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

        return self._provider.invoke(capability, payload)
