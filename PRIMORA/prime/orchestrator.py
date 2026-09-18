from typing import Optional
from .contract import PrimeRequest, PrimeResult
from .router import Router
from .execution import Execution
from .providers import Provider
from .capability import CapabilityRegistry


class Orchestrator:
    def __init__(self, provider: Provider, registry: Optional[CapabilityRegistry] = None):
        if provider is None:
            raise ValueError("Orchestrator requires a Provider")
        # Create default registry if not provided (for backward compatibility)
        if registry is None:
            from .capability import CapabilityRegistry
            registry = CapabilityRegistry()
        self._registry = registry
        self._router = Router(registry)
        self._execution = Execution(provider,registry)

    def orchestrate(self, request: PrimeRequest) -> PrimeResult:
        capability = self._router.route(request)
        # Validate capability is registered
        canonical = self._registry.validate_action(capability)
        if canonical is None:
            return PrimeResult.failure(f"Unknown capability: {capability}")
        return self._execution.execute(canonical, request.payload)
