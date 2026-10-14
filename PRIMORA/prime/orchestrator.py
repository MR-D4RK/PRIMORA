from typing import Optional
from .contract import PrimeRequest, PrimeResult
from .execution import Execution
from .providers import Provider
from .capability import CapabilityRegistry


class Orchestrator:
    def __init__(self, provider: Provider, registry: Optional[CapabilityRegistry] = None):
        if provider is None:
            raise ValueError("Orchestrator requires a Provider")
        # Create default registry if not provided (for backward compatibility)
        if registry is None:
            registry = CapabilityRegistry()
        self._registry = registry
        self._execution = Execution(provider, registry)

    def orchestrate(self, request: PrimeRequest) -> PrimeResult:
        # Prime.handle is the single authoritative validation path: it
        # rejects unknown actions with a message that names the registered
        # capabilities and the use_tool / consult_specialist wrappers, then
        # canonicalizes the action before dispatching here. The former
        # Router pass-through and a second validate_action at this layer
        # duplicated that check on every live request; a direct caller of
        # orchestrate() still gets a clean failure from Execution (specialist
        # dispatch, or the provider-backed fallback for a registered
        # provider capability).
        return self._execution.execute(request.action, request.payload)
