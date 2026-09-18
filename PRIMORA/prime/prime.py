from .contract import PrimeRequest, PrimeResult
from .orchestrator import Orchestrator
from .providers import Provider
from .capability import CapabilityRegistry


class Prime:
    def __init__(self, provider):
        if provider is None:
            raise ValueError("Prime requires a Provider")
        self._registry = CapabilityRegistry()
        self._orchestrator = Orchestrator(provider, self._registry)

    @property
    def registry(self) -> CapabilityRegistry:
        """Read-only access to the capability registry for Aurora's awareness."""
        return self._registry

    def handle(self, request: PrimeRequest) -> PrimeResult:
        if not isinstance(request, PrimeRequest):
            return PrimeResult.failure("Invalid request type")
        # Validate capability is registered
        canonical = self._registry.validate_action(request.action)
        if canonical is None:
            return PrimeResult.failure(f"Unknown capability: {request.action}")
        # Create a new request with canonical action
        canonical_request = PrimeRequest(
            action=canonical,
            payload=request.payload,
            context=request.context,
            metadata=request.metadata
        )
        return self._orchestrator.orchestrate(canonical_request)