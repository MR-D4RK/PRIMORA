from typing import Optional
from .contract import PrimeRequest
from .capability import CapabilityRegistry


class Router:
    def __init__(self, registry: CapabilityRegistry):
        self._registry = registry

    def route(self, request: PrimeRequest) -> str:
        return request.action
