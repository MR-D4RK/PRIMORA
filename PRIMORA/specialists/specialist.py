from abc import ABC, abstractmethod
from typing import Any
from PRIMORA.prime.contract import PrimeResult


class Specialist(ABC):
    """A unit of specialized work Prime can dispatch to.

    Specialists perform work; they do not route, orchestrate, or execute
    other capabilities. Selection stays with Prime's Router.
    """

    name: str

    @abstractmethod
    def handle(self, payload: dict[str, Any]) -> PrimeResult:
        """Perform this specialist's capability and return a PrimeResult."""
