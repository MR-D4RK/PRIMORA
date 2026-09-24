from abc import ABC, abstractmethod
from typing import Any


class VerseComponent(ABC):
    """Minimal contract for a Verse component.

    Components are tools/capabilities for Aurora to control: they perform
    experiential changes (show/hide/update) and never contain model logic,
    memory, cognition, or Prime routing.
    """

    name: str
    description: str = ""

    def __init__(self):
        self._visible = False

    def is_visible(self) -> bool:
        return self._visible

    @abstractmethod
    def show(self) -> None:
        """Make the component's experience present."""

    @abstractmethod
    def hide(self) -> None:
        """Withdraw the component's experience."""

    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        """Receive current render state. Default is a no-op."""
        _ = state
        return {}
