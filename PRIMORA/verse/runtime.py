from typing import Optional

from .components.base import VerseComponent
from .state import PresentationMode, VerseLifecycleState


_ALLOWED_TRANSITIONS: dict[VerseLifecycleState, set[VerseLifecycleState]] = {
    VerseLifecycleState.STARTING: {
        VerseLifecycleState.AWAKENING,
        VerseLifecycleState.CLOSING,
        VerseLifecycleState.CLOSED,
    },
    VerseLifecycleState.AWAKENING: {
        VerseLifecycleState.AWAKE,
        VerseLifecycleState.DEGRADED,
        VerseLifecycleState.CLOSING,
    },
    VerseLifecycleState.AWAKE: {
        VerseLifecycleState.ASLEEP,
        VerseLifecycleState.DEGRADED,
        VerseLifecycleState.CLOSING,
    },
    VerseLifecycleState.ASLEEP: {
        VerseLifecycleState.AWAKE,
        VerseLifecycleState.CLOSING,
    },
    VerseLifecycleState.DEGRADED: {
        VerseLifecycleState.AWAKE,
        VerseLifecycleState.ASLEEP,
        VerseLifecycleState.CLOSING,
    },
    VerseLifecycleState.CLOSING: {
        VerseLifecycleState.CLOSED,
    },
    VerseLifecycleState.CLOSED: set(),
}


class VerseRuntime:
    """Experience layer hosting one Aurora instance.

    Holds a reference to the existing Aurora object (never constructs one),
    tracks lifecycle state and presentation mode, and hosts registered
    Verse components. Contains no model, memory, cognition, routing,
    networking, or rendering logic.
    """

    def __init__(self, aurora):
        if aurora is None:
            raise ValueError("VerseRuntime requires an Aurora instance")
        self._aurora = aurora
        self._state = VerseLifecycleState.STARTING
        self._mode = PresentationMode.PORTAL
        self._components: dict[str, VerseComponent] = {}

    @property
    def aurora(self):
        """The single Aurora instance this runtime presents."""
        return self._aurora

    @property
    def state(self) -> VerseLifecycleState:
        return self._state

    @property
    def mode(self) -> PresentationMode:
        return self._mode

    def set_mode(self, mode: PresentationMode) -> None:
        if not isinstance(mode, PresentationMode):
            raise ValueError(f"Invalid presentation mode: {mode!r}")
        if self._state is VerseLifecycleState.CLOSED:
            raise ValueError("Cannot change mode on a closed runtime")
        self._mode = mode

    def transition_to(self, state: VerseLifecycleState) -> None:
        if not isinstance(state, VerseLifecycleState):
            raise ValueError(f"Invalid lifecycle state: {state!r}")
        allowed = _ALLOWED_TRANSITIONS[self._state]
        if state not in allowed:
            raise ValueError(
                f"Invalid lifecycle transition: "
                f"{self._state.value} -> {state.value}"
            )
        self._state = state

    def register_component(self, component: VerseComponent) -> None:
        if not isinstance(component, VerseComponent):
            raise ValueError(
                f"Invalid Verse component: {component!r}"
            )
        if not component.name:
            raise ValueError("Verse component requires a non-empty name")
        if component.name in self._components:
            raise ValueError(
                f"Verse component already registered: {component.name}"
            )
        self._components[component.name] = component

    def unregister_component(self, name: str) -> bool:
        if name in self._components:
            del self._components[name]
            return True
        return False

    def get_component(self, name: str) -> Optional[VerseComponent]:
        return self._components.get(name)

    def list_components(self) -> list[str]:
        return sorted(self._components.keys())
