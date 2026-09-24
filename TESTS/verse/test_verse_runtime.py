import sys
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.verse import (
    PresentationMode,
    VerseComponent,
    VerseLifecycleState,
    VerseRuntime,
)


class StubAurora:
    pass


class StubComponent(VerseComponent):
    name = "stub"
    description = "test component"

    def __init__(self):
        super().__init__()
        self.shown = 0
        self.hidden = 0

    def show(self) -> None:
        self.shown += 1
        self._visible = True

    def hide(self) -> None:
        self.hidden += 1
        self._visible = False


def test_verse_package_imports():
    assert VerseRuntime is not None
    assert VerseComponent is not None
    assert [s.value for s in VerseLifecycleState] == [
        "starting", "awakening", "awake", "asleep",
        "closing", "closed", "degraded",
    ]
    assert [m.value for m in PresentationMode] == ["immersive", "portal"]


def test_runtime_creation_requires_aurora():
    try:
        VerseRuntime(None)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "requires an Aurora instance" in str(e)


def test_runtime_holds_existing_aurora():
    aurora = StubAurora()
    runtime = VerseRuntime(aurora)
    assert runtime.aurora is aurora


def test_initial_state_and_mode():
    runtime = VerseRuntime(StubAurora())
    assert runtime.state is VerseLifecycleState.STARTING
    assert runtime.mode is PresentationMode.PORTAL


def test_valid_lifecycle_transitions():
    runtime = VerseRuntime(StubAurora())
    runtime.transition_to(VerseLifecycleState.AWAKENING)
    runtime.transition_to(VerseLifecycleState.AWAKE)
    runtime.transition_to(VerseLifecycleState.ASLEEP)
    runtime.transition_to(VerseLifecycleState.AWAKE)
    assert runtime.state is VerseLifecycleState.AWAKE
    runtime.transition_to(VerseLifecycleState.DEGRADED)
    runtime.transition_to(VerseLifecycleState.ASLEEP)
    runtime.transition_to(VerseLifecycleState.CLOSING)
    runtime.transition_to(VerseLifecycleState.CLOSED)
    assert runtime.state is VerseLifecycleState.CLOSED


def test_invalid_lifecycle_transitions_rejected():
    runtime = VerseRuntime(StubAurora())
    try:
        runtime.transition_to(VerseLifecycleState.AWAKE)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "starting -> awake" in str(e)
    runtime.transition_to(VerseLifecycleState.CLOSING)
    runtime.transition_to(VerseLifecycleState.CLOSED)
    try:
        runtime.transition_to(VerseLifecycleState.AWAKE)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    try:
        runtime.transition_to("awake")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_mode_changes():
    runtime = VerseRuntime(StubAurora())
    runtime.set_mode(PresentationMode.IMMERSIVE)
    assert runtime.mode is PresentationMode.IMMERSIVE
    runtime.set_mode(PresentationMode.PORTAL)
    assert runtime.mode is PresentationMode.PORTAL
    try:
        runtime.set_mode("immersive")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_component_registration():
    runtime = VerseRuntime(StubAurora())
    component = StubComponent()
    assert runtime.list_components() == []
    assert runtime.get_component("stub") is None
    runtime.register_component(component)
    assert runtime.get_component("stub") is component
    assert runtime.list_components() == ["stub"]
    assert component.is_visible() is False
    component.show()
    assert component.is_visible() is True
    component.hide()
    assert component.is_visible() is False


def test_duplicate_component_registration_rejected():
    runtime = VerseRuntime(StubAurora())
    runtime.register_component(StubComponent())
    try:
        runtime.register_component(StubComponent())
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "already registered" in str(e)
    assert runtime.list_components() == ["stub"]


def test_component_removal():
    runtime = VerseRuntime(StubAurora())
    assert runtime.unregister_component("missing") is False
    component = StubComponent()
    runtime.register_component(component)
    assert runtime.unregister_component("stub") is True
    assert runtime.get_component("stub") is None
    assert runtime.list_components() == []
    assert runtime.unregister_component("stub") is False


if __name__ == "__main__":
    test_verse_package_imports()
    test_runtime_creation_requires_aurora()
    test_runtime_holds_existing_aurora()
    test_initial_state_and_mode()
    test_valid_lifecycle_transitions()
    test_invalid_lifecycle_transitions_rejected()
    test_mode_changes()
    test_component_registration()
    test_duplicate_component_registration_rejected()
    test_component_removal()
    print("All Verse runtime tests passed!")
