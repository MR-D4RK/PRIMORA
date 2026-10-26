import os

from .context import ContextManager
from .context.mind import Mind
from .contract import PrimeRequest, PrimeResult
from .orchestrator import Orchestrator
from .providers import Provider
from .capability import CapabilityRegistry
from PRIMORA.consult.catalogue import SpecialistRegistry
from PRIMORA.research.sources import SourceRegistry
from PRIMORA.toolbox.base import ToolRegistry


def _configured_memory_db_path():
    """Return an explicitly configured runtime database path, if supplied.

    The default remains ContextStore's normal production path.  Acceptance
    runs can point the shared Prime/Mind/ContextManager boundary at a
    disposable file without creating a second memory system.
    """
    for name in ("PRIMORA_MEMORY_V2_DB", "PRIMORA_MEMORY_DB_PATH"):
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    return None


class Prime:
    def __init__(self, provider, research_provider=None, specialists=None,
                 jev=None, tools=None, context_manager=None, mind=None):
        if provider is None:
            raise ValueError("Prime requires a Provider")
        # One provider is shared by capability execution and opt-in Memory V2
        # typed formation.  Formation remains dormant while the existing
        # ContextManager V2 flags are off, so disabled runtime behavior is
        # unchanged and no second model/provider is constructed.
        self._provider = provider
        self._registry = CapabilityRegistry()
        if mind is not None and not isinstance(mind, Mind):
            raise TypeError("mind must be a Mind")
        if context_manager is not None and isinstance(context_manager, Mind):
            if mind is not None and mind is not context_manager:
                raise ValueError("context_manager Mind conflicts with mind")
            mind = context_manager
            context_manager = None
        if mind is not None:
            mechanisms = mind.mechanisms
            if context_manager is not None and mechanisms is not context_manager:
                raise ValueError("mind and context_manager must share mechanisms")
            self._mind = mind
            self._context = mechanisms
        else:
            db_path = _configured_memory_db_path()
            if context_manager is not None:
                self._context = context_manager
            elif db_path is None:
                # Keep the no-argument construction seam compatible with
                # lightweight test/runtime ContextManager factories.  The
                # provider is attached by _wire_formation_provider below.
                self._context = ContextManager()
            else:
                self._context = ContextManager(db_path=db_path)
            self._mind = Mind(self._context)
        self._wire_formation_provider()
        self._orchestrator = Orchestrator(provider, self._registry)
        # The recall capability answers from the Mind, and the registry is
        # built before the Mind exists, so bind it here. Every other
        # capability is self-contained and needs nothing.
        recall = self._registry.get_specialist("recall")
        if recall is not None:
            recall.bind(self._mind)
        # Milestone 3: the research backend is replaceable. Left unset, the
        # capability fails closed rather than reaching the network silently.
        #
        # The source registry is what turns a search's URLs into stable
        # numbers Aurora can cite and hand to research_read. It is created
        # here, once, and shared by both capabilities so a number always
        # names the same page within one interaction. The cognitive loop
        # clears it at the start of every session, so a number from an
        # earlier, unrelated search cannot be read or cited.
        self._sources = SourceRegistry()
        if research_provider is not None:
            research = self._registry.get_specialist("research")
            if research is not None:
                research.set_provider(research_provider)
        research = self._registry.get_specialist("research")
        if research is not None:
            research.set_sources(self._sources)
        read = self._registry.get_specialist("research_read")
        if read is not None:
            read.set_sources(self._sources)
        # The specialist catalogue is empty unless one is supplied. An
        # instrument nobody configured must not be reachable by name.
        self._specialists = (
            specialists if specialists is not None else SpecialistRegistry()
        )
        consult = self._registry.get_specialist("consult_specialist")
        if consult is not None:
            consult.set_provider(provider)
            consult.set_specialists(self._specialists)
        # Learning: the Mind admits the lesson, Jev structures it. Left
        # unset, learn() fails closed rather than guessing memory structure.
        learn = self._registry.get_specialist("learn")
        if learn is not None:
            learn.bind(self._mind)
            if jev is not None:
                learn.set_jev(jev)
        # Registered tools. Empty unless supplied, so no operation is
        # reachable until someone deliberately registers it.
        self._tools = tools if tools is not None else ToolRegistry()
        use_tool = self._registry.get_specialist("use_tool")
        if use_tool is not None:
            use_tool.set_tools(self._tools)

    def bind_memory_boundary(self, context_manager=None, mind=None) -> Mind:
        """Bind Prime and every memory capability to one Mind.

        This is the supported seam for embedding PRIMORA with an explicit
        ContextManager.  A capability must never retain the old default
        database after Aurora receives a caller-owned memory boundary.
        """
        if mind is not None and not isinstance(mind, Mind):
            raise TypeError("mind must be a Mind")
        if isinstance(context_manager, Mind):
            if mind is not None and mind is not context_manager:
                raise ValueError("context_manager Mind conflicts with mind")
            mind = context_manager
            context_manager = None
        if mind is None:
            if context_manager is None:
                raise ValueError("bind_memory_boundary requires a Mind or ContextManager")
            mind = Mind(context_manager)
        if context_manager is not None and mind.mechanisms is not context_manager:
            raise ValueError("mind and context_manager must share mechanisms")
        self._mind = mind
        self._context = mind.mechanisms
        self._wire_formation_provider()
        for name in ("recall", "learn"):
            specialist = self._registry.get_specialist(name)
            binder = getattr(specialist, "bind", None)
            if callable(binder):
                binder(mind)
        return mind

    def _wire_formation_provider(self) -> None:
        """Attach this Prime's provider to the one shared ContextManager.

        A caller may provide a ContextManager or Mind after Prime was built
        (Aurora's supported boundary-binding path).  Wire only when that
        manager has no formation provider, preserving an explicitly supplied
        test or specialist provider while making the configured runtime
        provider available to opt-in Memory V2 formation.
        """
        setter = getattr(self._context, "set_formation_provider", None)
        if not callable(setter):
            return
        if getattr(self._context, "_formation_provider", None) is None:
            setter(self._provider)

    @property
    def tools(self) -> ToolRegistry:
        """The registered-tool allowlist. Prime authorizes against this."""
        return self._tools

    @property
    def specialists(self) -> SpecialistRegistry:
        """The catalogue of consultable specialists. Instruments, not agents."""
        return self._specialists

    @property
    def sources(self) -> SourceRegistry:
        """The bounded set of citable search-result sources for this interaction.

        Owned by Prime, shared by the research capabilities, and read by the
        loop's citation check through the executor - never imported by the
        loop itself, which stays free of capability names. The loop clears
        it at the start of every session, so a number only ever names a page
        the current turn's search produced.
        """
        return self._sources

    @property
    def registry(self) -> CapabilityRegistry:
        """Read-only access to the capability registry for Aurora's awareness."""
        return self._registry

    @property
    def context(self) -> ContextManager:
        """Prime-owned conversation context infrastructure.

        The memory mechanisms. Mind is the cognitive boundary above it.
        """
        return self._context

    @property
    def mind(self) -> Mind:
        """The cognitive boundary over Prime's memory mechanisms."""
        return self._mind

    def handle(self, request: PrimeRequest) -> PrimeResult:
        if not isinstance(request, PrimeRequest):
            return PrimeResult.failure("Invalid request type")
# Validate capability is registered
        canonical = self._registry.validate_action(request.action)
        if canonical is None:
            # An unknown action is Aurora's mistake, and the executor hands
            # this text back to her as the tool result. Naming the action she
            # used, the registered set, and the wrappers for tools and
            # specialists turns a dead end into something she can correct on
            # the next cycle. A bare "unknown capability" told her nothing and
            # left her narrating instead of retrying.
            known = ", ".join(sorted(self._registry.get_names()))
            return PrimeResult.failure(
                f"Unknown capability: {request.action!r}. "
                f"Registered capabilities are: {known}. "
                "A tool name is never an action: use action 'use_tool' with "
                "the tool name in payload.name. A specialist name is never an "
                "action: use action 'consult_specialist' with the specialist "
                "name in payload.specialist."
            )
        # Create a new request with canonical action
        canonical_request = PrimeRequest(
            action=canonical,
            payload=request.payload,
            context=request.context,
            metadata=request.metadata
        )
        return self._orchestrator.orchestrate(canonical_request)
