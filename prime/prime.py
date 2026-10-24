from .context import ContextManager
from .context.mind import Mind
from .contract import PrimeRequest, PrimeResult
from .orchestrator import Orchestrator
from .providers import Provider
from .capability import CapabilityRegistry
from PRIMORA.consult.catalogue import SpecialistRegistry
from PRIMORA.research.sources import SourceRegistry
from PRIMORA.toolbox.base import ToolRegistry


class Prime:
    def __init__(self, provider, research_provider=None, specialists=None,
                 jev=None, tools=None):
        if provider is None:
            raise ValueError("Prime requires a Provider")
        self._registry = CapabilityRegistry()
        self._context = ContextManager()
        self._mind = Mind(self._context)
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