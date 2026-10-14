"""The ``consult_specialist`` capability: Aurora borrowing an instrument.

Milestone 3. The division of responsibility, unchanged from ``research``:

    Aurora decides WHEN and WHY she needs a specialist.
    Prime decides WHICH model and HOW to execute it.

Aurora sends a specialist name and a task. She never names a model, a
provider, a system prompt, a temperature, or a request parameter. All of that
belongs to the catalogue and the provider behind it.

A specialist is an instrument, not a sub-agent
---------------------------------------------
The specialist receives the task text and nothing else. It does not receive
Aurora's conversation, her memory, the user's message history, or any session
state; it has no identity, no goals, and no ability to act. It returns one
string, which comes back as a tool turn. Nothing about the consultation
persists, and nothing survives the call.

The model is executed through the existing
:class:`PRIMORA.prime.providers.Provider`, the same abstraction already used
for local model invocation, so the backend is replaceable and this file
contains no model name, no endpoint, and no transport.
"""

from typing import Any, Optional

from PRIMORA.consult.catalogue import SpecialistRegistry
from PRIMORA.prime.contract import PrimeResult, RequestStatus
from PRIMORA.specialists.specialist import Specialist

# A specialist answers one sub-problem. An unbounded task is how an
# instrument turns into an agent, so the input is capped.
MAX_TASK_CHARS = 8000


class ConsultSpecialist(Specialist):
    """Route a task to a named specialist via Prime's provider machinery."""

    name = "consult_specialist"

    def __init__(
        self,
        provider: Optional[Any] = None,
        specialists: Optional[SpecialistRegistry] = None,
    ) -> None:
        self._provider = provider
        self._specialists = specialists or SpecialistRegistry()

    def set_provider(self, provider: Any) -> None:
        if provider is None:
            raise ValueError("consult_specialist requires a Provider")
        self._provider = provider

    def set_specialists(self, specialists: SpecialistRegistry) -> None:
        if specialists is None:
            raise ValueError("specialists cannot be None")
        self._specialists = specialists

    @property
    def specialists(self) -> SpecialistRegistry:
        return self._specialists

    @property
    def is_configured(self) -> bool:
        """Whether a real catalogue of consultable specialists is wired.

        An empty catalogue fails closed on every request (``Unknown
        specialist... Available: none configured``), so advertising the
        capability would only invite a call that is guaranteed to fail.
        The capability instruction reads this and omits the entry,
        exactly as ``research`` and ``learn`` do when unconfigured.
        """
        return bool(self._specialists.names())

    def handle(self, payload: dict[str, Any]) -> PrimeResult:
        if not isinstance(payload, dict):
            return PrimeResult.failure("Invalid payload type")

        name = payload.get("specialist")
        if not isinstance(name, str) or not name.strip():
            return PrimeResult.failure(
                "Missing or invalid 'specialist' in payload"
            )

        task = payload.get("task")
        if not isinstance(task, str) or not task.strip():
            return PrimeResult.failure("Missing or invalid 'task' in payload")
        if len(task) > MAX_TASK_CHARS:
            return PrimeResult.failure(
                f"'task' is too long ({len(task)} characters, "
                f"maximum {MAX_TASK_CHARS}). Narrow the question."
            )

        spec = self._specialists.get(name)
        if spec is None:
            known = ", ".join(self._specialists.names()) or "none configured"
            return PrimeResult.failure(
                f"Unknown specialist: {name.strip()}. Available: {known}"
            )
        if not spec.available:
            return PrimeResult.failure(
                f"Specialist is unavailable: {spec.name}"
            )
        if self._provider is None:
            return PrimeResult.failure(
                "consult_specialist has no model provider configured"
            )

        # Exactly the task, and nothing else. No conversation, no memory, no
        # session: the specialist is an instrument, not a participant.
        messages: list[dict[str, str]] = []
        if spec.system_prompt:
            messages.append({"role": "system", "content": spec.system_prompt})
        messages.append({"role": "user", "content": task})

        call: dict[str, Any] = {"messages": messages}
        if spec.model:
            call["model"] = spec.model

        try:
            result = self._provider.invoke(spec.name, call)
        except Exception as exc:
            return PrimeResult.failure(
                f"specialist_failed: {spec.name}: {exc}"
            )

        # Check ``status``, not a ``success`` attribute: ``PrimeResult.success``
        # is a constructor classmethod, so a truthiness test on it is always
        # true and every failure would slip through as a success.
        if result is None or result.status is not RequestStatus.SUCCESS:
            error = getattr(result, "error", None) if result else None
            return PrimeResult.failure(
                f"specialist_failed: {spec.name}: {error or 'no result'}"
            )

        output = getattr(result, "output", None)
        text = output.get("content") if isinstance(output, dict) else output
        if not isinstance(text, str) or not text.strip():
            return PrimeResult.failure(
                f"specialist_returned_nothing: {spec.name}"
            )

        return PrimeResult.success(
            {"specialist": spec.name, "task": task, "result": text}
        )
