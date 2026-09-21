import json
import re
from typing import Optional
from PRIMORA.prime.context import BudgetConfig, ContextTooLargeError
from PRIMORA.prime.contract import PrimeRequest, PrimeResult
from PRIMORA.prime import Prime
from PRIMORA.aurora.brain import Brain, AuroraBrain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig , build_capability_instruction


_PRIME_REQUEST_PATTERN = re.compile(r"<prime_request>(.*?)</prime_request>", re.DOTALL)


def extract_prime_request(text: str) -> Optional[PrimeRequest]:
    if not isinstance(text, str):
        return None
    matches = _PRIME_REQUEST_PATTERN.findall(text)
    if len(matches) != 1:
        return None
    try:
        data = json.loads(matches[0].strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    action = data.get("action")
    if not isinstance(action, str) or not action:
        return None
    payload = data.get("payload")
    if not isinstance(payload, dict):
        return None
    context = data.get("context")
    if context is not None and not isinstance(context, dict):
        return None
    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return None
    return PrimeRequest(action=action, payload=payload, context=context, metadata=metadata)


class Aurora:
    def __init__(
        self,
        prime: Prime,
        brain: Brain | None = None,
        config: AuroraConfig | None = None,
        context_manager=None,
    ):
        self._prime = prime
        self._config = config or AuroraConfig.from_env()

        if brain is not None:
            self._brain = brain
        else:
            err = self._config.validate()
            if err:
                raise ValueError(err)
            self._brain = AuroraBrain(config=self._config)

        if context_manager is not None:
            self._context = context_manager
        else:
            self._context = prime.context

    def request(self, action: str, payload: dict = None, context: dict = None, metadata: dict = None) -> PrimeResult:
        req = PrimeRequest(
            action=action,
            payload=payload or {},
            context=context,
            metadata=metadata
        )
        return self._prime.handle(req)

    def warmup(self) -> Optional[str]:
        return self._brain.warmup()

    def think(self, messages: list[dict[str, str]]) -> BrainResult:
        return self._brain.generate(messages)

    def _with_instruction(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        capability_instruction = build_capability_instruction(self._prime)

        if messages and messages[0].get("role") == "system":
            system_content = messages[0]["content"]
            if capability_instruction:
                system_content += capability_instruction
            return [{"role": "system", "content": system_content}] + list(messages[1:])

        system_content = self._config.system_instruction + capability_instruction
        return [{"role": "system", "content": system_content}] + list(messages)

    def _budget(self) -> BudgetConfig:
        return BudgetConfig(
            input_target=self._config.context_target,
            input_ceiling=self._config.context_ceiling,
            output_reserve=self._config.context_output_reserve,
            total_budget=self._config.context_total,
        )

    def _token_counter(self):
        brain = self._brain

        def count(messages: list[dict[str, str]]) -> int:
            return brain.count_tokens(messages)

        return count

    def respond(
        self,
        messages: list[dict[str, str]],
        session_id: str = "default",
    ) -> BrainResult:
        instructed = self._with_instruction(messages)
        try:
            first_view = self._context.build(
                instructed,
                session_id,
                token_counter=self._token_counter(),
                max_new_tokens=self._config.max_new_tokens,
                budget=self._budget(),
            ).view
        except ContextTooLargeError as e:
            return BrainResult.failure(str(e), self._brain.get_model_name())
        first = self._brain.generate(
            first_view.messages,
            max_output_tokens=first_view.allowed_output,
        )
        if not first.success:
            return first
        req = extract_prime_request(first.content)
        if req is None:
            return first
        result = self.request(req.action, req.payload, req.context, req.metadata)
        tool_content = json.dumps(
            {
                "status": result.status.value,
                "output": result.output,
                "error": result.error,
            },
            default=str,
        )
        followup = instructed + [
            {"role": "assistant", "content": first.content},
            {"role": "tool", "content": tool_content},
        ]
        try:
            second_view = self._context.build(
                followup,
                session_id,
                token_counter=self._token_counter(),
                max_new_tokens=self._config.max_new_tokens,
                budget=self._budget(),
            ).view
        except ContextTooLargeError as e:
            return BrainResult.failure(str(e), self._brain.get_model_name())
        return self._brain.generate(
            second_view.messages,
            max_output_tokens=second_view.allowed_output,
        )

    def get_model_name(self) -> str:
        return self._brain.get_model_name()
