import json
import re
from typing import Optional
from PRIMORA.prime.context import BudgetConfig
from PRIMORA.prime.context.mind import CognitiveContext, Mind
from PRIMORA.prime.contract import PrimeRequest, PrimeResult
from PRIMORA.prime import Prime
from PRIMORA.aurora.brain import (
    Brain,
    AuroraBrain,
    BrainResult,
    OpenAIBrain,
    aurora_backend_from_env,
)
from PRIMORA.aurora.configuration import (
    AuroraConfig,
    GUIDANCE_HEADER,
    build_capability_instruction,
    build_guidance_instruction,
    build_specialist_instruction,
    build_tool_instruction,
)
from PRIMORA.aurora.identity import IDENTITY_HEADER, build_identity_instruction
from PRIMORA.aurora.loop import CognitiveLoop, CapabilityExecutor


_PRIME_REQUEST_PATTERN = re.compile(r"<prime_request>(.*?)</prime_request>", re.DOTALL)


def _string_spans(s: str) -> tuple[list[tuple[int, int]], bool]:
    """Locate double-quoted strings in s.

    Returns the [start, end) spans plus whether the final one was never
    closed, which is how a request truncated mid-string looks. Escape
    aware, so an escaped quote does not end its string.
    """
    spans: list[tuple[int, int]] = []
    unterminated = False
    i, n = 0, len(s)
    while i < n:
        if s[i] == '"':
            start = i
            i += 1
            closed = False
            while i < n:
                if s[i] == "\\":
                    i += 2
                    continue
                if s[i] == '"':
                    i += 1
                    closed = True
                    break
                i += 1
            spans.append((start, i))
            if not closed:
                unterminated = True
                break
        else:
            i += 1
    return spans, unterminated


def _outside_string(spans: list[tuple[int, int]], pos: int) -> bool:
    return not any(start <= pos < end for start, end in spans)


def _strip_trailing_commas(s: str, spans: list[tuple[int, int]]) -> str:
    """Drop a comma sitting immediately before a closer.

    A model that has learned the shape of the object still emits
    ``{"a":1,}``; strict JSON rejects the whole request over one
    character that carries no content.
    """
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "," and _outside_string(spans, i):
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j < n and s[j] in "}]" and _outside_string(spans, j):
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _balance_closers(s: str, spans: list[tuple[int, int]]) -> str:
    """Append the closers a truncated object is missing.

    Hitting max_tokens mid-request is the most common way a real
    request arrives unusable. The opening half and every key and value
    are already decided, so restoring the closers repairs structure
    without inventing any.
    """
    stack: list[str] = []
    for i, ch in enumerate(s):
        if not _outside_string(spans, i):
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    if not stack:
        return s
    return s + "".join(reversed(stack))


def _repair_json_object(raw: str) -> Optional[str]:
    """Repair syntax damage in a near-valid JSON object.

    Conservative by construction. Every repair restores structure the
    model already emitted - an unbalanced closer, a trailing comma, an
    unterminated string - and none invents a key, a value, or an
    action. A block the model never actually builds stays unusable and
    is still rejected.
    """
    candidate = raw.strip()
    if not candidate or candidate in ("null", "None", "none"):
        return None
    for _ in range(4):
        spans, unterminated = _string_spans(candidate)
        repaired = candidate
        if unterminated:
            repaired += '"'
            spans, _ = _string_spans(repaired)
        balanced = _balance_closers(_strip_trailing_commas(repaired, spans), spans)
        if balanced == candidate:
            return candidate
        candidate = balanced
    return candidate


def extract_prime_request(text: str) -> Optional[PrimeRequest]:
    if not isinstance(text, str):
        return None
    matches = _PRIME_REQUEST_PATTERN.findall(text)
    if len(matches) != 1:
        return None
    try:
        data = json.loads(matches[0].strip())
    except (json.JSONDecodeError, ValueError):
        # A local model most often damages the JSON it was already
        # building rather than inventing a different one, so repairing
        # syntax rescues the request. Anything still unparseable after
        # repair was never a request, and the loop reports it as such.
        repaired = _repair_json_object(matches[0])
        if repaired is None:
            return None
        try:
            data = json.loads(repaired, strict=False)
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
        mind: Mind | None = None,
    ):
        self._prime = prime
        self._config = config or AuroraConfig.from_env()

        if brain is not None:
            self._brain = brain
        else:
            backend = aurora_backend_from_env()
            if backend == "openai":
                # The OpenAI backend never loads the HF model or adapter,
                # so their directories must not gate construction; the
                # context-budget invariants still apply to both backends.
                err = self._config.validate(require_paths=False)
                if err:
                    raise ValueError(err)
                self._brain = OpenAIBrain(config=self._config)
            else:
                err = self._config.validate()
                if err:
                    raise ValueError(err)
                self._brain = AuroraBrain(config=self._config)

        # Milestone 1: Aurora is handed a Mind, the cognitive boundary, and
        # receives one unified CognitiveContext. A raw ContextManager is still
        # accepted and wrapped, so every existing call site keeps working.
        if mind is not None:
            self._mind = mind
        elif context_manager is not None:
            # An explicit manager means an isolated store (a test, a tool,
            # or a per-task runtime), so it gets its own Mind.
            self._mind = (
                context_manager if isinstance(context_manager, Mind)
                else Mind(context_manager)
            )
        else:
            # Default path: share Prime's single Mind, so the task-local
            # Working Memory belongs to the Prime/Aurora pair, not to a
            # wrapper created per Aurora instance.
            self._mind = getattr(prime, "mind", None) or Mind(prime.context)
        self._context = self._mind.mechanisms

        # Milestone 2: the cognitive loop. It reuses this Aurora's existing
        # brain and the same Mind, so there is no second agent. The executor
        # is the capability seam: today it is the Prime request path that
        # already existed, and Milestone 3 replaces it without changing the
        # loop's control flow.
        self._loop = CognitiveLoop(
            self._mind,
            self._brain,
            executor=CapabilityExecutor(self._prime),
            max_cycles=self._config.max_cognition_cycles,
            max_new_tokens=self._config.max_new_tokens,
            budget=self._budget(),
            token_counter=self._token_counter(),
            request_extractor=extract_prime_request,
        )
        # The learn capability checks a self-reported provenance against what
        # the turn actually gathered, which only the loop's executor records.
        learn = self._prime.registry.get_specialist("learn")
        if learn is not None:
            learn.set_evidence_source(self._loop.executor)

    @property
    def mind(self) -> Mind:
        """The cognitive boundary behind this Aurora instance."""
        return self._mind

    @property
    def loop(self) -> CognitiveLoop:
        """The cognitive loop behind this Aurora instance."""
        return self._loop

    def request(self, action: str, payload: dict = None, context: dict = None, metadata: dict = None) -> PrimeResult:
        req = PrimeRequest(
            action=action,
            payload=payload or {},
            context=context,
            metadata=metadata
        )
        return self._prime.handle(req)

    def warmup(self) -> Optional[str]:
        """Load everything the first turn would otherwise load inline.

        The brain first, then the local embedding model. The embedding
        model is the invisible half of the first-turn stall: it is loaded
        on its first retrieval, on CPU, and that first call measured at
        ~13 s inside an otherwise 7 s turn. Doing it here moves the cost to
        startup, where it belongs, and makes the model's own load the only
        thing the user waits for. A missing embedding model is not an
        error: retrieval degrades to lexical and warmup still succeeds.
        """
        err = self._brain.warmup()
        if err:
            return err
        try:
            from PRIMORA.prime.context import semantic as _semantic

            _semantic.load_model()
        except Exception:
            pass
        return None

    def think(self, messages: list[dict[str, str]]) -> BrainResult:
        return self._brain.generate(messages)

    def _with_instruction(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        capability_instruction = (
            build_capability_instruction(self._prime)
            + build_specialist_instruction(self._prime)
            + build_tool_instruction(self._prime)
        )
        # The identity is resolved from the live config on every call, so the
        # prompt describes the model and adapter this run actually uses. It
        # goes into both branches: a caller-supplied system message must not
        # be able to leave Aurora without an identity to answer from, and a
        # caller echoing an already-identified prompt must not get a second
        # copy of the block.
        identity_instruction = build_identity_instruction(self._config)
        # Placed last, after the catalogues, because it is a behavioural
        # nudge rather than a description: the nearest instruction to
        # generation is the one that has the most effect on whether a
        # request block is emitted at all. Empty for any base family it
        # does not apply to, so nothing is added by default.
        guidance_instruction = build_guidance_instruction(self._config)

        if messages and messages[0].get("role") == "system":
            system_content = messages[0]["content"]
            if IDENTITY_HEADER not in system_content:
                system_content += identity_instruction
            if capability_instruction:
                system_content += capability_instruction
            if guidance_instruction and GUIDANCE_HEADER not in system_content:
                system_content += guidance_instruction
            return [{"role": "system", "content": system_content}] + list(messages[1:])

        system_content = self._config.system_instruction
        if IDENTITY_HEADER not in system_content:
            system_content += identity_instruction
        system_content += capability_instruction
        if guidance_instruction and GUIDANCE_HEADER not in system_content:
            system_content += guidance_instruction
        return [{"role": "system", "content": system_content}] + list(messages)

    def _budget(self) -> BudgetConfig:
        return BudgetConfig(
            input_target=self._config.context_target,
            input_ceiling=self._config.context_ceiling,
            output_reserve=self._config.context_output_reserve,
            total_budget=self._config.context_total,
            state_reserve=self._config.context_state_reserve,
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
        """One logical response, possibly cognated over several segments.

        The loop owns the segment structure; this method owns only the
        instruction. The signature and return type are unchanged.
        """
        return self._loop.run(self._with_instruction(messages), session_id)

    def context_for(
        self,
        messages: list[dict[str, str]],
        session_id: str = "default",
    ) -> CognitiveContext:
        """The unified cognitive context Aurora would use for this turn.

        Exposed so a caller can inspect what Aurora is given without
        generating. Does not run the brain.
        """
        return self._mind.build(
            self._with_instruction(messages),
            session_id,
            token_counter=self._token_counter(),
            max_new_tokens=self._config.max_new_tokens,
            budget=self._budget(),
        )

    def get_model_name(self) -> str:
        return self._brain.get_model_name()
