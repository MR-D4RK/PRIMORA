import json
import os
import re
from typing import Callable, Optional
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
from PRIMORA.aurora import request_shape


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
    if matches:
        # A wrapper block is the documented format. Exactly one of them,
        # and only one: several blocks are ambiguous about which request
        # was being made, so none runs.
        if len(matches) != 1:
            return None
        body = matches[0].strip()
        try:
            data = json.loads(body)
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
        return request_shape.request_from_data(data)

    # No wrapper. A model trained for tool calling emits the request
    # object itself, and that is a request it really made: the whole reply
    # is the object the wrapper would have contained. It is not the raw
    # JSON string being echoed back, and it is not recognised because it
    # happens to be JSON - it is recognised because it is one whole
    # object with a valid action and payload, so it runs through the same
    # authorized path as a wrapped request and Prime still authorizes
    # every action and payload it carries.
    #
    # Strictly parsed: no repair here, because a bare object that does not
    # parse is evidence of nothing more than braces, and the loop's safe
    # recovery handles it. Only a whole-reply object qualifies, so JSON
    # quoted inside an answer never executes.
    bare = request_shape.bare_request_text(text)
    if bare is None:
        return None
    try:
        data = json.loads(bare, strict=False)
    except (json.JSONDecodeError, ValueError):
        return None
    return request_shape.request_from_data(data)


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
        if mind is not None or context_manager is not None:
            # Prime owns the boundary. Rebinding through its seam also updates
            # recall and learning handlers, so an explicit store cannot split
            # Aurora's conversation from the capability store.
            binder = getattr(prime, "bind_memory_boundary", None)
            if not callable(binder):
                raise TypeError("Prime does not support a shared memory boundary")
            self._mind = binder(context_manager=context_manager, mind=mind)
        else:
            # Default path: share Prime's single Mind, so the task-local
            # Working Memory belongs to the Prime/Aurora pair, not to a
            # wrapper created per Aurora instance.
            self._mind = getattr(prime, "mind", None) or Mind(prime.context)
        self._context = self._mind.mechanisms
        # Session-scoped, volatile confirmation for consequential forgetting.
        # Never persist or reuse a pending command after a restart.
        self._pending_forgets: dict[str, str] = {}

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
            # Aurora's normal model generation owns intent and memory
            # decisions.  Prime remains the only retrieval/admission boundary;
            # there is no compulsory semantic-planning generation before a
            # conversational turn.
            enable_memory_interpreter=False,
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
        capability_instruction = build_capability_instruction(self._prime)
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

        # The live llama.cpp path has a conservative character-based token
        # counter. A prompt that fits the model tokenizer may exceed that
        # estimate, leaving no room even for a two-word user turn. Fit the
        # *optional catalogues* before Mind.build applies the hard ceiling;
        # never raise the ceiling or truncate the user/identity/safety rules.
        from PRIMORA.aurora.prompt_fit import fit_system_prompt

        has_system = bool(messages and messages[0].get("role") == "system")
        core = (str(messages[0].get("content", ""))
                if has_system else self._config.system_instruction)
        # A caller may return a previously assembled system message.
        # Don't append the same catalogue again on each invocation.
        if "Available Prime Capabilities" in core:
            capability_instruction = ""
        specialist_instruction = (
            "" if "Available Specialists:" in core
            else build_specialist_instruction(self._prime)
        )
        tool_instruction = (
            "" if "Available Tools:" in core
            else build_tool_instruction(self._prime)
        )
        identity = "" if IDENTITY_HEADER in core else identity_instruction
        guidance = ("" if GUIDANCE_HEADER in core else guidance_instruction)

        def measured_tokens(content: str) -> int:
            try:
                amount = int(self._brain.count_tokens([
                    {"role": "system", "content": content}
                ]))
                if amount > 0:
                    return amount
            except Exception:
                pass
            # Exact counting is not available before HF model warmup. Match
            # OpenAIBrain's conservative estimator rather than undercount.
            return max(1, len(content) // 4) + 4

        system_content = fit_system_prompt(
            core, identity, capability_instruction,
            specialist_instruction, tool_instruction, guidance,
            count=measured_tokens,
            ceiling=self._config.context_ceiling,
            target=self._config.context_target,
        )
        return [{"role": "system", "content": system_content}] + (
            list(messages[1:]) if has_system else list(messages)
        )

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
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> BrainResult:
        """One logical response, possibly cognated over several segments.

        The loop owns the segment structure; this method owns only the
        instruction. When ``on_delta`` is supplied, user-facing content
        deltas are forwarded to it as they are generated (Verse streaming);
        when omitted, the existing non-streaming behavior is preserved.
        The signature and return type are unchanged for existing callers.
        """
        # Explicit forgetting is a Prime-owned memory mutation, not a
        # conversational promise. Process it before the model can see or
        # restate historical material; acknowledge only after the SQLite
        # transaction and verification complete.
        from PRIMORA.prime.context.forgetting import parse_forget_request
        latest_user = next((
            str(m.get("content", "")) for m in reversed(messages)
            if m.get("role") == "user"
        ), "")
        topic = parse_forget_request(latest_user)
        confirmation = re.fullmatch(
            r"\\s*confirm\\s+forget\\s+(.+?)\\s*[.!?]*\\s*",
            latest_user, re.IGNORECASE,
        )
        if topic is not None:
            self._pending_forgets[session_id] = topic
            response = BrainResult.success(
                f"Please confirm: suppress all stored memory about {topic}? "
                f"Reply exactly 'confirm forget {topic}' to proceed. "
                "This hides old memory from Aurora but does not physically "
                "erase the SQLite records or backups.",
                self._brain.get_model_name(),
            )
            if on_delta is not None:
                on_delta(response.content)
            return response
        if confirmation is not None:
            from PRIMORA.prime.context.forgetting import normalize_topic
            try:
                confirmed = normalize_topic(confirmation.group(1))
            except ValueError:
                confirmed = ""
            pending = self._pending_forgets.pop(session_id, None)
            if confirmed != pending or not pending:
                return BrainResult.success(
                    "No matching forget request is pending; nothing changed.",
                    self._brain.get_model_name(),
                )
            topic = pending
            try:
                receipt = self._context.forget_topic(topic)
                if receipt.get("status") != "suppressed":
                    raise RuntimeError("no verified suppression receipt")
            except Exception as exc:
                return BrainResult.failure(
                    f"Couldn't suppress that memory topic: {exc}",
                    self._brain.get_model_name(),
                )
            response = BrainResult.success(
                f"I've suppressed my stored memories about {topic} from "
                "conversation, memory retrieval, and active context. "
                "Original SQLite records and backups are not physically "
                "deleted. I won't use those old records in future replies.",
                self._brain.get_model_name(),
            )
            if on_delta is not None:
                on_delta(response.content)
            return response
        # Any other turn cancels pending authorization. A later unrelated
        # voice input can never accidentally confirm a stale command.
        self._pending_forgets.pop(session_id, None)
        # A broad or ambiguous request (e.g. "forget yesterday") must
        # never fall through to the language model, which could falsely
        # promise deletion without an operation.
        if re.match(r"^\\s*(?:hey\\s+aurora[,!.]?\\s*)?(?:please\\s+)?(?:forget|erase|delete)\\b",
                    latest_user, re.IGNORECASE):
            return BrainResult.success(
                "No memory was changed. To suppress a specific subject, "
                "say 'forget everything about [topic]'.",
                self._brain.get_model_name(),
            )
        try:
            safe_messages = self._context.filter_conversation_input(messages)
        except Exception as exc:
            return BrainResult.failure(
                f"Memory privacy filter unavailable: {exc}",
                self._brain.get_model_name(),
            )
        return self._loop.run(
            self._with_instruction(safe_messages), session_id,
            on_delta=on_delta)

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
