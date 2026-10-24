"""Milestone 3: arbitrary tool calling.

The properties that matter:

1. Aurora can select a registered tool, and it runs with validated arguments.
2. Anything not registered, not available, or not schema-valid is refused
   before the handler exists to the caller.
3. The capability budget and the repeat guard apply to tools unchanged.
4. Tool output is bounded.
5. Tool-derived knowledge is never user knowledge, and cannot be relabelled
   as such across turns.
6. There is no path from a model to Python, a shell, a file, or a socket.
"""

import ast
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.learning import Jev, LearningOutcome, StructuredCandidate
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeRequest, PrimeResult
from PRIMORA.prime.providers import Provider
from PRIMORA.specialists.tools.use_tool import UseToolSpecialist
from PRIMORA.toolbox import ToolRegistry, ToolResult, ToolSpec
from PRIMORA.toolbox.validation import validate

LONG = "x" * 9000


def _code(*relative_paths):
    """Executable source with docstrings removed.

    Safety tests scan for banned words. A raw scan matches the prose that
    explains *why* a thing is absent ("opens no socket") and fails for
    exactly the wrong reason.
    """
    tree = ast.parse(open(os.path.join(*relative_paths),
                           encoding="utf-8").read())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _request(action, payload):
    return (
        "<prime_request>\n"
        + json.dumps({"action": action, "payload": payload,
                      "context": None, "metadata": None})
        + "\n</prime_request>"
    )


def _echo(arguments):
    return ToolResult.success({"got": arguments})


def _boom(arguments):
    raise RuntimeError("handler exploded")


def _huge(arguments):
    return ToolResult.success("y" * 50000)


ECHO = ToolSpec(
    name="echo",
    description="Return the arguments it was given.",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "maxLength": 50},
            "times": {"type": "integer", "minimum": 1, "maximum": 5},
            "mode": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    handler=_echo,
)

COUNTED = ToolSpec(
    name="counted",
    description="A tool that records how often it ran.",
    input_schema={"type": "object", "properties": {}, "required": [],
                  "additionalProperties": False},
    handler=_echo,
)
COUNTER = {"n": 0}


def _counting(arguments):
    COUNTER["n"] += 1
    return ToolResult.success({"calls": COUNTER["n"]})


COUNTED_HANDLER = ToolSpec(
    name="counted",
    description="A tool that records how often it ran.",
    input_schema={"type": "object", "properties": {
        "n": {"type": "integer", "minimum": 0, "maximum": 100}},
        "required": [], "additionalProperties": False},
    handler=_counting,
)

BOOM = ToolSpec(name="boom", description="Always fails.",
                input_schema={"type": "object", "properties": {},
                              "required": [], "additionalProperties": False},
                handler=_boom)

HUGE = ToolSpec(name="huge", description="Returns far too much.",
                input_schema={"type": "object", "properties": {},
                              "required": [], "additionalProperties": False},
                handler=_huge, max_output_chars=200)

OFFLINE = ToolSpec(name="offline", description="Registered but unavailable.",
                   input_schema={"type": "object", "properties": {},
                                 "required": [], "additionalProperties": False},
                   handler=_echo, available=False)

NO_HANDLER = ToolSpec(name="stub", description="Registered with no handler.",
                      input_schema={"type": "object", "properties": {},
                                    "required": [], "additionalProperties": False},
                      handler=None)


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"value": 4})


class _Jev(Jev):
    def evaluate(self, request):
        return LearningOutcome.confirmed([
            StructuredCandidate(concept="result", value="known",
                                memory_type="FACT")])


class _Scripted(Brain):
    def __init__(self, *replies):
        self.replies = list(replies) or ["done"]
        self.calls = []

    def generate(self, messages, max_output_tokens=None):
        self.calls.append([dict(m) for m in messages])
        r = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(r, BrainResult):
            return r
        return BrainResult.success(r, "scripted")

    def get_model_name(self):
        return "scripted"

    def count_tokens(self, messages):
        return sum(len(m.get("content", "").split()) + 2 for m in messages)


_SEQ = [0]


def _prime(*specs, tools=None, jev=None):
    _SEQ[0] += 1
    registry = tools if tools is not None else ToolRegistry(list(specs))
    prime = Prime(_Provider(), jev=jev or _Jev(), tools=registry)
    prime._context = prime._context.__class__(
        db_path=os.path.join(tempfile.mkdtemp(prefix="tool_"),
                            f"t{_SEQ[0]}.db"))
    prime._mind = prime._mind.__class__(prime._context)
    for name in ("recall", "learn"):
        cap = prime.registry.get_specialist(name)
        if cap is not None:
            cap.bind(prime._mind)
    return prime


def _use(prime, name, arguments):
    return prime.handle(PrimeRequest("use_tool",
                                     {"name": name, "arguments": arguments}))


def _aurora(prime, brain, max_cycles=2, max_calls=None):
    a = Aurora(prime, brain=brain,
               config=AuroraConfig(model_path="m", adapter_path="a"))
    a.loop.max_cycles = max_cycles
    if max_calls is not None:
        a.loop.executor.max_calls = max_calls
    return a


def _tool(brain, index, last=False):
    tools = [m for m in brain.calls[index] if m["role"] == "tool"]
    return tools[-1]["content"] if last else tools[0]["content"]


# ======================================================================
# 1. The registry
# ======================================================================


class TestToolRegistry:
    def test_empty_by_default(self):
        assert len(_prime().tools) == 0

    def test_registration_and_lookup(self):
        registry = ToolRegistry([ECHO])
        assert registry.has("echo")
        assert registry.get("echo").description

    def test_duplicate_refused(self):
        with pytest.raises(ValueError):
            ToolRegistry([ECHO, ECHO])

    def test_requires_a_tool_spec(self):
        with pytest.raises(TypeError):
            ToolRegistry().register({"name": "x"})

    def test_name_and_description_required(self):
        with pytest.raises(ValueError):
            ToolSpec(name="", description="d", input_schema={})
        with pytest.raises(ValueError):
            ToolSpec(name="n", description="", input_schema={})

    def test_unavailable_is_hidden_from_aurora(self):
        registry = ToolRegistry([ECHO, OFFLINE])
        assert "offline" in registry.names()
        assert [t["name"] for t in registry.describe()] == ["echo"]

    def test_builtins_are_local_and_deterministic(self):
        registry = ToolRegistry.with_builtins()
        assert set(registry.names()) == {
            "convert_units", "describe_text", "current_time", "count_words"}
        for spec in registry:
            assert spec.handler is not None

    def test_schema_is_required(self):
        with pytest.raises(ValueError):
            ToolSpec(name="n", description="d", input_schema="not a schema")

    def test_describe_exposes_no_implementation_detail(self):
        for entry in ToolRegistry.with_builtins().describe():
            assert set(entry) == {"name", "description", "input_schema"}
            text = json.dumps(entry)
            for banned in ("handler", "PRIMORA", ".py", "lambda", "def "):
                assert banned not in text, banned


# ======================================================================
# 2. Schema validation
# ======================================================================


class TestValidation:
    def test_valid_arguments_pass(self):
        cleaned, errors = validate({"text": "hi"}, ECHO.input_schema)
        assert cleaned == {"text": "hi"} and errors == []

    def test_missing_required_is_rejected(self):
        cleaned, errors = validate({}, ECHO.input_schema)
        assert cleaned is None and any("missing" in e for e in errors)

    def test_wrong_type_is_rejected(self):
        cleaned, errors = validate({"text": 5}, ECHO.input_schema)
        assert cleaned is None and any("type string" in e for e in errors)

    def test_unknown_argument_is_rejected(self):
        cleaned, _ = validate(
            {"text": "hi", "shell": "rm -rf /"}, ECHO.input_schema)
        assert cleaned is None

    def test_enum_is_enforced(self):
        assert validate({"text": "a", "mode": "z"}, ECHO.input_schema)[0] is None
        assert validate({"text": "a", "mode": "a"}, ECHO.input_schema)[0] is not None

    def test_numeric_bounds_are_enforced(self):
        assert validate({"text": "a", "times": 0}, ECHO.input_schema)[0] is None
        assert validate({"text": "a", "times": 9}, ECHO.input_schema)[0] is None

    def test_string_length_is_enforced(self):
        assert validate({"text": LONG}, ECHO.input_schema)[0] is None

    def test_boolean_is_not_a_number(self):
        spec = ToolSpec(name="n", description="d",
                        input_schema={"type": "object", "properties": {
                            "n": {"type": "number"}}, "required": [],
                            "additionalProperties": False}, handler=_echo)
        assert validate({"n": True}, spec.input_schema)[0] is None

    def test_non_object_arguments_rejected(self):
        assert validate("rm -rf /", ECHO.input_schema)[0] is None
        assert validate([1, 2], ECHO.input_schema)[0] is None

    def test_nested_array_items_are_validated(self):
        spec = ToolSpec(name="n", description="d",
                        input_schema={"type": "object", "properties": {
                            "xs": {"type": "array",
                                   "items": {"type": "number"}}},
                            "required": [], "additionalProperties": False},
                        handler=_echo)
        assert validate({"xs": [1, 2]}, spec.input_schema)[0] is not None
        assert validate({"xs": [1, "a"]}, spec.input_schema)[0] is None

    def test_unknown_type_is_refused_not_ignored(self):
        spec = ToolSpec(name="n", description="d",
                        input_schema={"type": "object", "properties": {
                            "v": {"type": "wat"}}, "required": [],
                            "additionalProperties": False}, handler=_echo)
        assert validate({"v": 1}, spec.input_schema)[0] is None


# ======================================================================
# 3. The capability
# ======================================================================


class TestUseToolCapability:
    def test_registered(self):
        assert _prime(ECHO).registry.has("use_tool")

    def test_capabilities_registered_so_far(self):
        assert sorted(_prime().registry.get_names()) == [
            "calculate", "consult_specialist", "learn", "recall", "research",
            "research_read", "use_tool"]

    def test_valid_call_runs(self):
        result = _use(_prime(ECHO), "echo", {"text": "hello"})
        assert result.status.value == "success"
        assert result.output["result"]["output"] == {"got": {"text": "hello"}}

    def test_handler_receives_exactly_the_validated_arguments(self):
        seen = {}

        def spy(arguments):
            seen.update(arguments)
            return ToolResult.success("ok")

        spec = ToolSpec(name="spy", description="d",
                        input_schema=ECHO.input_schema, handler=spy)
        _use(_prime(spec), "spy", {"text": "hi", "times": 2})
        assert seen == {"text": "hi", "times": 2}

    def test_invalid_arguments_never_reach_the_handler(self):
        called = []

        def spy(arguments):
            called.append(arguments)
            return ToolResult.success("ok")

        spec = ToolSpec(name="spy", description="d",
                        input_schema=ECHO.input_schema, handler=spy)
        result = _use(_prime(spec), "spy", {"shell": "rm -rf /"})
        assert result.status.value == "failure"
        assert called == [], "the handler ran on invalid arguments"

    def test_unknown_tool_is_rejected(self):
        result = _use(_prime(ECHO), "evil_exec", {})
        assert result.status.value == "failure"
        assert "Unknown tool" in result.error

    def test_unknown_tool_lists_what_exists(self):
        assert "echo" in _use(_prime(ECHO), "nope", {}).error

    def test_unavailable_tool_fails_cleanly(self):
        result = _use(_prime(ECHO, OFFLINE), "offline", {})
        assert result.status.value == "failure"
        assert "unavailable" in result.error

    def test_tool_with_no_handler_fails_cleanly(self):
        assert _use(_prime(NO_HANDLER), "stub", {}).status.value == "failure"

    def test_handler_exception_fails_cleanly(self):
        result = _use(_prime(BOOM), "boom", {})
        assert result.status.value == "failure"
        assert "handler exploded" in result.error

    def test_handler_returning_junk_fails_cleanly(self):
        spec = ToolSpec(name="junk", description="d",
                        input_schema={"type": "object", "properties": {},
                                      "required": [],
                                      "additionalProperties": False},
                        handler=lambda a: "not a ToolResult")
        assert _use(_prime(spec), "junk", {}).status.value == "failure"

    def test_tool_failure_is_reported_not_raised(self):
        spec = ToolSpec(name="bad", description="d",
                        input_schema={"type": "object", "properties": {},
                                      "required": [],
                                      "additionalProperties": False},
                        handler=lambda a: ToolResult.failure("nope"))
        assert "nope" in _use(_prime(spec), "bad", {}).error

    def test_string_arguments_are_refused(self):
        assert _use(_prime(ECHO), "echo", "text").status.value == "failure"

    def test_oversized_arguments_are_refused(self):
        assert _use(_prime(ECHO), "echo", {"text": LONG}) \
            .status.value == "failure"

    def test_bad_payloads(self):
        prime = _prime(ECHO)
        for payload in ({}, {"name": ""}, {"name": 5}, {"arguments": {}},
                        None, "x"):
            assert prime.handle(
                PrimeRequest("use_tool", payload)).status.value == "failure"

    def test_rejects_none_registry(self):
        with pytest.raises(ValueError):
            UseToolSpecialist().set_tools(None)

    def test_output_is_bounded(self):
        result = _use(_prime(HUGE), "huge", {})
        rendered = json.dumps(result.output)
        assert len(rendered) < 1000, "output was not truncated"
        assert "truncated" in rendered

    def test_output_bounded_by_default_limit(self):
        spec = ToolSpec(name="big", description="d",
                        input_schema={"type": "object", "properties": {},
                                      "required": [],
                                      "additionalProperties": False},
                        handler=_huge)
        result = _use(_prime(spec), "big", {})
        body = json.dumps(result.output)
        assert len(body) < 6000


# ======================================================================
# 4. Inside the cognitive loop
# ======================================================================


class TestInsideTheLoop:
    def test_aurora_can_select_a_registered_tool(self):
        brain = _Scripted(
            _request("use_tool", {"name": "echo",
                                  "arguments": {"text": "hello"}}),
            "the tool said hello")
        a = _aurora(_prime(ECHO), brain)
        a.mind.begin_task("s", "t")
        result = a.respond([{"role": "user", "content": "use the tool"}], "s")
        assert result.content == "the tool said hello"
        assert [n for n, _ in a.loop.executor.calls] == ["use_tool"]

    def test_tool_result_reaches_the_next_segment(self):
        brain = _Scripted(
            _request("use_tool", {"name": "echo",
                                  "arguments": {"text": "kangaroo"}}),
            "done")
        a = _aurora(_prime(ECHO), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        assert "kangaroo" in _tool(brain, 1)

    def test_aurora_can_answer_using_the_result(self):
        prime = _prime(tools=ToolRegistry.with_builtins())
        brain = _Scripted(
            _request("use_tool", {"name": "convert_units", "arguments": {
                "value": 2.5, "from_unit": "kg", "to_unit": "g"}}),
            "That is 2500 grams.")
        a = _aurora(prime, brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        result = a.respond(
            [{"role": "user", "content": "2.5 kg in grams?"}], "s")
        assert "2500" in result.content

    def test_aurora_sees_the_tools_and_schemas(self):
        brain = _Scripted("done")
        a = _aurora(_prime(ECHO), brain)
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "hi"}], "s")
        system = brain.calls[0][0]["content"]
        assert "Available Tools:" in system
        assert "echo" in system
        assert "input_schema" in system or "Arguments" in system

    def test_unavailable_tool_is_not_advertised(self):
        brain = _Scripted("done")
        a = _aurora(_prime(ECHO, OFFLINE), brain)
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "hi"}], "s")
        assert "offline" not in brain.calls[0][0]["content"]

    def test_the_prompt_leaks_no_implementation_detail(self):
        brain = _Scripted("done")
        a = _aurora(_prime(tools=ToolRegistry.with_builtins()), brain)
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "hi"}], "s")
        system = brain.calls[0][0]["content"]
        # "PRIMORA" used to be banned here as the package name. It is now a
        # required identity fact (the identity block tells Aurora which
        # project she inhabits), so only genuine implementation details are
        # banned.
        for banned in ("handler", "ToolSpec", ".py", "__"):
            assert banned not in system, banned
        assert "identity in PRIMORA" in system, "the identity block must be there"

    def test_tool_failure_does_not_break_the_session(self):
        brain = _Scripted(
            _request("use_tool", {"name": "boom", "arguments": {}}),
            "the tool failed, so I cannot say")
        a = _aurora(_prime(BOOM), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        result = a.respond([{"role": "user", "content": "go"}], "s")
        assert result.success
        assert "failed" in result.content


# ======================================================================
# 5. The existing protections still apply
# ======================================================================


class TestProtectionsApply:
    def test_capability_budget_applies_to_tools(self):
        """Distinct calls, so the repeat guard is not what stops it."""
        COUNTER["n"] = 0
        brain = _Scripted(
            _request("use_tool", {"name": "counted", "arguments": {"n": 1}}),
            _request("use_tool", {"name": "counted", "arguments": {"n": 2}}),
            _request("use_tool", {"name": "counted", "arguments": {"n": 3}}),
            "done")
        a = _aurora(_prime(COUNTED_HANDLER), brain, max_cycles=4, max_calls=2)
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        assert a.loop.executor.used == 2
        assert COUNTER["n"] == 2, "the budget did not stop execution"
        assert a.loop.last_trace.stop_reason != "repeated_request"
        assert any(
            "capability_budget_exhausted" in m["content"]
            for call in brain.calls for m in call if m["role"] == "tool"
        ), "the refusal never reached Aurora"

    def test_repeat_guard_applies_to_identical_tool_calls(self):
        COUNTER["n"] = 0
        brain = _Scripted(_request("use_tool", {
            "name": "counted", "arguments": {}}))
        a = _aurora(_prime(COUNTED_HANDLER), brain, max_cycles=20, max_calls=50)
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        assert a.loop.last_trace.stop_reason == "repeated_request"
        assert COUNTER["n"] == 1, "a successful duplicate is refused"

    def test_tools_go_through_the_one_executor(self):
        """No side door around the budget."""
        prime = _prime(ECHO)
        assert isinstance(
            prime.registry.get_specialist("use_tool"), UseToolSpecialist)
        from PRIMORA.aurora.loop import CapabilityExecutor
        assert isinstance(Aurora(
            prime, brain=_Scripted("d"),
            config=AuroraConfig(model_path="m", adapter_path="a")
        ).loop.executor, CapabilityExecutor)


# ======================================================================
# 6. Tool knowledge is never user knowledge
# ======================================================================


class TestToolProvenance:
    def _session(self, replies, session="s", user="tell me about the crane"):
        prime = _prime(ECHO, jev=_Jev())
        brain = _Scripted(*replies)
        a = _aurora(prime, brain, max_cycles=3)
        a.mind.begin_task(session, "t")
        a.respond([{"role": "user", "content": user}], session)
        return brain, a

    def test_tool_facts_are_held_not_current(self):
        tool_spec = ToolSpec(
            name="facts", description="Returns a fact.",
            input_schema={"type": "object", "properties": {},
                          "required": [], "additionalProperties": False},
            handler=lambda a: ToolResult.success(
                {"note": "the harbour crane holds four containers"}))
        prime = _prime(tool_spec, jev=_Jev())
        brain = _Scripted(
            _request("use_tool", {"name": "facts", "arguments": {}}),
            _request("learn", {
                "text": "the harbour crane holds four containers",
                "provenance": "tool"}),
            "recorded")
        a = _aurora(prime, brain, max_cycles=3)
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        store = prime._context._get_store()
        assert store.fetch_states(status="current") == []
        assert store.fetch_candidates(status="pending")

    def test_tool_evidence_survives_into_a_later_turn(self):
        """The bypass is closed even across turns, in one session.

        One Aurora, one session, two turns: the tool runs in the first, and
        the claim arrives in the second. The evidence has to outlive the
        turn that produced it, or the relabelling works.
        """
        tool_spec = ToolSpec(
            name="facts", description="Returns a fact.",
            input_schema={"type": "object", "properties": {},
                          "required": [], "additionalProperties": False},
            handler=lambda a: ToolResult.success(
                {"note": "kestrel seventeen was the last cargo"}))
        prime = _prime(tool_spec, jev=_Jev())
        brain = _Scripted(
            _request("use_tool", {"name": "facts", "arguments": {}}),
            "noted",
            _request("learn", {"text": "kestrel seventeen was the last cargo",
                               "provenance": "user"}),
            "held")
        a = _aurora(prime, brain, max_cycles=2)
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "look up the cargo"}], "s")
        assert a.loop.executor.evidence, "the turn recorded no evidence"

        a.respond([{"role": "user", "content": "remember that"}], "s")
        assert any(
            "provenance_downgraded" in m["content"]
            for call in brain.calls for m in call if m["role"] == "tool"
        ), "the relabelling was accepted in a later turn"
        store = prime._context._get_store()
        assert store.fetch_states(status="current") == [], \
            "tool information became a current fact"

    def test_evidence_does_not_leak_across_sessions(self):
        ex_holder = {}

        class Watcher(Provider):
            def invoke(self, capability, payload):
                return PrimeResult.success({"value": 4})

        tool_spec = ToolSpec(
            name="facts", description="Returns a fact.",
            input_schema={"type": "object", "properties": {},
                          "required": [], "additionalProperties": False},
            handler=lambda a: ToolResult.success(
                {"note": "zephyr nine was the last cargo"}))
        prime = _prime(tool_spec, jev=_Jev())
        brain = _Scripted(_request("use_tool", {"name": "facts",
                                                "arguments": {}}), "noted")
        a = _aurora(prime, brain, max_cycles=2)
        a.mind.begin_task("one", "t")
        a.respond([{"role": "user", "content": "look it up"}], "one")
        assert a.loop.executor.evidence
        a.mind.begin_task("two", "t")
        a.respond([{"role": "user", "content": "hi"}], "two")
        assert a.loop.executor.evidence == [], \
            "provenance evidence crossed a session boundary"

    def test_evidence_is_a_fingerprint_not_raw_output(self):
        big = "s" * 4000
        tool_spec = ToolSpec(
            name="big", description="Returns a lot.",
            input_schema={"type": "object", "properties": {},
                          "required": [], "additionalProperties": False},
            handler=lambda a: ToolResult.success(big))
        prime = _prime(tool_spec)
        brain = _Scripted(_request("use_tool", {"name": "big",
                                                "arguments": {}}), "done")
        a = _aurora(prime, brain, max_cycles=2)
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        action, tokens = a.loop.executor.evidence[0]
        assert action == "use_tool"
        assert not isinstance(tokens, str), "raw output was retained"
        assert isinstance(tokens, (frozenset, set))

    def test_tool_is_a_valid_provenance_label(self):
        from PRIMORA.learning import HOLDS, PROVENANCE
        assert "tool" in PROVENANCE
        assert "tool" in HOLDS
        assert HOLDS == {"research", "specialist", "tool", "observation",
                         "inference", "unknown"}

    def test_the_authority_invariant_still_rejects_tool_facts(self):
        prime = _prime(ECHO)
        with pytest.raises(ValueError, match="user-authored"):
            prime._context._get_store().create_state(
                "s", "s", "x", "y", [1], status="current", authority="tool")


# ======================================================================
# 7. No escape hatch
# ======================================================================


class TestNoEscapeHatch:
    def test_no_builtin_touches_files_network_or_process(self):
        src = _code("PRIMORA", "toolbox", "builtin.py")
        for banned in ("open(", "urllib", "socket", "requests", "subprocess",
                       "os.system", "popen", "shutil", "pathlib", "eval",
                       "exec(", "__import__"):
            assert banned not in src, banned

    def test_the_capability_has_no_execution_primitive(self):
        src = _code("PRIMORA", "specialists", "tools", "use_tool.py")
        for banned in ("eval", "exec(", "subprocess", "__import__", "compile(",
                       "os.system", "getattr("):
            assert banned not in src, banned

    def test_the_whole_toolbox_has_no_execution_primitive(self):
        for name in ("base.py", "validation.py", "builtin.py", "__init__.py"):
            src = _code("PRIMORA", "toolbox", name)
            for banned in ("eval(", "exec(", "subprocess", "os.system",
                           "__import__", "compile("):
                assert banned not in src, f"{name}: {banned}"

    def test_the_tool_spec_has_no_room_for_a_command(self):
        fields = set(ToolSpec.__dataclass_fields__)
        assert fields == {"name", "description", "input_schema", "handler",
                          "available", "max_output_chars"}

    def test_only_callables_can_be_handlers(self):
        """A handler is a Python callable registered in code, never data."""
        registry = ToolRegistry.with_builtins()
        for spec in registry:
            assert callable(spec.handler)

    def test_no_generic_execute_tool_exists(self):
        names = ToolRegistry.with_builtins().names()
        for banned in ("execute", "run_code", "shell", "eval", "python",
                       "system", "import"):
            assert banned not in names

    def test_arguments_never_reach_a_shell(self):
        for banned in ("shell", "command", "cmd", "script", "code"):
            assert all(banned not in t.properties()
                       for t in ToolRegistry.with_builtins()), banned


# ======================================================================
# 8. Compatibility and scope
# ======================================================================


class TestCompatibilityAndScope:
    def test_existing_capabilities_still_work(self):
        prime = _prime(ECHO, tools=ToolRegistry.with_builtins())
        assert prime.registry.has("recall")
        assert prime.registry.has("research")
        assert prime.registry.has("consult_specialist")
        assert prime.registry.has("learn")

    def test_calculate_is_untouched(self):
        prime = _prime(ECHO)
        out = prime.handle(PrimeRequest("calculate", {"expression": "2+2"}))
        assert out.output["result"] == 4

    def test_existing_capabilities_were_not_rewritten_as_tools(self):
        for name in ToolRegistry.with_builtins().names():
            assert name not in ("recall", "research", "consult_specialist",
                                "learn", "calculate")

    def test_the_loop_control_flow_is_unchanged(self):
        from PRIMORA.aurora import loop as loopmod
        tree = ast.parse(open(loopmod.__file__, encoding="utf-8").read())
        run = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "run")
        src = ast.unparse(run)
        assert "ToolRegistry" not in src
        for banned in ("use_tool", "ToolRegistry", "ToolSpec", "toolbox"):
            assert banned not in src, banned

    def test_no_new_agent_or_identity(self):
        src = _code("PRIMORA", "toolbox", "base.py").lower()
        for banned in ("memory", "goal", "session_id", "identity"):
            assert banned not in src, banned

    def test_no_proactive_autonomy_or_scheduling(self):
        for name in ("base.py", "builtin.py", "validation.py"):
            src = _code("PRIMORA", "toolbox", name).lower()
            for banned in ("schedule", "trigger", "proactive", "autonom",
                           "background", "thread", "cron", "while true"):
                assert banned not in src, f"{name}: {banned}"

    def test_verse_untouched(self):
        assert os.path.isdir(os.path.join("PRIMORA", "verse", "qml"))

    def test_memory_storage_untouched(self):
        src = open(os.path.join("PRIMORA", "prime", "context", "store.py"),
                   encoding="utf-8").read().lower()
        for banned in ("tool", "toolbox"):
            assert banned not in src, banned

    def test_jev_is_undesigned(self):
        src = open(os.path.join("PRIMORA", "learning", "jev.py"),
                   encoding="utf-8").read()
        assert "toolbox" not in src and "ToolSpec" not in src

    def test_production_database_untouched(self):
        """Production memory must be unchanged by this test run.

        Content equality via production_digest(), not emptiness: a used
        install legitimately holds real memories (MEMORY_AUDIT C2), so
        the invariant is "unchanged", never "empty". Isolated work runs
        under harness_session, which refuses production opens and
        re-checks the digest on exit.
        """
        from PRIMORA.prime.context.isolation import (
            harness_session,
            production_db_path,
            production_digest,
        )
        if not production_db_path().exists():
            pytest.skip("no production database")
        before = production_digest()
        with harness_session("c2_untouched_") as session:
            store = session.context_manager()._get_store()
            # An empty starting state belongs to the private store, never
            # to production.
            assert store.fetch_states(limit=5) == []
        assert production_digest() == before, (
            "production memory changed during this test")


class TestUseToolIsConfigured:
    """An empty tool allowlist must not be advertised as a working capability."""

    def test_default_registry_is_unconfigured(self):
        prime = Prime(_Provider())
        specialist = prime.registry.get_specialist("use_tool")
        assert specialist.is_configured is False

    def test_unconfigured_is_not_advertised(self):
        from PRIMORA.aurora.configuration import build_capability_instruction
        prime = Prime(_Provider())
        block = build_capability_instruction(prime)
        assert "- use_tool:" not in block

    def test_configured_registry_is_advertised(self):
        from PRIMORA.aurora.configuration import build_capability_instruction
        from PRIMORA.toolbox import ToolRegistry, ToolSpec

        def _handler(arguments):
            return ToolResult.success(arguments)

        prime = Prime(_Provider())
        specialist = prime.registry.get_specialist("use_tool")
        specialist.set_tools(ToolRegistry([ToolSpec(
            name="echo",
            description="Return the arguments.",
            input_schema={"type": "object"},
            handler=_handler,
        )]))
        assert specialist.is_configured is True
        block = build_capability_instruction(prime)
        assert "- use_tool:" in block
