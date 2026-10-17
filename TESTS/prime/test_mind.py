"""Milestone 1: the Mind boundary.

Two properties matter most and are asserted first:

1. The unified context Aurora receives is byte-identical to what the raw
   ContextManager produced before the Mind existed. A façade that changes
   the message list is not a façade.
2. Working Memory stays runtime/task-local: nothing about it is persisted.

The rest covers the composition surface: Active Memory access, M6 retrieval
as an unchanged mechanism, and the compatibility of the existing Aurora and
Prime entry points.
"""

import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context import (
    BudgetConfig,
    CognitiveContext,
    ContextManager,
    Mind,
    RetrievalSummary,
    WorkingMemory,
)
from PRIMORA.prime.providers import Provider

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db(name=None):
    _SEQ[0] += 1
    return os.path.join(_TMP.name, f"{name or 'mind'}-{_SEQ[0]}.db")


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget():
    return BudgetConfig(
        input_target=600, input_ceiling=1200,
        output_reserve=64, total_budget=1400,
    )


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.failure("blocked")


class _StubBrain(Brain):
    """Records what it was asked to generate from."""

    def __init__(self, reply="ok"):
        self._reply = reply
        self.calls = []

    def generate(self, messages, max_output_tokens=None):
        self.calls.append([dict(m) for m in messages])
        return BrainResult.success(self._reply, "stub")

    def get_model_name(self):
        return "stub"

    def count_tokens(self, messages):
        return _counter(messages)


def _config():
    return AuroraConfig(
        model_path=os.path.join("RESOURCES", "aurora", "v0.1", "model"),
        adapter_path=os.path.join("RESOURCES", "aurora", "v0.1", "aurora_qlora"),
    )


def _code_only(*parts):
    """Source with comments and docstrings removed.

    Boundary tests must inspect executable code, not prose. A raw text scan
    of a module whose docstring explains *why* Jev is absent would match the
    word "Jev" and fail for the wrong reason.
    """
    import ast

    text = "\n".join(
        open(os.path.join("PRIMORA", *p), encoding="utf-8").read()
        for p in parts
    )
    tree = ast.parse(text)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                body.pop(0)
    return ast.unparse(tree)


def _aurora(manager=None, brain=None):
    prime = Prime(_Provider())
    if manager is not None:
        prime._context = manager
    return Aurora(prime, brain=brain or _StubBrain(), config=_config(),
                  context_manager=manager)


# ======================================================================
# 1. Parity: the unified context is byte-identical to the pre-Mind view
# ======================================================================
class TestParityWithContextManager:
    def _history(self):
        return [
            {"role": "user", "content": "The vault code is BLUE-FALCON-7."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "what is my favorite color?"},
        ]

    def test_messages_are_identical_to_the_raw_manager(self):
        # Same underlying store on both sides: the Mind wraps the very
        # manager whose output it must reproduce.
        manager = ContextManager(db_path=_db("parity"))
        manager.reconcile(
            [{"role": "user", "content": "My favorite color is blue."}], "s")
        history = self._history()

        raw = manager.build(
            history, "s", token_counter=_counter, max_new_tokens=64,
            budget=_budget(),
        ).view
        mind = Mind(manager)
        unified = mind.build(
            history, "s", token_counter=_counter, max_new_tokens=64,
            budget=_budget(),
        )

        assert unified.messages == raw.messages, (
            "Mind must not change what Aurora is given"
        )
        assert unified.input_tokens == raw.input_tokens
        assert unified.allowed_output == raw.allowed_output
        assert unified.session_id == raw.session_id

    def test_aurora_generates_from_identical_messages(self):
        history = self._history()

        a_brain = _StubBrain("same")
        a = _aurora(ContextManager(db_path=_db("p1")), brain=a_brain)
        a.respond(list(history), session_id="s")

        b_brain = _StubBrain("same")
        prime = Prime(_Provider())
        b = Aurora(prime, brain=b_brain, config=_config(),
                   mind=Mind(ContextManager(db_path=_db("p2"))))
        b.respond(list(history), session_id="s")

        assert a_brain.calls == b_brain.calls, (
            "the messages handed to the brain must be unchanged"
        )

    def test_unified_view_property_matches_build_result_view(self):
        manager = ContextManager(db_path=_db("view"))
        raw = manager.build(
            self._history(), "s", token_counter=_counter,
            max_new_tokens=64, budget=_budget(),
        ).view
        unified = Mind(ContextManager(db_path=_db("view"))).build(
            self._history(), "s", token_counter=_counter,
            max_new_tokens=64, budget=_budget(),
        ).view
        assert unified.messages == raw.messages
        assert unified.input_tokens == raw.input_tokens

    def test_budget_ceiling_still_raises(self):
        from PRIMORA.prime.context import ContextTooLargeError
        manager = ContextManager(db_path=_db("ceil"))
        mind = Mind(manager)
        huge = [{"role": "user", "content": "word " * 4000}]
        with pytest.raises(ContextTooLargeError):
            mind.build(huge, "s", token_counter=_counter,
                       max_new_tokens=64, budget=_budget())


# ======================================================================
# 2. Working Memory stays runtime and task-local
# ======================================================================
class TestWorkingMemory:
    def test_working_defaults_to_the_minds_own_set(self):
        manager = ContextManager(db_path=_db("wm"))
        manager.reconcile([{"role": "user", "content": "hello there"}], "s")
        mind = Mind(manager)
        mind.begin_task("s", "diagnose the memory")
        mind.working.set_objective("find the leak")

        ctx = mind.build(
            [{"role": "user", "content": "what is the task?"}], "s",
            token_counter=_counter, max_new_tokens=64, budget=_budget(),
        )
        blob = " ".join(m["content"] for m in ctx.messages)
        assert "[working task]" in blob
        assert "[working objective]" in blob
        assert ctx.working is mind.working

    def test_working_is_never_persisted(self):
        manager = ContextManager(db_path=_db("wm2"))
        mind = Mind(manager)
        mind.begin_task("s", "a private task")
        mind.working.note_decision("do not persist me")
        mind.working.add_evidence("user", "sensitive evidence")
        mind.build([{"role": "user", "content": "go"}], "s",
                   token_counter=_counter, max_new_tokens=64,
                   budget=_budget())
        con = sqlite3.connect(f"file:{manager._db_path}?mode=ro", uri=True)
        try:
            for table in ("states", "entities", "candidates"):
                rows = con.execute(f"SELECT * FROM {table}").fetchall()
                assert not any(
                    "do not persist me" in str(r) or "sensitive" in str(r)
                    for r in rows
                ), f"working memory leaked into {table}"
            body = " ".join(
                str(r) for r in con.execute("SELECT content FROM messages")
            )
            assert "do not persist me" not in body
        finally:
            con.close()

    def test_working_is_per_mind_not_global(self):
        a = Mind(ContextManager(db_path=_db("wm3")))
        b = Mind(ContextManager(db_path=_db("wm4")))
        a.begin_task("s", "task A")
        assert b.working is None, "working memory must not be shared"

    def test_end_task_clears_the_working_set(self):
        manager = ContextManager(db_path=_db("wm5"))
        mind = Mind(manager)
        mind.begin_task("s", "temporary")
        mind.end_task()
        assert mind.working is None
        ctx = mind.build(
            [{"role": "user", "content": "anything?"}], "s",
            token_counter=_counter, max_new_tokens=64, budget=_budget(),
        )
        assert "[working task]" not in " ".join(
            m["content"] for m in ctx.messages)

    def test_working_is_task_local_not_persistent_across_minds(self):
        manager = ContextManager(db_path=_db("wm6"))
        first = Mind(manager)
        first.begin_task("s", "original task")
        second = Mind(manager)
        ctx = second.build(
            [{"role": "user", "content": "hi"}], "s",
            token_counter=_counter, max_new_tokens=64, budget=_budget(),
        )
        assert "[working task]" not in " ".join(
            m["content"] for m in ctx.messages)


# ======================================================================
# 3. Active Memory access
# ======================================================================
class TestActiveMemoryAccess:
    def test_active_memory_is_reachable_through_the_mind(self):
        manager = ContextManager(db_path=_db("am"))
        mind = Mind(manager)
        mind.reconcile(
            [{"role": "user", "content": "My favorite color is blue."}], "s")
        active = mind.active_memory("s")
        assert any(item.concept == "favorite color" for item in active.items)

    def test_active_memory_is_included_in_the_unified_context(self):
        manager = ContextManager(db_path=_db("am2"))
        mind = Mind(manager)
        mind.reconcile(
            [{"role": "user", "content": "My favorite color is blue."}], "s")
        ctx = mind.build(
            [{"role": "user", "content": "what is my favorite color?"}], "s",
            token_counter=_counter, max_new_tokens=64, budget=_budget(),
        )
        assert ctx.active_memory is not None
        assert "The user's favorite color is blue." in " ".join(
            m["content"] for m in ctx.messages)

    def test_active_memory_can_be_excluded_from_the_summary(self):
        manager = ContextManager(db_path=_db("am3"))
        mind = Mind(manager)
        mind.reconcile(
            [{"role": "user", "content": "My favorite color is blue."}], "s")
        ctx = mind.build(
            [{"role": "user", "content": "hi"}], "s", token_counter=_counter,
            max_new_tokens=64, budget=_budget(),
            include_active_memory=False,
        )
        assert ctx.active_memory is None

    def test_select_active_delegates_to_the_existing_rule(self):
        manager = ContextManager(db_path=_db("am4"))
        mind = Mind(manager)
        mind.reconcile(
            [{"role": "user", "content": "My favorite color is blue."}], "s")
        chosen = mind.select_active("s", "favorite color", limit=3)
        assert [i.concept for i in chosen] == \
            [i.concept for i in manager.select_active_memory(
                "s", "favorite color", 3)]


# ======================================================================
# 4. M6 retrieval is used as a mechanism, unchanged
# ======================================================================
class TestRetrievalMechanism:
    def test_retrieval_summary_reports_m6_activity(self):
        manager = ContextManager(db_path=_db("m6"))
        mind = Mind(manager)
        mind.reconcile(
            [{"role": "user", "content": "the vault code is BLUE-FALCON-7"}],
            "s")
        mind.reconcile(
            [{"role": "user", "content": "unrelated chatter about lunch"}],
            "s")
        ctx = mind.build(
            [{"role": "user", "content": "what is the vault code?"}], "s",
            token_counter=_counter, max_new_tokens=64, budget=_budget(),
        )
        assert isinstance(ctx.retrieval, RetrievalSummary)
        assert ctx.retrieval.hybrid is True
        assert ctx.retrieval.fused_count > 0
        assert "lexical" in ctx.retrieval.channel_counts

    def test_mind_does_not_change_m6_thresholds(self):
        from PRIMORA.prime.context import hybrid as hybrid_mod
        from PRIMORA.prime.context import semantic as semantic_mod
        assert hybrid_mod.SEMANTIC_TAU == 0.45
        assert hybrid_mod.RRF_K == 60.0
        assert hybrid_mod.EVIDENCE_FLOOR_RATIO == 0.725
        assert semantic_mod.MODEL_ID == "sentence-transformers/all-MiniLM-L6-v2"

    def test_mind_adds_no_retrieval_channel(self):
        from PRIMORA.prime.context import hybrid as hybrid_mod
        assert set(hybrid_mod.CHANNELS) == {
            "lexical", "semantic", "entity", "relationship"}

    def test_mind_module_does_not_import_mechanisms_directly(self):
        """Mind composes; it must not re-implement or re-rank.

        Reading a read-only flag off the manager is allowed. Importing the
        retrieval engine, naming its calibration, or issuing SQL is not.
        """
        code = _code_only(("prime", "context", "mind.py"))
        for banned in ("SEMANTIC_TAU", "RRF_K", "EVIDENCE_FLOOR",
                       "import hybrid", "from .hybrid", "from .semantic",
                       "from .scoring", "import torch", "sentence_transformers",
                       "INSERT", "SELECT", "decision"):
            assert banned not in code, banned
        # The only M6 coupling is a read-only boolean.
        assert "hybrid_retrieval" in code


# ======================================================================
# 5. Composition and compatibility
# ======================================================================
class TestMindComposition:
    def test_mind_requires_a_manager(self):
        with pytest.raises(ValueError):
            Mind(None)

    def test_mechanisms_are_reachable_and_unchanged(self):
        manager = ContextManager(db_path=_db("mech"))
        mind = Mind(manager)
        assert mind.mechanisms is manager
        assert isinstance(mind.mechanisms, ContextManager)

    def test_reconcile_delegates_to_formation(self):
        manager = ContextManager(db_path=_db("rec"))
        mind = Mind(manager)
        mind.reconcile(
            [{"role": "user", "content": "The vault code is BLUE-FALCON-7"}],
            "s")
        con = sqlite3.connect(f"file:{manager._db_path}?mode=ro", uri=True)
        try:
            assert con.execute(
                "SELECT count(*) FROM messages").fetchone()[0] == 1
            assert con.execute(
                "SELECT count(*) FROM states").fetchone()[0] == 1
        finally:
            con.close()

    def test_m4_formation_still_runs_through_the_mind(self):
        manager = ContextManager(db_path=_db("m4"))
        mind = Mind(manager)
        mind.reconcile([{"role": "user", "content": "My name is Amir."}], "s")
        con = sqlite3.connect(f"file:{manager._db_path}?mode=ro", uri=True)
        try:
            assert con.execute(
                "SELECT count(*) FROM entities WHERE display_name='Amir'"
            ).fetchone()[0] == 1, "M5 formation must still work under Mind"
        finally:
            con.close()

    def test_last_context_is_retained(self):
        mind = Mind(ContextManager(db_path=_db("last")))
        assert mind.last_context is None
        mind.build([{"role": "user", "content": "hi"}], "s",
                   token_counter=_counter, max_new_tokens=64,
                   budget=_budget())
        assert isinstance(mind.last_context, CognitiveContext)

    def test_status_is_read_only(self):
        mind = Mind(ContextManager(db_path=_db("status")))
        status = mind.status()
        assert status["working"] is False
        assert status["has_context"] is False
        assert "semantic_index" in status
        mind.build([{"role": "user", "content": "hi"}], "s",
                   token_counter=_counter, max_new_tokens=64,
                   budget=_budget())
        assert mind.status()["has_context"] is True

    def test_context_for_is_an_alias_for_build(self):
        mind = Mind(ContextManager(db_path=_db("alias")))
        a = mind.build([{"role": "user", "content": "hi"}], "s",
                       token_counter=_counter, max_new_tokens=64,
                       budget=_budget())
        b = mind.context_for([{"role": "user", "content": "hi"}], "s",
                             token_counter=_counter, max_new_tokens=64,
                             budget=_budget())
        assert a.messages == b.messages

    def test_with_appended_preserves_the_first_leg(self):
        mind = Mind(ContextManager(db_path=_db("append")))
        first = mind.build([{"role": "user", "content": "compute"}], "s",
                           token_counter=_counter, max_new_tokens=64,
                           budget=_budget())
        second = first.with_appended(
            first.messages + [{"role": "tool", "content": "{}"}],
            ["tool_result"],
        )
        assert len(second.messages) == len(first.messages) + 1
        assert "tool_result" in second.diagnostics
        assert first.diagnostics == second.diagnostics[:-1]


class TestAuroraIntegration:
    def test_aurora_exposes_its_mind(self):
        aurora = _aurora(ContextManager(db_path=_db("a1")))
        assert isinstance(aurora.mind, Mind)

    def test_legacy_context_manager_argument_still_works(self):
        manager = ContextManager(db_path=_db("a2"))
        aurora = Aurora(Prime(_Provider()), brain=_StubBrain(),
                        config=_config(), context_manager=manager)
        assert isinstance(aurora.mind, Mind)
        assert aurora.mind.mechanisms is manager

    def test_mind_argument_is_accepted(self):
        mind = Mind(ContextManager(db_path=_db("a3")))
        aurora = Aurora(Prime(_Provider()), brain=_StubBrain(),
                        config=_config(), mind=mind)
        assert aurora.mind is mind

    def test_aurora_without_explicit_context_uses_prime_mind(self):
        prime = Prime(_Provider())
        aurora = Aurora(prime, brain=_StubBrain(), config=_config())
        assert aurora.mind is prime.mind

    def test_respond_still_works_through_the_mind(self):
        brain = _StubBrain("hello from the stub")
        aurora = _aurora(ContextManager(db_path=_db("a4")), brain=brain)
        result = aurora.respond(
            [{"role": "user", "content": "hello"}], session_id="s")
        assert result.success
        assert result.content == "hello from the stub"
        assert brain.calls, "the brain must have been called"

    def test_respond_uses_one_unified_context_per_leg(self):
        brain = _StubBrain()
        aurora = _aurora(ContextManager(db_path=_db("a5")), brain=brain)
        aurora.respond(
            [{"role": "user", "content": "what is my favorite color?"}],
            session_id="s")
        assert isinstance(aurora.mind.last_context, CognitiveContext)
        assert aurora.mind.last_context.active_memory is not None

    def test_context_for_exposes_the_context_without_generating(self):
        brain = _StubBrain()
        aurora = _aurora(ContextManager(db_path=_db("a6")), brain=brain)
        ctx = aurora.context_for(
            [{"role": "user", "content": "hello"}], session_id="s")
        assert isinstance(ctx, CognitiveContext)
        assert ctx.messages
        assert not brain.calls, "context_for must not run the brain"

    def test_prime_exposes_its_mind(self):
        prime = Prime(_Provider())
        assert isinstance(prime.mind, Mind)
        assert prime.mind.mechanisms is prime.context

    def test_aurora_legacy_context_attribute_still_points_at_mechanisms(self):
        manager = ContextManager(db_path=_db("a7"))
        aurora = Aurora(Prime(_Provider()), brain=_StubBrain(),
                        config=_config(), context_manager=manager)
        assert aurora._context is manager

    def test_context_for_large_message_still_raises(self):
        from PRIMORA.prime.context import ContextTooLargeError
        aurora = _aurora(ContextManager(db_path=_db("a8")))
        with pytest.raises(ContextTooLargeError):
            aurora.context_for(
                [{"role": "user", "content": "word " * 4000}],
                session_id="s")


class TestMilestoneBoundaries:
    """M1 introduces the boundary only. Later milestones are not here."""

    def test_no_cognitive_loop(self):
        code = _code_only(("prime", "context", "mind.py"))
        for banned in ("while True", "def step", "def tick",
                       "def observe", "def reflect", "def perceive"):
            assert banned not in code, banned

    def test_no_decision_layer_or_autonomy(self):
        """The M1 boundary, minus the part learning legitimately widens.

        Milestone 1 forbade Jev in the Mind outright. Milestone 3's
        ``learn`` capability needs a learning collaborator, so the ban is
        narrowed rather than dropped: the Mind may *pass a lesson to an
        opaque collaborator*, but it still decides nothing, advises nobody,
        and holds no goal of its own. The word may appear in a docstring; the
        decision verbs may not appear at all.
        """
        code = _code_only(
            ("prime", "context", "mind.py"), ("aurora", "aurora.py"),
        ).lower()
        for banned in ("decision", "decide", "advis", "autonom",
                       "proactive", "trigger", "goal"):
            assert banned not in code, banned

    def test_mind_does_not_import_the_jev_module(self):
        """The collaborator is passed through opaquely, not constructed."""
        import ast
        tree = ast.parse(
            open(os.path.join("PRIMORA", "prime", "context", "mind.py"),
                 encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "learning" not in (node.module or ""), node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "learning" not in (alias.name or ""), alias.name

    def test_no_new_model_or_dependencies(self):
        code = _code_only(("prime", "context", "mind.py"))
        for banned in ("torch", "transformers", "requests", "urllib",
                       "openai", "subprocess", "sentence_transformers"):
            assert banned not in code, banned

    def test_mind_persists_nothing_of_its_own(self):
        code = _code_only(("prime", "context", "mind.py"))
        for banned in ("CREATE TABLE", "INSERT INTO", "UPDATE ",
                       "DELETE FROM", "ContextStore"):
            assert banned not in code, banned
