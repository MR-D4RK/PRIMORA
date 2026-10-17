"""TEMPORARY profile: Aurora on the base model with NO LoRA adapter, running
against an isolated copy of production memory.

    python TESTS\\base_only_launch.py

ONE new file. Nothing existing is modified: no source file, no adapter, no
model weight, no config file, no production database. Isolation reuses the
repository's own fail-closed machinery in PRIMORA/prime/context/isolation.py.
"""
from __future__ import annotations

import dataclasses
import os
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ------------------------------------------------------------------ overrides
# Every value below is explicit; nothing relies on a default.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "true")  # PRIMORA/main.py:9
os.environ["PRIMORA_AURORA_MODEL_PATH"] = str(
    _ROOT / "RESOURCES" / "aurora" / "v0.1" / "model"
)
# "" -> identity.resolve_identity() reports configured=False (identity.py:77),
# so the system prompt says "you run on the base model alone".
os.environ["PRIMORA_AURORA_ADAPTER_PATH"] = ""
# Setdefault, so an explicit operator environment still wins.
# This harness exists to load the HF base-model path (no LoRA), so the
# entry point's openai default must not redirect the brain to llama-server.
os.environ.setdefault("PRIMORA_AURORA_BACKEND", "transformers")
# A test harness must never spawn a real llama-server; with the backend
# pinned above, `auto` would not manage the server either - this makes the
# intent explicit and covers operators who export an openai backend.
os.environ.setdefault("PRIMORA_MODEL_SERVER", "off")
os.environ.setdefault("PRIMORA_SESSION_ID", "base_only_test")  # PRIMORA/main.py:159
# Required by --cli (PRIMORA/main.py:78); inert, construction does no I/O.
os.environ.setdefault("PRIMORA_OLLAMA_MODEL", "qwen3-8b-16k:latest")

LAUNCH_MODE = "cli"          # "cli" or "gui"
SEED_FROM_PRODUCTION = True  # False -> empty store, zero contact with production

from PRIMORA.aurora.configuration import AuroraConfig, DEFAULT_SYSTEM_INSTRUCTION
from PRIMORA.prime.context.isolation import (
    harness_session,
    production_db_path,
    production_digest,
    require_isolated,
)
from PRIMORA.prime.context.manager import ContextManager

# Set inside the session body so the load-time report can see it.
_ISOLATED_ROOT: Path | None = None

# ============================================================================
# PATCH 1 - AuroraConfig.validate(): keep EVERY check, bypass ONLY
# "the adapter directory must exist".
#
# configuration.py:305-310 has two clauses: model_path is a directory, then
# adapter_path is a directory. When adapter_path is deliberately blank we hand
# the ORIGINAL method a frozen copy whose adapter_path is the model directory,
# so both clauses still execute verbatim and a bad model_path still fails
# first. A non-blank adapter_path receives the untouched original.
# ============================================================================
_orig_validate = AuroraConfig.validate


def _validate_base_only(self: AuroraConfig):
    if (self.adapter_path or "").strip():
        return _orig_validate(self)
    return _orig_validate(dataclasses.replace(self, adapter_path=self.model_path))


AuroraConfig.validate = _validate_base_only

# ============================================================================
# PATCH 2 - the adapter wrap at brain.py:186.
#     self._model = PeftModel.from_pretrained(
#         base_model, self._config.adapter_path, device_map={"": 0})
# brain.py does `from peft import PeftModel` inside _initialize(), so the
# attribute is looked up on the class at call time. Redirect it to hand back
# the base model untouched - the rest of the production load path (nf4 + double
# quant, device_map {"":0}, fp16, sdpa, tokenizer, logging) runs verbatim.
#
# brain._initialize() returns early when self._initialized is already True
# (brain.py:87), so this fires exactly once per process.
# ============================================================================
_intercepted: list[str] = []


def _return_base_model(base_model, *args, **kwargs):
    _intercepted.append(str(args[0] if args else kwargs.get("model_id")))
    return base_model


import peft

peft.PeftModel.from_pretrained = staticmethod(_return_base_model)

# ============================================================================
# PATCH 3 - proof the instant the weights are loaded. Installed at module
# scope: no interaction with production_guard().
# ============================================================================
from PRIMORA.aurora import Aurora

_orig_warmup = Aurora.warmup

# Shared verdict: True only after _verify_model() has run and every one of the
# 18 checks passed. It starts False and is NEVER set True by anything else, so
# an early exit, a skipped warmup or a raised exception all leave it closed.
_VERIFICATION: dict = {"ok": False, "reason": "verification has not run"}


def _abort_block(title: str, lines: list[str]) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)
    for line in lines:
        print(f"  {line}")
    print("=" * 74)
    print()


def _warmup_and_verify(self):
    """Run the original warmup, then gate startup on the verification result.

    Returning a non-empty string makes PRIMORA/main.py:116-118 print the error
    and sys.exit(1) - which is BEFORE serve() at line 123 and BEFORE run_cli()
    at line 129. That is the primary "do not enter the chat" gate.
    """
    err = _orig_warmup(self)
    if err:
        _VERIFICATION["ok"] = False
        _VERIFICATION["reason"] = f"warmup failed: {err}"
        return err
    try:
        ok = _verify_model(self)
    except BaseException as exc:  # _verify_model is documented never to raise;
        ok = False  # this second net exists because the gate is absolute.
        _VERIFICATION["ok"] = False
        _VERIFICATION["reason"] = f"verification raised {type(exc).__name__}: {exc}"
    if not ok:
        return (
            "base-only verification FAILED - startup aborted before chat "
            f"({ _VERIFICATION['reason'] })"
        )
    return None


Aurora.warmup = _warmup_and_verify


def _isolated(path) -> bool:
    try:
        require_isolated(path)
        return True
    except Exception:
        return False


def _verify_model(aurora) -> bool:
    """Run all 18 checks. Returns True only if every one of them passes.

    Never raises: an unexpected exception anywhere in here is itself recorded
    as a failure and returns False (requirement 4). The 4 production-DB
    isolation checks are a mandatory group: any one of them failing fails the
    whole gate, so interaction cannot begin without isolation proven.
    """
    try:
        import torch

        cfg, model = aurora._config, aurora._brain._model
        prod = production_db_path()
        raw = getattr(aurora._prime.context, "_db_path", None)
        runtime_db = Path(raw).resolve() if raw else None

        lora_modules = [n for n, _ in model.named_modules() if "lora_" in n]
        lora_params = [n for n, _ in model.named_parameters() if "lora_" in n]
        free, total = torch.cuda.mem_get_info()

        model_checks = [
            # --- bare base model, no LoRA ---
            ("PeftModel.from_pretrained intercepted exactly once", len(_intercepted) == 1),
            ("model class is the bare base model", type(model).__name__ == "Qwen3ForCausalLM"),
            ("model is NOT a peft.PeftModel", not isinstance(model, peft.PeftModel)),
            ("model exposes no peft_config attribute", not hasattr(model, "peft_config")),
            ("zero lora_* modules", not lora_modules),
            ("zero lora_* parameters", not lora_params),
            # --- configuration ---
            ("adapter_path blank => no adapter configured", not (cfg.adapter_path or "").strip()),
            (
                "model_path is v0.1/model",
                Path(cfg.model_path).resolve()
                == (_ROOT / "RESOURCES" / "aurora" / "v0.1" / "model").resolve(),
            ),
            ("system instruction unchanged", cfg.system_instruction == DEFAULT_SYSTEM_INSTRUCTION),
            (
                "context budgets unchanged",
                (cfg.context_target, cfg.context_ceiling, cfg.context_output_reserve, cfg.context_total)
                == (1500, 1600, 256, 1856),
            ),
            # --- runtime wiring preserved ---
            ("Prime present", type(aurora._prime).__name__ == "Prime"),
            ("Mind present", getattr(aurora, "_mind", None) is not None),
            ("CognitiveLoop present", type(aurora._loop).__name__ == "CognitiveLoop"),
            ("capability/tool executor wired", aurora._loop.executor is not None),
        ]
        # Mandatory isolation group: production memory must not be reachable.
        isolation_checks = [
            ("runtime db_path is not None", runtime_db is not None),
            (
                "runtime store is NOT the production database",
                runtime_db is not None and runtime_db != prod,
            ),
            ("runtime store passes require_isolated()", runtime_db is not None and _isolated(runtime_db)),
            (
                "runtime store is under the isolated session root",
                _ISOLATED_ROOT is not None
                and runtime_db is not None
                and runtime_db.is_relative_to(_ISOLATED_ROOT),
            ),
        ]
        checks = model_checks + isolation_checks

        print()
        print("=" * 74)
        print("BASE-ONLY PROFILE - LOAD VERIFICATION (no LoRA adapter, isolated memory)")
        print("=" * 74)
        print(f"  model_path   : {cfg.model_path}")
        print(f"  adapter_path : {cfg.adapter_path!r}   (blank = not configured)")
        print(f"  intercepted  : {_intercepted!r}")
        print(f"  model class  : {type(model).__module__}.{type(model).__name__}")
        print(f"  lora modules : {len(lora_modules)}   lora params: {len(lora_params)}")
        print(f"  runtime db   : {runtime_db}")
        print(f"  VRAM         : {(total - free) / 1024**3:.2f} GB used / {total / 1024**3:.2f} GB")
        for label, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        print(f"  checks       : {len(checks)} total "
              f"({len(model_checks)} model + {len(isolation_checks)} isolation)")
        print("=" * 74)
        print()
    except BaseException as exc:
        _VERIFICATION["ok"] = False
        _VERIFICATION["reason"] = f"verification raised {type(exc).__name__}: {exc}"
        _abort_block(
            "BASE-ONLY PROFILE - VERIFICATION ABORTED (treated as FAILURE)",
            [
                f"unexpected {type(exc).__name__}: {exc}",
                "startup aborts before chat.",
                "production memory was not written by this failure path.",
            ],
        )
        return False

    failed_model = [label for label, ok in model_checks if not ok]
    failed_iso = [label for label, ok in isolation_checks if not ok]
    _VERIFICATION["ok"] = not failed_model and not failed_iso
    _VERIFICATION["reason"] = "; ".join(failed_iso + failed_model) or "all 18 checks passed"

    if not _VERIFICATION["ok"]:
        lines = [f"FAILED CHECK: {label}" for label in failed_iso + failed_model]
        if failed_iso:
            lines.insert(
                0,
                "PRODUCTION-DB ISOLATION IS MANDATORY - it did not hold, "
                "so interaction is blocked.",
            )
        lines.append("startup aborts before chat.")
        _abort_block("BASE-ONLY PROFILE - VERIFICATION FAILED (startup aborted)", lines)
    return _VERIFICATION["ok"]


# ============================================================================
# PATCH 4 - redirect memory to the isolated store.
#
# *** MUST be installed INSIDE `with harness_session(...)`. ***
# production_guard() is already active by then (isolation.py:275), so it has
# already captured the real ContextManager.__init__ and installed
# guarded_manager. Our wrapper therefore becomes outermost and the order is
#     ours -> guarded_manager -> ContextManager.__init__
# so guarded_manager sees the substituted test path. Installed before entering
# the guard the order would be reversed and guarded_manager would raise
# IsolationError on db_path=None (isolation.py:195-202).
#
# ContextManager.__init__ (manager.py:177) stores db_path; _get_store()
# (manager.py:1287) passes it to ContextStore (store.py:231), which falls back
# to default_db_path() when falsy. Prime.__init__ (prime.py:18) is the only
# runtime construction of ContextManager(), and it passes no path - so that one
# default is the whole job. Any EXPLICIT production path is still refused by
# production_guard, and any ContextStore built by another route is refused too.
# ============================================================================
def _install_db_redirect(test_db: Path) -> None:
    original = ContextManager.__init__  # == production_guard's guarded_manager

    def _redirect(self, db_path=None, *args, **kwargs):
        if db_path is None:
            db_path = str(test_db)
        return original(self, db_path, *args, **kwargs)

    ContextManager.__init__ = _redirect


def _seed_store(target: Path) -> str:
    """Copy production memory into the isolated store.

    The source connection is opened READ-ONLY - the only production contact
    anywhere in this file, and it cannot write.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if not SEED_FROM_PRODUCTION:
        return "empty store (SEED_FROM_PRODUCTION = False)"
    src_path = production_db_path()
    if not src_path.exists():
        return "empty store (production database absent)"
    src = sqlite3.connect(src_path.as_uri() + "?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)  # sqlite3 backup API: WAL-safe, FTS5-safe
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()
    return f"copy of {src_path} ({target.stat().st_size} bytes)"


def _install_chat_gate() -> None:
    """Second net (defence in depth): refuse every chat surface unless
    verification passed.

    The PRIMARY gate is _warmup_and_verify() returning an error string, which
    makes PRIMORA/main.py:116-118 sys.exit(1) before serve() (line 123) and
    run_cli() (line 129). This net closes run_cli/run_gui too, in case that
    primary path is ever bypassed. Resolution happens in PRIMORA.main's module
    globals at call time, so assigning here is enough.
    """
    import PRIMORA.main as main_mod

    orig_run_cli = main_mod.run_cli
    orig_run_gui = main_mod.run_gui

    def _guarded_run_cli(aurora):
        if not _VERIFICATION["ok"]:
            print("ERROR: verification gate is CLOSED - refusing to enter chat.")
            print(f"  reason: {_VERIFICATION['reason']}")
            return 0
        return orig_run_cli(aurora)

    def _guarded_run_gui(aurora):
        if not _VERIFICATION["ok"]:
            print("ERROR: verification gate is CLOSED - refusing to open the UI.")
            print(f"  reason: {_VERIFICATION['reason']}")
            return 1
        return orig_run_gui(aurora)

    main_mod.run_cli = _guarded_run_cli
    main_mod.run_gui = _guarded_run_gui


def _run_main() -> int:
    sys.argv = [sys.argv[0]] + (["--cli"] if LAUNCH_MODE == "cli" else [])
    _install_chat_gate()
    import PRIMORA.main as main_mod

    return int(main_mod.main() or 0)


def main_launcher() -> int:
    # Fail closed on modes that cannot be gated before interaction: GUI warms
    # up on a worker thread AFTER the window opens (PRIMORA/verse/app.py:167),
    # so there is no pre-chat point at which to verify. Only 'cli' loads the
    # model synchronously inside main() and can abort before serve()/run_cli().
    if LAUNCH_MODE != "cli":
        print("ERROR: the verification gate only runs before chat in 'cli' mode.")
        print("       Set LAUNCH_MODE = 'cli'. Refusing to start.")
        return 1

    prod = production_db_path()
    before_digest = production_digest()
    before_stat = (prod.stat().st_size, prod.stat().st_mtime) if prod.exists() else (None, None)
    seed_note, failure, code = "<not run>", None, 1

    global _ISOLATED_ROOT
    try:
        with harness_session(prefix="aurora_base_only_") as session:
            _ISOLATED_ROOT = session.root
            test_db = require_isolated(session.db_path("aurora_context.db"))
            seed_note = _seed_store(test_db)
            _install_db_redirect(test_db)  # inside the guard - see PATCH 4
            print(f"[launcher] isolated store : {test_db}")
            print(f"[launcher] seeding        : {seed_note}")
            print(f"[launcher] mode           : {LAUNCH_MODE}\n")
            try:
                code = int(_run_main() or 0)
            except BaseException as exc:
                # harness_session skips its own after-check when the body
                # raises (isolation.py:276-281), so record it here; layer 2
                # below still verifies production either way.
                failure = f"{type(exc).__name__}: {exc}"
                code = 1
    finally:
        # Layer 2: independent of harness_session, runs even on a raise.
        after_digest = production_digest()
        unchanged = before_digest == after_digest
        print()
        print("=" * 74)
        print("PRODUCTION MEMORY - PROOF OF NO WRITES")
        print("=" * 74)
        print(f"  database             : {prod}")
        print(f"  content digest before: {before_digest[:16]}")
        print(f"  content digest after : {after_digest[:16]}")
        print(f"  UNCHANGED           : {unchanged}")
        if prod.exists():
            st = prod.stat()
            print(f"  size                 : {before_stat[0]} -> {st.st_size}   (info)")
            print(f"  mtime unchanged      : {before_stat[1] == st.st_mtime}   (corroborating)")
        print(f"  isolated store       : {seed_note}")
        print(
            "  chat gate            : "
            + ("OPEN (all 18 checks passed)"
               if _VERIFICATION["ok"]
               else f"CLOSED - {_VERIFICATION['reason']}")
        )
        print("=" * 74)
        print(
            "  " + ("PASS - production memory was not written."
                   if unchanged else "FAIL - PRODUCTION MEMORY CHANGED.")
        )
        if failure:
            print(f"  run raised: {failure}")
        print()
        if not unchanged:
            code = 2
        # Verification is a precondition for a clean exit: "never ran" and
        # "ran and failed" are both failures.
        if not _VERIFICATION["ok"] and code == 0:
            code = 3
    return code


if __name__ == "__main__":
    raise SystemExit(main_launcher())
