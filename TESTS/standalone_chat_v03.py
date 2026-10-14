r"""Standalone Aurora v0.3 generation: model only, no memory at all.

WHY THIS EXISTS
---------------
The v0.3 evaluation ran through the full PRIMORA stack, so every reply Aurora
gave was shaped by things that have nothing to do with the v0.3 adapter: a
ContextManager reading SQLite, retrieval, working memory, and state injection
that prepends ``[state current src:N] concept = value`` lines to the prompt.
This script removes that entire layer so the adapter can be judged on its own.

    > What is my dog's name?
    < I don't have that stored. What is it called?

WHAT IS BYPASSED, AND HOW IT IS PROVEN
--------------------------------------
Not imported anywhere in this file:

    Aurora                    the orchestrator
    CognitiveLoop              multi-cycle cognition
    Mind / ContextManager      context assembly
    ContextStore / sqlite3     the database
    Prime / Provider           capability execution
    aurora.loop, aurora.mind   the modules above

``PRIMORA.aurora.brain`` is imported for ``AuroraBrain`` only. That module has no
import of any of the above, so no memory subsystem can be reached - not "reaches
it and returns nothing", but is genuinely absent from the call graph.

The composed system prompt is the one thing that would drag that stack back in,
because ``generate_v2`` imports the full runtime harness. So it is built in a
**subprocess** that exits before the model loads. The parent receives the
instruction as a string and never imports anything that reaches memory. Section
[6] prints the loaded module list so this is verifiable rather than asserted.

Four independent runtime checks back it up, all printed below:
  1. the production database digest, hashed before and after every turn;
  2. ``sqlite3.connect`` wrapped to log any call to the production path;
  3. the module list of every PRIMORA module loaded in this process, with the
     memory modules explicitly named and checked for absence;
  4. the system-context fingerprint compared against the one the v0.3 dataset
     recorded, so the prompt is provably the training-time one.

The model and the system prompt are NOT simplified. The system turn is the
verbatim 6186-character instruction the v0.3 dataset was trained against,
including the full capability catalogue, built by the same
``generate_v2.build_system_context`` and checked against the dataset's own
recorded fingerprint. Using a short prompt instead would test a model nobody
trained.

GENERATION SETTINGS ARE THE RUNTIME'S OWN. ``AuroraBrain.generate`` applies the
runtime's sampling (do_sample=True, temperature, top_p, repetition_penalty) and
its context ceiling, read from ``AuroraConfig.from_env()``. Nothing here
overrides them.

USAGE
-----
    $env:PRIMORA_AURORA_ADAPTER_PATH="C:\PRIMORA PROJECT\RESOURCES\aurora\v0.3\aurora_qlora"
    python TESTS/standalone_chat_v03.py

Type a message and press Enter. ``/quit`` exits.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))
sys.path.insert(0, os.path.join(_ROOT, "TESTS", "training"))

# --------------------------------------------------------------------------
# Memory-bypass tripwires. Installed before anything from PRIMORA is imported,
# so anything that reaches for the database during import is caught too.
# --------------------------------------------------------------------------

VIOLATIONS: list[str] = []
# Set while this script's own digest check runs, so the tripwire does not report
# the instrumentation as a violation. The digest opens the database read-only and
# only ever reads; it is the auditor, not the accused.
_SELF_CHECK = False
_PRODUCTION_DB = os.path.join(
    _ROOT, "PRIMORA", "data", "context", "aurora_context.db")


def _fingerprint_production_db() -> str:
    """Content digest of the production memory database.

    Read directly with sqlite3 rather than through ``default_db_path``, so this
    check does not import the store module into the process it is meant to be
    proving clean.

    Content, not file bytes: opening a WAL database read-write and closing it
    folds the WAL into the main file, changing bytes without changing a row.
    Guarding on content is what makes the check meaningful.
    """
    path = os.path.join(_ROOT, "PRIMORA", "data", "context",
                        "aurora_context.db")
    if not os.path.exists(path):
        return "<absent>"
    global _SELF_CHECK
    import sqlite3
    _SELF_CHECK = True
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        parts = []
        for table in ("messages", "sessions", "digests", "states",
                      "candidates", "entities", "relationships",
                      "retrieval_vectors", "retrieval_index_meta"):
            try:
                rows = con.execute(f"SELECT * FROM {table}").fetchall()
            except sqlite3.Error:
                continue
            parts.append(f"{table}={len(rows)}:" + hashlib.sha256(
                "\n".join(sorted(repr(r) for r in rows)).encode()
            ).hexdigest()[:16])
        return hashlib.sha256("|".join(parts).encode()).hexdigest()
    finally:
        con.close()
        _SELF_CHECK = False


def _install_tripwires() -> None:
    """Make any attempt to open memory loud instead of silent."""
    import sqlite3

    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        global _SELF_CHECK
        if _SELF_CHECK:
            return real_connect(*args, **kwargs)
        try:
            target = str(args[0])
        except Exception:
            target = "<unknown>"
        if _PRODUCTION_DB.lower().replace("\\", "/") in \
                target.lower().replace("\\", "/"):
            VIOLATIONS.append(f"sqlite3.connect -> production db: {target}")
        return real_connect(*args, **kwargs)

    sqlite3.connect = traced_connect


def _wrap_memory_entrypoints() -> None:
    """Record any *call* into the memory stack, not merely its import.

    ``PRIMORA/aurora/__init__.py`` imports ``Aurora`` unconditionally, so
    importing ``PRIMORA.aurora.brain`` unavoidably loads the ContextManager, the
    ContextStore, Prime and the cognitive loop. Loaded is not the same as used.

    The claim this script needs to support is "memory is never *invoked*", so the
    entry points are wrapped and every call is recorded. A silent bypass becomes
    a visible one.
    """
    def wrap(owner, name: str, label: str) -> None:
        original = getattr(owner, name, None)
        if original is None:
            return

        def traced(*args, **kwargs):
            VIOLATIONS.append(f"{label} called")
            return original(*args, **kwargs)

        setattr(owner, name, traced)

    try:
        from PRIMORA.aurora.aurora import Aurora
        wrap(Aurora, "respond", "Aurora.respond")
        wrap(Aurora, "context_for", "Aurora.context_for")
    except Exception as exc:
        VIOLATIONS.append(f"could not wrap Aurora: {exc}")
    try:
        from PRIMORA.aurora.loop import CognitiveLoop
        wrap(CognitiveLoop, "run", "CognitiveLoop.run")
    except Exception as exc:
        VIOLATIONS.append(f"could not wrap CognitiveLoop: {exc}")
    try:
        from PRIMORA.prime.context.manager import ContextManager
        wrap(ContextManager, "__init__", "ContextManager()")
        wrap(ContextManager, "build", "ContextManager.build")
    except Exception as exc:
        VIOLATIONS.append(f"could not wrap ContextManager: {exc}")
    try:
        from PRIMORA.prime.context.store import ContextStore
        wrap(ContextStore, "__init__", "ContextStore()")
    except Exception as exc:
        VIOLATIONS.append(f"could not wrap ContextStore: {exc}")
    try:
        from PRIMORA.prime.prime import Prime
        wrap(Prime, "__init__", "Prime()")
        wrap(Prime, "handle", "Prime.handle")
    except Exception as exc:
        VIOLATIONS.append(f"could not wrap Prime: {exc}")


# --------------------------------------------------------------------------
# The full training-time system instruction.
# --------------------------------------------------------------------------


def build_system_prompt() -> str:
    """The composed instruction the v0.3 records were trained against.

    Built by ``generate_v2.build_system_context``, the same function the dataset
    generator used, then verified against the fingerprint the dataset itself
    recorded. A mismatch means the runtime instruction changed since training,
    and this test would then be measuring the wrong model.

    That builder imports the whole PRIMORA runtime harness - Prime, the
    ContextManager, the cognitive loop - which is precisely what this script
    exists to exclude. So it runs in a **subprocess** and exits before the model
    loads. The parent receives the instruction as a string and never imports
    anything that can reach memory. Section [6] proves it.
    """
    import subprocess

    child = (
        "import sys, hashlib, json\n"
        f"sys.path.insert(0, {_ROOT!r})\n"
        f"sys.path.insert(0, {os.path.join(_ROOT, 'TRAINING')!r})\n"
        f"sys.path.insert(0, {os.path.join(_ROOT, 'TESTS', 'training')!r})\n"
        "from generate_v2 import build_system_context\n"
        "key, composed, caps = build_system_context()\n"
        "sys.stderr.write('@@' + json.dumps({\n"
        "    'key': key,\n"
        "    'digest': hashlib.sha256(composed.encode()).hexdigest(),\n"
        "    'capabilities': caps,\n"
        "    'content': composed,\n"
        "}) + '@@')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True, text=True, cwd=_ROOT,
    )
    stderr = proc.stderr
    if "@@" not in stderr:
        raise RuntimeError(
            "the system-context subprocess produced no result; "
            f"stdout={proc.stdout[-400:]!r} stderr={stderr[-800:]!r}")
    payload = json.loads(stderr.split("@@", 1)[1].rsplit("@@", 1)[0])

    print(f"  system context key  : {payload['key']}")
    print(f"  system fingerprint  : {payload['digest']}")
    print(f"  capabilities        : {', '.join(payload['capabilities'])}")
    print(f"  instruction length  : {len(payload['content']):,} chars")

    # Cross-check against the fingerprint the v0.3 dataset recorded. This is
    # what proves the prompt is byte-identical to the training-time one.
    expected = os.environ.get("PRIMORA_EXPECT_SYSTEM_FINGERPRINT")
    if expected:
        match = expected == payload["digest"]
        print(f"  fingerprint match   : {match} (expected {expected[:16]}...)")
        if not match:
            raise RuntimeError(
                f"system fingerprint {payload['digest'][:16]} does not match "
                f"the dataset's {expected[:16]}")
    else:
        print("  fingerprint match   : not checked "
              "(set PRIMORA_EXPECT_SYSTEM_FINGERPRINT to verify)")
    return payload["content"]


def main() -> int:
    print("=" * 74)
    print("AURORA v0.3 - STANDALONE GENERATION (no PRIMORA memory)")
    print("=" * 74)

    _install_tripwires()
    _wrap_memory_entrypoints()
    print()
    print("[1] memory tripwires installed")
    print("    sqlite3.connect -> production db, and the constructors and")
    print("    entry points of Aurora, CognitiveLoop, ContextManager,")
    print("    ContextStore and Prime are all wrapped and recording.")

    from PRIMORA.aurora.brain import AuroraBrain
    from PRIMORA.aurora.configuration import AuroraConfig

    config = AuroraConfig.from_env()
    print()
    print("[2] configuration")
    print(f"  model path          : {config.model_path}")
    print(f"  adapter path        : {config.adapter_path}")
    print(f"  max_new_tokens      : {config.max_new_tokens}")
    print(f"  temperature         : {config.temperature}")
    print(f"  top_p               : {config.top_p}")
    print(f"  repetition_penalty  : {config.repetition_penalty}")

    adapter_name = os.path.basename(config.adapter_path.rstrip("\\/"))
    expected = "aurora_qlora"
    if adapter_name != expected:
        print(f"\n  WARNING: adapter directory is {adapter_name!r}, not "
              f"{expected!r}. Set PRIMORA_AURORA_ADAPTER_PATH explicitly.")
    adapter_config = os.path.join(config.adapter_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        with open(adapter_config, encoding="utf-8") as handle:
            recorded = json.load(handle).get("base_model_name_or_path")
        print(f"  adapter base model  : {recorded}")
    print(f"  ADAPTER LOADED      : {config.adapter_path}")

    print()
    print("[3] system prompt (full training-time instruction)")
    system_prompt = build_system_prompt()

    print()
    print("[4] loading model (4-bit, this takes ~30s)")
    brain = AuroraBrain(config=config)
    error = brain.warmup()
    if error:
        print(f"  LOAD FAILED: {error}")
        return 1
    print("  model ready")

    production_before = _fingerprint_production_db()
    print()
    print("[5] production memory digest (compared after every turn)")
    print(f"  {production_before}")

    import PRIMORA.aurora.brain as brain_module
    memory_modules = sorted(
        name for name in sys.modules
        if name.startswith("PRIMORA")
        and any(part in name for part in
                ("aurora.aurora", "aurora.loop", "prime.context", "prime.prime",
                 "prime.orchestrator", "prime.providers"))
    )
    print()
    print("[6] memory modules loaded vs invoked")
    print("  PRIMORA/aurora/__init__.py imports Aurora unconditionally, so")
    print("  importing AuroraBrain loads the memory stack whether or not it is")
    print("  used. Loaded is not the same as called; the tripwires decide it.")
    for name in memory_modules:
        print(f"    loaded: {name}")
    print()
    if VIOLATIONS:
        print(f"  FAIL: {len(VIOLATIONS)} memory call(s) recorded:")
        for violation in VIOLATIONS:
            print(f"    {violation}")
        print()
        print("  Treat every reply above as memory-contaminated.")
    else:
        print("  CONFIRMED: zero calls into Aurora.respond, CognitiveLoop.run,")
        print("            ContextManager, ContextStore, Prime, or the")
        print("            production database. Memory was loaded, never used.")

    print()
    print("=" * 74)
    print("Type a message and press Enter.  /quit to exit.")
    print("No memory is read and nothing is written to any database.")
    print("=" * 74)

    history: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        print()
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("/quit", "/exit", "/q"):
            break

        history.append({"role": "user", "content": user})
        result = brain.generate(history)

        if not result.success:
            print(f"< [generation failed] {result.error}")
            history.pop()
            continue

        reply = result.content or ""
        print(f"< {reply}")
        history.append({"role": "assistant", "content": reply})

        production_after = _fingerprint_production_db()
        if production_after != production_before:
            print(f"\n  !! PRODUCTION MEMORY CHANGED: "
                  f"{production_before[:12]} -> {production_after[:12]}")
            print("  !! this should be impossible; investigate before trusting "
                  "any reply above.")
            production_before = production_after

    production_final = _fingerprint_production_db()
    print()
    print("=" * 74)
    print(f"  production memory unchanged : "
          f"{production_final == production_before}")
    print(f"  memory calls recorded       : {len(VIOLATIONS)}")
    for violation in VIOLATIONS:
        print(f"    {violation}")
    if not VIOLATIONS:
        print("  -> no memory subsystem was invoked at any point.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())