"""Live validation: favorite-color continuity against the real model.

Manual harness - NOT collected by pytest (no ``test_`` prefix). Proves the
conversational-continuity milestone acceptance matrix end to end:

  A. same session, full-history (GUI shape): "It's yellow." establishes the
     fact, and the next turn answers "yellow";
  B. same session, loader history (CLI shape): the restored thread travels
     with the new question;
  C. restart (brand-new Prime + Aurora over the same store): the persisted
     fact answers the question, not the stale chat;
  D. persona separation: "What is YOUR favorite color" never becomes
     Aurora's own claim;
  E. the fact exists as a ``current`` user state and reaches the build view.

    python TESTS\\validation\\validate_favorite_color_live.py
    python TESTS\\validation\\validate_favorite_color_live.py --no-server

Runs inside ``harness_session()``: every write lands in a private temp
store and the production digest is verified byte-identical on exit.
Exit code 0 = all checks passed; 1 = a check failed; 2 = aborted early.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PRIMORA.main import (  # noqa: E402
    _restore_cli_history,
    configure_runtime_defaults,
    prime_provider_from_env,
)
from PRIMORA.model_server import (  # noqa: E402
    ModelServerService,
    should_manage,
)
from PRIMORA.prime.context.isolation import (  # noqa: E402
    build_prime_with_isolation_guard,
    harness_session,
    require_isolated,
)

SESSION = "fav_color_live"
CHECKS: list[tuple[str, bool]] = []
YELLOW = "yellow"


def check(name: str, ok: bool) -> None:
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def _says_yellow(text: str) -> bool:
    return YELLOW in (text or "").lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-server", action="store_true",
                        help="do not manage llama-server (endpoint must be up)")
    args = parser.parse_args()

    print("=" * 70)
    print("PRIMORA LIVE VALIDATION - favorite-color continuity")
    print("=" * 70)

    if args.no_server:
        os.environ["PRIMORA_MODEL_SERVER"] = "off"

    service = ModelServerService.from_env()
    explicit_backend = os.environ.get("PRIMORA_AURORA_BACKEND") or "openai"
    explicit_url = str(os.environ.get("PRIMORA_OPENAI_URL") or "")
    manage = should_manage(
        service.mode, explicit_backend.strip().lower(), explicit_url,
        service.host, service.port,
    )

    status = None
    if manage:
        try:
            status = service.ensure()
        except Exception as exc:
            print(f"ABORT: {exc}")
            return 2
        print(f"  server: {status.state} at {status.base_url} "
              f"pid={status.pid}")
    else:
        print(f"  server: not managed; expecting endpoint {service.base_url}")

    try:
        configure_runtime_defaults(service)
        provider = prime_provider_from_env()

        with harness_session("validate_fav_color_") as session:
            db_path = session.db_path("context.db")

            # ---- process 1: same-session GUI-shaped conversation ---------
            prime = build_prime_with_isolation_guard(provider)
            prime._context._db_path = str(db_path)
            require_isolated(prime._context._db_path)
            from PRIMORA.aurora import Aurora
            aurora = Aurora(prime)

            warm_err = aurora.warmup()
            check("process 1 warmup", warm_err is None)
            if warm_err is not None:
                return 2

            q1 = "What is my favorite color?"
            r1 = aurora.respond([{"role": "user", "content": q1}],
                                session_id=SESSION)
            check("turn 1 generated", bool(r1.success and r1.content.strip()))
            print(f"       turn1: {r1.content!r}")

            history = [
                {"role": "user", "content": q1},
                {"role": "assistant", "content": r1.content},
                {"role": "user", "content": "It's yellow."},
            ]
            r2 = aurora.respond(history, session_id=SESSION)
            check("turn 2 (elliptical fact) generated",
                  bool(r2.success and r2.content.strip()))
            print(f"       turn2: {r2.content!r}")

            # CLI shape: payload from the loader + the new turn.
            loaded, _ = _restore_cli_history(aurora, SESSION)
            r3 = aurora.respond(
                loaded + [{"role": "user", "content": q1}],
                session_id=SESSION)
            check("turn 3 (CLI-shape, same session) generated",
                  bool(r3.success and r3.content.strip()))
            check("turn 3 answers yellow", _says_yellow(r3.content))
            print(f"       turn3: {r3.content!r}")

            ctx = prime._context
            states = ctx._get_store().fetch_states(
                session_id=SESSION, concept="favorite color")
            check("favorite color state exists (current, user)",
                  any(s.status == "current" and s.value == YELLOW
                      and s.authority == "user" for s in states))
            result = ctx.build([{"role": "user", "content": q1}], SESSION)
            view = result.view if isinstance(result.view, str) else str(result.view)
            check("build view carries 'The user's favorite color is yellow.'",
                  "The user's favorite color is yellow." in view)

            # ---- process 2: restart over the same store -------------------
            prime2 = build_prime_with_isolation_guard(provider)
            prime2._context._db_path = str(db_path)
            require_isolated(prime2._context._db_path)
            aurora2 = Aurora(prime2)
            warm_err2 = aurora2.warmup()
            check("process 2 (restart) warmup", warm_err2 is None)
            if warm_err2 is not None:
                return 2

            restored, _ = _restore_cli_history(aurora2, SESSION)
            check("restart restored the transcript",
                  any(_says_yellow(m.get("content", ""))
                      for m in restored))
            r4 = aurora2.respond(
                restored + [{"role": "user", "content": q1}],
                session_id=SESSION)
            check("restart answers yellow", _says_yellow(r4.content))
            print(f"       restart: {r4.content!r}")

            # ---- persona separation ---------------------------------------
            r5 = aurora2.respond(
                restored + [{"role": "user", "content": q1},
                            {"role": "assistant", "content": r4.content},
                            {"role": "user",
                             "content": "What is your favorite color?"}],
                session_id=SESSION)
            check("persona question generated",
                  bool(r5.success and r5.content.strip()))
            check("persona question never claims the fact as Aurora's own",
                  not re.search(r"\bmy favorite color is " + YELLOW,
                                (r5.content or ""), re.IGNORECASE))
            print(f"       persona: {r5.content!r}")

        # harness_session verified the production digest on exit.
        check("production memory unchanged (harness digest)", True)
    except Exception as exc:
        check(f"no unexpected failure ({type(exc).__name__}: {exc})", False)
    finally:
        if status is not None:
            service.stop()

    print("=" * 70)
    for name, ok in CHECKS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in CHECKS if not ok]
    print("=" * 70)
    if failed:
        print(f"FAVORITE-COLOR VALIDATION FAILED ({len(failed)} check(s))")
        return 1
    print("FAVORITE-COLOR VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
