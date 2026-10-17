"""Live smoke harness: the official stack against the real model server.

Manual harness - NOT collected by pytest (no ``test_`` prefix). Migrates
the genuinely useful capability of the retired launcher pair
(``launch-aurora-llamacpp.py`` + ``launch-aurora-llamacpp.ps1``): run a
real full-loop conversation through Aurora on llama-server while proving
production memory is untouched.

    python TESTS\\live_smoke.py              # manage the server (auto)
    python TESTS\\live_smoke.py --no-server  # endpoint must already run
    python TESTS\\live_smoke.py --research   # add a research probe

What it does:
1.  Adopts or starts llama-server exactly as ``python -m PRIMORA.main``
    does (same ModelServerService; nothing is ever downloaded).
2.  Builds Prime + Aurora through the same composition functions main()
    uses, inside ``harness_session()`` so every write lands in a private
    temp store; the harness verifies the production digest before/after.
3.  Runs a two-turn conversation (real generation) and checks the private
    store received both turns, including the assistant answers.
4.  Stops only the server this run started (an adopted server survives).

Exit code 0 = all checks passed; 1 = a check failed; 2 = aborted early.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PRIMORA.main import (  # noqa: E402
    configure_runtime_defaults,
    prime_provider_from_env,
    research_provider_from_env,
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

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-server", action="store_true",
                        help="do not manage llama-server (endpoint must be up)")
    parser.add_argument("--research", action="store_true",
                        help="also probe the research provider")
    args = parser.parse_args()

    print("=" * 70)
    print("PRIMORA LIVE SMOKE - official stack, isolated memory")
    print("=" * 70)

    if args.no_server:
        os.environ["PRIMORA_MODEL_SERVER"] = "off"

    service = ModelServerService.from_env()
    # Same decision main() makes from explicit configuration.
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
        research = research_provider_from_env() if args.research else None

        with harness_session("live_smoke_") as session:
            path = session.db_path("context.db")
            prime = build_prime_with_isolation_guard(
                provider, research_provider=research)
            prime._context._db_path = str(path)
            require_isolated(prime._context._db_path)

            from PRIMORA.aurora import Aurora
            aurora = Aurora(prime)

            t0 = time.time()
            warm_err = aurora.warmup()
            check("warmup succeeded", warm_err is None)
            print(f"       warmup {time.time() - t0:.1f}s"
                  + (f" err={warm_err}" if warm_err else ""))

            question = ("In exactly one short sentence: what is the "
                        "capital of France?")
            t0 = time.time()
            r1 = aurora.respond([{"role": "user", "content": question}],
                                session_id="live_smoke")
            check("turn 1 generated", bool(r1.success and r1.content.strip()))
            print(f"       turn1 {time.time() - t0:.1f}s: {r1.content!r}")

            r2 = aurora.respond(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": r1.content},
                    {"role": "user", "content": "Say OK and nothing else."},
                ],
                session_id="live_smoke",
            )
            check("turn 2 generated", bool(r2.success and r2.content.strip()))
            print(f"       turn2: {r2.content!r}")

            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                rows = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                joined = "\n".join(
                    r[1] for r in con.execute(
                        "SELECT role, content FROM messages"))
            finally:
                con.close()
            check("private store received both turns", rows >= 3)
            check("assistant answers persisted in private store",
                  "paris" in joined.lower())

            if args.research:
                from PRIMORA.prime.contract import PrimeRequest, RequestStatus
                rr = prime.handle(PrimeRequest(
                    action="research",
                    payload={"query": "current stable Python version"},
                ))
                ok = rr.status == RequestStatus.SUCCESS or bool(rr.error)
                check("research returned results or a clear error", ok)
                print(f"       research status={rr.status.value} "
                      f"error={str(rr.error or '')[:120]}")

        # harness_session verified the production digest on exit.
        check("production memory unchanged (harness digest)", True)
    except Exception as exc:
        check(f"no unexpected failure ({type(exc).__name__}: {exc})", False)
    finally:
        if status is not None:
            service.stop()

    print("=" * 70)
    failed = [n for n, ok in CHECKS if not ok]
    for name, ok in CHECKS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 70)
    if failed:
        print(f"LIVE SMOKE FAILED ({len(failed)} check(s))")
        return 1
    print("LIVE SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
