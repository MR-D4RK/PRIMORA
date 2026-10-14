"""Dry-run: are the targets this dataset would teach actually valid?

A dataset can be internally consistent and still teach the model to emit
requests the runtime refuses. That is precisely the failure being fixed, so
reproducing it in the training data would be worse than useless.

This validates every canonical target the design implies, end to end:

    target text -> extract_prime_request -> Prime.handle -> capability runs

Nothing is generated at scale and nothing is trained. It answers one
question: is the thing we are about to teach, correct?

Isolation: every store here comes from ``harness_session``, and the
production digest is compared before and after.
"""

from __future__ import annotations

import json
import sys
import tempfile

sys.path.insert(0, r"C:\PRIMORA PROJECT")
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

from PRIMORA.aurora.aurora import extract_prime_request  # noqa: E402
from PRIMORA.prime import Prime  # noqa: E402
from PRIMORA.prime.contract import PrimeResult  # noqa: E402
from PRIMORA.prime.context.isolation import (  # noqa: E402
    IsolationError,
    production_digest,
)
from PRIMORA.prime.providers import Provider  # noqa: E402
from PRIMORA.toolbox import ToolRegistry  # noqa: E402


class _Blocked(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.failure("dry run: execution disabled")


def block(action: str, payload: dict) -> str:
    return (
        "<prime_request>\n"
        + json.dumps({"action": action, "payload": payload,
                      "context": None, "metadata": None})
        + "\n</prime_request>"
    )


# The targets the design implies, one per behaviour that emits a request.
TARGETS = [
    ("C1/C2 research", block("research", {"query": "latest Python release"})),
    ("C2/C4 recall then answer", block("recall", {"query": "berth code"})),
    ("C3 use_tool wrapper", block("use_tool", {
        "name": "count_words",
        "arguments": {"text": "the quick brown fox"}})),
    ("C6 specialist", block("consult_specialist", {
        "specialist": "reasoning", "task": "compare the two designs"})),
    ("C7 recall", block("recall", {"query": "which crane inspection"})),
    ("C3 learn with provenance", block("learn", {
        "text": "Amir prefers green tea.", "provenance": "user"})),
]

# The near-misses the current model actually produces. These must NOT be
# teachable, and proving they fail is what keeps them out of the dataset.
ANTI_TARGETS = [
    ("tool name as action", block("count_words", {"text": "x"})),
    ("two blocks in one reply",
     block("recall", {"query": "code"}) + "\n" +
     block("research", {"query": "python"})),
    ("unknown action", block("teleport", {})),
    ("payload not an object", block("research", "just a string")),
    ("missing required field", block("research", {})),
]


def main() -> int:
    before = production_digest()
    print("=" * 72)
    print("TARGET VALIDATION DRY RUN")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="targetdry_")
    if True:
        try:
            from PRIMORA.prime.context.isolation import harness_session
            with harness_session("dryrun_", root=tmp) as session:
                context = session.context_manager()
                from PRIMORA.consult import (
                    SpecialistRegistry,
                    SpecialistSpec,
                )
                from PRIMORA.prime.context.isolation import (
                    build_prime_with_isolation_guard,
                )
                from PRIMORA.research import (
                    ResearchProvider,
                    ResearchResult,
                )

                class _Web(ResearchProvider):
                    name = "dryrun"

                    def search(self, query, limit=5):
                        return [ResearchResult(
                            "t", "https://x.test/", "s", "x.test")]

                prime = build_prime_with_isolation_guard(
                    _Blocked(),
                    research_provider=_Web(),
                    specialists=SpecialistRegistry([
                        SpecialistSpec(name="reasoning",
                                       description="Hard analysis.",
                                       model="m")]),
                    tools=ToolRegistry.with_builtins())
                prime._context = context
                prime._mind = prime._mind.__class__(context)
                for name in ("recall", "learn"):
                    cap = prime.registry.get_specialist(name)
                    if cap is not None:
                        cap.bind(prime._mind)

                failures = 0
                # A target is valid when it parses, names the right
                # capability, and is not refused for a *format* reason.
                # Execution is deliberately blocked by _Blocked, so a
                # "specialist_failed: ... execution disabled" is a pass: the
                # request was well formed and reached the right capability.
                FORMAT_ERRORS = (
                    "Unknown capability", "Invalid arguments", "Unknown tool",
                    "Missing or invalid", "must be of type", "must be one of",
                    "unknown argument", "is not bound",
                )

                def format_error(result):
                    text = (result.error or "").lower()
                    return any(marker.lower() in text
                               for marker in FORMAT_ERRORS)

                print("\n  targets the dataset would teach:")
                for label, text in TARGETS:
                    request = extract_prime_request(text)
                    if request is None:
                        print(f"    FAIL  {label}: does not parse")
                        failures += 1
                        continue
                    result = prime.handle(request)
                    bad = format_error(result)
                    if bad:
                        failures += 1
                    note = "accepted" if result.status.value == "success" else (
                        "reached capability, execution blocked")
                    print(f"    {'FAIL' if bad else 'OK  '}  "
                          f"{label:24s} action={request.action:20s} {note}")

                print("\n  near-misses that must stay unteachable:")
                for label, text in ANTI_TARGETS:
                    request = extract_prime_request(text)
                    if request is None:
                        print(f"    OK   {label:28s} refused at parse")
                        continue
                    result = prime.handle(request)
                    refused = result.status.value == "failure"
                    if not refused:
                        failures += 1
                    print(f"    {'OK  ' if refused else 'FAIL'} "
                          f"{label:28s} "
                          f"{'refused' if refused else 'ACCEPTED - not a near miss'}")

                print("\n  isolation while validating targets:")
                try:
                    from PRIMORA.prime.context import ContextManager
                    ContextManager()
                    print("    FAIL  implicit production fallback was allowed")
                    failures += 1
                except IsolationError:
                    print("    OK    implicit production fallback refused")
        except IsolationError as exc:
            print(f"  isolation refused to run: {exc}")
            return 1

    after = production_digest()
    unchanged = after == before
    print(f"\n  production unchanged: {unchanged}")
    print(f"  result: {'PASS' if failures == 0 and unchanged else 'FAIL'}")
    return 0 if failures == 0 and unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
