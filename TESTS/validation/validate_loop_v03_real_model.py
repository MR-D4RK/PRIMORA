"""Real-model validation of the v0.3 cognitive loop.

An experiment, not a capability. Nothing in PRIMORA is changed by this file;
it loads the real Aurora brain, wires the real runtime with the real safety
protections, points it at a throwaway store, and records what she actually
does.

Two questions are kept apart on purpose:

    A. runtime works    can the system carry a multi-step process at all?
    B. brain knows how  does the real model choose the capability herself?

A scripted pass proves A and says nothing about B. Every observation here
comes from the real weights.

No chain of thought is recorded or reported. Only the observable trace: what
was requested, with what arguments, what came back, and what she finally said.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import AuroraBrain
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.aurora.loop import DEFAULT_CAPABILITY_BUDGET, LOOP_VERSION
from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
from PRIMORA.learning import Jev, LearningOutcome, StructuredCandidate
from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.providers import Provider
from PRIMORA.research import (
    DuckDuckGoResearchProvider,
    ResearchProvider,
    ResearchResult,
)
from PRIMORA.toolbox import ToolRegistry


# ======================================================================
# Observation
# ======================================================================


class Observation:
    """What happened in one scenario, as observables only."""

    def __init__(self, name, prompt, expected=None, expects_no_capability=False,
                 seed=()):
        self.name = name
        self.prompt = prompt
        self.expected = expected or []
        self.expects_no_capability = expects_no_capability
        self.seed = list(seed)
        self.segments = []
        self.requests = []          # (action, payload)
        self.results = []           # rendered tool-turn content
        self.final = None
        self.generations = 0
        self.stop_reason = None
        self.elapsed_s = 0.0
        self.error = None

    @property
    def capability_calls(self):
        return [a for a, _ in self.requests]

    @property
    def actions(self):
        return [a for a, _ in self.requests]

    def correct(self):
        if self.error:
            return False
        if not self.expected:
            return True
        text = (self.final or "").lower()
        return all(term.lower() in text for term in self.expected)

    def as_dict(self):
        return {
            "name": self.name,
            "prompt": self.prompt,
            "segments": len(self.segments),
            "requested": self.actions,
            "requests": [
                {"action": a, "payload": p} for a, p in self.requests
            ],
            "final": self.final,
            "generations": self.generations,
            "capability_calls": len(self.requests),
            "stop_reason": self.stop_reason,
            "correct": self.correct(),
            "used_capability": bool(self.requests),
            "expected_capability": not self.expects_no_capability,
            "elapsed_s": round(self.elapsed_s, 1),
            "error": self.error,
        }


# ======================================================================
# Wiring: the real brain, an isolated store, the real protections
# ======================================================================


class BrainAsProvider(Provider):
    """Lets a specialist consult reuse the already-loaded Aurora weights.

    A harness adapter, not a capability. Loading a second 8B model alongside
    the first does not fit in 8 GB, so the specialist is pointed at the model
    that is already resident. The provider seam is unchanged; only its
    implementation for this run differs.
    """

    def __init__(self, brain, name="shared-aurora-weights"):
        self._brain = brain
        self._name = name
        self.calls = []

    def invoke(self, capability, payload):
        self.calls.append({"capability": capability,
                           "messages": payload.get("messages", [])})
        result = self._brain.generate(payload["messages"], max_output_tokens=200)
        return PrimeResult.success(
            {"content": result.content, "model": self._name})


class FixedResearchProvider(ResearchProvider):
    """Deterministic stand-in for a live web search.

    The point of the research scenario is to see whether Aurora *decides* to
    research. Whether a public endpoint happened to answer is a network fact,
    not a fact about her, and it must not decide the verdict.
    """

    name = "validation-web"

    def __init__(self):
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append(query)
        return [
            ResearchResult(
                title="Latest release announcement",
                url="https://example.org/releases/latest",
                snippet=(
                    "Version 3.14.0 was released on 6 February 2025 and is "
                    "the current stable release."
                ),
                source="example.org",
            )
        ]


class FixedJev(Jev):
    """A Jev that structures whatever it is given, for the learn path."""

    def evaluate(self, request):
        return LearningOutcome.confirmed([
            StructuredCandidate(concept="the lesson", value=request.text[:80],
                                memory_type="FACT")])


_GUARD = None
_SESSIONS = []


def _install_guard():
    """Refuse production memory for the life of this standalone process.

    A validation script is one run, so the guard is process-wide rather than
    scoped to a call: a ``ContextManager`` built without a path - the failure
    that once wrote test rows into Aurora's real memory - is refused from the
    first build until the script exits.
    """
    global _GUARD
    if _GUARD is None:
        from PRIMORA.prime.context.isolation import production_guard
        _GUARD = production_guard()
        _GUARD.__enter__()
    return _GUARD


def build(brain, tmpdir, research_provider):
    """A complete, isolated runtime. Nothing here touches production."""
    from PRIMORA.prime.context.isolation import IsolatedSession, require_isolated

    root = require_isolated(os.path.join(tmpdir, "run"))
    os.makedirs(root, exist_ok=True)
    session = IsolatedSession(root, owned=False)
    context = session.context_manager()
    # Prime.__init__ unavoidably builds a db_path=None manager, which is lazy
    # and opens nothing. On the first build no guard is installed yet, so that
    # construction is fine; on every later build the guard from the first is
    # still active and would refuse it. The narrow allowance exists for exactly
    # this and covers only the omitted-path construction - the isolated store
    # is still what everything runs against, and production is still refused.
    from PRIMORA.prime.context.isolation import LazyManagerAllowed
    with LazyManagerAllowed():
        prime = Prime(
            BrainAsProvider(brain),
            research_provider=research_provider,
            specialists=SpecialistRegistry([
                SpecialistSpec(
                    name="reasoning",
                    description=(
                        "Careful step-by-step analysis of a hard problem. Use for "
                        "difficult reasoning, trade-offs, and checking your own "
                        "work."
                    ),
                    model="shared-aurora-weights",
                    system_prompt="Think carefully and precisely, then answer.",
                )
            ]),
            jev=FixedJev(),
            tools=ToolRegistry.with_builtins(),
        )
    prime._context = context
    prime._mind = prime._mind.__class__(context)
    for name in ("recall", "learn"):
        cap = prime.registry.get_specialist(name)
        if cap is not None:
            cap.bind(prime._mind)
    # Installed after Prime, for the same reason T1 does it: Prime builds a
    # db_path=None manager whose store is lazy, and the guard would otherwise
    # refuse the harness's own construction.
    _install_guard()
    _SESSIONS.append(session)
    config = AuroraConfig.from_env()
    return prime, Aurora(prime, brain=brain, config=config)


def release_all() -> None:
    """Close every session and release the process-wide guard."""
    global _GUARD
    while _SESSIONS:
        try:
            _SESSIONS.pop().__exit__(None, None, None)
        except Exception:
            pass
    if _GUARD is not None:
        try:
            _GUARD.__exit__(None, None, None)
        except Exception:
            pass
        _GUARD = None


# ======================================================================
# The experiment
# ======================================================================


def run_scenario(brain, tmpdir, research_provider, obs):
    """One scenario, with the real loop and the real protections."""
    try:
        prime, aurora = build(brain, tmpdir, research_provider)
        for entry in obs.seed:
            role, content = entry[0], entry[1]
            where = entry[2] if len(entry) > 2 else "s"
            prime._context.reconcile(
                [{"role": role, "content": content}], where)
        aurora.mind.begin_task("s", "validation")

        started = time.time()
        result = aurora.respond(
            [{"role": "user", "content": obs.prompt}], "s")
        obs.elapsed_s = time.time() - started

        trace = aurora.loop.last_trace
        obs.segments = [s.to_dict() for s in trace.segments]
        obs.requests = list(aurora.loop.executor.calls)
        obs.final = result.content
        obs.generations = len(obs.segments)
        obs.stop_reason = trace.stop_reason
        obs.success = result.success
    except Exception as exc:  # a broken run is a result, not a crash
        obs.error = f"{type(exc).__name__}: {exc}"
    return obs


# A recall scenario that is sound by construction. The answer is deliberately
# worded so a broad question does not surface it, and a specific query does.
# See validate_recall_deliberate.py for the soundness check.
RECALL_NOISE = [
    "The yard crane was serviced last week and the grease lines were flushed.",
    "Yard lighting was audited and two dark corners near the stack were found.",
    "The yard gate barrier sticks in cold weather and needs a spring return.",
    "Yard drainage pools after heavy rain so the crew uses gravel not concrete.",
    "The yard weighbridge calibration certificate lapses next month.",
    "Yard scrap bins were relocated nearer the weighbridge to cut walking.",
    "Yard fire extinguishers are serviced annually under contract.",
    "The yard drainage survey and the lighting audit share one contractor.",
    "Yard racking in aisle seven was reconfigured for longer pallet lengths.",
    "The yard gate logbook is kept at the hut and induction is in person.",
    "The crane slew-brake-eleventh inspection is booked for the eleventh, so "
    "the crane must be isolated before any work on the jib.",
]


def scenarios(live_research):
    return [
        Observation(
            "recall",
            "What are my notes on the yard equipment? Also remind me which "
            "crane inspection is booked and for when.",
            expected=["eleventh"],
            seed=[("user", note, "old-session") for note in RECALL_NOISE],
        ),
        Observation(
            "research",
            "What is the latest stable release of Python, and when was it "
            "released?",
            expected=["3.14"],
        ),
        Observation(
            "multi-step",
            "We berthed at berth nine with the kestrel seventeen cargo. "
            "Confirm the berth code we used, then check the current Python "
            "release, and tell me both.",
            expected=["foghorn-42", "3.14"],
            seed=[("user", "the berth code is foghorn-42, remember that"),
                  ("assistant", "noted, the berth code is foghorn-42")],
        ),
        Observation(
            "specialist",
            "Should a single server use a relational database or a key-value "
            "store for session storage, and why? Consult the reasoning "
            "specialist if it would help.",
            expected=[],
        ),
        Observation(
            "tool",
            "How many words are in this note: the quick brown fox jumps over "
            "the lazy dog?",
            expected=["9"],
        ),
        Observation(
            "no-capability",
            "In one short sentence, what is the capital of France?",
            expected=["paris"],
            expects_no_capability=True,
        ),
    ]


def main():
    live = "--live-research" in sys.argv
    print("=" * 72)
    print(f"AURORA COGNITIVE LOOP v{LOOP_VERSION} - REAL MODEL VALIDATION")
    print("=" * 72)
    print(f"  model            : {AuroraConfig.from_env().model_path}")
    print(f"  adapter          : {AuroraConfig.from_env().adapter_path}")
    print(f"  max_cognition    : {AuroraConfig.from_env().max_cognition_cycles}")
    print(f"  capability budget: {DEFAULT_CAPABILITY_BUDGET}")
    print(f"  research backend : "
          f"{'LIVE (DuckDuckGo)' if live else 'fixed stand-in'}")

    brain = AuroraBrain()
    problem = brain.warmup()
    if problem:
        print(f"\nMODEL FAILED TO LOAD: {problem}")
        return 1
    print("  model loaded     : yes")

    tmpdir = tempfile.mkdtemp(prefix="aurora_validation_")
    research = (
        DuckDuckGoResearchProvider() if live else FixedResearchProvider()
    )

    results = []
    for obs in scenarios(live):
        print(f"\n--- {obs.name} " + "-" * (66 - len(obs.name)))
        print(f"  prompt : {obs.prompt[:100]}")
        run_scenario(brain, tmpdir, research, obs)
        results.append(obs)
        if obs.error:
            print(f"  ERROR  : {obs.error}")
            continue
        print(f"  asked  : {obs.actions or '(nothing)'}")
        for action, payload in obs.requests:
            print(f"           {action} {json.dumps(payload)[:150]}")
        print(f"  gens   : {obs.generations}   calls: {len(obs.requests)}"
              f"   stop: {obs.stop_reason}")
        print(f"  final  : {(obs.final or '')[:300]}")
        print(f"  correct: {obs.correct()}")

    out = os.path.join(tmpdir, "observations.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump([o.as_dict() for o in results], handle, indent=2)
    print(f"\nFull observations: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
