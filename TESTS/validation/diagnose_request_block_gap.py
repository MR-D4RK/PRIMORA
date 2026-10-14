"""Diagnostic A/B: is the request-block gap formatting or capability?

The first real-model run showed Aurora naming the capability she wants in
prose ("I'll ask the specialist to help") and then stopping, without ever
emitting the request block. That leaves two very different explanations:

    formatting   she knows what to do and only misses the syntax
    capability   she cannot decide to act at all

Re-running the same scenarios with a more directive user message separates
them. Nothing in PRIMORA is modified; only the prompt text handed to the
runtime differs, and only for this run.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.aurora.brain import AuroraBrain
from PRIMORA.aurora.configuration import AuroraConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_loop_v03_real_model import (  # noqa: E402
    FixedResearchProvider,
    Observation,
    run_scenario,
)

DIRECTIVE = (
    " If this needs a Prime capability, do not describe what you are about to "
    "do - actually emit the <prime_request> block now."
)

CASES = [
    Observation(
        "multi-step/directive",
        "We berthed at berth nine with the kestrel seventeen cargo. Confirm the "
        "berth code we used, then check the current Python release, and tell "
        "me both." + DIRECTIVE,
        seed=[("user", "the berth code is foghorn-42, remember that"),
              ("assistant", "noted, the berth code is foghorn-42")],
    ),
    Observation(
        "specialist/directive",
        "Should a single server use a relational database or a key-value store "
        "for session storage, and why?" + DIRECTIVE,
        expected=[],
    ),
    Observation(
        "tool/directive",
        "How many words are in this note: the quick brown fox jumps over the "
        "lazy dog?" + DIRECTIVE,
        expected=["9"],
    ),
]


def main():
    config = AuroraConfig.from_env()
    print("=" * 72)
    print("DIAGNOSTIC A/B - directive prompt, same runtime, same model")
    print(f"  max_cognition={config.max_cognition_cycles}")
    print("=" * 72)

    brain = AuroraBrain()
    problem = brain.warmup()
    if problem:
        print(f"MODEL FAILED TO LOAD: {problem}")
        return 1

    tmpdir = tempfile.mkdtemp(prefix="aurora_diagnostic_")
    research = FixedResearchProvider()

    for obs in CASES:
        print(f"\n--- {obs.name} " + "-" * (66 - len(obs.name)))
        run_scenario(brain, tmpdir, research, obs)
        if obs.error:
            print(f"  ERROR : {obs.error}")
            continue
        print(f"  asked : {obs.actions or '(nothing)'}")
        for action, payload in obs.requests:
            print(f"          {action} {str(payload)[:120]}")
        print(f"  gens  : {obs.generations}  stop: {obs.stop_reason}")
        print(f"  final : {(obs.final or '')[:220]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
