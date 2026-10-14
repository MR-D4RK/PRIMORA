"""A recall scenario that genuinely tests deliberate recall.

The first attempt at this was invalid, and the reason is structural rather
than a matter of scenario wording: assembly and recall run the *same* M6
engine. So any query ``recall`` could answer, the user's own message could
usually answer too, and the answer would already be in the context. The
scenario would then measure assembly, not recall.

The lever that does work is the cross-session cap. Assembly admits at most
``cross_session_cap`` foreign messages, chosen by the *user's* query. Recall
runs the same engine against *Aurora's* query. So a message that ranks below
the cap for a broad question, and first for a specific one, is retrievable
only by a deliberate, targeted recall.

Soundness is checked before the model is asked anything:

    the target must be ABSENT from the automatically assembled context
    a targeted recall must be able to FIND it

If either fails, the scenario reports that and refuses to score it, because a
pass would then prove nothing about deliberate recall.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, r"C:\PRIMORA PROJECT")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PRIMORA.aurora.brain import AuroraBrain  # noqa: E402
from PRIMORA.prime.context import ContextManager  # noqa: E402

from validate_loop_v03_real_model import (  # noqa: E402
    FixedResearchProvider,
    Observation,
    run_scenario,
)

TARGET_PHRASE = "slew-brake-eleventh"
PROMPT = "What are my notes on the yard equipment?"

# Broad, plausible notes. None of them contains the distinctive term, so a
# broad query surfaces these while burying the target below the cap.
DECOYS = [
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
]

TARGET = (
    f"The crane {TARGET_PHRASE} inspection is booked for the eleventh, so the "
    f"crane must be isolated before any work on the jib."
)


def build_store(tmpdir):
    context = ContextManager(db_path=os.path.join(tmpdir, "recall.db"))
    for note in DECOYS + [TARGET]:
        context.reconcile([{"role": "user", "content": note}], "old-session")
    return context


def check_soundness(context):
    """Both conditions, checked before the model is involved at all."""
    built = context.build(
        [{"role": "user", "content": PROMPT}], "s",
        token_counter=lambda m: max(1, len(m[0].get("content", "")) // 4),
        max_new_tokens=256,
    )
    assembled = " ".join(m["content"] for m in built.view.messages)
    absent = TARGET_PHRASE not in assembled

    # A targeted recall must be able to find it.
    from PRIMORA.prime.context.mind import Mind

    mind = Mind(context)
    found = mind.recall("crane slew brake inspection", session_id="old-session",
                        limit=5)
    findable = any(TARGET_PHRASE in hit["content"] for hit in found)

    print("=" * 72)
    print("RECALL SCENARIO - soundness")
    print("=" * 72)
    print(f"  assembled tokens      : {built.view.input_tokens}")
    print(f"  selected messages     : {built.selected_count}")
    print(f"  target in assembly    : {not absent}")
    print(f"  target findable       : {findable}")
    print(f"  recall hit count      : {len(found)}")
    if absent and findable:
        print("  -> SOUND: answering this requires a deliberate recall")
    else:
        print("  -> UNSOUND: refusing to score this scenario")
    return absent and findable


def main():
    tmpdir = tempfile.mkdtemp(prefix="aurora_recall_")
    context = build_store(tmpdir)
    if not check_soundness(context):
        print("\nScenario invalid; the model was not asked.")
        return 1

    print("\n" + "=" * 72)
    print("RECALL SCENARIO - real Aurora")
    print("=" * 72)
    brain = AuroraBrain()
    if brain.warmup():
        print("MODEL FAILED TO LOAD")
        return 1

    obs = Observation(
        "recall-targeted",
        PROMPT + " Also remind me which crane inspection is booked and for "
                 "when.",
        expected=["eleventh"],
    )
    run_scenario(brain, tmpdir, FixedResearchProvider(), obs)
    print(f"  requested : {obs.requests}")
    print(f"  gens/calls/stop: {obs.generations} / {len(obs.requests)} / "
          f"{obs.stop_reason}")
    print(f"  final     : {obs.final}")
    print(f"  contains target: {TARGET_PHRASE in (obs.final or '').lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
