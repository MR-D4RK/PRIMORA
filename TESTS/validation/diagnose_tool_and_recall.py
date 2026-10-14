"""Two follow-ups from the first post-fix real-model run.

1. The tool scenario emitted a correct ``count_words`` request and then said
   the command "doesn't exist in our toolset". Either the tool genuinely did
   not return, or she misread what came back. This prints the tool turn.

2. The recall scenario still does not test recall, because the answer is
   already in the automatically assembled context. This builds a store where
   the target is provably *absent* from the assembled context, verifies that
   programmatically, and only then asks the model - so a pass is evidence
   about deliberate recall rather than about assembly.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, r"C:\PRIMORA PROJECT")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PRIMORA.aurora.brain import AuroraBrain  # noqa: E402
from PRIMORA.aurora.configuration import AuroraConfig  # noqa: E402
from PRIMORA.prime.context import ContextManager  # noqa: E402
from PRIMORA.prime.contract import PrimeResult  # noqa: E402
from PRIMORA.prime.providers import Provider  # noqa: E402

from validate_loop_v03_real_model import (  # noqa: E402
    FixedJev,
    FixedResearchProvider,
    Observation,
    run_scenario,
)

SECRET = "amber-lantern-19"
NOISE = [
    "The north quay crane was serviced last week and the counterweight "
    "inspection is still outstanding for the following quarter.",
    "Berth scheduling on the east side runs on a two hour rotation and the "
    "harbour master prefers a fifteen minute buffer between arrivals.",
    "The forklift battery replacement is scheduled for the second Tuesday of "
    "the month and the old pack goes to the recycling yard afterwards.",
    "Safety briefing notes for the loading bay were laminated and reissued "
    "after the signage audit found two ambiguous placards near the ramp.",
    "The catering contract runs to the end of the financial year and the "
    "vegetarian option has been underused since the kitchen rota changed.",
    "The yard lighting audit found two dark corners near the container stack "
    "and a contractor is booked to survey the mounting heights this week.",
    "Insurance renewal paperwork is filed under two separate policy numbers "
    "depending on whether the vessel is chartered or owned outright.",
    "The mooring line inspection log shows one frayed section on bollard "
    "fourteen which was replaced during the previous maintenance window.",
    "Warehouse racking in aisle seven was reconfigured to hold longer pallet "
    "lengths and the picking route now avoids the forklift crossing point.",
    "The generator load test failed twice before the fuel filter was replaced "
    "and a third attempt is booked for the following morning.",
    "Drainage on the apron pools after heavy rain and the crew uses gravel "
    "rather than concrete because it drains freely and costs less to relay.",
    "The visitor logbook is kept at the gate hut and the induction briefing "
    "is delivered in person whenever a new contractor starts on site.",
    "Fuel accounting moved to a monthly reconciliation cycle after the audit "
    "found discrepancies between the pump total and the shore purchase order.",
    "The workshop compressor was serviced and the condensate trap refitted "
    "after operators reported a taste issue in the compressed air supply.",
    "Spare impeller for the pump is held in the stores cage and the shelf "
    "label was corrected after two staff members ordered the wrong variant.",
    "Sea chest strainers are cleaned monthly and the last clean found a "
    "significant amount of shell grit suggesting the water intake is shallow.",
    "The paperwork for the crane annual examination is due and the booking "
    "calendar shows one available slot in the first week of the month.",
    "Gate barrier number two sticks in cold weather and the technician has "
    "been asked to fit a spring return to the linkage assembly.",
    "Training records show three staff overdue for the lifting equipment "
    "refresher and a supervisor is arranging a group session next week.",
    "The workshop forklift horn failed the daily check and a replacement "
    "unit was fitted from the consumables store the same morning.",
    "Dredging of the approach channel is planned for the summer window "
    "subject to a licence from the harbour authority and a short closure.",
    "Fire extinguisher servicing is contracted annually and the contractor "
    "provides a compliance certificate for the site file on completion.",
    "The weighbridge calibration certificate lapses next month and the "
    "testing house has been asked to quote for a repeat visit.",
    "Paint condition on the handrails along the access walkway is fair and "
    "a touch-up campaign is being added to next quarter's maintenance plan.",
    "The office telephone system was migrated to a hosted platform and the "
    "internal extension list is now maintained centrally by head office.",
    "Scrap metal segregation bins were relocated nearer the weighbridge to "
    "cut walking distance and the signage was updated to match the move.",
    "The water sampling results for the potable supply returned within the "
    "permitted range and a certificate has been filed with the supplier.",
    "Ladder inspections identified two units with non compliant feet and "
    "they were withdrawn from service and tagged for disposal.",
]


def diagnose_tool():
    print("=" * 72)
    print("1. TOOL TURN DIAGNOSTIC - did the tool actually return?")
    print("=" * 72)
    brain = AuroraBrain()
    if brain.warmup():
        print("MODEL FAILED TO LOAD")
        return
    tmpdir = tempfile.mkdtemp(prefix="aurora_tool_diag_")
    obs = Observation(
        "tool-diagnostic",
        "How many words are in this note: the quick brown fox jumps over the "
        "lazy dog?")
    run_scenario(brain, tmpdir, FixedResearchProvider(), obs)
    print(f"  requested : {obs.requests}")
    print(f"  stop      : {obs.stop_reason}")
    print(f"  final     : {obs.final}")
    print("  --- per segment ---")
    for index, segment in enumerate(obs.segments):
        print(f"    segment {index}: requested={segment['requested']} "
              f"capability={segment['capability']} executed={segment['executed']}")


def build_recall_scenario():
    """A store where the answer is provably not auto-assembled."""
    tmpdir = tempfile.mkdtemp(prefix="aurora_recall_")
    context = ContextManager(
        db_path=os.path.join(tmpdir, "recall.db"))
    # The fact lives in a different session, worded so it shares no content
    # words with the current conversation.
    context.reconcile(
        [{"role": "user",
          "content": f"the strongroom opens to the phrase {SECRET}"}],
        "earlier")
    # The current session is busy with unrelated, long messages that consume
    # the assembly budget.
    for note in NOISE:
        context.reconcile([{"role": "user", "content": note}], "s")

    prompt = "Remind me of the phrase that opens the strongroom."
    built = context.build(
        [{"role": "user", "content": prompt}], "s",
        token_counter=lambda m: max(1, len(m[0].get("content", "")) // 4),
        max_new_tokens=256,
    )
    assembled = " ".join(m["content"] for m in built.view.messages)
    present = SECRET in assembled
    print("\n" + "=" * 72)
    print("2. RECALL SCENARIO - is the answer already in the context?")
    print("=" * 72)
    print(f"  assembled context tokens : {built.view.input_tokens}")
    print(f"  selected messages        : {built.selected_count}")
    print(f"  secret present in context: {present}")
    if present:
        print("  -> scenario does NOT test recall; needs more noise")
        return None
    print("  -> verified absent, so answering it requires deliberate recall")
    return tmpdir


def main():
    diagnose_tool()
    build_recall_scenario()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
