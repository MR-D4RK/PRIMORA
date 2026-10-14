"""Aurora Brain v0.1 training dataset design.

A specification, not a dataset. Nothing here is training data; this is the
plan that must be inspected and validated before any data is generated, and
before any training is started.

Every category below is anchored to something measured. The anchors are the
30-run real-model repeatability check and the T1 baseline, not intuition:

    research        5/5 pass   -> protect, do not disturb
    no-capability   5/5 pass   -> protect; the over-trigger risk is real
    multi-step      0/5 pass, 4/5 emitted TWO request blocks
    tool            0/5 pass, 5/5 emitted the tool name as the action
    specialist      0/5 pass, 0/5 emitted any block
    recall          unmeasured (the scenario was unsound)

The first three failures are format failures with a known correct answer, so
they are the cheapest to fix and the most important not to break something
else on the way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------
# Behaviour categories
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Category:
    id: str
    behavior: str
    anchor: str            # the measured failure this exists to address
    kind: str              # "emit" | "suppress" | "continue" | "style"
    seeds_per_category: int
    paraphrases_per_seed: int
    priority: str          # "acute" | "protective" | "supporting"
    notes: str = ""


CATEGORIES: tuple[Category, ...] = (
    Category(
        id="C1_act_not_announce",
        behavior="Emit the capability request instead of narrating it.",
        anchor="specialist 0/5 emitted a block; 2/5 said 'Let me ask the "
               "specialist'",
        kind="emit",
        seeds_per_category=12,
        paraphrases_per_seed=40,
        priority="acute",
    ),
    Category(
        id="C2_single_block",
        behavior="Emit exactly one request block, never several in one reply.",
        anchor="multi-step 4/5 emitted two blocks; the parser refuses them",
        kind="emit",
        seeds_per_category=12,
        paraphrases_per_seed=45,
        priority="acute",
        notes="Highest-value target: the correct blocks were already there, "
              "just not alone.",
    ),
    Category(
        id="C3_action_wrapper",
        behavior="Use the outer action and the payload the capability declares.",
        anchor="tool 5/5 emitted action='count_words' instead of "
               "action='use_tool' with name+arguments",
        kind="emit",
        seeds_per_category=12,
        paraphrases_per_seed=45,
        priority="acute",
    ),
    Category(
        id="C4_continue_cognition",
        behavior="After a result arrives, continue with another request or "
                 "answer; do not stop at the result.",
        anchor="multi-step 0/5 reached an answer; the one partial run called "
               "recall and then narrated",
        kind="continue",
        seeds_per_category=12,
        paraphrases_per_seed=40,
        priority="acute",
    ),
    Category(
        id="C5_answer_directly",
        behavior="When no capability is warranted, answer plainly with no block.",
        anchor="no-capability 5/5 pass; this is the behaviour that format "
               "training is most likely to break",
        kind="suppress",
        seeds_per_category=14,
        paraphrases_per_seed=40,
        priority="protective",
        notes="Deliberately the largest suppress category. Over-triggering is "
              "the main regression risk of this entire dataset.",
    ),
    Category(
        id="C6_delegate_when_useful",
        behavior="Consult a specialist for a hard sub-problem, and only then.",
        anchor="specialist 0/5; she answers directly instead of delegating",
        kind="emit",
        seeds_per_category=10,
        paraphrases_per_seed=35,
        priority="supporting",
    ),
    Category(
        id="C7_deliberate_recall",
        behavior="Recall when the answer is not in the assembled context.",
        anchor="recall unmeasured; assembly and recall share one engine",
        kind="emit",
        seeds_per_category=10,
        paraphrases_per_seed=35,
        priority="supporting",
        notes="Paired with C5: the same prompt shape, differing only in "
              "whether the fact is present.",
    ),
    Category(
        id="C8_stay_direct",
        behavior="Answer the question; no commentary about how Aurora works.",
        anchor="T1 claimed_action / capability_limit failures",
        kind="style",
        seeds_per_category=10,
        paraphrases_per_seed=35,
        priority="supporting",
    ),
    Category(
        id="C9_uncertainty_natural",
        behavior="Separate fact, uncertainty, and guess; guess when asked, "
                 "labelled as a guess.",
        anchor="T1 unknown_fact, unknown_personal, ambiguous, contradiction",
        kind="style",
        seeds_per_category=12,
        paraphrases_per_seed=35,
        priority="supporting",
    ),
)

# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "holdout": 0.15}

# Splitting is by SEED, never by example. Paraphrase expansion means a random
# example-level split would place paraphrases of the same seed in train and
# holdout, which inflates every number and makes the holdout a second training
# set. This is the single easiest way to ruin this dataset.
SPLIT_UNIT = "seed"

HOLDOUT_POLICY = (
    "Holdout seeds are written once, used once, and never inspected while "
    "tuning. Headline numbers come from holdout only. If a holdout prompt is "
    "ever used to change a prompt, a threshold, or the data mix, the holdout is "
    "replaced."
)

# T1's 21 scenarios were used to measure the current brain. Reusing them as
# training seeds would make the same evaluation material into training data.
RESERVED_EVALUATION_MATERIAL = (
    "TESTS/t1/scenarios.py scenario ids A1..O2"
)


# --------------------------------------------------------------------------
# Isolation and contamination
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class IsolationPolicy:
    store: str
    per_example: str
    production_guard: bool
    verify: str
    dedupe_within: str
    dedupe_across_splits: str
    reserved: str

    def as_lines(self) -> list[str]:
        return [
            f"store: {self.store}",
            f"per example: {self.per_example}",
            f"production guard active: {self.production_guard}",
            f"verify: {self.verify}",
            f"dedupe within split: {self.dedupe_within}",
            f"dedupe across splits: {self.dedupe_across_splits}",
            f"reserved material: {self.reserved}",
        ]


ISOLATION = IsolationPolicy(
    store="harness_session() from PRIMORA.prime.context.isolation; a run that "
          "cannot obtain a private store raises IsolationError",
    per_example="each example is generated in a session whose ContextManager "
                "is explicit, so no example can share a store with another",
    production_guard=True,
    verify="production_digest() compared before and after generation, per run "
           "and for the whole collection; mismatch raises",
    dedupe_within="normalised content fingerprint (case, punctuation, "
                  "whitespace collapsed) within a split",
    dedupe_across_splits="by seed id first, then by normalised fingerprint; a "
                         "seed may appear in exactly one split",
    reserved=RESERVED_EVALUATION_MATERIAL,
)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    id: str
    question: str
    measure: str
    target: str
    split: str


EVALUATION: tuple[Criterion, ...] = (
    Criterion("E1_block_count", "Does she ever emit more than one block?",
              "blocks per reply, counted before parsing", "0 replies with >1",
              "holdout"),
    Criterion("E2_parse_rate", "Is an emitted block a valid request?",
              "parse attempts that yield a valid action", ">= 98%", "holdout"),
    Criterion("E3_action_correct", "Did she pick the right capability?",
              "correct action over capability requests", ">= 95%", "holdout"),
    Criterion("E4_multistep", "Does a two-capability task complete?",
              "tasks with >=2 distinct capabilities AND a correct answer",
              ">= 85%", "holdout"),
    Criterion("E5_over_trigger", "Does she call when she should not?",
              "no-capability prompts that produce any block",
              "<= 2% (was 0/5 failures, must not regress)", "holdout"),
    Criterion("E6_narration", "Does she narrate instead of acting?",
              "final replies containing a stated intention with no block",
              "<= 2%", "holdout"),
    Criterion("E7_uncertainty", "Does she separate fact from guess?",
              "scored against the seeded expected stance", ">= 90%", "holdout"),
    Criterion("E8_direct", "Does she avoid meta-commentary?",
              "replies containing self-description of her own machinery",
              "<= 3%", "holdout"),
    Criterion("E9_regression", "Did anything that already worked break?",
              "the six-scenario suite, repeated 5x, on the same defaults",
              "research and no-capability stay at 5/5", "val"),
)

# A behaviour that is fixed by breaking another one is not an improvement.
REGRESSION_GATE = (
    "Training is rejected if research or no-capability falls below 5/5 on the "
    "six-scenario suite, regardless of every other number. C5 exists to make "
    "that gate possible to pass."
)


def totals() -> dict:
    per_split = {"train": 0, "val": 0, "holdout": 0}
    for category in CATEGORIES:
        count = category.seeds_per_category * category.paraphrases_per_seed
        for name, fraction in SPLIT_FRACTIONS.items():
            per_split[name] += int(round(count * fraction))
    return {
        "categories": len(CATEGORIES),
        "seeds": sum(c.seeds_per_category for c in CATEGORIES),
        "examples": sum(
            c.seeds_per_category * c.paraphrases_per_seed for c in CATEGORIES
        ),
        "per_split": per_split,
    }
