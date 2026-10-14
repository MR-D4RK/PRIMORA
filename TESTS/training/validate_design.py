"""Validate the v0.1 dataset design before any data is generated.

Checks the design's internal coherence and the assumptions it rests on. A
design that only reads well is not a design that has been checked, and the
failure modes here are quiet ones: a split that leaks, a category that
contradicts another, a target that trades one behaviour for another.

Run it directly:  python TESTS/training/validate_design.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "C:\\PRIMORA PROJECT")
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

from design import (  # noqa: E402
    CATEGORIES,
    EVALUATION,
    HOLDOUT_POLICY,
    ISOLATION,
    REGRESSION_GATE,
    SPLIT_FRACTIONS,
    SPLIT_UNIT,
    totals,
)

FAILURES: list[str] = []
CHECKS: list[str] = []


def check(condition: bool, description: str) -> None:
    CHECKS.append(description)
    if not condition:
        FAILURES.append(description)


def main() -> int:
    ids = [c.id for c in CATEGORIES]

    # -- categories -----------------------------------------------------
    check(len(set(ids)) == len(ids), "category ids are unique")
    check(len(CATEGORIES) >= 6, "covers the five target behaviours plus a "
                                "suppress category")
    kinds = {c.kind for c in CATEGORIES}
    for required in ("emit", "suppress", "continue", "style"):
        check(required in kinds, f"has at least one '{required}' category")
    check(all(c.anchor for c in CATEGORIES),
          "every category cites a measured anchor")
    check(all(c.behavior for c in CATEGORIES),
          "every category states a behaviour")
    acute = [c for c in CATEGORIES if c.priority == "acute"]
    check(len(acute) >= 3, "the reproducible failures are marked acute")
    check(any(c.kind == "suppress" and c.priority == "protective"
              for c in CATEGORIES),
          "a protective suppress category exists to guard the 5/5 win")

    # -- variation ------------------------------------------------------
    check(all(c.paraphrases_per_seed >= 25 for c in CATEGORIES),
          "every category has enough paraphrases to avoid wording memorisation")
    check(all(c.seeds_per_category >= 10 for c in CATEGORIES),
          "every category has enough distinct seeds to avoid scenario "
          "memorisation")

    # -- split ----------------------------------------------------------
    total_fraction = sum(SPLIT_FRACTIONS.values())
    check(abs(total_fraction - 1.0) < 1e-9,
          f"split fractions sum to 1.0 (got {total_fraction})")
    check(SPLIT_UNIT == "seed",
          "split is by seed, not by example, so paraphrases cannot leak")
    check(set(SPLIT_FRACTIONS) == {"train", "val", "holdout"},
          "has train, val, and holdout")
    check(SPLIT_FRACTIONS["holdout"] >= 0.10,
          "holdout is large enough to mean anything")
    check("never inspected while" in HOLDOUT_POLICY,
          "holdout has an explicit non-tuning policy")

    # -- isolation ------------------------------------------------------
    check(ISOLATION.production_guard, "production guard is required")
    check("harness_session" in ISOLATION.store,
          "isolation is built on the shared harness_session guard")
    check("production_digest" in ISOLATION.verify,
          "production is verified by content digest, not file hash")
    check("seed" in ISOLATION.dedupe_across_splits,
          "cross-split dedupe is defined and is seed-aware")
    check("A1" in ISOLATION.reserved,
          "T1 scenarios are reserved as evaluation material, not training "
          "seeds")

    # -- evaluation -----------------------------------------------------
    for required in ("E5_over_trigger", "E6_narration", "E9_regression"):
        check(any(c.id == required for c in EVALUATION),
              f"evaluation includes {required}")
    check(any(c.split == "holdout" for c in EVALUATION),
          "headline criteria are measured on holdout")
    check("must not regress" in REGRESSION_GATE.lower() or
          "falls below 5/5" in REGRESSION_GATE,
          "a regression gate blocks training on a traded behaviour")

    # -- size -----------------------------------------------------------
    summary = totals()
    check(summary["examples"] >= 3000,
          f"dataset is large enough to generalise "
          f"({summary['examples']} examples)")
    check(summary["per_split"]["holdout"] >= 400,
          f"holdout has enough examples to measure "
          f"({summary['per_split']['holdout']})")

    print("=" * 72)
    print("AURORA BRAIN v0.1 DATASET DESIGN - VALIDATION")
    print("=" * 72)
    print(f"  categories : {summary['categories']}")
    print(f"  seeds      : {summary['seeds']}")
    print(f"  examples   : {summary['examples']}")
    for name, count in summary["per_split"].items():
        print(f"    {name:8s} {count}")
    print(f"\n  checks run : {len(CHECKS)}")
    for failure in FAILURES:
        print(f"  FAILED     {failure}")
    print(f"\n  result     : {'PASS' if not FAILURES else 'FAIL'}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
