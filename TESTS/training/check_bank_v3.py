"""Structural check of the v0.3 seed bank.

Everything here is decided before a single record is generated, because these
are the gates the design makes non-negotiable: 211 seeds with the exact ids the
design names, domain and register spread inside every category, no verbatim
target shared between seeds, and enough distinct final targets to fill the
record budgets without repeating a target more than three times.

The split is derived here too, deterministically, so the capacity arithmetic
below is the same arithmetic the generator will use.

Run:  python check_bank_v3.py
Exit code 0 means every gate held; any failure prints a FAIL line and exits 1.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import audit_v2 as A2
import builders_v2 as B2
import design_v2 as D2
import design_v3 as D3
import new_seeds_v3a as NA
import new_seeds_v3b as NB
import seed_bank_v3 as SB3

PRIORITY_RANK = {"critical": 0, "high": 1, "normal": 2}

# The seven categories that take a second validation seed: the largest budgets
# that still have train-side slack. C11 is deliberately absent. It has 18 seeds
# and a budget of 160, which at the 10x ceiling needs all 16 of its non-eval
# seeds, so it can afford exactly two evaluation seeds and no more.
EXTRA_VAL: tuple[str, ...] = ("C10", "C8", "C12", "C13", "C9", "C14", "C5")

# Every evaluation seed must fill a whole depth on its own.
EVAL_DEPTH = 8

EXPECTED_BY_CATEGORY = {
    "C1": 10, "C2": 8, "C3": 10, "C4": 9, "C5": 14, "C6": 8, "C7": 16,
    "C8": 20, "C9": 16, "C10": 20, "C11": 18, "C12": 18, "C13": 18,
    "C14": 16, "C15": 10,
}

SLOT_RE = re.compile(r"\{(\w+)\}")
BANKS = {
    "item": B2.ITEMS,
    "value": SB3.ALL_VALUES,
    "day": B2.DAYS,
    "note": SB3.NOTES,
    "query": tuple(B2.QUERY.values()),
}


def normalise(text: str) -> str:
    return A2.normalise(text)


def new_seeds() -> list[SB3.SeedV3]:
    """The 75 newly authored seeds, shaped like the retained ones."""
    seeds: list[SB3.SeedV3] = []
    for raw in NA.NEW_SEEDS_A + NB.NEW_SEEDS_B:
        variants = tuple(SB3.Variant.ask_answer(ask, target)
                         for ask, target in raw["variants"])
        seeds.append(SB3.SeedV3(
            seed_id=raw["seed_id"],
            category=raw["category"],
            kind="prose",
            scenario=raw["scenario"],
            ask=raw["ask"],
            behavior=raw["behavior"],
            target_shape=raw["shape"],
            why=raw["why"],
            multi_turn=False,
            priority=raw["priority"],
            domain=raw["domain"],
            register=raw["register"],
            provenance="new",
            training_allowed=True,
            train_on=None,
            variants=variants,
        ))
    return seeds


def split_of(seeds: list[SB3.SeedV3], problems: list[str]) -> dict[str, str]:
    """Whole-seed split: holdout, then validation, then train.

    Three rules bind the ordering. Seeds inside a ``REQUIRED_ADJACENT_PAIRS``
    pair go to train together, because the contrast only teaches anything when
    both halves are supervised and adjacent in the emitted order. Evaluation
    seeds must reach ``EVAL_DEPTH`` records on their own, so a seed that can
    only fill three records is never spent on a split it cannot fill. What is
    left is ordered by priority then id, and the first seed of each category
    becomes the holdout representative, the next validation, and the one after
    that a second validation seed for the seven extra categories.
    """
    pairs = {sid: pair for pair in D3.REQUIRED_ADJACENT_PAIRS
             for sid in pair}
    out: dict[str, str] = {}
    by_category: dict[str, list[SB3.SeedV3]] = defaultdict(list)
    for seed in seeds:
        if seed.training_allowed:
            by_category[seed.category].append(seed)

    for category, rows in by_category.items():
        rows.sort(key=lambda s: (PRIORITY_RANK.get(s.priority, 3), s.seed_id))
        slots = 2 + (1 if category in EXTRA_VAL else 0)
        free = [s for s in rows if s.seed_id not in pairs]
        evalable = [s for s in free if capacity(s) >= EVAL_DEPTH]
        if len(evalable) < slots:
            problems.append(
                f"{category}: {len(evalable)} seeds can fill {EVAL_DEPTH} "
                f"records, need {slots} for the evaluation splits")
        picked = evalable[:slots]
        if picked:
            out[picked[0].seed_id] = "holdout"
        for seed in picked[1:]:
            out[seed.seed_id] = "validation"
        for seed in rows:
            out.setdefault(seed.seed_id, "train")
    return out


def distinct_and_fills(seed: SB3.SeedV3) -> tuple[int, int]:
    """(distinct literal final targets, distinct slot-filled final targets)."""
    literal: set[str] = set()
    fills = 0
    for variant in seed.variants:
        target = variant.target
        slots = sorted(set(SLOT_RE.findall(target)))
        if not slots:
            if target.strip():
                literal.add(normalise(target))
            continue
        ways = 1
        for name in slots:
            ways *= len(BANKS.get(name, ()))
        fills += ways
    return len(literal), fills


def capacity(seed: SB3.SeedV3) -> int:
    """Records this seed can contribute under the design ceilings."""
    literal, fills = distinct_and_fills(seed)
    distinct = literal + fills
    if not distinct:
        return 0
    return min(D3.MAX_EXPANSION, D3.MAX_TARGET_REPEAT * distinct)


def main() -> int:
    problems: list[str] = []

    def fail(message: str) -> None:
        problems.append(message)

    retained = SB3.build_retained(SB3.build_extras())
    fresh = new_seeds()
    seeds = retained + fresh
    by_id = {s.seed_id: s for s in seeds}

    # -- identity ---------------------------------------------------------
    want = set(D3.RETAINED_IDS) | set(D3.NEW_IDS)
    if len(seeds) != D3.SEED_BUDGET:
        fail(f"seed count {len(seeds)} != {D3.SEED_BUDGET}")
    if set(by_id) != want:
        fail(f"id mismatch: missing {sorted(want - set(by_id))}, "
             f"extra {sorted(set(by_id) - want)}")
    if set(D3.RETAINED_IDS) & set(D3.NEW_IDS):
        fail("RETAINED_IDS and NEW_IDS overlap")
    counts = Counter(s.category for s in seeds)
    for category, expected in EXPECTED_BY_CATEGORY.items():
        if counts[category] != expected:
            fail(f"{category} has {counts[category]} seeds, expected {expected}")

    # -- domains, registers ----------------------------------------------
    for seed in seeds:
        if seed.domain not in D3.DOMAINS:
            fail(f"{seed.seed_id} domain {seed.domain!r} not in design")
        # The bank labels a few registers more finely than the design does;
        # fold them first, then require the folded label to be one the design
        # recognises. An unfoldable label would silently stop counting.
        folded = SB3.fold_register(seed.register)
        if folded not in D3.REGISTERS:
            fail(f"{seed.seed_id} register {seed.register!r} folds to "
                 f"{folded!r}, which is not in design")
    for category in EXPECTED_BY_CATEGORY:
        rows = [s for s in seeds if s.category == category]
        domains = {s.domain for s in rows}
        registers = {SB3.fold_register(s.register) for s in rows}
        if len(domains) < D3.MIN_DOMAINS_PER_CATEGORY:
            fail(f"{category} spans {len(domains)} domains, "
                 f"need {D3.MIN_DOMAINS_PER_CATEGORY}")
        if len(registers) < D3.MIN_REGISTERS_PER_CATEGORY:
            fail(f"{category} spans {len(registers)} registers, "
                 f"need {D3.MIN_REGISTERS_PER_CATEGORY}")

    # -- authored material -------------------------------------------------
    for seed in seeds:
        if not seed.variants:
            if seed.training_allowed:
                fail(f"{seed.seed_id} has no variants")
            continue
        if seed.provenance == "new" and len(seed.variants) < 6:
            fail(f"{seed.seed_id} has {len(seed.variants)} variants, need >= 6")
        targets: Counter[str] = Counter()
        for variant in seed.variants:
            ask, target = variant.ask, variant.target
            if not ask.strip():
                fail(f"{seed.seed_id} has an empty ask")
            if not target.strip():
                fail(f"{seed.seed_id} has an empty target")
            if SLOT_RE.search(target):
                continue
            if normalise(ask) == normalise(target):
                fail(f"{seed.seed_id} ask equals its target: {ask!r}")
            targets[normalise(target)] += 1
        for text, times in targets.items():
            if times > 1:
                fail(f"{seed.seed_id} repeats one final target {times}x: "
                     f"{text[:60]!r}")

    # -- verbatim across seeds -------------------------------------------
    target_home: dict[str, str] = {}
    ask_home: dict[str, str] = {}
    for seed in seeds:
        for variant in seed.variants:
            text = variant.target
            if not SLOT_RE.search(text) and text.strip():
                key = normalise(text)
                if key in target_home and target_home[key] != seed.seed_id:
                    fail(f"target shared by {target_home[key]} and "
                         f"{seed.seed_id}: {text[:60]!r}")
                target_home.setdefault(key, seed.seed_id)
        key = normalise(seed.ask)
        if key in ask_home and ask_home[key] != seed.seed_id:
            fail(f"ask shared by {ask_home[key]} and {seed.seed_id}: "
                 f"{seed.ask!r}")
        ask_home.setdefault(key, seed.seed_id)

    # -- ascii -------------------------------------------------------------
    for seed in fresh:
        for text in [seed.ask, seed.behavior, seed.target_shape, seed.why] + [
                part for v in seed.variants for part in (v.ask, v.target)]:
            if any(ord(ch) > 127 for ch in text):
                fail(f"{seed.seed_id} has non-ascii text: {text[:50]!r}")

    # -- correction tables stay pointed at retained seeds ------------------
    for name, table in (
        ("PROSE_TARGET_OVERRIDES", SB3.PROSE_TARGET_OVERRIDES),
        ("EMIT_PAYLOAD_OVERRIDES", SB3.EMIT_PAYLOAD_OVERRIDES),
        ("TOOL_ANSWER_OVERRIDES", SB3.TOOL_ANSWER_OVERRIDES),
        ("TURNS_OVERRIDES", SB3.TURNS_OVERRIDES),
        ("KIND_OVERRIDES", SB3.KIND_OVERRIDES),
    ):
        for sid in table:
            if sid not in D3.RETAINED_IDS:
                fail(f"{name} names {sid}, which is not a retained seed")

    # -- request payloads match the runtime's own contract -------------------
    # A payload the capability refuses is a record teaching Aurora to fail:
    # consult_specialist fails on a missing specialist before it reads the
    # task, and learn fails on anything but 'text' plus 'provenance'. One
    # deliberate exception - C15-R01 sends an unknown capability on purpose so
    # the record can show the tool refusing it and Aurora recovering.
    deliberate = {("C15-R01", "count_words")}
    try:
        from PRIMORA.prime.capability import CapabilityRegistry
        registry = CapabilityRegistry()
        known = set(registry.get_names())
    except Exception as exc:  # the gate is worthless if the runtime is missing
        registry, known = None, set()
        fail(f"capability registry unavailable: {exc}")

    if registry:
        for seed in seeds:
            for variant in seed.variants:
                for turn in variant.turns:
                    if turn[0] not in ("R", "X"):
                        continue
                    action, payload = turn[1], turn[2]
                    if action not in known:
                        if (seed.seed_id, action) not in deliberate:
                            fail(f"{seed.seed_id} requests unknown capability "
                                 f"{action!r}")
                        continue
                    spec = registry.get(action)
                    missing = [name for name in spec.required_fields
                               if name not in payload]
                    extra = sorted(set(payload)
                                   - set(spec.required_fields)
                                   - set(spec.optional_fields))
                    if missing or extra:
                        fail(f"{seed.seed_id} {action} payload "
                             f"missing={missing} extra={extra}")

    # -- split and capacity ------------------------------------------------
    split = split_of(seeds, problems)
    train_rows = [s for s in seeds if split.get(s.seed_id) == "train"]
    val_rows = [s for s in seeds if split.get(s.seed_id) == "validation"]
    hold_rows = [s for s in seeds if split.get(s.seed_id) == "holdout"]
    if len(val_rows) != 22:
        fail(f"validation has {len(val_rows)} seeds, expected 22")
    if len(hold_rows) != 15:
        fail(f"holdout has {len(hold_rows)} seeds, expected 15")

    for pair in D3.REQUIRED_ADJACENT_PAIRS:
        sides = {split.get(sid, "blocked") for sid in pair}
        if len(sides) != 1:
            fail(f"adjacent pair {pair} split across {sorted(sides)}")

    # Evaluation splits are fixed-depth: every seed there must reach the depth
    # on its own, or the split cannot be filled without repeating a target.
    for rows, depth, budget, label in (
        (hold_rows, 8, D3.HOLDOUT_SIZE, "holdout"),
        (val_rows, 8, D3.VALIDATION_SIZE, "validation"),
    ):
        reach = 0
        for seed in rows:
            reach += capacity(seed)
            if capacity(seed) < depth:
                fail(f"{label} seed {seed.seed_id} reaches only "
                     f"{capacity(seed)} records, needs {depth}")
        if reach < budget:
            fail(f"{label} capacity {reach} < {budget}")

    train_counts = Counter(s.category for s in train_rows)
    for problem in D3.capacity_problems(dict(train_counts)):
        fail(problem)

    # Per-category placement: every budget must be reachable inside train at
    # the per-seed ceiling, and the records must come with enough distinct
    # final targets to satisfy MIN_DISTINCT_FINAL_TARGETS and the repeat cap.
    #
    # Allocation runs in three passes because distinct targets, not records,
    # are what the gate counts. Every train seed first receives the minimum
    # expansion, then records go where they buy a new distinct target, and only
    # then where they repeat one - so 900 distinct is reached with the least
    # material spent.
    allocation: dict[str, int] = {}
    for category, budget in D3.RECORD_BUDGET.items():
        rows = [s for s in train_rows if s.category == category]
        caps = {s.seed_id: capacity(s) for s in rows}
        spans = {s.seed_id: sum(distinct_and_fills(s)) for s in rows}
        if sum(caps.values()) < budget:
            fail(f"{category}: train capacity {sum(caps.values())} < "
                 f"budget {budget}")
            continue
        order = sorted(rows, key=lambda s: s.seed_id)
        left = budget
        for ceiling in (
            lambda sid: min(D3.MIN_EXPANSION, caps[sid]),
            lambda sid: min(caps[sid], spans[sid]),
            lambda sid: caps[sid],
        ):
            if not left:
                break
            progressed = True
            while left and progressed:
                progressed = False
                for seed in order:
                    sid = seed.seed_id
                    current = allocation.get(sid, 0)
                    take = min(ceiling(sid) - current, left)
                    if take > 0:
                        allocation[sid] = current + take
                        left -= take
                        progressed = True
                        if not left:
                            break
        if left:
            fail(f"{category}: could not place {left} records")

    distinct_total = 0
    for sid, records in allocation.items():
        distinct_total += min(records, sum(distinct_and_fills(by_id[sid])))
    if distinct_total < D3.MIN_DISTINCT_FINAL_TARGETS:
        fail(f"train distinct final targets {distinct_total} < "
             f"{D3.MIN_DISTINCT_FINAL_TARGETS}")

    if sum(allocation.values()) != D3.TRAIN_SIZE:
        fail(f"train plan places {sum(allocation.values())} records, "
             f"expected {D3.TRAIN_SIZE}")

    # -- report ------------------------------------------------------------
    split_records = {
        "train": sum(allocation.values()),
        "validation": D3.VALIDATION_SIZE,
        "holdout": D3.HOLDOUT_SIZE,
    }
    tightest = sorted(
        ((sid, allocation[sid], sum(distinct_and_fills(by_id[sid])))
         for sid in allocation),
        key=lambda row: (row[2] - row[1], row[0]),
    )[:12]

    report = {
        "seeds": len(seeds),
        "by_provenance": dict(Counter(s.provenance for s in seeds)),
        "by_kind": dict(Counter(s.kind for s in seeds)),
        "split_seeds": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "holdout": len(hold_rows),
            "blocked": [s.seed_id for s in seeds if not s.training_allowed],
        },
        "split_records": dict(split_records),
        "train_priority": dict(Counter(s.priority for s in train_rows)),
        "eval_priority": dict(Counter(
            s.priority for s in val_rows + hold_rows)),
        "distinct_train_targets": distinct_total,
        "min_variants": min(len(s.variants) for s in seeds),
        "min_capacity": min(capacity(s) for s in seeds if s.training_allowed),
        "seeds_short_of_distinct": {
            sid: {"records": records,
                  "distinct": sum(distinct_and_fills(by_id[sid]))}
            for sid, records, _span in tightest
            if records > sum(distinct_and_fills(by_id[sid]))
        },
        "capacity_by_kind": {
            kind: sorted({s.seed_id for s in seeds
                          if s.kind == kind and capacity(s) < 4})[:12]
            for kind in ("prose", "emit", "tool", "cycle")
        },
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    if problems:
        print("\nFAILURES:")
        for problem in problems:
            print(f"  FAIL {problem}")
        return 1
    print("\nOK: every seed-bank gate held.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
