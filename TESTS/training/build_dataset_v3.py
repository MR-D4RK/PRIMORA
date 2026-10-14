"""Build the Aurora Brain v0.3 dataset from the approved v0.3 seed bank.

This is the simple path. It replaces the token-ratio machinery of
``generate_v3`` and nothing else:

*   the seed bank is still ``seed_bank_v3`` + ``check_bank_v3.new_seeds`` - the
    approved 211 seeds, with their ids and provenance untouched;
*   the record builder is still ``builders_v3.build`` - same turn lists, same
    slot filling, same loss masking, same ``<prime_request>`` rendering;
*   the wire format is still ``generate_v2.SCHEMA_VERSION`` and the system
    context is still built by ``generate_v2.build_system_context``, so the
    emitted files load through ``aurora.dataset_v2`` unchanged.

What changed is only *allocation*. v0.3 no longer chases a 65/35
supervised-token split, so there is no token objective to optimise, no DP over
token lengths, and no tokenizer call during generation at all. Each train seed
simply contributes as many distinct draws as it has, up to a per-band ceiling,
and conversational seeds get the larger ceiling because Aurora is primarily a
conversational companion. That gives conversational records the majority by
construction rather than by arithmetic against a token target.

Run: python TESTS/training/build_dataset_v3.py [outdir]
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import builders_v3 as B3  # noqa: E402
import check_bank_v3 as C3  # noqa: E402
import design_v3 as D3  # noqa: E402
import seed_bank_v3 as SB3  # noqa: E402
from generate_v2 import SCHEMA_VERSION, build_system_context  # noqa: E402

GENERATOR_VERSION = "aurora-brain-v0.3-build-1"
DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "v3")

SPLIT_ORDER = ("train", "validation", "holdout")

# Per-seed record ceilings. Conversational behaviour is the point of Aurora, so
# a conversational seed may use up to ten of its distinct draws and a capability
# seed up to five. These are coverage decisions, not a token budget: the two
# bands differ because capability records carry embedded JSON and would
# otherwise dominate the corpus by volume.
CONVERSATIONAL_EXPANSION = 10
CAPABILITY_EXPANSION = 5

# Evaluation seeds fill a fixed depth so a split cannot be padded by repeating a
# target from a seed that ran out of material.
EVAL_DEPTH = 8

# The design fixes validation at 180 records and there are 22 validation seeds, so
# 8 each reaches 176 and the first four by priority take one more. The rule is
# kept exactly so the split sizes match the approved design without any record
# budget arithmetic.
VALIDATION_TOP_UP = 4

# One distinct final target may be used this many times inside one seed. Kept
# because the value is the behaviour being taught - it must not be memorised as
# the only reply - not because a global ratio demanded it.
MAX_TARGET_REPEAT = 3

# ``check_bank_v3.split_of`` sends the first seed of each category to holdout by
# priority, which puts C15-R01 - the only deliberate capability-failure case in
# the bank - into holdout. The failure is exactly the behaviour that most needs
# supervising, so it is moved to train and the next C15 seed takes its holdout
# place. Seed-level separation is unaffected: both seeds still live wholly in
# one split.
SPLIT_OVERRIDES = {
    "C15-R01": "train",
    "C15-R04": "holdout",
}


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


def seeds() -> list[SB3.SeedV3]:
    """The approved bank, in bank order. 211 seeds, one review-only anchor."""
    return SB3.build_retained(SB3.build_extras()) + C3.new_seeds()


def split_of(all_seeds: list[SB3.SeedV3]) -> dict[str, str]:
    """Whole-seed split, reusing the bank's own rule plus two overrides."""
    split = C3.split_of(all_seeds, [])
    for seed_id, side in SPLIT_OVERRIDES.items():
        if seed_id in split:
            split[seed_id] = side
    return split


def capacity(seed: SB3.SeedV3) -> int:
    """Records this seed can contribute without exceeding the repeat cap."""
    draws = len(B3.materialise(seed))
    if not draws:
        return 0
    return MAX_TARGET_REPEAT * draws


def record_counts(split: dict[str, str],
                  all_seeds: list[SB3.SeedV3]) -> dict[str, int]:
    """How many records each seed emits.

    Train seeds use their band ceiling, capped by distinct-target capacity.
    Evaluation seeds use a fixed depth, also capped by capacity - a seed that
    cannot fill the depth is reported rather than silently padded.
    """
    topped_up = {
        seed.seed_id for seed in sorted(
            (s for s in all_seeds if split.get(s.seed_id) == "validation"),
            key=lambda s: (C3.PRIORITY_RANK.get(s.priority, 3), s.seed_id)
        )[:VALIDATION_TOP_UP]
    }

    counts: dict[str, int] = {}
    for seed in all_seeds:
        if not seed.training_allowed or not seed.variants:
            continue
        side = split[seed.seed_id]
        if side == "train":
            ceiling = (CONVERSATIONAL_EXPANSION
                       if seed.category in D3.CONVERSATIONAL_CATEGORIES
                       else CAPABILITY_EXPANSION)
        elif seed.seed_id in topped_up:
            ceiling = EVAL_DEPTH + 1
        else:
            ceiling = EVAL_DEPTH
        counts[seed.seed_id] = max(0, min(ceiling, capacity(seed)))
    return counts


def plan() -> tuple[list[SB3.SeedV3], dict[str, str], dict[str, int],
                   dict[str, list[B3.Draw]]]:
    """The whole generation plan, resolved once in bank order.

    Returns ``(seeds, split, counts, draws)``. Splitting, per-seed record counts
    and target ownership are all decided here, so generation and validation
    derive the same plan from the same deterministic walk.
    """
    all_seeds = seeds()
    split = split_of(all_seeds)
    counts = record_counts(split, all_seeds)
    draws, _owner = assign_draws(counts, all_seeds)
    return all_seeds, split, counts, draws


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def record(example: B3.Example, seed: SB3.SeedV3) -> dict:
    """The v2 wire shape, unchanged, plus the seed's provenance."""
    return {
        "id": example.id,
        "seed_id": example.seed_id,
        "category": example.category,
        "split": example.split,
        "messages": example.messages,
        "train_on": example.train_on,
        "expect_block": example.expect_block,
        "expect_action": example.expect_action,
        "note": example.note,
        "variant": example.variant,
        "seed_priority": example.seed_priority,
        "seed_provenance": seed.provenance,
        "system_ref": example.system_ref,
        "system_fingerprint": example.system_fingerprint,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
    }


def usable_draws(seed: SB3.SeedV3, owner: dict[str, str]) -> list[B3.Draw]:
    """Draws this seed may claim, excluding targets another seed already owns.

    The bank forbids two seeds *authoring* the same literal target, but a slot
    can still land on one: ``C5-A08``'s target is ``{value}.`` and the value
    bank contains ``nine``, which is exactly ``C15-R01``'s authored reply. One
    owner per target, first claim in bank order winning, is the whole rule.
    Because the map is global it also makes train/validation/holdout target
    isolation structural rather than something the validator has to catch.
    """
    return [draw for draw in B3.materialise(seed)
            if owner.get(draw.key, seed.seed_id) == seed.seed_id]


def assign_draws(counts: dict[str, int],
                 all_seeds: list[SB3.SeedV3]) -> tuple[dict[str, list[B3.Draw]],
                                                       dict[str, str]]:
    """Choose every seed's draws in one bank-order pass, claiming as it goes.

    Only the draws a seed actually uses are claimed, so a seed with twenty-eight
    authored variants and a five-record ceiling does not block a later seed from
    the twenty-three it never reached.
    """
    owner: dict[str, str] = {}
    chosen: dict[str, list[B3.Draw]] = {}
    for seed in all_seeds:
        want = counts.get(seed.seed_id, 0)
        if want <= 0:
            continue
        usable = usable_draws(seed, owner)
        # One pass over the distinct draws first, so the first records of a
        # seed are its distinct authored replies rather than repetitions.
        picked = list(usable[:want])
        # Then top up by repeating those same draws in order, never exceeding
        # the repeat cap on any one of them.
        if picked and len(picked) < want:
            uses = collections.Counter(draw.key for draw in picked)
            for draw in picked:
                while (len(picked) < want
                       and uses[draw.key] < MAX_TARGET_REPEAT):
                    picked.append(draw)
                    uses[draw.key] += 1
        for draw in picked:
            owner.setdefault(draw.key, seed.seed_id)
        chosen[seed.seed_id] = picked
    return chosen, owner


def build_seed(seed: SB3.SeedV3, draws: list[B3.Draw], side: str,
               system_key: str, fingerprint: str) -> list[B3.Example]:
    """One record per assigned draw, with the framing prefix rotated.

    The draws are already distinct-first and already respect the repeat cap (see
    ``assign_draws``). The prefix rotates independently of the draw, which is
    what varies the opening without authoring a second word of the seed's own
    material.
    """
    examples: list[B3.Example] = []
    for index, draw in enumerate(draws):
        options = B3.prefix_options(seed, draw)
        prefix = options[index % len(options)][0]
        example = B3.build(seed, draw, prefix, index)
        example.split = side
        example.system_ref = system_key
        example.system_fingerprint = fingerprint
        examples.append(example)
    return examples


def _pair_blocks(seed_ids: tuple[str, ...],
                 remaining: collections.defaultdict) -> list[B3.Example]:
    """Emit one required contrast pair as an alternating block."""
    sides = {sid: remaining.get(sid, []) for sid in seed_ids}
    for sid, rows in sides.items():
        if not rows:
            return []
    block: list[B3.Example] = []
    turn = 0
    while any(sides[sid] for sid in seed_ids):
        for sid in seed_ids:
            if sides[sid]:
                block.append(sides[sid].pop(0))
        turn += 1
        if turn > sum(len(v) for v in sides.values()):
            break
    return block


def order_examples(examples: list[B3.Example]) -> list[B3.Example]:
    """Category by category; required contrast pairs first; then round-robin.

    Round-robin across the remaining seeds means two consecutive records in a
    category come from different seeds by construction, so no long run of one
    seed's wording can reach the trainer. That is the whole ordering policy -
    there is no construction scheduler and nothing to tune.
    """
    by_category: dict[str, list[B3.Example]] = collections.OrderedDict()
    for category in D3.CATEGORIES:
        rows = [e for e in examples if e.category == category.key]
        if rows:
            by_category[category.key] = rows

    ordered: list[B3.Example] = []
    for rows in by_category.values():
        remaining: dict[str, list[B3.Example]] = collections.defaultdict(list)
        for example in rows:
            remaining[example.seed_id].append(example)
        placed: list[B3.Example] = []
        for pair in D3.REQUIRED_ADJACENT_PAIRS:
            if not any(example.seed_id in pair for example in rows):
                continue
            block = _pair_blocks(pair, remaining)
            placed.extend(block)
        lanes = [remaining[sid] for sid in sorted(remaining) if remaining[sid]]
        while lanes:
            for lane in lanes:
                if lane:
                    placed.append(lane.pop(0))
            lanes = [lane for lane in lanes if lane]
        ordered.extend(placed)
    return ordered


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def build() -> tuple[dict[str, list[dict]], list[str], dict]:
    """Generate, validate-free, and write the three splits.

    Returns ``(records, problems)``. Nothing is written if ``problems`` is
    non-empty; ``write`` is the only thing that touches the disk.
    """
    problems: list[str] = []
    all_seeds, split, counts, draws = plan()
    by_id = {s.seed_id: s for s in all_seeds}

    system_key, composed, capabilities = build_system_context()
    fingerprint = hashlib.sha256(composed.encode("utf-8")).hexdigest()

    examples: list[B3.Example] = []
    for seed in all_seeds:
        want = counts.get(seed.seed_id, 0)
        if want <= 0:
            continue
        side = split[seed.seed_id]
        available = draws[seed.seed_id]
        if len(available) != want:
            problems.append(
                f"{seed.seed_id}: {side} seed has {want} records planned but "
                f"{len(available)} drawable targets")
            continue
        if side in ("validation", "holdout") and want < EVAL_DEPTH:
            problems.append(
                f"{seed.seed_id}: {side} seed fills {want} records, short of "
                f"the {EVAL_DEPTH} evaluation depth")
        examples.extend(
            build_seed(seed, available, side, system_key, fingerprint))

    records: dict[str, list[dict]] = {s: [] for s in SPLIT_ORDER}
    for side in SPLIT_ORDER:
        rows = [e for e in examples if e.split == side]
        for example in order_examples(rows):
            records[side].append(record(example, by_id[example.seed_id]))

    meta = {"system_ref": system_key, "capabilities": capabilities}
    return records, problems, meta


def write(records: dict[str, list[dict]], meta: dict,
          outdir: str = DEFAULT_OUTDIR) -> dict[str, int]:
    os.makedirs(outdir, exist_ok=True)
    written: dict[str, int] = {}
    for side in SPLIT_ORDER:
        path = os.path.join(outdir, f"{side}.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for row in records[side]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[side] = len(records[side])
    with open(os.path.join(outdir, "HOLDOUT_PROTECTED.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "split": "holdout",
            "count": written["holdout"],
            "status": "protected - evaluation only",
            "rule": ("Not used during generation, tuning or wording decisions. "
                     "Read only when the holdout is actually being scored."),
            "generator_version": GENERATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
        }, handle, indent=2)
    with open(os.path.join(outdir, "build_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "generator_version": GENERATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
            "system_ref": meta["system_ref"],
            "capabilities": meta["capabilities"],
            "counts": written,
            "expansion": {
                "conversational": CONVERSATIONAL_EXPANSION,
                "capability": CAPABILITY_EXPANSION,
                "evaluation_depth": EVAL_DEPTH,
                "max_target_repeat": MAX_TARGET_REPEAT,
            },
            "split_overrides": SPLIT_OVERRIDES,
        }, handle, indent=2)
    return written


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTDIR
    records, problems, meta = build()
    if problems:
        print(f"REFUSING TO WRITE: {len(problems)} problems")
        for problem in problems[:20]:
            print(f"  FAIL {problem}")
        return 1
    written = write(records, meta, outdir)
    print("=" * 72)
    print("AURORA BRAIN v0.3 - DATASET BUILD (simple path)")
    print("=" * 72)
    for side in SPLIT_ORDER:
        print(f"  {side:<11}: {written[side]}")
    print(f"  output     : {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
