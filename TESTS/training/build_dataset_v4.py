"""Build the Aurora Brain v0.4 behavioral-correction dataset.

The v0.4 build has one job: turn the approved 240-seed behavioral bank
(``seed_bank_v4``) plus the authored response bank (``responses_v4``) into
exactly 1,440 training records under a leak-resistant whole-seed split.

What it does:

*   grouping     - seeds that share a distinctive value (a time, a dotted
                   version, a path, thousands, four-plus digits, a weekday, a
                   month, an ordinal, or a named value word such as
                   ``helios``) or an identical normalised ask are joined by
                   union-find. A group is atomic: it can never straddle two
                   splits, so a value trained in one split cannot be tested
                   in another;
*   split        - groups are walked in a canonical order (cell index, then
                   difficulty, then category) and filled greedily into
                   holdout (30 seeds), then validation (30 seeds), then
                   train; pass two moves whole train groups into whatever
                   eval quota is still short, so the split is exactly
                   180/30/30 seeds = 1080/180/180 records;
*   records      - six per seed: each of the three authored replies in bare
                   and framed form. Context seeds open on their context plus
                   a short acknowledgement, prior seeds on their prior
                   exchange. Loss lands on the final assistant turn only;
*   gates        - exact counts, per-split difficulty/category/idk_ok
                   coverage, whole-seed atomicity, disjoint distinctive-value
                   pools, target repetition (one exact final target may be
                   supervised at most twice). Any gate failing means
                   REFUSE TO WRITE.

No token-ratio machinery, no genetic algorithm, no dynamic programming, no
slot banks, no record budgets: allocation is grouping plus quota fill.

Run: python TESTS/training/build_dataset_v4.py [outdir]
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import sys

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import builders_v2 as B2  # noqa: E402
import builders_v3 as B3  # noqa: E402
import responses_v4 as RV4  # noqa: E402
import seed_bank_v4 as SB4  # noqa: E402
from generate_v2 import SCHEMA_VERSION, build_system_context  # noqa: E402

GENERATOR_VERSION = "aurora-brain-v0.4-build-1"
DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "v4")

SPLIT_ORDER = ("train", "validation", "holdout")
CATEGORIES = tuple(SB4.CATEGORY_SPECS)
DIFFICULTIES = (1, 2, 3, 4)

# Whole-seed quota for each evaluation split. Train takes whatever remains.
SEED_QUOTA = {"validation": 30, "holdout": 30}
RECORDS_PER_SEED = 6

# Seed difficulty maps to record priority. Nothing else sets it.
PRIORITY = {4: "critical", 3: "high", 2: "normal", 1: "normal"}

# One exact final target may be supervised at most twice: the bare record and
# the framed record of the same authored reply. A higher count means the same
# sentence is being drilled past the ceiling the gate exists to hold.
TARGET_REPEAT_LIMIT = 2


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------

_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
_DOTTED = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")
_THOUSANDS = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_ORDINAL = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\b", re.IGNORECASE)
_LONGNUM = re.compile(r"\b\d{4,}\b")
_PATH = re.compile(r"[\w.-]+(?:/[\w.-]+)+")
_WEEKDAY = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE)
_MONTH = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b", re.IGNORECASE)
_WORDVAL = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in sorted(RV4.VALUE_WORDS))
    + r")\b", re.IGNORECASE)

# Run in this order and mask each match, so a fragment of one value (the "400"
# inside "384,400") can never come back out as a second, spurious value.
_VALUE_PATTERNS = (_TIME, _DOTTED, _THOUSANDS, _ORDINAL, _LONGNUM, _PATH,
                   _WEEKDAY, _MONTH, _WORDVAL)


def spec_texts(seed_id: str) -> list[str]:
    """Every authored text of a seed, i.e. the whole of its user-visible world."""
    spec = RV4.SPECS[seed_id]
    return [spec.ask, spec.context or "", spec.prior_ask or "",
            spec.prior_say or "", *spec.replies]


def extract_values(text: str) -> set[str]:
    """Distinctive-value candidates inside one string, casefolded."""
    values: set[str] = set()
    masked = text
    for pattern in _VALUE_PATTERNS:
        for match in pattern.finditer(masked):
            values.add(match.group(0).casefold())
        masked = pattern.sub(lambda m: " " * len(m.group(0)), masked)
    return values


def normalise_ask(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def seed_values(seed_id: str) -> set[str]:
    values: set[str] = set()
    for text in spec_texts(seed_id):
        values |= extract_values(text)
    return values


class _Union:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def build_groups() -> tuple[list[tuple[str, ...]], dict[str, tuple[str, ...]],
                            dict]:
    """Connected components over shared values and shared normalised asks.

    Returns ``(groups_in_touch_order, seed_to_group, stats)``. Values owned by
    a single seed link nothing - they are context, not an obligation.
    """
    seed_ids = [seed["id"] for seed in SB4.SEEDS]
    union = _Union(seed_ids)

    value_owners: dict[str, list[str]] = collections.defaultdict(list)
    ask_owners: dict[str, list[str]] = collections.defaultdict(list)
    for seed in SB4.SEEDS:
        sid = seed["id"]
        for value in seed_values(sid):
            value_owners[value].append(sid)
        ask_owners[normalise_ask(RV4.SPECS[sid].ask)].append(sid)

    shared_values = 0
    for owners in value_owners.values():
        if len(owners) > 1:
            shared_values += 1
            for other in owners[1:]:
                union.union(owners[0], other)
    for owners in ask_owners.values():
        if len(owners) > 1:
            for other in owners[1:]:
                union.union(owners[0], other)

    components: dict[str, list[str]] = collections.defaultdict(list)
    for sid in seed_ids:
        components[union.find(sid)].append(sid)
    by_group = {root: tuple(sorted(members))
                for root, members in components.items()}
    seed_to_group = {sid: group for group in by_group.values()
                     for sid in group}

    order = canonical_order(seed_to_group)
    stats = {
        "groups": len(order),
        "largest_group": max(len(g) for g in order),
        "shared_values": shared_values,
        "multi_seed_groups": sum(1 for g in order if len(g) > 1),
        "seeds_in_multi_groups": sum(len(g) for g in order if len(g) > 1),
    }
    return order, seed_to_group, stats


def canonical_order(seed_to_group: dict[str, tuple[str, ...]]
                    ) -> list[tuple[str, ...]]:
    """Cell-major interleave: round index, then difficulty, then category.

    Cells are (difficulty, category) buckets of seeds in bank order. Walking
    them by row index first means the first groups touched already spread
    across every difficulty and category, which is what lets both eval splits
    satisfy the coverage gates under a simple greedy fill.
    """
    cells: dict[tuple[int, str], list[str]] = collections.defaultdict(list)
    for seed in SB4.SEEDS:
        cells[(seed["difficulty"], seed["category"])].append(seed["id"])

    seen: set[tuple[str, ...]] = set()
    order: list[tuple[str, ...]] = []
    max_cell = max(len(v) for v in cells.values())
    for index in range(max_cell):
        for difficulty in DIFFICULTIES:
            for category in CATEGORIES:
                cell = cells.get((difficulty, category), [])
                if index >= len(cell):
                    continue
                group = seed_to_group[cell[index]]
                if group not in seen:
                    seen.add(group)
                    order.append(group)
    return order


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def assign_split(order: list[tuple[str, ...]]) -> tuple[dict[str, str],
                                                        list[str], dict]:
    """Greedy quota fill, then whole-group moves from train into any deficit."""
    problems: list[str] = []
    split: dict[str, str] = {}
    counts = {side: 0 for side in SPLIT_ORDER}

    for group in order:
        side = "train"
        for target in ("holdout", "validation"):
            if counts[target] + len(group) <= SEED_QUOTA[target]:
                side = target
                break
        counts[side] += len(group)
        for sid in group:
            split[sid] = side

    moved = 0
    for target in ("holdout", "validation"):
        deficit = SEED_QUOTA[target] - counts[target]
        while deficit > 0:
            candidates = [
                (i, group) for i, group in enumerate(order)
                if len(group) <= deficit
                and all(split[sid] == "train" for sid in group)
            ]
            if not candidates:
                problems.append(
                    f"cannot fill {target}: {deficit} seed(s) short and no "
                    f"whole train group that small exists")
                break
            _, best = max(candidates, key=lambda item: (len(item[1]), -item[0]))
            for sid in best:
                split[sid] = target
            counts[target] += len(best)
            counts["train"] -= len(best)
            deficit -= len(best)
            moved += 1

    stats = {"counts": counts, "groups_moved": moved}
    return split, problems, stats


def split_problems(split: dict[str, str],
                   seed_to_group: dict[str, tuple[str, ...]]) -> list[str]:
    """The write gates: exact counts, coverage, atomicity, disjoint pools."""
    problems: list[str] = []
    counts = collections.Counter(split.values())
    for side, want in (("train", 180), ("validation", 30), ("holdout", 30)):
        if counts[side] != want:
            problems.append(f"{side} holds {counts[side]} seeds, need {want}")

    by_id = {seed["id"]: seed for seed in SB4.SEEDS}
    for side in SPLIT_ORDER:
        sids = [sid for sid, s in split.items() if s == side]
        difficulties = {by_id[sid]["difficulty"] for sid in sids}
        categories = {by_id[sid]["category"] for sid in sids}
        idk = {RV4.SPECS[sid].idk_ok for sid in sids}
        missing_d = sorted(set(DIFFICULTIES) - difficulties)
        missing_c = sorted(set(CATEGORIES) - categories)
        if missing_d:
            problems.append(f"{side}: difficulty cells empty: {missing_d}")
        if missing_c:
            problems.append(f"{side}: categories empty: {missing_c}")
        if idk != {True, False}:
            problems.append(f"{side}: idk_ok coverage is {sorted(idk)}")

    for group in seed_to_group.values():
        sides = {split[sid] for sid in group}
        if len(sides) > 1:
            problems.append(
                f"atomicity broken for group {group[:4]}... -> {sorted(sides)}")

    pools: dict[str, set[str]] = {side: set() for side in SPLIT_ORDER}
    asks: dict[str, set[str]] = {side: set() for side in SPLIT_ORDER}
    for sid, side in split.items():
        pools[side] |= seed_values(sid)
        asks[side].add(normalise_ask(RV4.SPECS[sid].ask))
    for i, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[i + 1:]:
            shared = pools[left] & pools[right]
            if shared:
                problems.append(
                    f"{left}/{right} share distinctive value(s): "
                    f"{sorted(shared)[:5]}")
            shared_asks = asks[left] & asks[right]
            if shared_asks:
                problems.append(
                    f"{left}/{right} share a normalised ask: "
                    f"{sorted(shared_asks)[0][:50]!r}")
    return problems


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def priority_of(seed: dict) -> str:
    return PRIORITY[seed["difficulty"]]


def seed_messages(spec: RV4.Spec, reply: str, final_ask: str,
                  ack: str) -> list[dict[str, str]]:
    """One record's turn list. Context and prior seeds get their grounding
    turns; every record ends on the reply under supervision."""
    if spec.context:
        return [
            {"role": "user", "content": spec.context},
            {"role": "assistant", "content": ack},
            {"role": "user", "content": final_ask},
            {"role": "assistant", "content": reply},
        ]
    if spec.prior_ask:
        return [
            {"role": "user", "content": spec.prior_ask},
            {"role": "assistant", "content": spec.prior_say},
            {"role": "user", "content": final_ask},
            {"role": "assistant", "content": reply},
        ]
    return [
        {"role": "user", "content": final_ask},
        {"role": "assistant", "content": reply},
    ]


def build_seed_records(seed: dict, side: str, system_key: str,
                       fingerprint: str) -> list[dict]:
    """Six records: reply 0..2 x bare/framed. Framed records take FRAMES[j],
    which for odd j is never the empty frame, so bare and framed never
    collide."""
    sid = seed["id"]
    spec = RV4.SPECS[sid]
    records: list[dict] = []
    for j in range(RECORDS_PER_SEED):
        reply_idx = j // 2
        framed = j % 2 == 1
        frame = B3.FRAMES[j] if framed else ""
        final_ask = frame + spec.ask
        ack = B2.ACKS[j % len(B2.ACKS)]
        messages = seed_messages(spec, spec.replies[reply_idx], final_ask, ack)
        assistant_turns = sum(1 for m in messages
                              if m["role"] == "assistant")
        records.append({
            "id": f"{sid}-{j:02d}",
            "seed_id": sid,
            "category": seed["category"],
            "split": side,
            "messages": messages,
            "train_on": [assistant_turns - 1],
            "expect_block": False,
            "expect_action": None,
            "note": seed["behavior_target"],
            "variant": reply_idx,
            "seed_priority": priority_of(seed),
            "seed_provenance": "seed_bank_v4",
            "system_ref": system_key,
            "system_fingerprint": fingerprint,
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
        })
    return records


def order_records(rows: list[dict]) -> list[dict]:
    """Category-major (A..G); inside a category, round-robin across that
    split's seeds in id order, so no run of one seed's wording is contiguous."""
    ordered: list[dict] = []
    for category in CATEGORIES:
        lanes: collections.OrderedDict[str, list[dict]] = collections.OrderedDict()
        for row in rows:
            if row["category"] == category:
                lanes.setdefault(row["seed_id"], []).append(row)
        while lanes:
            for sid in list(lanes):
                ordered.append(lanes[sid].pop(0))
                if not lanes[sid]:
                    del lanes[sid]
    return ordered


# --------------------------------------------------------------------------
# Build / write
# --------------------------------------------------------------------------


def target_of(row: dict) -> str:
    """The one assistant turn this record supervises."""
    assistants = [m for m in row["messages"] if m["role"] == "assistant"]
    return assistants[row["train_on"][-1]]["content"]


def target_repetition_gate(
        records: dict[str, list[dict]]) -> list[str]:
    """No exact final target supervised more than TARGET_REPEAT_LIMIT times."""
    counts: collections.Counter = collections.Counter(
        target_of(row) for side in SPLIT_ORDER
        for row in records.get(side, []))
    problems: list[str] = []
    for text, times in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if times > TARGET_REPEAT_LIMIT:
            problems.append(
                f"target repeated {times} times (limit "
                f"{TARGET_REPEAT_LIMIT}): {text[:60]!r}")
    return problems


def plan() -> tuple[list[dict], dict[str, str], dict]:
    """Resolve bank, grouping and split once. Returns (seeds, split, info)."""
    seeds = [dict(seed) for seed in SB4.SEEDS]
    order, seed_to_group, group_stats = build_groups()
    split, problems, split_stats = assign_split(order)
    problems += split_problems(split, seed_to_group)
    info = {"grouping": group_stats, "split": split_stats,
            "problems": problems, "seed_to_group": seed_to_group,
            "split_map": split,
            "seed_counts": dict(collections.Counter(split.values()))}
    return seeds, split, info


def build() -> tuple[dict[str, list[dict]], list[str], dict]:
    """Generate every record. Nothing touches disk; ``write`` does that."""
    RV4.self_check()
    seeds, split, info = plan()
    problems: list[str] = list(info["problems"])

    system_key, composed, capabilities = build_system_context()
    fingerprint = hashlib.sha256(composed.encode("utf-8")).hexdigest()

    records: dict[str, list[dict]] = {side: [] for side in SPLIT_ORDER}
    for seed in seeds:
        side = split.get(seed["id"])
        if side is None:
            problems.append(f"{seed['id']}: no split assigned")
            continue
        records[side].extend(
            build_seed_records(seed, side, system_key, fingerprint))
    for side in SPLIT_ORDER:
        records[side] = order_records(records[side])

    problems += target_repetition_gate(records)

    info["system_ref"] = system_key
    info["capabilities"] = capabilities
    return records, problems, info


def write(records: dict[str, list[dict]], info: dict,
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
            "seeds": info["seed_counts"]["holdout"],
            "status": "protected - evaluation only",
            "rule": ("Not used during generation, tuning or wording decisions. "
                     "Read only when the holdout is actually being scored."),
            "generator_version": GENERATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
        }, handle, indent=2)

    split = info["split_map"]
    seeds_by_side = {side: sorted(sid for sid, s in split.items() if s == side)
                     for side in SPLIT_ORDER}
    with open(os.path.join(outdir, "build_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "generator_version": GENERATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
            "system_ref": info["system_ref"],
            "capabilities": info["capabilities"],
            "record_counts": written,
            "seed_counts": info["seed_counts"],
            "seeds": seeds_by_side,
            "records_per_seed": RECORDS_PER_SEED,
            "seed_quota": SEED_QUOTA,
            "grouping": info["grouping"],
            "split_moved_groups": info["split"]["groups_moved"],
        }, handle, indent=2)
    return written


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTDIR
    records, problems, info = build()
    if problems:
        print(f"REFUSING TO WRITE: {len(problems)} problem(s)")
        for problem in problems[:30]:
            print(f"  FAIL {problem}")
        return 1
    written = write(records, info, outdir)
    print("=" * 72)
    print("AURORA BRAIN v0.4 - BEHAVIORAL DATASET BUILD")
    print("=" * 72)
    for side in SPLIT_ORDER:
        print(f"  {side:<11}: {written[side]:>5} records  "
              f"({info['seed_counts'][side]} seeds)")
    print(f"  grouping   : {info['grouping']['groups']} groups, "
          f"largest {info['grouping']['largest_group']}, "
          f"{info['grouping']['shared_values']} shared values")
    print(f"  output     : {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
