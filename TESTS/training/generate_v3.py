"""Generate the Aurora Brain v0.3 dataset from the approved 211-seed bank.

Not a trainer. This produces the material and refuses to produce anything it
cannot verify - every record passes ``validate_v3.audit`` before a single byte
is written, so the dataset that lands on disk is the dataset the gates already
approved.

Relationship to ``generate_v2`` -------------------------------------------
The wire format, the system-context builder and the refusal discipline are
v2's, unchanged. What is new, because the v0.3 gates demand it:

*   the split is ``check_bank_v3``'s split, recomputed here;
*   allocation is token-target-forced per design section 12: the record budgets
    are fixed, but per-seed counts are chosen to satisfy the supervised-token
    band while respecting the per-seed bounds [3, min(10, 3×draws)] and the
    global distinct-target gate. ``check_bank_v3`` proves feasibility; this
    module optimises within identical bounds.
*   expansion is planned per category against exact draw capacity, then each
    record *selects* a draw - never repeats a final target across seeds, and
    never more than three times inside one;
*   opening framing is chosen greedily by opening frequency, so
    ``MIN_DISTINCT_OPENINGS`` is reached by construction rather than by audit;
*   the emitted order is built, not incidental: required adjacent pairs come
    first inside their category, and no two consecutive records in a category
    open their supervised span with the same first-person construction.

Run: python TESTS/training/generate_v3.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
from typing import Optional

sys.path.insert(0, r"C:\PRIMORA PROJECT")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import builders_v3 as B3  # noqa: E402
import check_bank_v3 as C3  # noqa: E402
import design_v3 as D3  # noqa: E402
import seed_bank_v3 as SB3  # noqa: E402
from generate_v2 import SCHEMA_VERSION, build_system_context  # noqa: E402

GENERATOR_VERSION = "aurora-brain-v0.3-seedgen-1"
DEFAULT_BASE_MODEL = r"C:\PRIMORA PROJECT\RESOURCES\aurora\v0.1\model"
DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "v3")

SPLIT_ORDER = ("train", "validation", "holdout")

PAYLOAD_ALLOWLIST = {("C15-R01", "count_words")}


# --------------------------------------------------------------------------
# Tokenizer and draw-length cache.
# --------------------------------------------------------------------------


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
        from aurora.config import TrainerConfig
        config = TrainerConfig()
        return AutoTokenizer.from_pretrained(
            DEFAULT_BASE_MODEL, trust_remote_code=True), config
    except Exception as exc:
        raise RuntimeError(f"tokenizer unavailable with DEFAULT_BASE_MODEL={DEFAULT_BASE_MODEL}: {exc}") from exc


def _supervised_tokens(seed: SB3.SeedV3, draw: B3.Draw,
                       tokenizer, config, system_content: str) -> int:
    """Supervised tokens for one draw, using authoritative masking."""
    from aurora.masking import build_mask
    example = B3.build(seed, draw, "", 0)
    messages = [{"role": "system", "content": system_content}] + example.messages
    result = build_mask(
        tokenizer, messages, example.train_on, max_length=config.max_seq_length)
    return result.trained_tokens


def _build_length_cache(seeds: list[SB3.SeedV3],
                        system_content: str) -> dict[tuple[str, int], int]:
    """Cache (seed_id, draw_ordinal) -> supervised token count for all seeds."""
    tokenizer, config = _load_tokenizer()
    cache: dict[tuple[str, int], int] = {}
    for seed in seeds:
        draws = B3.materialise(seed)
        for draw in draws:
            cache[(seed.seed_id, draw.ordinal)] = _supervised_tokens(
                seed, draw, tokenizer, config, system_content)
    return cache


# --------------------------------------------------------------------------
# Planning: split, allocation (token-objective DP), evaluation depths.
# --------------------------------------------------------------------------


def _draw_lens_for_seed(seed: SB3.SeedV3, length_cache: dict) -> list[int]:
    """Return supervised token lengths for each draw of this seed, in ordinal order."""
    draws = B3.materialise(seed)
    return [length_cache[(seed.seed_id, d.ordinal)] for d in draws]


def _seed_cap(seed: SB3.SeedV3) -> int:
    """Per-seed ceiling: min(MAX_EXPANSION, MAX_TARGET_REPEAT * num_draws)."""
    draws = B3.materialise(seed)
    return min(D3.MAX_EXPANSION, D3.MAX_TARGET_REPEAT * len(draws))


def _allocate_dp(train_rows: list[SB3.SeedV3],
                 length_cache: dict[tuple[str, int], int],
                 problems: list[str]) -> dict[str, int]:
    """
    Per-category DP to maximize conversational supervised tokens / minimize
    capability supervised tokens, subject to:
      - exact RECORD_BUDGET per category
      - per-seed n in [3, min(10, 3 * num_draws)]
      - global distinct targets >= 900 (across all categories)
    Priority is only a tie-break (lower PRIORITY_RANK = higher priority).
    """
    by_id = {s.seed_id: s for s in train_rows}
    # Prepare per-category items
    cat_items: dict[str, list[dict]] = {}
    for category, budget in D3.RECORD_BUDGET.items():
        rows = [s for s in train_rows if s.category == category]
        items = []
        for seed in rows:
            draws = B3.materialise(seed)
            if not draws:
                continue
            lens = _draw_lens_for_seed(seed, length_cache)
            cap = _seed_cap(seed)
            items.append({
                "sid": seed.seed_id,
                "lens": sorted(lens, reverse=(category in D3.CONVERSATIONAL_CATEGORIES)),
                "cap": cap,
                "len": len(lens),
                "priority": C3.PRIORITY_RANK.get(seed.priority, 3),
            })
        cat_items[category] = items

    def realize(lens_pref: list[int], n: int, nd: int) -> tuple[int, int] | None:
        """Token sum and distinct-target count that ``select_records`` will realise
        for n records drawn from one seed whose draw lengths are already sorted in
        the preferred direction (desc for conversational, asc for capability).

        The rule selection implements, and the rule the DP scores, are the same:

        * the first ``k = min(n, nd)`` records take k *distinct* draws, in
          preferred order -- this is what keeps the distinct-target count honest;
        * the remaining ``n - k`` records top those same draws up to
          ``MAX_TARGET_REPEAT`` uses each, again in preferred order.

        Returns ``None`` when n cannot be placed within the repeat cap.
        """
        k = min(n, nd)
        if k <= 0:
            return None
        if n > D3.MAX_TARGET_REPEAT * nd:
            return None
        total = 0
        for i in range(k):
            total += lens_pref[i]
        remaining = n - k
        for i in range(k):
            if remaining <= 0:
                break
            # one use already taken above; cap is MAX_TARGET_REPEAT
            add = min(D3.MAX_TARGET_REPEAT - 1, remaining)
            total += add * lens_pref[i]
            remaining -= add
        if remaining > 0:
            return None
        return total, k

    # Per-category DP. State is (records_used, distinct_used) so the category
    # record budget is an exact constraint rather than an accident; the token
    # objective can therefore actually bind.
    # Each state -> (conv_tokens, cap_tokens, prio_sum, choices)
    cat_frontiers = {}
    for category, items in cat_items.items():
        cat_budget = D3.RECORD_BUDGET[category]
        is_conv = category in D3.CONVERSATIONAL_CATEGORIES
        dp = {(0, 0): (0, 0, 0, [])}
        for it in items:
            ndp = {}
            lens_pref = it["lens"]
            cap = it["cap"]
            nd = it["len"]
            rank = it["priority"]
            sid = it["sid"]
            for (r_used, d_used), (conv, cap_tok, prio, choices) in dp.items():
                for n in range(D3.MIN_EXPANSION, cap + 1):
                    r_new = r_used + n
                    if r_new > cat_budget:
                        break
                    got = realize(lens_pref, n, nd)
                    if got is None:
                        continue
                    s, k = got
                    d_new = d_used + k
                    if is_conv:
                        conv_new, cap_new = conv + s, cap_tok
                    else:
                        conv_new, cap_new = conv, cap_tok + s
                    prio_new = prio + n * rank
                    key = (r_new, d_new)
                    cur = ndp.get(key)
                    if (cur is None
                            or conv_new + cap_new > cur[0] + cur[1]
                            or (conv_new + cap_new == cur[0] + cur[1]
                                and prio_new < cur[2])):
                        ndp[key] = (conv_new, cap_new, prio_new, choices + [(sid, n)])
            dp = ndp
        # exact category budget is mandatory; distinct becomes the frontier axis
        cat_frontiers[category] = {
            d: (conv, cap_tok, prio, choices)
            for (r, d), (conv, cap_tok, prio, choices) in dp.items()
            if r == cat_budget
        }

    for category, frontier in cat_frontiers.items():
        if not frontier:
            problems.append(
                f"{category}: no allocation places the exact budget "
                f"{D3.RECORD_BUDGET[category]} within per-seed bounds")
            return _allocate_3pass(train_rows, problems)

    # Global stage. Each category is wholly conversational or wholly capability,
    # so the only coupling left is the global distinct-target floor. Keep the
    # Pareto set on (conv_tokens, cap_tokens) per total-distinct so the token
    # band is a real constraint on the outcome rather than a hoped-for accident.
    # state -> list of (conv, cap_tok, prio, alloc)
    global_dp = {0: [(0, 0, 0, {})]}
    for category in cat_items:
        frontier = cat_frontiers[category]
        ndp = {}
        for d_used, entries in global_dp.items():
            for d_cat, (conv_c, cap_c, prio_c, choices) in frontier.items():
                d_new = d_used + d_cat
                bucket = ndp.setdefault(d_new, [])
                seen = {(e[0], e[1]) for e in bucket}
                for conv_g, cap_g, prio_g, alloc_g in entries:
                    key = (conv_g + conv_c, cap_g + cap_c)
                    if key in seen:
                        continue
                    seen.add(key)
                    new_alloc = alloc_g.copy()
                    for sid, n in choices:
                        new_alloc[sid] = n
                    bucket.append((key[0], key[1], prio_g + prio_c, new_alloc))
                bucket.sort(key=lambda e: (-(e[0] + e[1]), e[2]))
                del bucket[64:]
        global_dp = ndp

    low = D3.CONVERSATIONAL_TOKEN_TARGET - D3.TOKEN_RATIO_TOLERANCE
    high = D3.CONVERSATIONAL_TOKEN_TARGET + D3.TOKEN_RATIO_TOLERANCE
    target = D3.CONVERSATIONAL_TOKEN_TARGET

    def ratio_of(conv: int, cap_tok: int) -> float:
        total = conv + cap_tok
        return conv / total if total else 0.0

    best = None
    best_key = None
    for d, entries in global_dp.items():
        if d < D3.MIN_DISTINCT_FINAL_TARGETS:
            continue
        for conv, cap_tok, prio, alloc in entries:
            r = ratio_of(conv, cap_tok)
            in_band = low <= r <= high
            # prefer in-band, then closeness to target, then more distinct,
            # then lower priority weight
            key = (0 if in_band else 1, abs(r - target), -d, prio)
            if best_key is None or key < best_key:
                best_key, best = key, (d, conv, cap_tok, prio, alloc)

    if best is None:
        problems.append(
            f"no allocation reaches {D3.MIN_DISTINCT_FINAL_TARGETS} distinct targets")
        return _allocate_3pass(train_rows, problems)

    d_best, conv_best, cap_best, _prio_best, best_alloc = best
    ratio_best = ratio_of(conv_best, cap_best)
    if not (low <= ratio_best <= high):
        problems.append(
            f"token-target-forced allocation could not hold the conversational "
            f"share inside [{low:.2f}, {high:.2f}]: best is {ratio_best:.4f}")

    # budgets and bounds are exact by construction; assert rather than assume
    by_category = {cat: 0 for cat in D3.RECORD_BUDGET}
    for sid, n in best_alloc.items():
        by_category[by_id[sid].category] += n
        draws = B3.materialise(by_id[sid])
        if not (D3.MIN_EXPANSION <= n <= _seed_cap(by_id[sid])):
            problems.append(f"{sid}: allocated {n} outside per-seed bounds")
    for category, budget in D3.RECORD_BUDGET.items():
        if by_category[category] != budget:
            problems.append(
                f"{category}: allocated {by_category[category]}, budget {budget}")
    if sum(best_alloc.values()) != D3.TRAIN_SIZE:
        problems.append(
            f"allocation places {sum(best_alloc.values())}, expected {D3.TRAIN_SIZE}")
    if d_best < D3.MIN_DISTINCT_FINAL_TARGETS:
        problems.append(
            f"train distinct final targets {d_best} < {D3.MIN_DISTINCT_FINAL_TARGETS}")
    if problems:
        return _allocate_3pass(train_rows, problems)

    return best_alloc


def _allocate_3pass(train_rows: list[SB3.SeedV3], problems: list[str]) -> dict[str, int]:
    """Fallback: the original 3-pass allocation from check_bank_v3."""
    allocation: dict[str, int] = {}
    by_id = {s.seed_id: s for s in train_rows}
    for category, budget in D3.RECORD_BUDGET.items():
        rows = [s for s in train_rows if s.category == category]
        caps = {s.seed_id: C3.capacity(s) for s in rows}
        spans = {s.seed_id: sum(C3.distinct_and_fills(s)) for s in rows}
        if sum(caps.values()) < budget:
            problems.append(
                f"{category}: train capacity {sum(caps.values())} < budget {budget}")
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
            problems.append(f"{category}: could not place {left} records")

    distinct_total = sum(
        min(records, sum(C3.distinct_and_fills(by_id[sid])))
        for sid, records in allocation.items())
    if distinct_total < D3.MIN_DISTINCT_FINAL_TARGETS:
        problems.append(
            f"train distinct final targets {distinct_total} < {D3.MIN_DISTINCT_FINAL_TARGETS}")
    if sum(allocation.values()) != D3.TRAIN_SIZE:
        problems.append(
            f"train plan places {sum(allocation.values())} records, expected {D3.TRAIN_SIZE}")
    return allocation


def allocate(train_rows: list[SB3.SeedV3],
             problems: list[str]) -> dict[str, int]:
    """Per-seed record counts for the train split using token-objective DP.

    The DP maximises conversational supervised tokens and minimises capability
    supervised tokens (design: "Holding the token target is what forces this
    allocation"), while respecting the per-seed bounds [3, min(10, 3×draws)],
    exact category budgets, and global distinct-target >= 900. Priority is
    only a tie-break.
    """
    # This function is kept for backward compatibility; the real work
    # happens in _allocate_dp which is called from plan() after length cache.
    return _allocate_3pass(train_rows, problems)


def evaluation_depths(split_of: dict[str, str],
                      seeds: list[SB3.SeedV3]) -> dict[str, int]:
    """Fixed depth for every evaluation seed: eight records each, with the
    validation total of 180 reached by giving the first four validation seeds
    (priority rank, then id) a ninth record."""
    val_rows = sorted(
        (s for s in seeds if split_of.get(s.seed_id) == "validation"),
        key=lambda s: (C3.PRIORITY_RANK.get(s.priority, 3), s.seed_id))
    extra = {s.seed_id for s in val_rows[:4]}
    depths: dict[str, int] = {}
    for seed in seeds:
        split = split_of.get(seed.seed_id)
        if split == "holdout":
            depths[seed.seed_id] = C3.EVAL_DEPTH
        elif split == "validation":
            depths[seed.seed_id] = C3.EVAL_DEPTH + (
                1 if seed.seed_id in extra else 0)
    return depths


# --------------------------------------------------------------------------
# Selection: one draw and one opening per record.
# --------------------------------------------------------------------------


def select_records(seeds: list[SB3.SeedV3],
                   split_of: dict[str, str],
                   allocation: dict[str, int],
                   depths: dict[str, int],
                   system_key: str,
                   fingerprint: str,
                   length_cache: dict[tuple[str, int], int],
                   problems: list[str]) -> list[B3.Example]:
    draws_cache = {
        s.seed_id: B3.materialise(s) for s in seeds if s.variants}
    planned: dict[str, int] = {}
    for seed in seeds:
        split = split_of.get(seed.seed_id)
        if split == "train":
            planned[seed.seed_id] = allocation.get(seed.seed_id, 0)
        elif split:
            planned[seed.seed_id] = depths.get(seed.seed_id, 0)
    category_totals: dict[tuple[str, str], int] = {}
    for seed in seeds:
        split = split_of.get(seed.seed_id)
        if split:
            key = (split, seed.category)
            category_totals[key] = (
                category_totals.get(key, 0) + planned.get(seed.seed_id, 0))

    usage: dict[tuple[str, str], int] = {}
    owner_main: dict[str, str] = {}
    owner_hold: dict[str, str] = {}
    openings_main: collections.Counter = collections.Counter()
    openings_hold: collections.Counter = collections.Counter()
    chosen_cons: dict[tuple[str, str], collections.Counter] = (
        collections.defaultdict(collections.Counter))
    cons_cache: dict[tuple[str, str], str | None] = {}
    examples: list[B3.Example] = []

    for split in SPLIT_ORDER:
        rows = [s for s in seeds if split_of.get(s.seed_id) == split]
        rows.sort(key=lambda s: (C3.PRIORITY_RANK.get(s.priority, 3),
                                 s.seed_id))
        shared = split in ("train", "validation")
        owner = owner_main if shared else owner_hold
        openings = openings_main if shared else openings_hold
        for seed in rows:
            draws = draws_cache.get(seed.seed_id, [])
            planned_count = planned.get(seed.seed_id, 0)
            if planned_count <= 0:
                problems.append(
                    f"{seed.seed_id}: no planned record count for {split}")
                continue
            ceiling = min(D3.MAX_EXPANSION,
                          D3.MAX_TARGET_REPEAT * len(draws)) if draws else 0
            if planned_count > ceiling:
                problems.append(
                    f"{seed.seed_id}: plan asks for {planned_count} records "
                    f"but material reaches {ceiling}")
                continue
            constructions = [
                cons_cache.setdefault(
                    (seed.seed_id, draw.ordinal),
                    B3.draw_construction(seed, draw))
                for draw in draws]
            # Draw lengths for this seed (for length rank in selection)
            draw_lens = {draw.ordinal: length_cache.get((seed.seed_id, draw.ordinal), 0)
                         for draw in draws}
            bound = (category_totals.get((split, seed.category), 0) + 1) // 2
            bucket = chosen_cons[(split, seed.category)]
            is_conv = seed.category in D3.CONVERSATIONAL_CATEGORIES
            # Pin the failure draw for C15-R01 if in allowlist
            pinned_draw = None
            for draw in draws:
                if (seed.seed_id, draw.key) in PAYLOAD_ALLOWLIST:
                    pinned_draw = draw
                    break

            distinct_here = 0
            for record_index in range(planned_count):
                # For record 0, if there's a pinned draw, force it
                if record_index == 0 and pinned_draw is not None:
                    draw = pinned_draw
                    cons = next(c for d, c in zip(draws, constructions) if d is draw)
                else:
                    # Realise the same rule _allocate_dp scored: spend the first
                    # k = min(planned_count, len(draws)) records on k *distinct*
                    # draws in preferred order, then top those draws up to
                    # MAX_TARGET_REPEAT. Spreading first is what preserves the
                    # distinct-target count; preferring length is what moves the
                    # supervised-token share.
                    spread_left = max(0, min(planned_count, len(draws)) - distinct_here)
                    best = None
                    for draw, cons in zip(draws, constructions):
                        used = usage.get((seed.seed_id, draw.key), 0)
                        if used >= D3.MAX_TARGET_REPEAT:
                            continue
                        holder = owner.get(draw.key)
                        if holder is not None and holder != seed.seed_id:
                            continue
                        picked = bucket[cons] if cons is not None else 0
                        violates = (
                            1 if cons is not None and picked + 1 > bound else 0)
                        penalty = picked if cons is not None else 0
                        length = draw_lens.get(draw.ordinal, 0)
                        length_rank = -length if is_conv else length
                        # prefer a still-unused draw while we still owe spread
                        spread_rank = 0 if (used == 0 and spread_left > 0) else 1
                        rank = (violates, spread_rank, length_rank, penalty, draw.ordinal)
                        if best is None or rank < best[0]:
                            best = (rank, draw, cons)
                    if best is None:
                        problems.append(
                            f"{seed.seed_id}: no drawable final target left at "
                            f"record {record_index}")
                        break
                    draw = best[1]
                    cons = best[2]
                state = usage.get((seed.seed_id, draw.key), 0)
                usage[(seed.seed_id, draw.key)] = state + 1
                if state == 0:
                    distinct_here += 1
                owner.setdefault(draw.key, seed.seed_id)
                bucket[cons] += 1
                options = B3.prefix_options(seed, draw)
                pick = min(range(len(options)),
                           key=lambda i: (openings[options[i][1]], i))
                prefix, opening = options[pick]
                openings[opening] += 1
                example = B3.build(seed, draw, prefix, record_index)
                example.split = split
                example.system_ref = system_key
                example.system_fingerprint = fingerprint
                examples.append(example)
    return examples


# --------------------------------------------------------------------------
# Emission order: pair blocks, then a construction-safe greedy sequence.
# --------------------------------------------------------------------------


def _compatible(cons: str | None, last: str | None) -> bool:
    return cons is None or last is None or cons != last


def _pick_index(items: list[tuple[B3.Example, str | None]],
                counts: collections.Counter,
                last: str | None) -> Optional[int]:
    best: Optional[tuple[tuple, int]] = None
    for index, (_, cons) in enumerate(items):
        if not _compatible(cons, last):
            continue
        if cons is None:
            rank = (1, 0)
        else:
            rank = (0, -counts.get(cons, 0))
        key = (rank, items[index][0].id)
        if best is None or key < best[0]:
            best = (key, index)
    return None if best is None else best[1]


def _greedy_sequence(
    items: list[tuple[B3.Example, str | None]],
    counts: collections.Counter,
    last: str | None,
) -> Optional[list[tuple[B3.Example, str | None]]]:
    remaining = list(items)
    ordered: list[tuple[B3.Example, str | None]] = []
    while remaining:
        index = _pick_index(remaining, counts, last)
        if index is None:
            return None
        item = remaining.pop(index)
        counts[item[1]] -= 1
        ordered.append(item)
        last = item[1]
    return ordered


def _pair_block(
    side_a: list[tuple[B3.Example, str | None]],
    side_b: list[tuple[B3.Example, str | None]],
    counts: collections.Counter,
    last: str | None,
) -> Optional[list[tuple[B3.Example, str | None]]]:
    left_a, left_b = list(side_a), list(side_b)
    ordered: list[tuple[B3.Example, str | None]] = []
    last_side: Optional[str] = None
    while left_a or left_b:
        prefer_a = last_side != "a"
        chosen = None
        for use_a in ((True, False) if prefer_a else (False, True)):
            side = left_a if use_a else left_b
            if not side:
                continue
            index = _pick_index(side, counts, last)
            if index is not None:
                chosen = (use_a, index)
                break
        if chosen is None:
            return None
        use_a, index = chosen
        item = (left_a if use_a else left_b).pop(index)
        counts[item[1]] -= 1
        ordered.append(item)
        last = item[1]
        last_side = "a" if use_a else "b"
    return ordered


def order_examples(
    examples: list[B3.Example],
) -> tuple[list[B3.Example], list[str]]:
    """Emit categories in budget order; inside a category the required pairs
    first (adjacent, alternating where the constructions allow), then the rest
    as a sequence that never repeats a first-person construction across two
    consecutive records of that category."""
    problems: list[str] = []
    by_category: dict[str, list[B3.Example]] = {}
    for example in examples:
        by_category.setdefault(example.category, []).append(example)

    ordered: list[B3.Example] = []
    for category in D3.RECORD_BUDGET:
        group = by_category.get(category, [])
        if not group:
            continue
        items = [(ex, B3.construction(ex.messages, ex.train_on))
                 for ex in group]
        counts: collections.Counter = collections.Counter(
            cons for _, cons in items)
        placed: list[tuple[B3.Example, str | None]] = []
        rest = list(items)
        for pair in D3.REQUIRED_ADJACENT_PAIRS:
            first = [it for it in rest if it[0].seed_id == pair[0]]
            second = [it for it in rest if it[0].seed_id == pair[1]]
            if not first or not second:
                continue
            rest = [it for it in rest
                    if it[0].seed_id not in (pair[0], pair[1])]
            last = placed[-1][1] if placed else None
            block = _pair_block(first, second, counts, last)
            if block is None:
                problems.append(
                    f"{category}: pair {pair[0]}/{pair[1]} cannot be placed "
                    f"without repeating a first-person construction")
                block = first + second
            placed.extend(block)
        last = placed[-1][1] if placed else None
        tail = _greedy_sequence(rest, counts, last)
        if tail is None:
            problems.append(
                f"{category}: remaining records cannot be ordered without "
                f"repeating a first-person construction")
            tail = list(rest)
        placed.extend(tail)
        ordered.extend(example for example, _ in placed)
    return ordered, problems


# --------------------------------------------------------------------------
# Record shape and writing.
# --------------------------------------------------------------------------


def record(example: B3.Example) -> dict:
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
        "system_ref": example.system_ref,
        "system_fingerprint": example.system_fingerprint,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
    }


def write_datasets(records: dict[str, list[dict]], outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    written: dict[str, int] = {}
    for split in SPLIT_ORDER:
        rows = records[split]
        path = os.path.join(outdir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[split] = len(rows)
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
    return written


def plan(problems: list[str]) -> tuple[
        list[SB3.SeedV3], dict[str, str], dict[str, int], dict[str, int],
        dict[tuple[str, int], int]]:
    retained = SB3.build_retained(SB3.build_extras())
    seeds = retained + C3.new_seeds()
    split_of = C3.split_of(seeds, problems)
    if problems:
        return seeds, split_of, {}, {}, {}
    train_rows = [s for s in seeds if split_of.get(s.seed_id) == "train"]

    # Build length cache for all seeds (train + val + holdout) before allocation
    # Need system content for masking; use placeholder since system_ref is same
    system_content = "System prompt placeholder"
    length_cache = _build_length_cache(seeds, system_content)

    allocation = _allocate_dp(train_rows, length_cache, problems)
    depths = evaluation_depths(split_of, seeds)
    return seeds, split_of, allocation, depths, length_cache


def main() -> int:
    from PRIMORA.prime.context.isolation import production_digest
    import validate_v3

    before = production_digest()
    problems: list[str] = []

    system_key, composed, capabilities = build_system_context()
    fingerprint = hashlib.sha256(composed.encode("utf-8")).hexdigest()

    seeds, split_of, allocation, depths, length_cache = plan(problems)

    examples: list[B3.Example] = []
    if not problems:
        examples = select_records(
            seeds, split_of, allocation, depths,
            system_key, fingerprint, length_cache, problems)

    records: dict[str, list[dict]] = {split: [] for split in SPLIT_ORDER}
    if not problems:
        by_split = {
            split: [ex for ex in examples if ex.split == split]
            for split in SPLIT_ORDER}
        for split in SPLIT_ORDER:
            ordered, order_problems = order_examples(by_split[split])
            problems.extend(order_problems)
            records[split] = [record(ex) for ex in ordered]

    audit: dict = {"problems": [], "report": {}}
    if not problems:
        audit = validate_v3.audit(records)
        problems.extend(audit["problems"])

    report = audit.get("report", {})
    print("=" * 72)
    print("AURORA BRAIN v0.3 - DATASET GENERATION (seed bank v3)")
    print("=" * 72)
    print(f"  seeds in bank        : {len(seeds)}")
    print(f"  system reference key : {system_key}")
    print(f"  capabilities         : {', '.join(capabilities)}")
    print(f"  planned train records: {sum(allocation.values())}")
    print(f"  train / val / holdout: "
          f"{len(records['train'])} / {len(records['validation'])} / "
          f"{len(records['holdout'])}")
    for key in ("distinct_train_targets", "distinct_openings_train",
                "opening_max_train", "token_ratio_conversational"):
        if key in report:
            print(f"  {key:<21}: {report[key]}")

    if problems:
        print(f"  REFUSING TO WRITE: {len(problems)} problems")
        for problem in problems[:20]:
            print(f"      {problem}")
    else:
        outdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTDIR
        written = write_datasets(records, outdir)
        print(f"  written              : {written}")
        print(f"  output directory     : {outdir}")

    print(f"  production memory unchanged: {production_digest() == before}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
