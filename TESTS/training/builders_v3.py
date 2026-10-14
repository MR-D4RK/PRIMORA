"""Aurora Brain v0.3 - deterministic record builders over the 211-seed bank.

Read-only with respect to v2: the material comes from ``seed_bank_v3`` and the
wire-format helpers (``request_block``, ``tool_turn``, ``_tool_payload``) are
the very functions v2 already ships, so a v3 record is byte-parseable by the
same runtime parser. Nothing here edits a v2 or v3 bank table.

What this module adds is the deterministic draw model the v0.3 gates need. A
seed's variants are enumerated once into *draws* - one per distinct filled
final target - so a record is a draw plus a framing prefix. Expansion can then
be planned against exact distinct-target capacity before anything is emitted,
which is what ``MIN_DISTINCT_FINAL_TARGETS`` and ``MAX_TARGET_REPEAT`` measure.

Slots fill per draw from the shared banks; slots inside the final target are
the draw's axis (they are what makes two draws distinct), slots that only
appear in the ask, payload or tool text are fixed per variant so payload,
result and answer can never disagree about what was drawn.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import audit_v2 as A2
import builders_v2 as B2
import seed_bank_v3 as SB3

SLOT_RE = re.compile(r"\{(\w+)\}")

# Framing prefixes for the first user turn of a non-prior record. The empty
# entry is deliberate: the plain ask is a situation of its own, and a bank
# without it would force a wrapper onto every record in the corpus.
FRAMES: tuple[str, ...] = (
    "",
    "Quick: ",
    "Right - ",
    "So: ",
    "Now, ",
    "Question: ",
    "Sorry - ",
    "Honestly: ",
    "Straight up: ",
    "First: ",
    "One thing: ",
    "Genuine question: ",
)

# Opening leads for prior-context records. These seeds open on an authored
# memory blob, where "Quick: " would read as the user performing; a lead is
# the speaker picking the thread back up instead.
PRIOR_LEADS: tuple[str, ...] = (
    "",
    "Right - ",
    "So - ",
    "Anyway, ",
    "Now - ",
    "Picking this up: ",
    "Back to it: ",
    "Where we were: ",
    "To continue: ",
    "Carrying on: ",
)


@dataclass
class Example:
    """One record under construction. Field-for-field the v2 record shape."""

    id: str
    seed_id: str
    category: str
    split: str
    messages: list[dict[str, str]] = field(default_factory=list)
    train_on: list[int] = field(default_factory=list)
    expect_block: bool = False
    expect_action: Optional[str] = None
    note: str = ""
    variant: int = 0
    system_ref: str = ""
    system_fingerprint: str = ""
    seed_priority: str = "normal"

    @property
    def assistant_turn_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "assistant")

    def trained_assistant_indexes(self) -> list[int]:
        return [i for i in self.train_on if i < self.assistant_turn_count]


@dataclass(frozen=True)
class Draw:
    """One distinct final target of one variant, with its slot assignment."""

    variant_index: int
    slots: tuple[tuple[str, str], ...]
    key: str
    ordinal: int


def fill(text: str, slots: dict[str, str]) -> str:
    """Substitute slot placeholders. Unknown braces are left untouched, so a
    payload carrying literal JSON braces cannot be mangled the way str.format
    would mangle it."""
    return SLOT_RE.sub(lambda m: slots.get(m.group(1), m.group(0)), text)


def fill_payload(payload: Any, slots: dict[str, str]) -> Any:
    if isinstance(payload, dict):
        return {k: fill_payload(v, slots) for k, v in payload.items()}
    if isinstance(payload, list):
        return [fill_payload(v, slots) for v in payload]
    if isinstance(payload, str):
        return fill(payload, slots)
    return payload


def payload_names(payload: Any) -> set[str]:
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, str):
            found.update(SLOT_RE.findall(node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found


def _target_names(variant: SB3.Variant) -> set[str]:
    return set(SLOT_RE.findall(variant.target))


def _needed_names(variant: SB3.Variant) -> set[str]:
    found: set[str] = set()
    for turn in variant.turns:
        if turn[0] in ("U", "A", "T"):
            found.update(SLOT_RE.findall(turn[1]))
        elif turn[0] in ("R", "X"):
            found.update(payload_names(turn[2]))
    return found


def _axis_pool(name: str) -> tuple[str, ...]:
    if name == "item":
        return tuple(B2.ITEMS)
    if name == "value":
        return SB3.ALL_VALUES
    if name == "day":
        return tuple(B2.DAYS)
    if name == "note":
        return SB3.NOTES
    return tuple(B2.QUERY.values()) or ("reference",)


def _extra_value(name: str, seed: SB3.SeedV3, variant_index: int) -> str:
    if name == "query":
        authored = B2.QUERY.get(seed.seed_id)
        if authored:
            return authored
        pool = tuple(B2.QUERY.values()) or ("reference",)
    elif name == "item":
        pool = tuple(B2.ITEMS)
    elif name == "value":
        pool = SB3.ALL_VALUES
    elif name == "day":
        pool = tuple(B2.DAYS)
    else:
        pool = SB3.NOTES
    return pool[variant_index % len(pool)]


def materialise(seed: SB3.SeedV3) -> list[Draw]:
    """Every distinct final target this seed can emit, in round-robin order.

    The order walks the variants together - first combo of every variant,
    then second - so the earliest draws already span the seed's authored
    replies. A filled target normalising to the same text as an earlier draw
    contributes nothing new and is skipped, which keeps the draw count equal
    to ``check_bank_v3.distinct_and_fills`` exactly.
    """
    per_variant: list[list[dict[str, str]]] = []
    for index, variant in enumerate(seed.variants):
        axis = [name for name in SB3.SLOT_NAMES if name in _target_names(variant)]
        extras = sorted(_needed_names(variant) - set(axis))
        combos: list[dict[str, str]] = []
        for values in itertools.product(*[_axis_pool(name) for name in axis]):
            slots = dict(zip(axis, values))
            for name in extras:
                slots[name] = _extra_value(name, seed, index)
            combos.append(slots)
        per_variant.append(combos)

    draws: list[Draw] = []
    seen: set[str] = set()
    rung = 0
    progressed = True
    while progressed:
        progressed = False
        for index, combos in enumerate(per_variant):
            if rung >= len(combos):
                continue
            slots = combos[rung]
            key = A2.normalise(fill(seed.variants[index].target, slots))
            if key and key not in seen:
                seen.add(key)
                ordered = tuple(
                    (name, slots[name])
                    for name in SB3.SLOT_NAMES if name in slots)
                draws.append(Draw(index, ordered, key, len(draws)))
                progressed = True
        rung += 1
    return draws


def is_prior(seed: SB3.SeedV3, variant: SB3.Variant) -> bool:
    first = variant.turns[0] if variant.turns else None
    if not seed.multi_turn or not first or first[0] != "U":
        return False
    return SB3._is_prior_context(first[1])


def first_user_content(seed: SB3.SeedV3, draw: Draw) -> str:
    slots = dict(draw.slots)
    variant = seed.variants[draw.variant_index]
    for turn in variant.turns:
        if turn[0] == "U":
            return fill(turn[1], slots)
    return ""


def prefix_options(seed: SB3.SeedV3, draw: Draw) -> list[tuple[str, str]]:
    """(prefix, opening key) for every framing this record may take, so the
    generator can choose by opening frequency without rebuilding the text."""
    variant = seed.variants[draw.variant_index]
    bank = PRIOR_LEADS if is_prior(seed, variant) else FRAMES
    base = first_user_content(seed, draw)
    return [(prefix, opening_key(prefix + base)) for prefix in bank]


def opening_key(text: str) -> str:
    return A2.normalise(text)[:18]


def resolve_train_on(
    seed: SB3.SeedV3,
    variant: SB3.Variant,
    assistant_count: int,
    failure_preceding: set[int],
) -> list[int]:
    kind = seed.kind
    if kind == "prose":
        base = [assistant_count - 1] if assistant_count else []
    elif kind == "emit":
        base = [0] if assistant_count else []
    else:
        declared = variant.train_on
        if declared is None:
            declared = seed.train_on
        if declared is None:
            base = list(range(assistant_count))
        else:
            base = [i for i in declared if 0 <= i < assistant_count]
    resolved = sorted(set(base) - failure_preceding)
    if not resolved:
        raise ValueError(
            f"{seed.seed_id}: loss mask is empty after removing the request "
            f"before a failed tool turn")
    if max(resolved) != assistant_count - 1:
        raise ValueError(
            f"{seed.seed_id}: loss stops at assistant turn {max(resolved)} "
            f"but the record ends on {assistant_count - 1}")
    return resolved


def build(
    seed: SB3.SeedV3,
    draw: Draw,
    prefix: str,
    record_index: int,
) -> Example:
    """One record: a draw's turns, framed by ``prefix``, loss-resolved."""
    if not seed.training_allowed or not seed.variants:
        raise ValueError(f"{seed.seed_id} must not be emitted")
    variant = seed.variants[draw.variant_index]
    slots = dict(draw.slots)

    messages: list[dict[str, str]] = []
    failure_preceding: set[int] = set()
    assistant_count = 0
    first_user = True

    for turn in variant.turns:
        if turn[0] == "U":
            text = fill(turn[1], slots)
            if first_user:
                text = prefix + text
                first_user = False
            messages.append({"role": "user", "content": text})
        elif turn[0] == "A":
            messages.append(
                {"role": "assistant", "content": fill(turn[1], slots)})
            assistant_count += 1
        elif turn[0] in ("R", "X"):
            payload = fill_payload(turn[2], slots)
            messages.append(
                {"role": "assistant",
                 "content": B2.request_block(turn[1], payload)})
            assistant_count += 1
        elif turn[0] == "T":
            status, output, error = B2._tool_payload(fill(turn[1], slots))
            messages.append(
                {"role": "tool", "content": B2.tool_turn(status, output, error)})
            if status == "failure" and assistant_count:
                failure_preceding.add(assistant_count - 1)

    train_on = resolve_train_on(
        seed, variant, assistant_count, failure_preceding)

    first_request = next(
        (t for t in variant.turns if t[0] in ("R", "X")), None)
    if seed.kind == "prose":
        expect_block, expect_action = False, None
    elif seed.kind == "cycle":
        expect_block, expect_action = True, None
    else:
        expect_block = True
        expect_action = first_request[1] if first_request else None

    return Example(
        id=f"{seed.seed_id}-{record_index:02d}",
        seed_id=seed.seed_id,
        category=seed.category,
        split="train",
        messages=messages,
        train_on=train_on,
        expect_block=expect_block,
        expect_action=expect_action,
        note=seed.scenario,
        variant=draw.variant_index,
        seed_priority=seed.priority,
    )


def construction(messages: list[dict[str, str]],
                 train_on: list[int]) -> Optional[str]:
    """Opening word of the supervised span, or None where the span opens on a
    request block, a number, or punctuation and the gate does not apply."""
    assistants = [m for m in messages if m.get("role") == "assistant"]
    if not train_on or min(train_on) >= len(assistants):
        return None
    content = assistants[min(train_on)].get("content", "")
    if "<prime_request>" in content:
        return None
    match = re.match(r"[A-Za-z']+", content.strip())
    if not match:
        return None
    return match.group(0).lower()


def draw_construction(seed: SB3.SeedV3, draw: Draw) -> Optional[str]:
    """Construction of the record this draw would produce, independent of the
    framing prefix. Selection uses it to keep a category's construction
    multiset inside the bound the emission order requires."""
    example = build(seed, draw, "", 0)
    return construction(example.messages, example.train_on)
