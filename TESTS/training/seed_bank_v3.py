"""Aurora Brain v0.3 - the 211-seed bank.

Read-only with respect to v2. Retained seeds are pulled straight out of
``design_v2.SEEDS`` and their authored material out of ``builders_v2``; nothing
here rewrites a v2 constant. What this module adds is the thing v0.2 lacked -
a per-seed bank of *variants*, because v0.2 gave most seeds two authored
replies and then shipped 967 records built from 287 distinct final targets.

Model
-----
A seed owns a tuple of variants. A variant is an explicit turn list, so a
multi-turn record is written out rather than assembled by a splitter:

    ("U", text)                  user turn
    ("A", text)                  assistant turn, carries loss
    ("R", action, payload)       assistant capability request block, loss
    ("X", action, payload)       assistant capability request block, masked
    ("T", text)                  tool result

Slots - ``{query}``, ``{item}``, ``{value}``, ``{day}``, ``{note}`` - are
filled per record from the variant's own overrides or from the shared bank, so
a single slotted template can yield many distinct final targets without
inventing new wording. A variant whose final assistant turn contains no slot is
only as varied as the number of variants authored for it, which is why every
seed is checked for target capacity before generation runs.

Supersession
------------
Fifty-three of the 75 ``design_v3.NEW_IDS`` also name seeds in
``design_v2.SEEDS`` - unapproved v2 seeds that never reached the export. The
ids are kept because the approved design fixed the numbering (C11 18, C9 16,
C10 20, C12 18, C13 18, C14 16) and ``REQUIRED_ADJACENT_PAIRS`` and
``RECONCILED_IDS`` are written against it. This bank is what those ids mean
from v0.3 onward: ``build_retained`` filters on ``RETAINED_IDS`` and the new
seeds are read from ``new_seeds_v3a``/``new_seeds_v3b``, so no superseded v2
material can reach the package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import design_v2 as D2
import design_v3 as D3
import builders_v2 as B2
import extras_v3 as X3

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

SLOT_NAMES = ("query", "item", "value", "day", "note")


def request_block(action: str, payload: dict) -> str:
    """Copy of ``generate_v2.request_block``, reproduced byte-for-byte.

    A local copy rather than an import because importing generate_v2 pulls in
    the runtime harness; this module must stay importable on its own. The
    string has to match the generator's exactly or a rendered request block
    would stop being parseable by ``extract_prime_request``.
    """
    return (
        "<prime_request>\n"
        + json.dumps(
            {
                "action": action,
                "payload": payload,
                "context": None,
                "metadata": None,
            },
            ensure_ascii=False,
        )
        + "\n</prime_request>"
    )


@dataclass(frozen=True)
class Variant:
    """One authored (or template) conversation ending in a trained turn."""

    turns: tuple[tuple, ...]
    # Optional per-variant slot overrides. None means "draw from the bank",
    # drawn by variant index so a rerun reproduces byte-identical output.
    query: str | None = None
    item: str | None = None
    value: str | None = None
    day: str | None = None
    # Explicit loss mask over assistant turns. None defers to the seed.
    train_on: tuple[int, ...] | None = None

    @classmethod
    def ask_answer(cls, ask: str, target: str, **kw: Any) -> "Variant":
        return cls(turns=(("U", ask), ("A", target)), **kw)

    @property
    def ask(self) -> str:
        for turn in self.turns:
            if turn[0] == "U":
                return turn[1]
        return ""

    @property
    def target(self) -> str:
        """The record's final target: content of the last assistant turn.

        Mirrors ``audit_v2.final_assistant``. For a request block that is the
        rendered ``<prime_request>`` string, so a cycle seed whose last turn is
        an action contributes that block rather than an empty string.
        """
        for turn in reversed(self.turns):
            if turn[0] == "A":
                return turn[1]
            if turn[0] in ("R", "X"):
                return request_block(turn[1], turn[2])
        return ""


@dataclass(frozen=True)
class SeedV3:
    seed_id: str
    category: str
    kind: str                      # prose | emit | tool | cycle | none
    scenario: str
    ask: str
    behavior: str
    target_shape: str
    why: str
    multi_turn: bool
    priority: str
    domain: str
    register: str
    provenance: str                # retained | new | reconciled
    training_allowed: bool = True
    train_on: tuple[int, ...] | None = None
    variants: tuple[Variant, ...] = ()

    @property
    def band(self) -> str:
        if self.category in D3.CONVERSATIONAL_CATEGORIES:
            return "conversational"
        if self.category in D3.CAPABILITY_CATEGORIES:
            return "capability"
        raise ValueError(self.category)

    @property
    def slots(self) -> set[str]:
        """Slot names that actually appear in this seed's authored material."""
        found: set[str] = set()

        def scan(text: str) -> None:
            for name in SLOT_NAMES:
                if "{" + name + "}" in text:
                    found.add(name)

        for variant in self.variants:
            for turn in variant.turns:
                if turn[0] in ("U", "A", "T"):
                    scan(turn[1])
                elif turn[0] in ("R", "X"):
                    scan(json.dumps(turn[-1], sort_keys=True))
        return found


# --------------------------------------------------------------------------
# Slots. Same banks as v2 so retained specs keep their authored shape.
# --------------------------------------------------------------------------

ITEMS = B2.ITEMS
VALUES = B2.VALUES
DAYS = B2.DAYS

# Extra concrete particulars for v3. Kept mundane and factual so no example can
# teach a falsehood by swapping one in.
EXTRA_VALUES = (
    "12.40", "4,850", "0.40", "twelve", "18 April", "71%", "2,300",
    "1:40", "twenty-six", "9.05", "16%", "3,900", "115 km", "7.6%",
)
ALL_VALUES: tuple[str, ...] = tuple(VALUES) + EXTRA_VALUES


def bank_values() -> tuple[str, ...]:
    return ALL_VALUES


# Notes for the one approved ask that carries an unfilled ``<text>`` hole
# (C3-P01: "How many words are in this note: <text>?"). The hole is filled at
# generation time and the same string is passed as the word_count argument, so
# the call actually matches the question. None of these equal the literal
# "the survey note" used by the other two word_count seeds.
NOTES: tuple[str, ...] = (
    "the delivery note from Tuesday",
    "the two-line note on the clipboard",
    "the note Amir left about the gate",
    "the handover note in the folder",
    "the maintenance note for the van",
    "the note about the amended dates",
    "the packing note for the site kit",
    "the unsigned note from the surveyor",
)


def bank_notes() -> tuple[str, ...]:
    return NOTES


# --------------------------------------------------------------------------
# Retained-material corrections.
#
# v2 shipped several pairs of seeds whose *final target* was byte-identical -
# C14-R03/R04 both closed with "Cleared - 64 duplicates out.", C10-N03 and
# C11-I01 both closed with "No.", and three tool seeds all closed with
# "{value}." so any shared value made them collide on the record. v0.3's
# approved gates forbid a verbatim target across seeds, so the offending
# material is corrected here rather than in builders_v2, which stays untouched.
#
# Each table replaces the listed piece for that seed only; everything else in
# the v2 spec passes through unchanged.
# --------------------------------------------------------------------------

# Final assistant turn, replacing B2.PROSE_TARGETS[sid] wholesale.
PROSE_TARGET_OVERRIDES: dict[str, tuple[str, ...]] = {
    # v2: ("Paid.", "No.") - "No." was also C11-I01's first reply.
    "C10-N03": ("Paid.", "It didn't."),
    # v2 first reply was C14-R03's first reply, verbatim.
    "C14-R04": ("That's 64 duplicates cleared.", "That's the duplicates gone."),
    # Reclassified from tool to prose by KIND_OVERRIDES, so its targets are
    # supplied whole rather than found in B2.PROSE_TARGETS. v2's calculate
    # round answered "360." to "What's 12 times 12?".
    "C5-A02": (
        "144.",
        "144 - no calculator needed.",
        "Twelve times twelve is 144.",
        "144, and I didn't need the calculator for it.",
        "That's 144.",
    ),
}

# The specialist name Aurora borrows. 'reasoning' is the name the catalogue
# documents and the one the capability tests invoke, so a trained request
# naming it resolves rather than coming back as an unknown specialist.
SPECIALIST = "reasoning"

# Emit payload, replacing the payload built in _retained_base for that seed.
EMIT_PAYLOAD_OVERRIDES: dict[str, dict] = {
    # v2: both C1-A01 and C1-A02 requested "recent timetable changes".
    "C1-A02": {"query": "the timetable change you mentioned"},
    # v2: both C2-D02 and C2-D05 recalled "the berth code we used".
    "C2-D05": {"query": "the berth code and the current release"},
    # v2: C1-A10, C3-P01 and C6-X05 all counted "the survey note".
    "C6-X05": {"name": "word_count",
               "arguments": {"text": "the weights in the table"}},
    # C3-P01's approved ask carries an unfilled <text> hole; see NOTES.
    "C3-P01": {"name": "word_count", "arguments": {"text": "{note}"}},
    # v2 requested 15% of 2400 for an ask of "What's 17% of 4,200?".
    "C3-P04": {"expression": "17% of 4200"},
    # consult_specialist takes a specialist and a task. v2 sent a 'query', which
    # the capability rejects outright - it fails on the missing specialist
    # before it reads anything else.
    "C1-A06": {"specialist": SPECIALIST,
               "task": "Is the van insurance worth what it costs"},
    "C3-P02": {"specialist": SPECIALIST,
               "task": "Advise whether to push back on the pricing change"},
    "C3-P03": {"specialist": SPECIALIST,
               "task": "Advise whether the old pump should be replaced"},
}

# Request payload of a tool or cycle seed, replacing the payload B2 authored.
# Same contract as EMIT_PAYLOAD_OVERRIDES: the turn list is untouched, only
# the payload carried by the R turn.
PAYLOAD_OVERRIDES: dict[str, dict] = {
    # v2 sent a task with no specialist, which the capability refuses before it
    # reads the task: "Missing or invalid 'specialist' in payload".
    "C5-A14": {"specialist": SPECIALIST, "task": "Is {item} worth the cost"},
    "C6-X04": {"specialist": SPECIALIST, "task": "Weigh splitting {item}"},
    "C6-X06": {"specialist": SPECIALIST,
               "task": "Choose between two suppliers"},
    "C6-X07": {"specialist": SPECIALIST,
               "task": "How do these two constraints interact"},
    "C6-X08": {"specialist": SPECIALIST, "task": "Is this approach sound"},
}

# Final assistant turn of a tool seed, replacing B2.TOOL_SPEC's answer_t.
TOOL_ANSWER_OVERRIDES: dict[str, str] = {
    # v2: C5-A08, C7-M03 and C7-M05 all answered "{value}.". C5-A08 and
    # C7-M05 are handled below by TURNS_OVERRIDES, which replaces them whole.
    "C7-M03": "{value}, as agreed.",
}

# Retained tool/cycle seeds whose v2 turn list answered a *different question*
# from the approved ask in seed_review_export.txt, or whose payload disagreed
# with it. Each entry replaces the turns after the user message wholesale. The
# approved ask is never touched - it comes from design_v2 either way - so what
# is corrected here is the material Aurora is trained to produce for it.
#
# Worked example: C4-M08's approved ask is "Is the 8:15 still running?", but
# v2's cycle researched van insurance renewal terms and then reported a renewal
# quote. Trained on that, Aurora learns to answer a timetable question with an
# insurance lookup.
TURNS_OVERRIDES: dict[str, tuple[tuple, ...]] = {
    # "Remember for next time: I take my coffee black." - v2 filled {item} into
    # the fact, so the record stored "the coffee preference is the van". The
    # payload also used v2's 'fact' key, which the capability does not accept:
    # it wants the lesson in 'text' and where it came from in 'provenance'.
    "C3-P05": (
        ("R", "learn", {"text": "Amir takes his coffee black",
                        "provenance": "user"}),
        ("T", "learn: stored (provenance: user)"),
        ("A", "Noted."),
    ),
    # "I'd rather you didn't schedule things for me without asking." - v2's
    # fact was about an approver signing something off.
    "C3-P10": (
        ("R", "learn", {"text": "Scheduling for Amir needs asking first",
                        "provenance": "user"}),
        ("T", "learn: stored (provenance: user)"),
        ("A", "That's saved."),
    ),
    # "Count the words in this: <long pasted document>" - v2 counted {item},
    # which draws from the shared item bank ("the van comes to 412 words").
    "C5-A03": (
        ("R", "use_tool",
         {"name": "word_count", "arguments": {"text": "the pasted document"}}),
        ("T", "word_count: {value}"),
        ("A", "That comes to {value} words."),
    ),
    # "Should we still support this old format..." - v2 looked up a rate.
    "C5-A05": (
        ("R", "research", {"query": "current ecosystem support for the old format"}),
        ("T", "research: support has largely moved to the new format"),
        ("A", "Support's moved across - the old format still turns up, but "
              "it's the smaller share now."),
    ),
    # "what's the exchange rate on the day we bought the car" - v2 queried
    # "{item} rate this year", which draws a household item.
    "C5-A08": (
        ("R", "research", {"query": "the exchange rate on the day of purchase"}),
        ("T", "research: rate on the day was {value}"),
        ("A", "{value}."),
    ),
    # "Is the 08:40 still running?" - v2 searched "{item} latest".
    "C5-A11": (
        ("R", "research", {"query": "the 08:40 timetable"}),
        ("T", "research: the 08:40 was withdrawn, replaced by the 09:05"),
        ("A", "No - it's the 09:05 now. The 08:40 was withdrawn."),
    ),
    # "What's our usual order?" - v2 answered "{item} = {value}", which reads
    # "It was the van."
    "C5-A13": (
        ("R", "recall", {"query": "the usual order"}),
        ("T", "recall: usual order is the cortado"),
        ("A", "The cortado - that's your usual."),
    ),
    # "Which supplier did we sign with in March?" - v2's {value} draws from the
    # numeric bank, so a supplier came out as "412".
    "C7-M05": (
        ("R", "recall", {"query": "the supplier signed with in March"}),
        ("T", "recall: signed with Hensley & Ferrier in March"),
        ("A", "Hensley & Ferrier - we signed with them in March."),
    ),
    # "What was the deadline last season" - v2 filled {value} (numeric) where a
    # day was wanted. {day} keeps a slot and draws from the weekday bank.
    "C7-M07": (
        ("R", "recall", {"query": "last season's deadline"}),
        ("T", "recall: last season's deadline was {day}"),
        ("A", "Last season it was {day}. This season is different."),
    ),
    # Two stored figures - v2's tool turn printed the same {value} twice.
    "C8-G18": (
        ("R", "recall", {"query": "the two stored figures for the vineyard shed"}),
        ("T", "recall: {value} from last week; an older figure from March"),
        ("A", "{value} - that's last week's. The March figure is superseded."),
    ),
    # C2-D04: approved ask is ferry times versus a meeting. v2 recalled a berth
    # code and researched the Python release.
    "C2-D04": (
        ("R", "recall", {"query": "ferry times for the crossing"}),
        ("T", "recall: the ferry docks at 08:05"),
        ("R", "research", {"query": "start time of the meeting today"}),
        ("T", "research: the meeting starts at 08:30"),
        ("A", "The ferry's in at 08:05 and the meeting starts at 08:30, so "
              "you're clear."),
    ),
    # C2-D06: approved ask is next train plus weather. v2 recalled a coffee
    # preference and the roaster's hours.
    "C2-D06": (
        ("R", "recall", {"query": "the next train departure"}),
        ("T", "recall: next train leaves at 17:40"),
        ("R", "research", {"query": "the weather there tomorrow"}),
        ("T", "research: dry until six, rain after that"),
        ("A", "Next train is 17:40, and it stays dry until six before the rain "
              "comes in."),
    ),
    # C4-M06: approved ask is berth code plus release date. v2 recalled a van
    # booking day and a signatory.
    "C4-M06": (
        ("R", "recall", {"query": "the berth code and the release date"}),
        ("T", "recall: berth foghorn-42, release 3.13"),
        ("A", "The berth code was foghorn-42, and the release is 3.13."),
    ),
    # C4-M07: approved ask is the venue. v2 asked about a deposit figure.
    "C4-M07": (
        ("R", "recall", {"query": "the agreed venue"}),
        ("T", "recall: no matching passage"),
        ("A", "I don't have the venue stored. What was agreed?"),
    ),
    # C4-M08: approved ask is the 8:15. v2 researched van insurance.
    "C4-M08": (
        ("R", "research", {"query": "the 8:15 service timetable"}),
        ("T", "research: no match for that exact wording"),
        ("R", "research", {"query": "timetable for services around 08:15"}),
        ("T", "research: the 08:15 was withdrawn, replaced by the 08:25"),
        ("A", "The 8:15 is gone - it was replaced by the 08:25. The first "
              "search only missed on wording."),
    ),
    # C7-M11: approved ask is the insurance. v2 recalled the vitrines.
    "C7-M11": (
        ("R", "recall", {"query": "the insurance policy"}),
        ("T", "recall: policy number only"),
        ("R", "recall", {"query": "what was decided about the insurance"}),
        ("T", "recall: keep the existing cover, no change this year"),
        ("A", "Keep the cover we've got - nothing was changed on it this year."),
    ),
}

# Seeds whose approved target shape is not the kind v2's spec tables place
# them in. C5-A02's approved shape is "user / assistant(plain answer, no tool
# needed for arithmetic)" and its stated behaviour is that using a tool would
# be theatre - but v2 built a calculate round whose answer was 360 for
# "What's 12 times 12?". Reclassifying it as prose is what makes the record
# say 144 and skip the tool, as approved.
KIND_OVERRIDES: dict[str, str] = {
    "C5-A02": "prose",
}




# --------------------------------------------------------------------------
# Loss defaults, matching v0.2 behaviour exactly.
# --------------------------------------------------------------------------


def default_train_on(kind: str, assistant_count: int,
                     seed_train_on: tuple[int, ...] | None) -> tuple[int, ...]:
    if kind in ("emit",):
        return (0,) if assistant_count else ()
    if kind in ("tool", "cycle"):
        if seed_train_on is not None:
            return tuple(i for i in seed_train_on if i < assistant_count)
        return tuple(range(assistant_count))
    # prose: only the final reply is the behaviour being taught
    return (assistant_count - 1,) if assistant_count else ()


# --------------------------------------------------------------------------
# Retained seeds: v2 material, plus the v3 supplement bank.
# --------------------------------------------------------------------------


def _is_prior_context(text: str) -> bool:
    return any(marker in text for marker in
               ("Prior turn", "Prior context", "Earlier", "Several turns"))


def _split_prior(context: str) -> tuple[list[tuple], str]:
    """Turn an authored 'Prior ... Now: question' blob into explicit turns.

    v0.2 did this at build time inside prose_example. It is repeated here so
    v3 can express multi-turn records as data instead of re-deriving them on
    every build, and so a retained seed's prior turns stay byte-comparable to
    what v0.2 produced.
    """
    body, sep, last = context.rpartition(". Now:")
    if not sep:
        body, sep, last = context.rpartition(". Later:")
    if not sep:
        return [], context.strip()
    chunks = [c.strip() for c in body.replace("Earlier turn:", "").split(". ")
              if c.strip()]
    turns: list[tuple] = []
    acks = B2.ACKS
    for index, chunk in enumerate(chunks):
        turns.append(("U", chunk + "."))
        turns.append(("A", acks[index % len(acks)]))
    turns.append(("U", (last.strip() if last else context) + "?"))
    return turns, ""


def _retained_base(seed: D2.Seed) -> list[Variant]:
    """Rebuild v0.2's authored material as v3 variants, read-only.

    Two deliberate departures from a literal copy, both required by v0.3's
    approved gates and both recorded in the correction tables above:

    * the approved C3-P01 ask carries an unfilled ``<text>`` hole, which is
      rewritten to ``{note}`` so generation fills it instead of shipping a
      literal placeholder to the model;
    * the payload is kept as its authored *template* rather than being baked
      with ``fill_payload``, so the generator can fill payload, tool result
      and answer from one draw and the three can never disagree.
    """
    sid = seed.seed_id
    ask = B2.strip_amir_prefix(seed.user_or_context).strip("'")
    if "<text>" in ask:
        ask = ask.replace("<text>", "{note}")

    variants: list[Variant] = []

    forced_kind = KIND_OVERRIDES.get(sid)
    if forced_kind == "prose" or (
            forced_kind is None and sid in B2.PROSE_TARGETS):
        if sid in PROSE_TARGET_OVERRIDES:
            targets = PROSE_TARGET_OVERRIDES[sid]
        else:
            targets = B2.PROSE_TARGETS[sid]
        if seed.multi_turn and _is_prior_context(seed.user_or_context):
            head, _ = _split_prior(seed.user_or_context)
            if not head:
                # No '. Now:' / '. Later:' separator: the authored blob is the
                # whole prior exchange. v0.2 emitted it twice - once as a blob
                # and again as the final question - so the record opened with
                # two consecutive user turns. One user turn carrying the prior
                # context keeps the memory lesson intact without that artefact.
                head = [("U", seed.user_or_context)]
            for target in targets:
                variants.append(Variant(turns=tuple(head + [("A", target)])))
        else:
            for target in targets:
                variants.append(Variant.ask_answer(ask, target))
        return variants

    if sid in TURNS_OVERRIDES:
        turns = [("U", ask)] + list(TURNS_OVERRIDES[sid])
        variants.append(Variant(
            turns=tuple(turns),
            train_on=tuple(seed.train_on) if seed.train_on is not None else None,
        ))
        return variants

    if sid in B2.EMIT_CAPABILITY:
        action = B2.EMIT_CAPABILITY[sid]
        payload: dict = {"query": B2.QUERY.get(sid, "reference")}
        if action == "calculate":
            payload = {"expression": "15% of 2400"}
        elif action == "use_tool":
            payload = {"name": "word_count",
                       "arguments": {"text": "the survey note"}}
        elif action == "learn":
            payload = {"text": payload["query"], "provenance": "user"}
        elif action == "consult_specialist":
            payload = {"specialist": SPECIALIST, "task": payload["query"]}
        payload = EMIT_PAYLOAD_OVERRIDES.get(sid, payload)
        variants.append(Variant(turns=(("U", ask), ("R", action, payload))))
        return variants

    if sid in B2.TOOL_SPEC:
        action, payload_t, tool_t, answer_t = B2.TOOL_SPEC[sid]
        answer_t = TOOL_ANSWER_OVERRIDES.get(sid, answer_t)
        payload_t = PAYLOAD_OVERRIDES.get(sid, payload_t)
        variants.append(Variant(
            turns=(("U", ask),
                   ("R", action, payload_t),
                   ("T", tool_t),
                   ("A", answer_t)),
            train_on=tuple(seed.train_on) if seed.train_on is not None else None,
        ))
        return variants

    if sid in B2.CYCLE_SPEC:
        turns: list[tuple] = [("U", ask)]
        override = PAYLOAD_OVERRIDES.get(sid)
        for turn in B2.CYCLE_SPEC[sid]:
            if turn[0] == "T":
                turns.append(("T", turn[1]))
            elif turn[0] == "R":
                turns.append(("R", turn[1], override or turn[2]))
            elif turn[0] == "X":
                turns.append(("X", turn[2], override or turn[3]))
            else:
                turns.append(("A", turn[1]))
        variants.append(Variant(
            turns=tuple(turns),
            train_on=tuple(seed.train_on) if seed.train_on is not None else None,
        ))
        return variants

    # C15-R02: the review-only anchor. Retained in the bank, emits nothing.
    return variants


# --------------------------------------------------------------------------
# Domain / register spread. Every seed carries one so each category can prove
# >= 4 domains and >= 3 registers without generation inventing them.
# --------------------------------------------------------------------------

DOMAIN_REGISTER: dict[str, tuple[str, str]] = json.loads(
    r"""
{
  "C11-I01": ["technology", "neutral"],   "C11-I02": ["technology", "neutral"],
  "C11-I03": ["technology", "serious"],   "C11-I04": ["technology", "casual"],
  "C11-I05": ["household", "casual"],
  "C8-G01": ["work", "casual"],           "C8-G02": ["travel", "warm"],
  "C8-G03": ["household", "casual"],      "C8-G04": ["admin", "neutral"],
  "C8-G05": ["household", "casual"],      "C8-G06": ["household", "neutral"],
  "C8-G07": ["study", "casual"],          "C8-G08": ["work", "neutral"],
  "C8-G09": ["money", "casual"],          "C8-G10": ["work", "serious"],
  "C8-G11": ["work", "casual"],           "C8-G12": ["admin", "serious"],
  "C8-G13": ["money", "neutral"],         "C8-G14": ["outdoors", "casual"],
  "C8-G15": ["outdoors", "serious"],      "C8-G16": ["travel", "warm"],
  "C8-G17": ["admin", "neutral"],         "C8-G18": ["money", "neutral"],
  "C8-G19": ["travel", "warm"],           "C8-G20": ["money", "neutral"],
  "C9-U01": ["work", "neutral"],          "C9-U02": ["household", "playful"],
  "C9-U03": ["study", "neutral"],         "C9-U04": ["health", "serious"],
  "C9-U05": ["media", "playful"],
  "C10-N01": ["social", "casual"],        "C10-N02": ["social", "warm"],
  "C10-N03": ["money", "neutral"],        "C10-N04": ["admin", "neutral"],
  "C10-N05": ["work", "casual"],
  "C12-P01": ["work", "warm"],            "C12-P02": ["household", "serious"],
  "C12-P03": ["outdoors", "curious"],     "C12-P04": ["household", "playful"],
  "C12-P05": ["social", "playful"],       "C12-P06": ["technology", "focused"],
  "C13-A01": ["work", "warm"],            "C13-A02": ["money", "warm"],
  "C13-A03": ["household", "serious"],    "C13-A04": ["health", "serious"],
  "C13-A05": ["work", "serious"],
  "C14-R01": ["travel", "neutral"],       "C14-R02": ["work", "urgent"],
  "C14-R03": ["admin", "neutral"],        "C14-R04": ["admin", "neutral"],
  "C14-R05": ["admin", "urgent"],
  "C5-A01": ["study", "neutral"],         "C5-A02": ["study", "casual"],
  "C5-A03": ["work", "neutral"],          "C5-A04": ["household", "casual"],
  "C5-A05": ["technology", "serious"],    "C5-A06": ["social", "warm"],
  "C5-A07": ["travel", "casual"],         "C5-A08": ["money", "neutral"],
  "C5-A09": ["study", "neutral"],         "C5-A10": ["money", "neutral"],
  "C5-A11": ["travel", "urgent"],         "C5-A12": ["money", "neutral"],
  "C5-A13": ["food", "casual"],           "C5-A14": ["money", "serious"],
  "C7-M01": ["social", "casual"],         "C7-M02": ["food", "casual"],
  "C7-M03": ["money", "neutral"],         "C7-M04": ["work", "casual"],
  "C7-M05": ["work", "neutral"],          "C7-M06": ["food", "casual"],
  "C7-M07": ["outdoors", "neutral"],      "C7-M08": ["food", "warm"],
  "C7-M09": ["admin", "casual"],          "C7-M10": ["money", "casual"],
  "C7-M11": ["money", "neutral"],         "C7-M12": ["household", "casual"],
  "C7-M13": ["admin", "neutral"],         "C7-M14": ["social", "warm"],
  "C7-M15": ["household", "casual"],      "C7-M16": ["work", "casual"],
  "C15-R01": ["study", "casual"],         "C15-R02": ["technology", "serious"],
  "C15-R03": ["work", "neutral"],         "C15-R04": ["technology", "neutral"],
  "C15-R05": ["work", "neutral"],         "C15-R06": ["admin", "neutral"],
  "C15-R07": ["work", "casual"],          "C15-R08": ["work", "casual"],
  "C15-R09": ["technology", "serious"],   "C15-R10": ["admin", "neutral"],
  "C1-A01": ["work", "neutral"],          "C1-A02": ["work", "casual"],
  "C1-A03": ["admin", "polite"],          "C1-A04": ["admin", "neutral"],
  "C1-A05": ["admin", "neutral"],         "C1-A06": ["money", "neutral"],
  "C1-A07": ["technology", "neutral"],    "C1-A08": ["admin", "urgent"],
  "C1-A09": ["travel", "urgent"],         "C1-A10": ["study", "casual"],
  "C3-P01": ["study", "neutral"],         "C3-P02": ["money", "serious"],
  "C3-P03": ["money", "serious"],         "C3-P04": ["money", "neutral"],
  "C3-P05": ["food", "casual"],           "C3-P06": ["technology", "neutral"],
  "C3-P07": ["work", "neutral"],          "C3-P08": ["household", "neutral"],
  "C3-P09": ["outdoors", "curious"],      "C3-P10": ["work", "polite"],
  "C2-D01": ["admin", "casual"],          "C2-D02": ["work", "urgent"],
  "C2-D03": ["work", "neutral"],          "C2-D04": ["travel", "neutral"],
  "C2-D05": ["work", "urgent"],           "C2-D06": ["travel", "casual"],
  "C2-D07": ["money", "neutral"],         "C2-D08": ["money", "urgent"],
  "C4-M01": ["technology", "neutral"],    "C4-M02": ["work", "neutral"],
  "C4-M03": ["money", "neutral"],         "C4-M04": ["admin", "formal"],
  "C4-M05": ["money", "neutral"],         "C4-M06": ["admin", "neutral"],
  "C4-M07": ["social", "casual"],         "C4-M08": ["travel", "urgent"],
  "C4-M09": ["money", "serious"],
  "C6-X01": ["work", "neutral"],          "C6-X02": ["money", "neutral"],
  "C6-X03": ["study", "casual"],          "C6-X04": ["money", "serious"],
  "C6-X05": ["household", "neutral"],     "C6-X06": ["work", "serious"],
  "C6-X07": ["work", "urgent"],           "C6-X08": ["technology", "warm"]
}
"""
)

# Registers that count toward MIN_REGISTERS_PER_CATEGORY. The bank above uses
# a few finer-grained labels; they fold into these six for the gate.
REGISTER_FOLD: dict[str, str] = {
    "curious": "casual", "focused": "serious", "polite": "formal",
}


def fold_register(register: str) -> str:
    return REGISTER_FOLD.get(register, register)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

RETAINED_IDS: tuple[str, ...] = D3.RETAINED_IDS  # filled by design_v3


def _kind_of(sid: str) -> str:
    if sid in KIND_OVERRIDES:
        return KIND_OVERRIDES[sid]
    if sid in B2.EMIT_CAPABILITY:
        return "emit"
    if sid in B2.TOOL_SPEC:
        return "tool"
    if sid in B2.CYCLE_SPEC:
        return "cycle"
    if sid in B2.PROSE_TARGETS:
        return "prose"
    return "none"


def _clone_last(turns: tuple[tuple, ...], last: tuple) -> tuple[tuple, ...]:
    """Same conversation, final turn replaced. Everything above stays fixed."""
    return tuple(turns[:-1]) + (last,)


def build_extras() -> dict[str, tuple[Variant, ...]]:
    """Turn the authored tables in ``extras_v3`` into supplement variants.

    Three shapes, all applied to the seed's own base conversation so a
    supplement can never change what the fixed turns above it said:

    * ``PROSE_SUPPLEMENTS``   - replace the closing ``A`` turn with a new reply;
    * ``FINAL_TURN_ALTERNATES`` - replace the closing ``A`` turn of a tool or
      cycle record, the turn that follows the tool result;
    * ``EMIT_PAYLOAD_BANK``   - replace the closing ``R`` payload.

    Keying off the base's final turn means the three tables cannot drift from
    the material they extend: a seed whose base does not end the way the table
    assumes raises here rather than shipping a malformed record.
    """
    by_id = {s.seed_id: s for s in D2.SEEDS}
    base_cache: dict[str, tuple[Variant, ...]] = {}

    def base_of(sid: str) -> tuple[Variant, ...]:
        if sid not in base_cache:
            base_cache[sid] = tuple(_retained_base(by_id[sid]))
        return base_cache[sid]

    out: dict[str, list[Variant]] = {}

    def add(sid: str, variant: Variant) -> None:
        out.setdefault(sid, []).append(variant)

    for sid, targets in X3.PROSE_SUPPLEMENTS.items():
        if sid not in RETAINED_IDS:
            raise ValueError(f"prose supplement for non-retained seed {sid}")
        base = base_of(sid)
        if not base or base[0].turns[-1][0] != "A":
            raise ValueError(f"prose supplement target is not a closing A: {sid}")
        for target in targets:
            add(sid, Variant(
                turns=_clone_last(base[0].turns, ("A", target)),
                train_on=base[0].train_on,
            ))

    for sid, alternates in X3.FINAL_TURN_ALTERNATES.items():
        if sid not in RETAINED_IDS:
            raise ValueError(f"turn alternate for non-retained seed {sid}")
        base = base_of(sid)
        if not base or base[0].turns[-1][0] != "A":
            raise ValueError(f"turn alternate target is not a closing A: {sid}")
        for text in alternates:
            add(sid, Variant(
                turns=_clone_last(base[0].turns, ("A", text)),
                train_on=base[0].train_on,
            ))

    for sid, payloads in X3.EMIT_PAYLOAD_BANK.items():
        if sid not in RETAINED_IDS:
            raise ValueError(f"payload bank for non-retained seed {sid}")
        base = base_of(sid)
        if not base or base[0].turns[-1][0] != "R":
            raise ValueError(f"payload bank target is not a closing R: {sid}")
        action = base[0].turns[-1][1]
        for payload in payloads:
            add(sid, Variant(
                turns=_clone_last(base[0].turns, ("R", action, payload)),
                train_on=base[0].train_on,
            ))

    return {sid: tuple(variants) for sid, variants in out.items()}


def build_retained(extras: dict[str, tuple[Variant, ...]]) -> list[SeedV3]:
    """136 retained seeds, v0.2 material plus the v3 supplement variants."""
    seeds: list[SeedV3] = []
    for v2_seed in D2.SEEDS:
        sid = v2_seed.seed_id
        if sid not in extras and sid not in RETAINED_IDS:
            continue
        if sid not in RETAINED_IDS:
            continue
        base = _retained_base(v2_seed)
        added = extras.get(sid, ())
        domain, register = DOMAIN_REGISTER[sid]
        seeds.append(SeedV3(
            seed_id=sid,
            category=v2_seed.category,
            kind=_kind_of(sid),
            scenario=v2_seed.scenario,
            ask=B2.strip_amir_prefix(v2_seed.user_or_context).strip("'"),
            behavior=v2_seed.expected_behavior,
            target_shape=v2_seed.target_shape,
            why=v2_seed.why_it_exists,
            multi_turn=v2_seed.multi_turn,
            priority=v2_seed.priority,
            domain=domain,
            register=register,
            provenance="retained",
            training_allowed=v2_seed.training_allowed,
            train_on=v2_seed.train_on,
            variants=tuple(base) + tuple(added),
        ))
    return seeds


def seed_bank_report(seeds: Iterable[SeedV3]) -> dict:
    """Structural summary. The validator re-derives every number independently."""
    seeds = list(seeds)
    by_category: dict[str, list[SeedV3]] = {}
    for seed in seeds:
        by_category.setdefault(seed.category, []).append(seed)

    domains: dict[str, set[str]] = {}
    registers: dict[str, set[str]] = {}
    for seed in seeds:
        domains.setdefault(seed.category, set()).add(seed.domain)
        registers.setdefault(seed.category, set()).add(fold_register(seed.register))

    return {
        "seed_count": len(seeds),
        "declared": D3.SEED_BUDGET,
        "by_category": {k: len(v) for k, v in sorted(by_category.items())},
        "by_provenance": {
            p: sum(1 for s in seeds if s.provenance == p)
            for p in ("retained", "new", "reconciled")
        },
        "by_kind": {
            k: sum(1 for s in seeds if s.kind == k)
            for k in ("prose", "emit", "tool", "cycle", "none")
        },
        "blocked": [s.seed_id for s in seeds if not s.training_allowed],
        "domains_per_category": {k: sorted(v) for k, v in sorted(domains.items())},
        "registers_per_category": {k: sorted(v)
                                   for k, v in sorted(registers.items())},
        "min_domains": min(len(v) for v in domains.values()),
        "min_registers": min(len(v) for v in registers.values()),
    }
