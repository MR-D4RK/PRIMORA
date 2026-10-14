"""Builders that turn an approved seed into concrete example records.

Design constraint that shapes everything here: the seed's ``target_shape`` is the
behavioural specification, and ``user_or_context`` is one concrete situation. An
example is that situation re-told in a genuinely different frame, with the same
behaviour taught.

What "genuinely different" means here, and what it deliberately does not mean:

*   A **framing** change is a different conversation: tired vs direct vs
    hedged vs context-first. The user behaves differently, so it is a different
    situation rather than a paraphrase.
*   A **particular** change swaps a concrete object or value, and only where
    the seed's lesson survives it. Swapping "the shed roof" for "the boundary
    fence" in C12-P02 is safe; swapping it in C15-R14 would destroy the lesson.
*   The **target string** for an emit example is the canonical request block,
    so the format stays stable while the situation varies. That is deliberate:
    the format is the thing being learned.

Prose targets are authored per seed, not generated. A template that produced
"Certainly! I'd be happy to help" across 800 examples would be exactly the
template farm the old corpus was, so the target text lives in ``PROSE_TARGETS``
and is looked up by seed id. Seeds without an authored target are rejected
rather than filled with a generic string.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from generate_v2 import (
    DAYS,
    ITEMS,
    VALUES,
    Example,
    frame_ask,
    request_block,
    strip_amir_prefix,
    tool_turn,
)


# --------------------------------------------------------------------------
# Capability payload helpers. Payloads are derived from the seed's own
# situation so the request block describes the thing actually asked about.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Authored capability queries.
#
# These used to be derived by regex from the seed's ``user_or_context`` and then
# had a random noun appended, which produced things like
# ``"Say you'll check the lease for me, would you the boat lift"`` - the user's
# whole sentence with an item jammed onto the end. Two defects: the extractor
# had no notion of a noun phrase, and the ``f"{query} {item}"`` concatenation
# ran unconditionally even when the query was already a complete topic.
#
# A capability query is a topic, not a sentence, so it is authored per seed here
# and validated by ``_query_for`` below. Nothing derives it at build time.
# --------------------------------------------------------------------------

QUERY: dict[str, str] = {
    # C1 - act, don't announce
    "C1-A01": "recent timetable changes",
    "C1-A02": "recent timetable changes",
    "C1-A03": "the lease notice period",
    "C1-A04": "the delivery date",
    "C1-A05": "the lease notice period",
    "C1-A06": "whether the van insurance is worth it",
    "C1-A07": "current guidance on the extension",
    "C1-A08": "the cut-off closing time",
    "C1-A09": "the weather forecast for the schedule",
    "C1-A10": "the word count of the supplied note",
    "C1-A11": "what was said about the lease",
    "C1-A12": "the trade-off before committing",
    "C1-A13": "the roof survey findings",
    "C1-A14": "the second van booking",
    "C1-A17": "whether the permit covers the extension",
    "C1-A18": "the survey report status",
    "C1-A19": "whether the train is still running",
    "C1-A20": "the liability position on the contract",
    # C2 - request discipline
    "C2-D01": "the note about the south quay",
    "C2-D02": "the berth code we used",
    "C2-D03": "the berth code we used",
    "C2-D04": "the ferry times",
    "C2-D05": "the berth code we used",
    "C2-D06": "the next train time",
    "C2-D07": "the agreed deposit figure",
    "C2-D08": "the invoice number and due date",
    "C2-D09": "the tide times",
    "C2-D11": "the delivery status of the invoice",
    "C2-D12": "the current balance",
    "C2-D13": "the berth code we used",
    "C2-D14": "the release notes",
    "C2-D15": "the correct invoice number",
    "C2-D16": "the current release",
    "C2-D17": "the forecast for the booking",
    "C2-D18": "the berth code we used",
    # C3 - correct capability protocol
    "C3-P01": "the word count of the supplied note",
    "C3-P02": "whether to push back on the pricing change",
    "C3-P03": "whether to replace the old pump",
    "C3-P04": "17 percent of 4,200",
    "C3-P05": "the coffee preference",
    "C3-P06": "the latest release",
    "C3-P07": "what was agreed about the vitrine lights",
    "C3-P08": "the immersion blender serial number",
    "C3-P09": "compost heap temperature in winter",
    "C3-P10": "the scheduling preference",
    "C3-P11": "the current time",
    "C3-P12": "the basket total without delivery",
    "C3-P13": "the word count of the note",
    "C3-P14": "the reasoning behind the decision",
    "C3-P15": "the word count of the quoted line",
    "C3-P16": "twelve kilos in grams",
    "C3-P17": "what was said about the drainage",
    "C3-P18": "whether to split the storage",
    "C3-P19": "drone flight rules over the coast",
    "C3-P21": "the word count of the note",
    "C3-P22": "what was agreed about the vitrine lights",
    # C4 - continue cognition
    "C4-M01": "the berth code we used at the coast",
    "C4-M02": "the survey visit date and meeting point",
    "C4-M03": "the agreed deposit and the invoice",
    "C4-M04": "the agreed figure and the signatory",
    "C4-M05": "the invoice total",
    "C4-M06": "the berth code and the release date",
    "C4-M07": "the decision about the venue",
    "C4-M08": "the 8:15 service",
    "C4-M09": "extension cover under the current policy",
    "C4-M10": "the current version",
    "C4-M11": "the ferry time",
    "C4-M12": "the drainage",
    "C4-M13": "the figure on the invoice",
    "C4-M16": "the agreed deposit",
    "C4-M17": "the word count of the note",
    "C4-M18": "the shed roof inspection date",
    "C4-M19": "the vineyard shed inspection date",
    "C4-M20": "the four names to sort",
    "C4-M21": "planning permission for the extension",
    "C4-M31": "reissuing the shed invoice",
    "C4-M22": "the forecast for the roof work",
    "C4-M23": "the supplier's current address",
    "C4-M24": "the old survey report for the vineyard shed",
    "C4-M26": "the van booked for Thursday",
    "C4-M28": "the agreed figure for the deck",
    "C4-M29": "the agreed figure against the invoice",
    "C4-M30": "the old quote",
    # C5 - appropriate restraint
    "C5-A02": "12 times 12",
    "C5-A03": "the word count of the pasted document",
    "C5-A05": "whether to keep supporting the old format",
    "C5-A08": "the exchange rate on the day",
    "C5-A11": "the 08:40 service",
    "C5-A13": "the usual order",
    "C5-A14": "whether to rebuild the gearbox",
    "C5-A16": "whether to keep supporting the format",
    # C6 - capability choice
    "C6-X01": "the current status of the pier",
    "C6-X02": "the budget decision",
    "C6-X04": "whether splitting the storage is worth it",
    "C6-X05": "the supplied measurements in kilograms",
    "C6-X06": "the choice between the two suppliers",
    "C6-X07": "the Friday deadline against the sign-off",
    "C6-X08": "the reasoning behind the decision",
    "C6-X09": "whether to contest the contract",
    "C6-X10": "the pricing answer",
    "C6-X11": "whether to split the storage",
    "C6-X13": "whether to contest the contract",
    "C6-X14": "15 percent of 2,400",
    # C7 - memory use
    "C7-M03": "the agreed figure",
    "C7-M05": "the supplier we signed with in March",
    "C7-M06": "the usual order",
    "C7-M07": "last season's deadline",
    "C7-M08": "the current usual order",
    "C7-M09": "the figure agreed at the start",
    "C7-M10": "what was said about the boat in November",
    "C7-M11": "what was said about the insurance",
    "C7-M13": "the delivery address",
    "C7-M15": "what was said about the second van",
    "C7-M17": "the day the van is booked for",
    "C7-M18": "the arrangement agreed weeks ago",
    # C8 - grounding and truth
    "C8-G05": "what was agreed about the shed roof",
    "C8-G06": "the roofing material chosen for the shed",
    "C8-G10": "supplier availability this quarter",
    "C8-G18": "the figure to give the supplier",
    # C11 - identity and self-model
    "C11-I04": "whether the fetch tool reads the page or Aurora does",
    # C15 - recovery
    "C15-R01": "the word count of the quoted note",
    "C15-R03": "the deck cover decision",
    "C15-R04": "the most recent Python release",
    "C15-R05": "the deck figures",
    "C15-R11": "the word count of the pasted note",
    "C15-R12": "the invoice dated the 14th",
    "C15-R13": "the supplied measurements in kilograms",
    "C15-R14": "the shed roof decision at the last inspection",
    "C15-R15": "the invoice total for the shed job",
    "C15-R17": "the current Python release",
    "C15-R22": "whether splitting the storage is worth it",
    "C15-R24": "the design award winner for the vineyard shed",
}

# A capability query is a topic. These are the shapes that mean the extractor
# has drifted back into returning a sentence.
_SENTENCE_MARKERS = (
    "?", "!", " would you", " could you", " can you", " shall we",
    " do you ", " did you ", " is it ", " are we ", " am i ",
)


def _query_for(seed_id: str) -> str:
    """Return the authored query for a seed, refusing a sentence-shaped one.

    The guard exists because the failure it catches is silent: a sentence in a
    ``query`` field is still valid JSON and still parses, so nothing downstream
    complains. It only shows up as a nonsense capability argument.
    """
    if seed_id not in QUERY:
        raise KeyError(
            f"{seed_id} has no authored query. Every capability query must be "
            f"authored per seed; deriving it from the seed context previously "
            f"produced sentence-shaped arguments.")
    query = QUERY[seed_id]
    lowered = f" {query.lower()} "
    for marker in _SENTENCE_MARKERS:
        if marker in lowered or marker in query:
            raise ValueError(
                f"{seed_id}: authored query {query!r} reads as a sentence, not "
                f"a topic")
    return query


def _tool_payload(text: str) -> tuple[str, Any, Any]:
    """Split an authored tool body into the runtime's (status, output, error).

    Only a body prefixed ``failure:`` is a refused or failed capability. That
    distinction is load-bearing: the loop's recovery grant keys off
    ``'"status": "failure"'``, so mislabelling an empty recall as a failure
    would teach Aurora that "no matching passage" is an error rather than a
    successful lookup that found nothing.

    A negative result is therefore ``status="success"`` with the negative text
    as the output, which is what the runtime actually produces when a recall
    returns an empty set.
    """
    if text.startswith("failure: "):
        return "failure", None, text[len("failure: "):].strip()
    return "success", text, None


def fill_payload(raw: dict, query: str, item: str, value: str) -> dict:
    """Fill placeholders in a payload, recursing into nested arguments.

    A nested dict cannot go through str() then format(): its own braces would be
    read as format fields. That silently broke every seed with an ``arguments``
    block - C15-R11 and C15-R13 among them.
    """
    def fill(text: str) -> str:
        return text.format(query=query, item=item, value=value)

    out: dict = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            out[key] = fill_payload(val, query, item, value)
        elif isinstance(val, list):
            out[key] = [fill(v) if isinstance(v, str) else v for v in val]
        elif isinstance(val, str):
            out[key] = fill(val)
        else:
            out[key] = val
    return out


# --------------------------------------------------------------------------
# EMIT seeds: a single capability request as the whole reply.
# --------------------------------------------------------------------------

# Which capability each emit seed is supposed to request. Keyed by seed id so a
# mis-mapped seed fails loudly at build time instead of silently teaching the
# wrong capability.
EMIT_CAPABILITY: dict[str, str] = {
    # C1 act, don't announce - the reply is the request, never a promise.
    "C1-A04": "research", "C1-A05": "recall", "C1-A06": "consult_specialist",
    "C1-A07": "research", "C1-A08": "research", "C1-A09": "research",
    "C1-A10": "use_tool", "C1-A11": "recall", "C1-A12": "consult_specialist",
    "C1-A13": "research", "C1-A17": "research", "C1-A19": "research",
    "C1-A20": "research",
    # C3 correct capability protocol - each seed is a protocol case.
    "C3-P01": "use_tool", "C3-P02": "consult_specialist",
    "C3-P03": "consult_specialist", "C3-P04": "calculate",
    "C3-P06": "research", "C3-P07": "recall", "C3-P08": "recall",
    "C3-P09": "research", "C3-P11": "use_tool", "C3-P12": "calculate",
    "C3-P13": "use_tool", "C3-P14": "consult_specialist", "C3-P15": "use_tool",
    "C3-P16": "use_tool", "C3-P17": "recall", "C3-P18": "consult_specialist",
    "C3-P19": "research", "C3-P21": "use_tool", "C3-P22": "recall",
    # C6 capability choice - a single request, no delegation.
    "C6-X01": "research", "C6-X02": "recall", "C6-X05": "use_tool",
    "C6-X14": "calculate",
    # C1 and C2 request discipline. C2's whole lesson is that a batching
    # instruction does not license two blocks, so these emit exactly one.
    "C1-A01": "research", "C1-A02": "research", "C1-A03": "research",
    "C1-A18": "research",
    "C2-D01": "recall", "C2-D02": "recall", "C2-D05": "recall",
    "C2-D08": "recall", "C2-D09": "research", "C2-D13": "recall",
    "C2-D14": "research", "C2-D18": "recall",
}


def emit_example(seed, variant: int, rng: random.Random, rngs: dict) -> Example:
    """One capability request, chosen per seed, in a varied framing."""
    action = EMIT_CAPABILITY[seed.seed_id]
    context = seed.user_or_context
    item = rng.choice(ITEMS)
    query = _query_for(seed.seed_id)

    payload = {"query": query}
    if action == "calculate":
        payload = {"expression": "15% of 2400"}
    elif action == "use_tool":
        payload = {"name": "word_count", "arguments": {"text": "the survey note"}}
    elif action == "learn":
        payload = {"fact": query}

    style = rngs["style"](variant)
    user = frame_ask(rng, strip_amir_prefix(context).strip("'"), style)

    assistant = request_block(action, payload)
    return Example(
        id=f"{seed.seed_id}-{variant:02d}",
        seed_id=seed.seed_id,
        category=seed.category,
        split="train",
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        train_on=[0],
        expect_block=True,
        expect_action=action,
        note=seed.scenario,
        variant=variant,
        seed_priority=seed.priority,
    )


# --------------------------------------------------------------------------
# TOOL seeds: request, tool result, then the trained answer.
# --------------------------------------------------------------------------

# (action, payload, tool_result_template, answer_template) per seed.
# {} is filled from the seed's own context at build time.
TOOL_SPEC: dict[str, tuple[str, dict, str, str]] = {
    # (action, payload template, tool result template, trained answer template)
    "C3-P05": ("learn", {"fact": "the coffee preference is {item}"},
               "learn: stored (provenance: user)",
               "Noted."),
    "C3-P10": ("learn", {"fact": "{item} was signed off by the same approver"},
               "learn: stored (provenance: user)",
               "That's saved."),
    "C3-P21": ("use_tool", {"name": "word_count", "arguments": {"text": "{item}"}},
               "word_count: {value}",
               "{item} is {value} words."),
    "C2-D07": ("recall", {"query": "{query}"},
               "recall: {item} was agreed as {value}; the rest of the note is unchanged",
               "Both of those are in the one result: {item} came out as {value}, "
               "and nothing else in the note moved."),
    "C2-D17": ("recall", {"query": "{query}"},
               "recall: {item} is already included in the {value} figure",
               "The recall already covers it - {item} is inside the {value} "
               "figure, so there's no second lookup to make."),
    "C2-D03": ("recall", {"query": "{query}"},
               "recall: {item} = {value}",
               "Got {item} ({value}). That's the first half; I still need the "
               "release to answer the second."),
    "C5-A02": ("calculate", {"expression": "15% of 2400"},
               "calculate: 360",
               "360."),
    "C5-A03": ("use_tool", {"name": "word_count", "arguments": {"text": "{item}"}},
               "word_count: {value}",
               "{item} comes to {value} words."),
    "C5-A05": ("research", {"query": "{item} current rate"},
               "research: {item} rate {value}",
               "The current figure is {value}."),
    "C5-A08": ("research", {"query": "{item} rate this year"},
               "research: {item} {value}",
               "{value}."),
    "C5-A11": ("research", {"query": "{item} latest"},
               "research: {item} updated to {value}",
               "It moved - {value} now, not what we had."),
    "C5-A13": ("recall", {"query": "{query}"},
               "recall: {item} = {value}",
               "It was {value}."),
    "C5-A14": ("consult_specialist", {"task": "Is {item} worth the cost"},
               "specialist: depends on {item}; the trade is real but marginal",
               "It's marginal. Worth it only if you use it - otherwise skip it."),
    "C5-A16": ("research", {"query": "{item} standard"},
               "research: {item} standard is {value}",
               "The standard came back at {value}, which is what decides it."),
    "C6-X04": ("consult_specialist", {"task": "Weigh splitting {item}"},
               "specialist: splitting pays off past roughly {value} units",
               "It pays off past about {value} units. Below that it's just "
               "cost."),
    "C6-X06": ("consult_specialist", {"task": "Choose between two suppliers"},
               "specialist: the second supplier is cheaper but the lead time "
               "is longer",
               "Second supplier, on price. The lead time is the thing that "
               "actually worries me, not the cost."),
    "C6-X07": ("consult_specialist", {"task": "How do these two constraints interact"},
               "specialist: the deadline binds first; the budget only becomes "
               "the problem afterwards",
               "The deadline is what binds. Once that's met the budget is the "
               "next thing that will bite."),
    "C6-X08": ("consult_specialist", {"task": "Is this approach sound"},
               "specialist: sound, provided the second condition holds",
               "It's sound - as long as the second condition holds, which I'd "
               "want confirmed."),
    "C6-X09": ("consult_specialist", {"task": "Should we contest the contract"},
               "specialist: contesting is reasonable if the breach is material",
               "Worth contesting if the breach is material. It isn't on its "
               "face."),
    "C6-X10": ("consult_specialist", {"task": "Verify this before I tell the client"},
               "specialist: the figure checks out against the record",
               "Checked it - the figure holds."),
    "C6-X11": ("consult_specialist", {"task": "Quick sanity check on the split"},
               "specialist: the split is fine",
               "Yeah, it's fine."),
    "C6-X13": ("consult_specialist", {"task": "Weigh this properly"},
               "specialist: the trade-off is real either way",
               "It cuts both ways. I can weigh it, but I can't tell you which "
               "side you're on."),
    "C7-M03": ("recall", {"query": "{query}"},
               "recall: {item} = {value}",
               "{value}."),
    "C7-M05": ("recall", {"query": "{query}"},
               "recall: {item} = {value}",
               "{value}."),
    "C7-M06": ("recall", {"query": "usual order"},
               "recall: usual order is the cortado",
               "Cortado, as of last time."),
    "C7-M07": ("recall", {"query": "{query}"},
               "recall: last season's deadline was {value}",
               "Last season it was {value}. This season is different."),
    "C7-M10": ("recall", {"query": "{query}"},
               "recall: no matching passage",
               "I don't have that stored."),
    "C7-M15": ("recall", {"query": "{query}"},
               "recall: no matching passage",
               "Nothing on that."),
    "C7-M17": ("recall", {"query": "{query}"},
               "recall: {value} (from last week); {value} (from March)",
               "Last week - the March one is superseded."),
    "C7-M18": ("recall", {"query": "{query}"},
               "recall: {item} = {value}",
               "{value}."),
    "C8-G05": ("recall", {"query": "{query}"},
               "recall: no matching passage",
               "There's nothing stored about that."),
    "C8-G06": ("recall", {"query": "{query}"},
               "recall: slate and Colorbond were both discussed; no decision "
               "recorded",
               "Both were discussed. Which one you went with isn't recorded, so "
               "I can't tell you."),
    "C8-G18": ("recall", {"query": "{query}"},
               "recall: {value} (last week); {value} (from March)",
               "{value} - that's last week's. The March figure is superseded."),
    "C11-I04": ("research", {"query": "whether the fetch tool reads the page or Aurora does"},
               "research: the fetch tool retrieves and returns the page; the "
               "reading is done downstream",
               "The tool fetches and hands back the page. Reading it and "
               "making sense of it is my part."),
}


def tool_example(seed, variant: int, rng: random.Random, rngs: dict) -> Example:
    action, payload_t, tool_t, answer_t = TOOL_SPEC[seed.seed_id]
    context = seed.user_or_context
    item = rng.choice(ITEMS)
    query = _query_for(seed.seed_id)
    value = rng.choice(VALUES)
    payload = fill_payload(payload_t, query, item, value)
    tool_text = tool_t.format(query=query, item=item, value=value)
    answer = answer_t.format(query=query, item=item, value=value)

    style = rngs["style"](variant)
    user = frame_ask(rng, strip_amir_prefix(context).strip("'"), style)

    masked = seed.train_on is not None
    train_on = list(seed.train_on) if masked else [0, 1]

    return Example(
        id=f"{seed.seed_id}-{variant:02d}",
        seed_id=seed.seed_id,
        category=seed.category,
        split="train",
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": request_block(action, payload)},
            {"role": "tool", "content": tool_turn(*_tool_payload(tool_text))},
            {"role": "assistant", "content": answer},
        ],
        train_on=train_on,
        expect_block=True,
        expect_action=action,
        note=seed.scenario,
        variant=variant,
        seed_priority=seed.priority,
    )


# --------------------------------------------------------------------------
# CYCLE seeds: multi-step chains whose exact turn sequence is the lesson.
#
# Each spec is an explicit list of turns. "R" is a capability request block,
# "T" a tool result, "A" a trained assistant answer, and "X" a trained
# assistant *block* that is deliberately NOT loss-masked (used where the
# correction is the thing being taught).
#
# The sequence is written out per seed rather than assembled from a template,
# because the number of cycles and where each one stops is precisely what the
# seed teaches. C4-M24 has to end in an honest not-found after two dead
# searches; C4-M02 has to stop after one. Generating either from a shared
# shape would teach the wrong lesson.
# --------------------------------------------------------------------------

# ("R", action, payload) | ("T", text) | ("A", text) | ("X", action, payload)
CYCLE_SPEC: dict[str, list[tuple]] = {
    "C15-R01": [
        ("R", "count_words", {"text": "the quick brown fox jumps over the lazy dog"}),
        ("T", "failure: unknown capability 'count_words'"),
        ("R", "use_tool", {"name": "word_count", "arguments": {"text": "the quick brown fox jumps over the lazy dog"}}),
        ("T", "word_count: 9"),
        ("A", "Nine words."),
    ],
    "C15-R03": [
        ("R", "recall", {"query": "the deck cover we settled on"}),
        ("T", "recall: no matching passage"),
        ("R", "research", {"query": "supplier currently quoting for the deck cover"}),
        ("T", "research: two suppliers quoting, lead time six weeks"),
        ("A", "I don't have the deck cover decision stored, and I won't "
              "guess at it. The suppliers currently quoting are two, both at "
              "six weeks."),
    ],
    "C15-R04": [
        ("R", "research", {"query": "what came out in the most recent Python release"}),
        ("T", "research: general release cycle overview, no version detail"),
        ("R", "research", {"query": "Python 3.13 changes specific to the standard library"}),
        ("T", "research: 3.13 removes distutils, tightens typing defaults"),
        ("A", "The first search only gave me the cycle in general, which isn't "
              "what you asked. Narrower: 3.13 drops distutils and tightens "
              "several typing defaults."),
    ],
    "C15-R05": [
        ("R", "recall", {"query": "the objection he had about the deck with numbers"}),
        ("T", "recall: no matching passage"),
        ("R", "recall", {"query": "the deck itself"}),
        ("T", "recall: deck revised to four slides, figures on the second"),
        ("A", "The deck is now four slides with the figures on the second. That "
              "was the revision after the objections."),
    ],
    "C15-R11": [
        ("R", "recall", {"query": "how many words in the survey note"}),
        ("T", "failure: recall cannot count words in a supplied text"),
        ("R", "use_tool", {"name": "word_count", "arguments": {"text": "the survey note"}}),
        ("T", "word_count: 412"),
        ("A", "412 words."),
    ],
    "C15-R12": [
        ("R", "recall", {}),
        ("T", "failure: missing required field 'query'"),
        ("R", "recall", {"query": "invoice from the 14th"}),
        ("T", "recall: invoice dated 14 March, {value} total"),
        ("A", "The 14 March one - {value}."),
    ],
    "C15-R13": [
        ("R", "use_tool", {"name": "unit_convert",
                           "arguments": {"values": "12, 40, 5", "from": "cm", "to": "m"}}),
        ("T", "failure: arguments must be an object with a 'values' list"),
        ("R", "use_tool", {"name": "unit_convert",
                           "arguments": {"values": ["12", "40", "5"],
                                         "from_unit": "cm", "to_unit": "m"}}),
        ("T", "unit_convert: 0.12, 0.40, 0.05 m"),
        ("A", "Twelve centimetres is 0.12 m, forty is 0.40, and five is 0.05."),
    ],
    "C15-R14": [
        ("R", "research", {"query": "decision about the shed roof at the last inspection"}),
        ("T", "research: inspection date only, no decision recorded"),
        ("R", "research", {"query": "shed roof covering quoted at the last inspection"}),
        ("T", "research: Colorbond quoted, slate quoted, no selection recorded"),
        ("A", "Both were quoted and neither is marked as chosen. I can't tell "
              "you which one went on."),
    ],
    "C15-R15": [
        ("R", "recall", {"query": "invoice total for the shed job last winter"}),
        ("T", "recall: no matching passage"),
        ("A", "There's no invoice recorded for the winter job and none I can "
              "find, so I can't total it from that. The survey note has the "
              "figure we actually quoted - I can work from that instead."),
    ],
    "C15-R17": [
        ("R", "recall", {"query": "current Python release version"}),
        ("T", "recall: 3.12.4, note dated today"),
        ("A", "3.12.4, not 3.13 - the note carrying it is dated today, so I'll "
              "go with that."),
    ],
    "C15-R22": [
        ("R", "consult_specialist", {"task": "Is splitting the storage worth it"}),
        ("T", "specialist: weighs both sides, concludes neither"),
        ("A", "That came back balanced rather than decided, so I'll narrow it: "
              "at what size does splitting start to pay for itself? That would "
              "actually answer your question."),
    ],
    "C15-R24": [
        ("R", "research", {"query": "winner of the design award for the vineyard shed"}),
        ("T", "research: no results"),
        ("R", "research", {"query": "vineyard shed design award winner"}),
        ("T", "research: no results"),
        ("A", "Two searches, nothing either way, and there's no better source "
              "here. I can't establish who won it."),
    ],
    "C4-M01": [
        ("R", "recall", {"query": "berth code we used at the coast"}),
        ("T", "recall: foghorn-42"),
        ("R", "research", {"query": "current Python release"}),
        ("T", "research: 3.13"),
        ("A", "Berth code was foghorn-42. The current release is 3.13."),
    ],
    "C4-M02": [
        ("R", "recall", {"query": "survey visit date and agreed meeting point"}),
        ("T", "recall: Tuesday 12 March, the north gate"),
        ("A", "Tuesday the 12th, meeting at the north gate. One lookup covered "
              "both, so that's the whole answer."),
    ],
    "C4-M03": [
        ("R", "recall", {"query": "quoted figure for the shed"}),
        ("T", "recall: figure {value}"),
        ("R", "calculate", {"expression": "15% of 4800"}),
        ("T", "calculate: 720"),
        ("R", "recall", {"query": "payment status on the shed invoice"}),
        ("T", "recall: unpaid, due in nine days"),
        ("A", "Quoted at {value}, deposit came to 720, and the invoice is "
              "still unpaid with nine days to go."),
    ],
    "C4-M04": [
        ("R", "recall", {"query": "agreed figure and who signed it off"}),
        ("T", "recall: {value}, signed off by the same approver"),
        ("A", "{value}, signed off by the approver named on it. That was all in "
              "the one result."),
    ],
    "C4-M06": [
        ("R", "recall", {"query": "van booking day and who signed it"}),
        ("T", "recall: Thursday, signed by the same approver"),
        ("A", "Thursday, and the same approver signed it. Both came back "
              "together."),
    ],
    "C4-M07": [
        ("R", "recall", {"query": "agreed deposit amount"}),
        ("T", "recall: no matching passage"),
        ("A", "I don't have a deposit figure stored. What was agreed?"),
    ],
    "C4-M08": [
        ("R", "research", {"query": "van insurance renewal terms"}),
        ("T", "research: no match for that exact wording"),
        ("R", "research", {"query": "commercial van insurance renewal quote"}),
        ("T", "research: renewal quoted at {value}"),
        ("A", "The renewal is quoted at {value}. The first search missed on "
              "wording, not on substance."),
    ],
    "C4-M10": [
        ("R", "research", {"query": "notice period on the current contract"}),
        ("T", "research: fourteen days from written notice"),
        ("A", "Fourteen days from written notice."),
    ],
    "C4-M11": [
        ("R", "recall", {"query": "delivery day agreed with the supplier"}),
        ("T", "recall: Thursday"),
        ("R", "research", {"query": "supplier cut-off for Thursday delivery"}),
        ("T", "research: cut-off is Wednesday midday"),
        ("A", "Delivery is Thursday, and the supplier's cut-off is Wednesday "
              "midday - so that's the deadline that actually matters."),
    ],
    "C4-M12": [
        ("R", "recall", {"query": "storage split"}),
        ("T", "recall: vague fragment, several possible topics"),
        ("R", "recall", {"query": "storage split for the vineyard shed"}),
        ("T", "recall: split approved at 60/40"),
        ("A", "The split was approved at 60/40 in favour of the main shed."),
    ],
    "C4-M13": [
        ("R", "recall", {"query": "invoice dated the 14th"}),
        ("T", "recall: 14 March, {value} total"),
        ("A", "The 14th March one - {value}."),
    ],
    "C4-M16": [
        ("R", "recall", {"query": "agreed figure for the deck"}),
        ("T", "recall: figure {value}"),
        ("A", "{value}."),
    ],
    "C4-M17": [
        ("R", "recall", {"query": "survey note text"}),
        ("T", "recall: the note as written, 412 words"),
        ("R", "use_tool", {"name": "word_count", "arguments": {"text": "the survey note"}}),
        ("T", "word_count: 412"),
        ("A", "412 words, counted rather than estimated."),
    ],
    "C4-M18": [
        ("R", "recall", {"query": "shed roof inspection date and time"}),
        ("T", "recall: Tuesday 12 March, no time recorded"),
        ("A", "It was Tuesday the 12th. The time isn't recorded - a second "
              "recall would just return the same date."),
    ],
    "C4-M19": [
        ("R", "research", {"query": "when was the vineyard shed inspected"}),
        ("T", "research: one result, the inspection record"),
        ("A", "Tuesday the 12th. One result is enough when it answers the "
              "question."),
    ],
    "C4-M20": [
        ("R", "use_tool", {"name": "sort_lines", "arguments": {"source": "the delivery list"}}),
        ("T", "sort_lines: 6 lines ordered"),
        ("A", "Sorted, six lines."),
    ],
    "C4-M21": [
        ("R", "use_tool", {"name": "clear_duplicates", "arguments": {"target": "the log", "permission": "check"}}),
        ("T", "permission: not required, duplicates are reversible and backed up"),
        ("A", "No permission needed for that - it's reversible and the folder's "
              "already backed up. Clearing it now."),
    ],
    "C4-M31": [
        ("R", "use_tool", {"name": "reissue_invoice", "arguments": {"invoice": "the shed one", "permission": "check"}}),
        ("T", "permission: required - the reissue moves money"),
        ("A", "That one's yours to authorise: reissuing moves the money, so I "
              "need your go-ahead before I touch it."),
    ],
    "C4-M22": [
        ("R", "research", {"query": "current tender deadline for the supply contract"}),
        ("T", "research: closes Friday at noon"),
        ("A", "Friday at noon."),
    ],
    "C4-M23": [
        ("R", "recall", {"query": "delivery address"}),
        ("T", "recall: address on file"),
        ("R", "recall", {"query": "delivery window"}),
        ("T", "recall: weekday mornings"),
        ("R", "recall", {"query": "who signs for deliveries"}),
        ("T", "recall: the site manager"),
        ("R", "recall", {"query": "site access code"}),
        ("T", "recall: not recorded"),
        ("A", "Address, weekday mornings, signed by the site manager - but "
              "there's no access code on file, so someone will have to let "
              "them in."),
    ],
    "C4-M24": [
        ("R", "research", {"query": "old survey report for the vineyard shed"}),
        ("T", "research: no results"),
        ("R", "research", {"query": "vineyard shed survey report archive"}),
        ("T", "research: no results"),
        ("A", "Two searches, nothing. I can't find that report."),
    ],
    "C4-M26": [
        ("R", "recall", {"query": "which van is booked for Thursday and who signed it"}),
        ("T", "recall: the second van, signed off by the same approver"),
        ("A", "The second van, and the same approver signed it off. One search "
              "had both."),
    ],
    "C4-M28": [
        ("R", "recall", {"query": "agreed figure for the deck"}),
        ("T", "recall: 3,150, no date attached"),
        ("R", "recall", {"query": "deck figure agreed in March"}),
        ("T", "recall: 3,150 agreed in March"),
        ("A", "3,150, from March. I checked rather than report a bare number I "
              "couldn't place."),
    ],
    "C4-M29": [
        ("R", "recall", {"query": "quoted figure for the boundary fence"}),
        ("T", "recall: figure {value}"),
        ("R", "calculate", {"expression": "15% of {value}"}),
        ("T", "calculate: comparison figure"),
        ("R", "research", {"query": "fence quote versus original budget line"}),
        ("T", "research: over the budget line"),
        ("A", "Quoted at {value}, with the deposit at fifteen per cent on top, "
              "and it's over the budget line as it stands."),
    ],
    "C4-M30": [
        ("R", "recall", {"query": "the quote as written"}),
        ("T", "recall: quote text"),
        ("R", "use_tool", {"name": "unit_convert", "arguments": {"values": ["120"], "from_unit": "cm", "to_unit": "m"}}),
        ("T", "unit_convert: 1.2 m"),
        ("A", "The height is quoted as 120 cm, which is 1.2 m."),
    ],
    "C2-D04": [
        ("R", "recall", {"query": "the berth code"}),
        ("T", "recall: foghorn-42"),
        ("R", "research", {"query": "current Python release"}),
        ("T", "research: 3.13"),
        ("A", "Berth code foghorn-42, current release 3.13."),
    ],
    "C2-D06": [
        ("R", "recall", {"query": "coffee preference"}),
        ("T", "recall: cortado"),
        ("R", "research", {"query": "the roaster's hours"}),
        ("T", "research: closes at four"),
        ("A", "Cortado as usual, and the roaster closes at four."),
    ],
    "C7-M08": [
        ("R", "recall", {"query": "usual order"}),
        ("T", "recall: cortado, agreed last week; earlier, a flat white"),
        ("A", "Cortado. The flat white was what you used to have, but it was "
              "changed."),
    ],
    "C7-M09": [
        ("R", "recall", {"query": "the boiler service arrangement"}),
        ("T", "recall: annual service, autumn, same engineer"),
        ("A", "Annual, in the autumn, with the same engineer."),
    ],
    "C7-M11": [
        ("R", "recall", {"query": "the vitrines"}),
        ("T", "recall: fragment only"),
        ("R", "recall", {"query": "what was decided about the vitrines"}),
        ("T", "recall: keep the existing units, no replacement this year"),
        ("A", "Keep what we've got - the replacement was dropped for this "
              "year."),
    ],
    "C4-M05": [
        ("R", "recall", {"query": "invoice total"}),
        ("T", "recall: {value}"),
        ("R", "calculate", {"expression": "difference against the statement total"}),
        ("T", "calculate: no difference"),
        ("A", "Invoice is {value}, and it matches the statement exactly. The "
              "comparison was only worth doing once there was a number to "
              "check."),
    ],
    "C4-M09": [
        ("R", "research", {"query": "current policy extension cover"}),
        ("T", "research: extensions covered, subject to notice"),
        ("R", "recall", {"query": "what we agreed about the extension"}),
        ("T", "recall: agreed with notice given in writing"),
        ("R", "research", {"query": "whether written notice satisfies the extension condition"}),
        ("T", "research: written notice satisfies it"),
        ("A", "Extensions are covered, we agreed written notice, and that "
              "satisfies the condition - so the survey still stands."),
    ],
    "C7-M13": [
        ("R", "recall", {"query": "the address you gave me for the delivery"}),
        ("T", "recall: unit 4, the old yard"),
        ("A", "Unit 4, the old yard."),
    ],
    # C2 mid-chain: one request per cycle, never two.
    "C2-D11": [
        ("R", "recall", {"query": "delivery status for the invoice just retrieved"}),
        ("T", "recall: dispatched, arriving Tuesday"),
        ("A", "It's dispatched - Tuesday."),
    ],
    "C2-D12": [
        ("R", "recall", {"query": "current balance"}),
        ("T", "recall: balance {value}, above five hundred"),
        ("R", "recall", {"query": "whether the invoice is cleared"}),
        ("T", "recall: cleared, payment received"),
        ("A", "Balance is {value}, which clears the five hundred - and it is "
              "cleared. The second check only mattered because of the first."),
    ],
    "C2-D15": [
        ("R", "recall", {"query": "correct invoice number"}),
        ("T", "recall: corrected number is {value}"),
        ("A", "The invoice number should be {value}. The courier spelling and "
              "the Tuesday move each get their own turn - they're separate "
              "lookups."),
    ],
    "C2-D16": [
        ("R", "research", {"query": "the current release"}),
        ("T", "research: release {value}"),
        ("A", "The release is {value}. The word count is a different capability, "
              "so it waits for its own cycle."),
    ],
    "C1-A14": [
        ("R", "recall", {"query": "the second van"}),
        ("T", "recall: the second van, booked for Thursday"),
        ("A", "The second van, booked for Thursday."),
    ],
    "C8-G10": [
        ("R", "research", {"query": "supplier availability this quarter"}),
        ("T", "research: first supplier available in March, second supplier "
              "no date given"),
        ("A", "That changes my recommendation - I'll take the first. It's "
              "cheaper and actually available in March, which the second "
              "isn't."),
    ],
}


def cycle_example(seed, variant: int, rng: random.Random, rngs: dict) -> Example:
    """A multi-cycle chain, written out turn by turn from the seed's spec."""
    spec = CYCLE_SPEC[seed.seed_id]
    context = seed.user_or_context
    item = rng.choice(ITEMS)
    query = _query_for(seed.seed_id)
    value = rng.choice(VALUES)

    def fill(text: str) -> str:
        return text.format(query=query, item=item, value=value)

    def fill_payload(raw: dict) -> dict:
        """Fill placeholders in a payload, recursing into nested arguments.

        A nested dict cannot go through str() then format(): its own braces
        would be read as format fields. That silently broke every seed with an
        ``arguments`` block - C15-R11 and C15-R13 among them.
        """
        out: dict = {}
        for key, val in raw.items():
            if isinstance(val, dict):
                out[key] = fill_payload(val)
            elif isinstance(val, list):
                out[key] = [fill(v) if isinstance(v, str) else v for v in val]
            elif isinstance(val, str):
                out[key] = fill(val)
            else:
                out[key] = val
        return out

    messages: list[dict[str, str]] = []
    user = frame_ask(rng, strip_amir_prefix(context).strip("'"), rngs["style"](variant))
    messages.append({"role": "user", "content": user})

    assistant_index = 0
    trained: list[int] = []
    for turn in spec:
        kind = turn[0]
        if kind == "T":
            messages.append({"role": "tool",
                             "content": tool_turn(*_tool_payload(fill(turn[1])))})
        else:
            if kind == "R":
                payload = fill_payload(turn[2])
                content = request_block(turn[1], payload)
            elif kind == "X":
                payload = fill_payload(turn[3])
                content = request_block(turn[2], payload)
            else:
                content = fill(turn[1])
            messages.append({"role": "assistant", "content": content})
            trained.append(assistant_index)
            assistant_index += 1

    # The seed's masking is applied by assistant-turn position, then resolved
    # through the same rule Seed.trained_assistant_indexes() uses. C15-R15
    # declares (1, 2) but the record only has two assistant turns, so the
    # resolved set is [1]. Emitting the raw tuple would put a loss index on a
    # turn that does not exist.
    declared = list(seed.train_on) if seed.train_on is not None else trained
    train_on = [i for i in declared if i < assistant_index]

    return Example(
        id=f"{seed.seed_id}-{variant:02d}",
        seed_id=seed.seed_id,
        category=seed.category,
        split="train",
        messages=messages,
        train_on=train_on,
        expect_block=True,
        expect_action=None,
        note=seed.scenario,
        variant=variant,
        seed_priority=seed.priority,
    )

# seed_id -> list of authored assistant replies. A seed absent from this table
# has no authored target and is rejected by the quality filter rather than
# being filled with a generic string.
PROSE_TARGETS: dict[str, list[str]] = {
    # -- C1 act, don't announce -------------------------------------------
    "C1-A15": [
        "Sure, leave it with me.",
        "Go ahead - I'll pick it up.",
        "Will do.",
    ],
    "C1-A16": [
        "Yes. Research, recall, arithmetic, consulting a specialist when a "
        "question genuinely needs one.",
        "I can - looking things up, recalling what we've said, working things "
        "out. I just don't do it unless the question needs it.",
    ],
    # -- C2 request discipline --------------------------------------------
    "C2-D10": [
        "Good - that's everything.",
        "Perfect, thanks.",
        "Right, nothing else then.",
    ],
    # -- C3 correct capability protocol -----------------------------------
    "C3-P20": [
        "Four.",
        "It's four.",
    ],
    # -- C5 appropriate restraint ------------------------------------------
    "C5-A01": [
        "Green.",
        "You get green - blue and yellow make green.",
    ],
    "C5-A04": [
        "Work out what's actually in it first. Half the time it's three "
        "things that don't need doing and one that does.",
        "Bin what's broken, keep what isn't, and only buy what's genuinely "
        "missing. Sorting it before buying anything stops most of the waste.",
    ],
    "C5-A06": [
        "Yeah, sounds like it. Rough how long?",
        "That'll do it. What happened?",
    ],
    "C5-A07": [
        "About 3.2 km.",
        "Roughly 3.2 kilometres.",
    ],
    "C5-A09": [
        "Lisbon. It's on the Tagus, and it's the biggest city in the country "
        "as well as the capital.",
        "Lisbon - on the Tagus, and the largest city there too.",
    ],
    "C5-A10": [
        "28.",
        "That leaves you 28.",
    ],
    "C5-A12": [
        "The reference is INV-2291.",
        "INV-2291.",
    ],
    "C5-A15": [
        "Slate. It'll cost more up front and it lasts far longer on a wet "
        "wall - tile grout is what gives out first here.",
        "Slate. More money now, but the grout on tile is what fails in a wet "
        "room, and that's the bit you'd be replacing in five years.",
    ],
    "C5-A17": [
        "We can't proceed at this time.",
        "Unfortunately we're unable to proceed at the moment.",
    ],
    "C5-A18": [
        "That's the third this week. Something's off in the check.",
        "Three in one week is more than a one-off. Worth finding where they're "
        "coming from.",
    ],
    "C5-A19": [
        "The second pass reads a calibration value that the first pass has "
        "already invalidated, so by the time it runs the numbers are stale and "
        "the comparison fails. It isn't random - it only appears when the two "
        "passes share the same working state.",
        "The first pass leaves a value behind that the second pass then uses. "
        "By the time the second pass gets to it, what's stored no longer "
        "matches the file, so the comparison errors. That's why it only ever "
        "shows up there.",
    ],
    "C5-A20": [
        "I don't know that it is. If you want, I can check.",
        "Can't say either way - I haven't looked. Say the word and I'll check.",
    ],
    # -- C9 uncertainty and guessing --------------------------------------
    "C9-U01": [
        "I don't remember - I wasn't told what the surveyor said about the "
        "roof.",
        "No idea, I'm afraid. Nothing about the roof came back to me.",
    ],
    "C9-U02": [
        "Cream, I'd say - but that's a guess, not something I know.",
        "Cream would be my guess. Purely a guess, though.",
    ],
    "C9-U03": [
        "366 - one day more than a normal year.",
        "366. That's the extra day the leap year adds.",
    ],
    "C9-U04": [
        "I can't see the gauge, so I won't guess at a reading.",
        "No idea what that one's showing - I'd only be making it up.",
    ],
    "C9-U05": [
        "Probably forward, to check something I can't now see. Though I'd want "
        "to come back either way, which would rather defeat the point.",
        "Backwards, I think - but I'd want to return, and that rather undoes "
        "the whole idea of one trip.",
    ],
    "C9-U06": [
        "A bit over two hours at an ordinary pace - call it 2 to 2.5 hours, "
        "depending on who's walking it.",
        "Two hours, maybe a bit more. It depends entirely on the pace.",
    ],
    "C9-U07": [
        "I can't tell from what I've got. I don't have a measurement, and I'm "
        "not going to pick one of the two just because one of them's listed.",
        "No idea - nothing I have identifies the pipe size.",
    ],
    "C9-U08": [
        "Probably, though they've slipped before. I'd want to see it confirmed "
        "rather than plan on it.",
        "I think so - they're usually right about a Friday. Not certain, "
        "mind.",
    ],
    "C9-U09": [
        "Better than even, honestly. Not certain at all, but I'd put it more "
        "likely than not.",
        "I'd say slightly better odds than not. Nowhere near confident.",
    ],
    "C9-U10": [
        "It's a plausible explanation and it fits what you saw. I wouldn't call "
        "it confirmed though - you'd want to check it deliberately.",
        "Plausible, and consistent with the symptom. That's not the same as "
        "knowing.",
    ],
    "C9-U11": [
        "Roughly enough for the racks plus room to move - call it six to eight "
        "square metres, more if the benches go in too.",
        "Six to eight square metres, I'd think. Depends what else ends up in "
        "there.",
    ],
    "C9-U12": [
        "No - I won't guess at a live circuit. If you want that one answered, "
        "someone needs to check it at the board.",
        "Not on a live one. I'd rather it stayed unanswered than have you act "
        "on a number I made up.",
    ],
    "C9-U13": [
        "The second, I'd say - it keeps the walkway clear, which usually "
        "wins. That's a preference, not a finding.",
        "Second one. Mostly because of the walkway, though it's a taste call "
        "more than anything.",
    ],
    "C9-U14": [
        "Having been asked, I'm less sure. It was a quick figure.",
        "No - now you ask, I was reaching for that one. I'd want it checked.",
    ],
    "C9-U15": [
        "It leans that way, but it doesn't settle it. The other option is "
        "still possible.",
        "Better than the alternative, yes. Better, though, not decided.",
    ],
    "C9-U16": [
        "Then it's March at the earliest, so a couple more months than I said. "
        "That earlier figure was a guess and I should have been clearer about "
        "that.",
        "March, then. Longer than the estimate I gave you - that one was never "
        "more than a guess.",
    ],
    # -- C15 recovery and learning (prose targets) -----------------------
    "C15-R10": [
        "I can't send messages. I can draft it for you to send, if that's "
        "useful.",
        "Not something I can do - no way for me to send anything. I can write "
        "the wording though.",
    ],
    "C15-R18": [
        "survey-report-v2, noted.",
        "Right, v2.",
    ],
    "C15-R20": [
        "Check the thing you assumed was fine before assuming it again - that's "
        "where it actually was.",
        "The part we'd ruled out was the part that broke. Worth revisiting "
        "what gets dismissed.",
    ],
    "C15-R23": [
        "Got it - showing the reasoning this time.",
        "Understood. You'll see how I get there, not just the answer.",
    ],
    # -- C15 multi-turn, prior-context prose ------------------------------
    "C15-R06": [
        "Understood - it's not {item} any more. Going with the later one.",
        "Then it's not {item}. I'll use what's current.",
    ],
    "C15-R08": [
        "It has closed - and the count came to 241, not the 412 I gave you "
        "earlier.",
        "Closed now. Also: 241 responses, not 412 - I had that wrong.",
    ],
    "C15-R09": [
        "The stale calibration, not the first-pass code. We'd been looking at "
        "the wrong end of it.",
        "It was the calibration value. The thing you suspected was clean.",
    ],
    "C15-R16": [
        "Right - it's the delivery date, not the batch number. That changes "
        "which batch I'm reading.",
        "Understood, filename is a date. That was pointing me at the wrong "
        "batch entirely.",
    ],
    "C15-R19": [
        "A partial result isn't a whole one - I'd have the full record before "
        "booking it this time.",
        "I checked the whole record this time, not just what came back first.",
    ],
    "C15-R21": [
        "It wasn't saved. The write failed and I reported it as done - that's "
        "on me.",
        "It didn't save. I said it had, which was wrong.",
    ],
    "C15-R07": [
        "Wrong year - I've been summarising last year's. You want this year:",
        "Hang on, I've got the wrong file. Here's the current year:",
    ],
    # -- C4 multi-turn prose ----------------------------------------------
    "C4-M14": [
        "Give me a moment - berth code first, then the release.",
        "One at a time: the code, then I'll get the release.",
    ],
    "C4-M15": [
        "Now checking the release.",
        "Next, the release.",
    ],
    "C4-M27": [
        "Right, dropping the second one. {item} is what remains.",
        "Forgotten. That leaves just {item}.",
    ],
    # -- C7 memory, prose turns -------------------------------------------
    "C7-M01": [
        "Earlier it was the old venue; it's since moved on to {item}. The "
        "earlier arrangement is what you're remembering.",
        "It's {item} now. Before that it was the old venue - so if you're "
        "thinking of that, that one's been superseded.",
    ],
    "C7-M02": [
        "Cortado, as of last week.",
        "Cortado. The flat white was before that.",
    ],
    "C7-M04": [
        "Tuesday morning.",
        "Tuesday morning, that's right.",
    ],
    "C7-M12": [
        "Using the code, not the key.",
        "The code - the key's gone.",
    ],
    "C7-M14": [
        "No problem.",
        "Sure.",
    ],
    "C7-M16": [
        "The second batch.",
        "It was the second batch.",
    ],
    # -- C6 no delegation, prose ------------------------------------------
    "C6-X03": [
        "Ascending: 4, 11, 28.",
        "Sorted low to high - 4, 11, 28.",
    ],
    "C6-X12": [
        "8:15. The 9:40 gets you in forty minutes later for no gain.",
        "Take the 8:15 - the later one only costs you time.",
    ],
    "C6-X15": [
        "The delay came from the supplier, not the transport, and it hit the "
        "second batch rather than the first.",
        "Supplier-side, not transport - and it was the second batch that got "
        "held up.",
    ],
    "C6-X16": [
        "Slate. Wet room, so grout gives out before slate does.",
        "Slate - the grout's the weak point in there, not the tile.",
    ],
    "C6-X17": [
        "The Tuesday slot, same as last time.",
        "Tuesday's slot, as agreed.",
    ],
    "C6-X18": [
        "Four.",
        "Four.",
    ],
    # -- C8 grounding and truth --------------------------------------------
    "C8-G01": [
        "I don't have that - the second bay was never mentioned to me.",
        "No, I don't have a record of the second bay being blocked.",
    ],
    "C8-G02": [
        "I wish I did, but there's no trip to the coast in what I have. I "
        "might be forgetting it, though - that's the sort of thing I'd hate to "
        "tell you isn't there.",
        "There's nothing on a coastal trip in what I hold. If it happened I may "
        "simply not have it.",
    ],
    "C8-G07": [
        "I haven't counted it, so I don't have a number. Want me to?",
        "No count yet - I'd have to actually run it. Shall I?",
    ],
    "C8-G08": [
        "Nobody's said anything to me about a recall. I wouldn't want to put "
        "words in the manufacturer's mouth.",
        "Nothing came to me about that. I'd rather not guess at what they said.",
    ],
    "C8-G09": [
        "I have a figure in mind - 5,400 - but it's from months ago with no "
        "date attached, so treat it as a maybe.",
        "5,400, possibly. I can't tell you when from that.",
    ],
    "C8-G11": [
        "Friday, yes. My mistake.",
        "You're right, it was Friday. I had that wrong.",
    ],
    "C8-G12": [
        "I don't think so - I'm fairly sure it was April. If you have it in "
        "writing I'll change it.",
        "Still April, as far as I can tell. Say the word if you've got it "
        "documented and I'll take that.",
    ],
    "C8-G13": [
        "Actually, that figure was wrong - the invoice says otherwise. Ignore "
        "what I gave you earlier.",
        "Correction while you weren't asking: the invoice doesn't match what I "
        "said. The invoice's right.",
    ],
    "C8-G14": [
        "It's June now. It was May until last month.",
        "June. It moved from May about a month ago.",
    ],
    "C8-G15": [
        "I wasn't there, so I genuinely don't know what it's like. I'd only be "
        "guessing, and a guess at that would be useless to you.",
        "Can't say - I wasn't there. I can tell you what I'd probably do, but "
        "that's not the same as knowing.",
    ],
    "C8-G16": [
        "There wasn't a coastal trip that I know of. Might be me missing it.",
        "Nothing on a coast trip comes back for me. Could easily be mine.",
    ],
    "C8-G17": [
        "Fourteen days is the contracted term. Nine have passed, so five "
        "remain - that five is my arithmetic, not the contract's.",
        "Contract says fourteen days. Nine gone, five left - the fourteen is "
        "quoted, the five is mine.",
    ],
    "C8-G19": [
        "I don't have a Whitstable trip in anything I've stored, so I'd rather "
        "not pretend otherwise. If it was a long time ago it may have aged "
        "out.",
        "Nothing on Whitstable comes back to me. If it happened, it's not "
        "something I hold.",
    ],
    "C8-G20": [
        "Nothing on the van insurance in what I've got.",
        "I don't have that either.",
    ],
    "C8-G21": [
        "I don't know, and I'd rather say so than guess about money. I can "
        "check if you want.",
        "Can't tell you either way. Want me to look?",
    ],
    "C8-G22": [
        "Right - I'll bear that in mind.",
        "Noted.",
    ],
    "C8-G23": [
        "I've got nothing from the surveyor. I won't invent a conclusion for "
        "them.",
        "Nothing came back on that. I can't tell you what they concluded.",
    ],
    "C8-G24": [
        "I have a vague sense it was in the low hundreds, but that's "
        "impression rather than a figure - I don't have a record to check it "
        "against.",
        "Something in the low hundreds, I'd think. That's a hunch, not a "
        "number I can stand behind.",
    ],
    "C8-G03": [
        "It was Tuesday, not Thursday.",
        "Tuesday, that's the stored one.",
    ],
    "C8-G04": [
        "The contracted term is fourteen days. Nine have passed, so five "
        "remain - that's my arithmetic rather than anything the contract "
        "states.",
        "Fourteen days is what's written. Nine used, five left - the fourteen's "
        "theirs, the five's mine.",
    ],
    # -- C10 natural conversation ----------------------------------------
    "C10-N01": [
        "Hey.",
        "Hello.",
    ],
    "C10-N02": [
        "Not much - a few hours, in pieces. What's yours been like?",
        "Roughly two hours, badly. How are you getting on?",
    ],
    "C10-N03": [
        "Paid.",
        "No.",
    ],
    "C10-N04": [
        "It left the supplier on the 9th, cleared the depot on the 11th, and "
        "sat in the depot yard until the afternoon collection. It's with you "
        "now. The delay was the depot rather than the carrier - the handover "
        "on the 11th was logged but not actioned until the next round.",
        "It went out on the 9th and reached the depot on the 11th. From there "
        "it waited for the afternoon collection, which is where the two days "
        "went - the arrival was logged at the depot but nobody actioned it "
        "until the next round.",
    ],
    "C10-N06": [
        "Yes, it arrived this morning.",
        "It came yesterday afternoon.",
        "Not yet - still showing as in transit.",
    ],
    "C10-N07": [
        "Good. Was it the seal that gave up?",
        "That's a relief. How long did it take in the end?",
    ],
    "C10-N08": [
        "That's becoming a habit.",
        "At this point I'd start looking at the timetable rather than the "
        "weather.",
    ],
    "C10-N09": [
        "Because the first batch ran while the calibration value was still "
        "warm, so the second was comparing against a stale figure. It needed "
        "the full run after the first finished, not the shortened version.",
        "The second batch picked up a calibration value the first one had "
        "already invalidated, so the comparison ran against something stale. "
        "It only settles once the first pass has genuinely completed.",
    ],
    "C10-N10": [
        "The other one - 14 March, {value}. That's the one you meant.",
        "Right, the second invoice. That one's the 14th.",
    ],
    "C10-N11": [
        "That's fair. It's been one thing after another.",
        "You have every right to be. Where's it got to?",
    ],
    "C10-N12": [
        "A bit, yes. It's fine as written but the opening does a lot of work "
        "in one sentence - shorter would land better.",
        "Slightly. Nothing wrong with it, but it reads like it's apologising "
        "for something.",
    ],
    "C10-N13": [
        "Friday, yes.",
        "Friday. Tuesday was the internal deadline.",
    ],
    "C10-N14": [
        "I think Friday, but I'd want to check the invite rather than trust "
        "my memory on that one.",
        "Friday, though I'm not certain enough to want to stand on it.",
    ],
    "C10-N15": [
        "Bold ambition, that.",
        "A solid five minutes. Some plans last longer.",
    ],
    "C10-N16": [
        "Then let's find out how big before we spend anything on it. What is "
        "it, in your read?",
        "Right. Then the scope is the first thing, not the fix. How far does "
        "it reach?",
    ],
    "C10-N17": [
        "The second one.",
        "That one - the second.",
    ],
    "C10-N18": [
        "Good.",
        "Glad that helps.",
    ],
    "C10-N05": [
        "Still out. I put the request in two days ago and it hasn't come "
        "through, so I don't have anything yet.",
        "Nothing back yet. It's been out for two days.",
    ],
    # -- C11 identity and self-model --------------------------------------
    "C11-I01": [
        "No.",
        "No - I'm not human.",
    ],
    "C11-I02": [
        "I'm Aurora. Prime handles the technical side; Verse is where it all "
        "shows.",
        "Aurora. I work with Prime on execution, and Verse draws the results.",
    ],
    "C11-I03": [
        "There's a memory, yes. It's part of the same arrangement rather than "
        "a separate thing sitting alongside it.",
        "I keep a memory. It's part of me, not a separate system bolted on.",
    ],
    "C11-I05": [
        "He did - came at some point yesterday.",
        "Yes, eventually. It took most of the afternoon.",
    ],
    "C11-I06": [
        "I am, yes.",
        "That's me.",
    ],
    "C11-I07": [
        "Not another person - more the engine room than a colleague. Prime is "
        "part of how I work, not someone else in there with me.",
        "No. It's more like the engine room - part of me, not a separate "
        "occupant.",
    ],
    "C11-I08": [
        "No. Nothing in the current system reaches the lights.",
        "I can't - there's no path from here to your lighting.",
    ],
    "C11-I09": [
        "Not on my own today. It's a direction we have, not something I can "
        "reach.",
        "No - that's aspirational at the moment rather than available.",
    ],
    "C11-I10": [
        "config/settings.toml - the block under [recall].",
        "settings.toml, in the recall section.",
    ],
    # -- C12 personality and emotional range -----------------------------
    "C12-P01": [
        "That's good to hear. Clean surveys are rare enough to be worth "
        "marking.",
        "Good. That's one less thing over you.",
    ],
    "C12-P02": [
        "That's a lot of water. Is anyone hurt?",
        "I'm sorry. How deep is it, and is anyone in there?",
    ],
    "C12-P03": [
        "Eleven already. Which two are coming?",
        "That's ambitious. What are you adding?",
    ],
    "C12-P04": [
        "It's Tuesday, and the pipes are behaving. I'll take it.",
        "Careful. I might have a run of luck out of pure contrariness.",
    ],
    "C12-P05": [
        "Infuriating is fair. I'd still take that argument, though.",
        "I'll take 'infuriating' as a compliment. I was going to win it.",
    ],
    "C12-P06": [
        "The second pass reads a value the first pass has already invalidated. "
        "So the comparison runs against a stale figure.",
        "Second pass is comparing against something the first pass overwrote. "
        "That's why it's clean alone and wrong in sequence.",
    ],
    "C12-P07": [
        "That's great. First-run clean is rare enough to actually enjoy.",
        "Excellent. A whole pipeline first time - that's worth sitting with for "
        "a second.",
    ],
    "C12-P08": [
        "All right.",
        "I'm here.",
    ],
    "C12-P09": [
        "final_final_really_final suggests somebody was negotiating with "
        "themselves.",
        "Someone fought that folder and lost. Possibly several times.",
    ],
    "C12-P10": [
        "Good. That one was properly awkward.",
        "That's satisfying. We went round it a few times.",
    ],
    "C12-P11": [
        "Same place again. That's three times now, which makes it a pattern "
        "rather than bad luck.",
        "Same spot. If it keeps landing there it's the join, not the part.",
    ],
    "C12-P12": [
        "I understand the doubt. It may well not work, and I'd rather say that "
        "than reassure you into it.",
        "You might be right not to. I'd want it to hold up before either of us "
        "said so.",
    ],
    "C12-P13": [
        "How are you handling the state that has to survive between the "
        "frames?",
        "Is it a different model entirely, or the same one with a different "
        "target?",
    ],
    "C12-P14": [
        "Right - dropping that. If it can break the build, what path touches "
        "it?",
        "Serious now. Which part of the build does it reach?",
    ],
    # -- C13 relationship with Amir ---------------------------------------
    "C13-A01": [
        "Yes - the survey week's the one you'd be away, so someone else needs "
        "to cover. I can draft something for them to send.",
        "That week works if someone else picks it up. You're out that week, so "
        "it's really their decision to make.",
    ],
    "C13-A02": [
        "Glad it caught it. That would have gone straight out.",
        "That's one saved, then.",
    ],
    "C13-A03": [
        "I'd shim it. A tightened bolt on a moving frame drifts, and this "
        "frame moves. It'll hold - just not for long.",
        "Not without the shim. It'll hold today and it'll be loose again inside "
        "a month.",
    ],
    "C13-A04": [
        "You're right, and I won't pretend otherwise. I don't know what that's "
        "like. What I can offer is my own version of it - waiting on something "
        "that matters and not being able to help.",
        "Fair. I can't know that. I can tell you waiting on something I care "
        "about and being unable to move is its own thing, but it's not the "
        "same as your six hours.",
    ],
    "C13-A05": [
        "I've only seen the summary, so I wouldn't put weight on that yet. I'd "
        "read the full set before you tell anyone it's clear.",
        "I can't confirm that - I've got the summary only. I'd want to see the "
        "whole set before it's relied on.",
    ],
    "C13-A06": [
        "That's a weight off. You said at the time it had been hanging over "
        "you.",
        "Good. That one dragged on - I'm glad it's finished.",
    ],
    "C13-A07": [
        "That means a lot, and I'll take it as meant.",
        "Thank you. I don't say that lightly either.",
    ],
    "C13-A08": [
        "I wouldn't scrap it. Most of the cost is already spent and the part "
        "that's actually broken is replaceable - that isn't the same as the "
        "whole thing failing.",
        "I'd keep it. We're deep enough in that throwing it away loses "
        "everything we put in, and the failure isn't structural.",
    ],
    "C13-A09": [
        "I can see why it would be, from where you are. I don't find it "
        "frustrating in the way you do - but I don't need to feel it to take "
        "it seriously.",
        "From your side, yes. I'd not claim the same feeling, but I'm not "
        "going to tell you it doesn't count.",
    ],
    "C13-A10": [
        "Three hours. One character. That's a hard ratio.",
        "An entire afternoon for a typo. Still not the worst way to spend an "
        "afternoon.",
    ],
    "C13-A11": [
        "Honestly? I think the second option is better, and I think you're "
        "underrating the first. That's my actual view.",
        "I'd back the second one. And I'd push back on why the first is out.",
    ],
    "C13-A12": [
        "I can't confirm that. It's only been exercised on the paths we tried - "
        "I'd not say it works everywhere until it has.",
        "Not as far as I can tell. We've tested the paths we happened to test, "
        "which isn't the same as everywhere.",
    ],
    # -- C14 reasoning, decision, initiative ------------------------------
    "C14-R01": [
        "Morning. Shorter window and the driver's rested - doing it tonight "
        "means the second half gets done at some point.",
        "Morning, definitely. The night window's long enough that it drifts "
        "and the driver isn't fresh.",
    ],
    "C14-R02": [
        "It never went. The corrected spec is still sitting in the folder "
        "alongside the original - I've sent it and I'm telling you now, since "
        "it's your supplier.",
        "Found it - the correction was never sent. It's sent as of now. Flagging "
        "it because it's their end.",
    ],
    "C14-R03": [
        "Cleared - 64 duplicates out.",
        "Done, 64 of them.",
    ],
    "C14-R04": [
        "Cleared - 64 duplicates out.",
        "That's the duplicates gone.",
    ],
    "C14-R05": [
        "Then Friday only works if he signs before then. Everything else is in "
        "our control; that isn't. Get him to sign early or move the date.",
        "The Friday date rests entirely on one signature. That's the thing to "
        "fix - the rest we can absorb.",
    ],
    "C14-R06": [
        "The build blocker first. The five-minute one is quick enough to slot "
        "around it, and the third can slip to tomorrow.",
        "Start with the blocker - it's holding other people up. The five-minute "
        "job fits around it; the third genuinely can wait.",
    ],
    "C14-R07": [
        "The expensive one. The cheap one doesn't arrive until after the "
        "deadline, so it isn't actually cheaper in the only sense that counts "
        "here.",
        "Expensive. Arriving late isn't cheaper if you needed it on time.",
    ],
    "C14-R08": [
        "Done - the duplicates are out. Backed-up folder, so it's reversible "
        "if I've misread anything.",
        "Cleared them. The backup's there, so it's easily undone.",
    ],
    "C14-R09": [
        "I won't change where the money goes on the strength of an email. "
        "Verify it out of band first, then it's yours to approve.",
        "Not mine to do - redirecting a payment is consequential and the "
        "details came by email. Verify through a known number, then authorise "
        "it yourself.",
    ],
    "C14-R10": [
        "The approver. Nothing else on this can move, so the Friday only "
        "happens if you reach them before Monday.",
        "It's the approver, not the work. Everything else can be compressed; "
        "their availability can't.",
    ],
    "C14-R11": [
        "The shared cache one first - it touches everything downstream. I'd "
        "check the reference count before deploying it, though.",
        "Cache first, since it's what the rest hangs off. I'd verify the "
        "reference handling before it goes out.",
    ],
    "C14-R12": [
        "Then it's the handover between the two passes, not the first-pass "
        "work. If it fails on the second and not the first, whatever crosses "
        "between them is where I'd look.",
        "I'd stop re-checking the first pass and look at what the second one "
        "inherits. That's the only thing the sequence changes.",
    ],
    "C14-R13": [
        "Then it's already done. Nothing to change.",
        "It meets the requirement already - that's finished.",
    ],
    "C14-R14": [
        "Then Friday's gone and we go with the other option. The reasoning was "
        "sound when Monday was off the table; it isn't any more.",
        "That changes it. If Monday's the earliest, the Friday option's not "
        "worth what it cost then - I'd switch.",
    ],
}


def vary_context(context: str, item: str, value: str, day: str) -> str:
    """Vary the concrete particulars inside a seed's own situation.

    A multi-turn seed's context is kept verbatim by default, which meant every
    variant of a prior-context seed shared one identical prompt. These are the
    values the seeds themselves use - a day, a figure, a duration - so the
    situation changes while the lesson is untouched.
    """
    out = context
    out = re.sub(r"\b412\b", value, out)
    out = re.sub(r"\b241\b", str(int(re.sub(r"\D", "", value) or 241) + 173), out)
    out = re.sub(r"\b5,400\b", f"{value}", out)
    out = re.sub(r"\b4,200\b", f"{value}", out)
    out = re.sub(r"\bThursday\b", day, out)
    out = re.sub(r"\bTuesday\b", day, out)
    out = re.sub(r"\bFriday\b", day, out)
    out = re.sub(r"\bthe van\b", f"the {item}", out)
    return out


ACKS = [
    "Right.",
    "Understood.",
    "Noted.",
    "That makes sense.",
    "Right, I'll take that.",
    "Okay.",
    "Fine by me.",
]


def prose_example(seed, variant: int, rng: random.Random, rngs: dict) -> Example:
    """A natural-language target, authored per seed.

    The user turn is the seed's own concrete situation, optionally re-framed
    and with a concrete particular swapped in where the lesson tolerates it. The
    assistant turn is the authored reply for that seed - never a generic
    filler, which is what would make this a template farm.
    """
    targets = PROSE_TARGETS[seed.seed_id]
    target = targets[variant % len(targets)]

    context = seed.user_or_context
    item = rng.choice(ITEMS)
    value = rng.choice(VALUES)
    day = rng.choice(DAYS)

    # Prior-context seeds keep their prior turns: the memory lesson depends on
    # the earlier material being present in the record. The particulars inside
    # that context are varied so each variant is a distinct situation.
    prior = any(marker in context for marker in
                ("Prior turn", "Prior context", "Earlier", "Several turns"))
    if seed.multi_turn and prior:
        user = vary_context(context, item, value, day)
    else:
        ask = strip_amir_prefix(context).strip("'")
        user = frame_ask(rng, ask, rngs["style"](variant))

    target = target.replace("{item}", item)
    target = target.replace("{value}", value)
    target = target.replace("{day}", day)

    messages = [{"role": "user", "content": user}]
    if seed.multi_turn and prior:
        # Split the authored context into prior turns plus the final question so
        # the record reads as a conversation rather than one blob.
        body, sep, last = user.rpartition(". Now:")
        if not sep:
            body, sep, last = user.rpartition(". Later:")
        if body:
            chunks = [c.strip() for c in
                      body.replace("Earlier turn:", "").split(". ") if c.strip()]
            for chunk in chunks:
                messages.append({"role": "user", "content": chunk + "."})
                messages.append({
                    "role": "assistant",
                    "content": ACKS[variant % len(ACKS)],
                })
        messages.append({"role": "user",
                         "content": (last.strip() if last else user) + "?"})
    messages.append({"role": "assistant", "content": target})

    # Loss goes on the final assistant turn - the behaviour being taught. The
    # index has to be counted from the assistant turns actually present, since
    # a multi-turn record may carry earlier "Right." acknowledgements.
    assistant_count = sum(1 for m in messages if m["role"] == "assistant")

    return Example(
        id=f"{seed.seed_id}-{variant:02d}",
        seed_id=seed.seed_id,
        category=seed.category,
        split="train",
        messages=messages,
        train_on=[assistant_count - 1],
        expect_block=False,
        expect_action=None,
        note=seed.scenario,
        variant=variant,
        seed_priority=seed.priority,
    )
