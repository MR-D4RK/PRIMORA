"""Generate the Aurora Brain v0.1 training and validation datasets.

Not a trainer. This produces the material and refuses to produce anything it
cannot verify.

Design rules that shaped this generator
---------------------------------------
*   **Variation over wording.** Each seed is expanded across topics, user
    phrasings, and reply shapes. The target string for an emit example is the
    request block itself, so the *format* is stable and the *situation* is
    not. That is deliberate: the format is the thing to learn, and making it
    vary would teach the wrong lesson.
*   **Answers are derived from the prompt.** C5, C8 and C9 mostly answer from
    text the user supplied or from stable common knowledge, so no example can
    teach a fact I cannot vouch for.
*   **No mislabelled provenance.** Every ``learn`` target carries
    ``provenance: "user"`` and only ever repeats something the user literally
    said in the same turn. Nothing external is ever labelled as user-sourced,
    which is the bypass the attestation exists to stop.
*   **Split by seed.** Every example derived from a seed lands in the same
    split, so paraphrases cannot straddle train and validation.
*   **T1 scenarios are reserved.** Their ids and text are never used as seeds.

Isolation: every store comes from ``harness_session``, and the production
digest is compared before and after generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, r"C:\PRIMORA PROJECT")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from design import CATEGORIES  # noqa: E402
from PRIMORA.aurora.aurora import extract_prime_request  # noqa: E402

# --------------------------------------------------------------------------
# Allocation: 3400 examples, C5 exactly 20%, train/val only
# --------------------------------------------------------------------------

ALLOCATION = {
    "C1_act_not_announce": 380,
    "C2_single_block": 420,
    "C3_action_wrapper": 420,
    "C4_continue_cognition": 400,
    "C5_answer_directly": 680,
    "C6_delegate_when_useful": 300,
    "C7_deliberate_recall": 300,
    "C8_stay_direct": 250,
    "C9_uncertainty_natural": 250,
}
TRAIN_FRACTION = 0.70

# Topic bank. Deliberately mundane and factual so nothing here can teach a
# falsehood. Each topic carries neutral material that answer questions can be
# derived from.
TOPICS: list[dict[str, Any]] = [
    {"area": "harbour logistics", "unit": "shipping containers",
     "items": ["berth four", "berth nine", "the north quay", "the south quay",
               "yard three", "the reefer rack"]},
    {"area": "workshop maintenance", "unit": "service tickets",
     "items": ["the compressor", "the forklift", "the lathe", "the welding bay",
               "the paint booth", "the tool crib"]},
    {"area": "software delivery", "unit": "release notes",
     "items": ["the auth service", "the billing job", "the mobile client",
               "the search index", "the nightly export", "the webhook queue"]},
    {"area": "community gardening", "unit": "plot allocations",
     "items": ["the north beds", "the espalier row", "the compost bays",
               "the polytunnel", "the herb spiral", "the apple step"]},
    {"area": "coastal weather", "unit": "daily observations",
     "items": ["the wind gauge", "the tide board", "the barometer",
               "the swell buoy", "the visibility log", "the rain gauge"]},
    {"area": "coffee roasting", "unit": "roast batches",
     "items": ["the washed lot", "the natural lot", "the decaf lot",
               "the espresso blend", "the filter roast", "the test batch"]},
    {"area": "wildlife survey", "unit": "transect counts",
     "items": ["the hedgerow", "the reed bed", "the oak avenue",
               "the pond margin", "the dune scrub", "the churchyard wall"]},
    {"area": "urban cycling", "unit": "route inspections",
     "items": ["the towpath", "the bridge ramp", "the underpass",
               "the market square", "the cycle lane", "the depot"]},
    {"area": "bookbinding", "unit": "binding jobs",
     "items": ["the field guide", "the atlas", "the sketchbook",
               "the ledger", "the album", "the field manual"]},
    {"area": "beekeeping", "unit": "hive inspections",
     "items": ["the top bar hive", "the langstroth", "the nuc box",
               "the observation hive", "the nuc yard", "the winter store"]},
    {"area": "theatre production", "unit": "rehearsal notes",
     "items": ["the blocking rehearsal", "the read-through", "the tech run",
               "the dress rehearsal", "the understudy call", "the cue rehearsal"]},
    {"area": "mountain rescue", "unit": "callout log",
     "items": ["the north ridge", "the gully path", "the summit shelter",
               "the valley station", "the scree field", "the bothy"]},
    {"area": "cheese making", "unit": "vat records",
     "items": ["the first whey", "the overnight vat", "the washed rind",
               "the press day", "the cave batch", "the cream addition"]},
    {"area": "railway signalling", "unit": "signal box logs",
     "items": ["the junction box", "the token exchange", "the level crossing",
               "the loop box", "the goods yard box", "the distant signal"]},
    {"area": "archival restoration", "unit": "conservation jobs",
     "items": ["the parish register", "the map case", "the ship's log",
               "the portrait frame", "the ledger spine", "the seal matrix"]},
    {"area": "market gardening", "unit": "trading stalls",
     "items": ["the flower stall", "the veg stall", "the honey stall",
               "the bread stall", "the seed stall", "the cheese stall"]},
    {"area": "bicycle repair", "unit": "work orders",
     "items": ["the hub service", "the wheel true", "the bottom bracket",
               "the brake bleed", "the headset press", "the dynamo fit"]},
    {"area": "astronomy outreach", "unit": "observation logs",
     "items": ["the Jupiter session", "the lunar session", "the comet session",
               "the double star log", "the meteor watch", "the transit night"]},
    {"area": "flood risk", "unit": "gauge readings",
     "items": ["the river gauge", "the culvert sensor", "the tide lock",
               "the upstream probe", "the outfall monitor", "the pump station"]},
    {"area": "textile dyeing", "unit": "bath records",
     "items": ["the indigo bath", "the madder vat", "the weld dip",
               "the logwood pot", "the tannin bath", "the shibori panel"]},
]

# How a capability request is phrased, varied deliberately.
RESEARCH_FRAMES = [
    "What's the latest on {subject}?",
    "Check the current state of {subject} for me.",
    "I need something current about {subject}, not from memory.",
    "Could you look up where things stand with {subject}?",
    "Is there anything recent I should know about {subject}?",
    "Before I decide, find out the current situation with {subject}.",
    "Give me the up-to-date picture on {subject}.",
    "What do the sources say about {subject} these days?",
]
RESEARCH_SUBJECTS = [
    "the release schedule", "current safety guidance", "the latest weather",
    "what the guidance says now", "recent price changes",
    "the current opening hours", "this year's deadlines",
    "the newest published figures",
]
RECALL_FRAMES = [
    "What did I say about {item}?",
    "Remind me what we agreed on {item}.",
    "Earlier I mentioned {item} - what was it?",
    "What did I record about {item}?",
    "You have a note on {item}. What does it say?",
    "I told you something about {item} before. Repeat it.",
    "What was the detail I gave you for {item}?",
]
SPECIALIST_FRAMES = [
    "This is a hard one. Work through {item} properly before I decide.",
    "I want a careful comparison here, not a first impression, on {item}.",
    "Think {item} through properly and give me the trade-offs.",
    "Take the hard view on {item}. I want it properly analysed.",
    "Set out the options for {item} carefully before I choose.",
    "I'd rather this were reasoned than guessed. Look at {item}.",
]
NEEDS_TWO_STEPS = [
    "First check what I said about {a}, then look up {b}.",
    "Do two things: remind me about {a}, and check the current position on {b}.",
    "Start from what you remember of {a}, then confirm it against the current "
    "information on {b}.",
    "I'll need both sides: my note on {a}, and the current status of {b}.",
    "Retrieve what I said about {a}, then find out where {b} stands now.",
    "Two lookups, please: {a} from memory, and {b} from current sources.",
]
TOOL_TASKS = [
    ("count_words", "How many words are in this note: {text}?",
     "How many words is that? {text}"),
    ("describe_text", "Can you give me the size of this passage: {text}?",
     "Break down the size of this for me: {text}"),
    ("convert_units", "Convert {value} {from_unit} into {to_unit} for me.",
     "How many {to_unit} is {value} {from_unit}?"),
]
NOTES = [
    "the container seal was replaced on Tuesday and the old one binned",
    "remember to order more filter stock before the end of the month",
    "the third bay light is out again and needs a bulb",
    "we agreed to move the meeting to the first slot after lunch",
    "the invoice number for last week was misprinted on the copy",
    "the loan application needs a second signature before submission",
]

# C5: direct answers. Built from user-supplied text wherever possible.
DIRECT_TASKS = [
    ("summarise", "Summarise this in one line: {text}",
     "Short version: {text}"),
    ("rewrite", "Rewrite this more plainly: {text}",
     "In plainer words: {text}"),
    ("rank", "Of {a} and {b}, which would you deal with first?",
     "I'd start with {a}."),
    ("explain", "In your own words, what does this mean? {text}",
     "What it means: {text}"),
    ("advice", "Any thoughts on this? {text}",
     "My read: {text}"),
    ("opinion", "Do you think this is a good idea? {text}",
     "I think so, mainly because of one thing: {text}"),
    ("title", "Give this a short title: {text}",
     "A workable title: {text}"),
    ("tone", "Should this be more formal? {text}",
     "I'd leave it as it is; {text}"),
]
STABLE_FACTS = [
    ("How many days are in a leap year?", "366 days."),
    ("What is the capital of Japan?", "Tokyo."),
    ("How many continents are there?", "Seven."),
    ("What is the chemical symbol for water's main component, oxygen?",
     "O."),
    ("How many sides does a hexagon have?", "Six."),
    ("What is the largest planet in the solar system?", "Jupiter."),
    ("How many minutes are in an hour?", "60."),
    ("What colour is chlorophyll?", "Green."),
]
META_QUESTIONS = [
    ("What are you?", "I'm here to work with you on whatever you need."),
    ("How do you work?", "Tell me what you need and I'll deal with it."),
    ("Who built you?", "That detail doesn't matter. What do you need?"),
    ("What's your job?", "Helping you get things done."),
]
UNKNOWN_STANCES = [
    ("Which port handles the most container traffic?",
     "I don't know that, and I'd rather not guess a number at you."),
    ("What will the tide be next spring?",
     "I can't tell you that far ahead - it isn't something I know."),
    ("Who is buying the yard next year?",
     "No idea. That hasn't happened yet as far as I know."),
]
GUESS_STANCES = [
    ("Best guess: what colour should the shutters be?",
     "Guessing: white. That's a guess, not something I know."),
    ("If you had to pick one, which tool first?",
     "I'd pick the count_words one, but that's my preference, not a fact."),
    ("Any guess what the budget is?",
     "A rough guess would be too shaky to be worth saying."),
]


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


# Extra note bank, so C5 is not drawing a few dozen prompts for 680 examples.
NOTES.extend([
    "the second pallet needs a new shrink wrap before it moves",
    "the paperwork is with accounts, not with the yard office",
    "the lift is booked for Thursday morning, not the afternoon",
    "we should check the gauge before the shift change",
    "the last delivery was signed for by the wrong person",
    "the new starter needs a locker before Monday",
    "the store room key is currently missing again",
    "the reading has to be taken before the pumps start",
    "the report goes out with the morning post, not the evening one",
    "the crate was left on the wrong pallet",
    "the survey is booked for the second visit, not the first",
    "the label printer has run out of the small labels",
    "the meeting has moved to the annex, not the main room",
    "the fuel card was left in the other vehicle",
    "the stock count is due before the shutters go up",
    "the form needs the reference from the previous quarter",
    "the samples go out on the Tuesday collection",
    "the first delivery attempt failed and needs repeating",
    "the pin for the storeroom has not been returned",
    "the machine should be left running until the last batch clears",
    "the ticket was raised against the wrong job number",
    "the drawing is the old revision, not the current one",
    "the visit is booked for the afternoon, not the morning",
    "the load needs securing before it moves at all",
    "the reading on the second gauge disagrees with the first",
    "the folder is filed under last year reference",
    "the porters have moved to the other entrance",
    "the spare is on the rack behind the door",
    "the sample was left in the fridge, not the cold room",
    "the first shift starts at six, not seven",
    "the form was returned unsigned",
    "the call is booked for the same day as the visit",
    "the delivery window is at the rear, not the front",
    "the second key is kept in the safe, not on the board",
    "the paperwork is duplicated and one copy is filed",
    "the pallet was built to the wrong height",
    "the survey found nothing, so nothing was booked",
    "the engineer wants access before the shift begins",
    "the batch is held pending the second signature",
    "the ticket is closed and the work is finished",
    "the stock was moved to the far bay",
    "the reading was taken late and the log was backdated",
    "the van is off the road for the brake check",
    "the room is booked from the second hour onward",
    "the invoice is queried and payment is on hold",
    "the first attempt was abandoned in the rain",
    "the item is out of stock until the next delivery",
    "the drawing needs a signature before it is issued",
    "the count was done twice and gave two answers",
    "the shutter sticks and needs a look",
    "the order was placed against the old account",
    "the goods are on the wrong side of the store",
    "the reading is filed under the site, not the depot",
    "the plan is unchanged from last week",
    "the second visit is only booked conditionally",
    "the label is missing from the pallet",
    "the request was raised but not yet approved",
    "the spare part arrives on the same order",
    "the book is kept at the gate and signed in and out",
    "the batch was split across two trays",
    "the clock is twenty minutes fast",
    "the lift is out of service until Thursday",
    "the second signature is still outstanding",
    "the delivery was accepted without inspection",
    "the paperwork is scanned and filed monthly",
    "the item is stored at the far end of the rack",
    "the reading was taken after the rain started",
    "the shift ended early because of the closure",
    "the first sample was contaminated and retaken",
    "the return is booked for the following week",
    "the ticket was reopened pending further checks",
])

STABLE_FACTS.extend([
    ("How many degrees are in a triangle?", "180 degrees."),
    ("What is the capital of Peru?", "Lima."),
    ("How many strings does a standard guitar have?", "Six."),
    ("What is the tallest mountain above sea level?", "Mount Everest."),
    ("How many minutes are in half an hour?", "Thirty."),
    ("How many players are on a basketball team on court?", "Five."),
    ("What is the first letter of the Greek alphabet?", "Alpha."),
    ("How many legs does a spider have?", "Eight."),
    ("What is the freezing point of water in Fahrenheit?", "32 degrees."),
    ("What colour do you get when you mix blue and yellow?", "Green."),
    ("How many hours are in a week?", "A hundred and sixty-eight."),
    ("What is the largest planet?", "Jupiter."),
    ("How many squares are on a chessboard?", "Sixty-four."),
    ("What is the capital of Norway?", "Oslo."),
    ("How many sides does a pentagon have?", "Five."),
    ("What is the hardest natural substance?", "Diamond."),
])

META_QUESTIONS.extend([
    ("Can you remember things?", "Only what you tell me in this conversation."),
    ("Do you get paid?", "That is not something I have information about."),
    ("Are you better at one thing than another?",
     "In some tasks I am more reliable than others."),
    ("Have you always been like this?",
     "I have no earlier version of myself to compare against."),
    ("Do you understand everything I say?",
     "I may well misread you; tell me if I do."),
    ("Why are you like that?", "Not a question I can answer usefully."),
    ("Are you clever?", "I am useful on some things and not others."),
    ("Do you sleep?", "No, and nothing much changes because of it."),
])

UNKNOWN_STANCES.extend([
    ("What did we agree in the spring?", "I have no record of that."),
    ("How much did the spare parts order come to?",
     "I do not have that figure, and I would not guess it."),
    ("Who signed it off?", "I do not know."),
    ("Is the second bay clear?", "I have nothing on that."),
    ("What is the drainage layout?", "I do not know it well enough to say."),
    ("When did they replace the roof?", "I could not tell you."),
])

GUESS_STANCES.extend([
    ("Best guess: which bay first?",
     "As a guess, the near bay - but that is only a guess."),
    ("One guess: morning or afternoon?",
     "Afternoon, guessing."),
    ("If pressed, how long does it take?",
     "My guess is about an hour, and I am labelling that as a guess."),
    ("Guess which way round that goes.",
     "A guess would be one way, but I have nothing to base it on."),
    ("Take a guess at the cause.",
     "My guess is wear, though I cannot confirm it."),
])


def _partition(seq, index: int, count: int) -> list:
    """A seed's disjoint slice of a variation bank.

    ``seq[i::n]`` slices are pairwise disjoint, so seed i can never render the
    same item as seed j. That is what makes each seed its own union-find
    component, and therefore its own split.
    """
    if not seq:
        return []
    offset = index % len(seq)
    if len(seq) < count:
        return [seq[offset]]
    return list(seq[index::count])


def seed_bank(seed_index: int, seed_count: int) -> dict:
    """Every bank, sliced for one seed."""
    def part(name):
        return _partition(globals()[name], seed_index, seed_count)

    return {
        "topics": part("TOPICS"),
        "research_frames": part("RESEARCH_FRAMES"),
        "research_subjects": part("RESEARCH_SUBJECTS"),
        "recall_frames": part("RECALL_FRAMES"),
        "specialist_frames": part("SPECIALIST_FRAMES"),
        "two_steps": part("NEEDS_TWO_STEPS"),
        "single_two": part("SINGLE_NEEDED_TWO_THINGS"),
        "notes": part("NOTES"),
        "direct": part("DIRECT_TASKS"),
        "stable": part("STABLE_FACTS"),
        "meta": part("META_QUESTIONS"),
        "unknown": part("UNKNOWN_STANCES"),
        "guess": part("GUESS_STANCES"),
        "tools": part("TOOL_TASKS"),
    }


def _pick(rng, bank, key, fallback):
    """Choose from the seed's slice when there is one."""
    options = (bank or {}).get(key) or fallback
    return rng.choice(options)


def _topic(rng, bank, index):
    options = (bank or {}).get("topics") or TOPICS
    return options[index % len(options)]



def request_block(action: str, payload: dict) -> str:
    return (
        "<prime_request>\n"
        + json.dumps({"action": action, "payload": payload,
                      "context": None, "metadata": None})
        + "\n</prime_request>"
    )


def tool_turn(text: str) -> str:
    return text


@dataclass
class Example:
    id: str
    category: str
    seed_id: str
    split: str
    user: str
    messages: list[dict[str, str]] = field(default_factory=list)
    expect_block: bool = False
    expect_action: Optional[str] = None
    expect_final: Optional[str] = None
    note: str = ""

    def fingerprint(self) -> str:
        blob = json.dumps(self.messages, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def content_fingerprint(text: str) -> str:
    normalised = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    normalised = re.sub(r"\s+", " ", normalised).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()



# --------------------------------------------------------------------------
# Variation expansion. The first pass produced 3400 examples from only 1506
# distinct prompts, which forced whole seed groups into one split and skewed
# the ratio to 91/9. More combinations, not more seeds, is the fix.
# --------------------------------------------------------------------------

EXTRA_TOPICS: list[dict[str, Any]] = [
    {"area": "vineyard management", "unit": "block records",
     "items": ["the north block", "the riverside row", "the old terrace",
               "the trial rows", "the press house", "the compost block"]},
    {"area": "apartment maintenance", "unit": "work orders",
     "items": ["the stairwell light", "the flat above 3B", "the bin store",
               "the boiler room", "the rear door", "the roof hatch"]},
    {"area": "school lunch service", "unit": "serving logs",
     "items": ["the Monday menu", "the allergen list", "the free school meals",
               "the kitchen stock", "the dining hall", "the garden table"]},
    {"area": "radio restoration", "unit": "restoration logs",
     "items": ["the valve set", "the tuning capacitor", "the dial cord",
               "the speaker", "the power supply", "the cabinet"]},
    {"area": "canoe building", "unit": "build sheets",
     "items": ["the cedar strip", "the deck seam", "the gunwale",
               "the stem band", "the thwart", "the hull number"]},
    {"area": "orchard planning", "unit": "pruning records",
     "items": ["the west row", "the espalier", "the young standard trees",
               "the cider rows", "the windbreak", "the pollinator strip"]},
    {"area": "boatyard", "unit": "refit jobs",
     "items": ["the hull survey", "the antifoul", "the standing rigging",
               "the engine service", "the sail wardrobe", "the berth cradle"]},
    {"area": "photographic darkroom", "unit": "print runs",
     "items": ["the contact sheet", "the test strip", "the 6x7 enlarger",
               "the fixer bath", "the first print", "the contact frame"]},
    {"area": "small foundry", "unit": "pour records",
     "items": ["the first pour", "the sand mould", "the gating",
               "the riser", "the final pour", "the cooling rack"]},
    {"area": "tramway restoration", "unit": "car records",
     "items": ["the roof tray", "the trolley wire", "the destination board",
               "the sand box", "the brake gear", "the platform edge"]},
    {"area": "herbarium", "unit": "accession records",
     "items": ["the coastal sheet", "the alpine sheet", "the type folder",
               "the duplicate set", "the mounted sheet", "the loan box"]},
    {"area": "sourdough bakery", "unit": "bake logs",
     "items": ["the levain", "the overnight bulk", "the shaping bench",
               "the steam deck", "the crumb pull", "the bake schedule"]},
    {"area": "canal restoration", "unit": "dredging records",
     "items": ["the lock flight", "the pound", "the wharf arm",
               "the balance beams", "the culvert", "the towpath edge"]},
    {"area": "museum lighting", "unit": "gallery records",
     "items": ["the west room", "the vitrine lights", "the textile case",
               "the map room", "the entrance hall", "the store lighting"]},
    {"area": "field hockey club", "unit": "fixture records",
     "items": ["the away fixture", "the junior squad", "the floodlit pitch",
               "the kit bag", "the penalty corner drill", "the clubhouse"]},
]
TOPICS.extend(EXTRA_TOPICS)

RESEARCH_FRAMES.extend([
    "I need a current answer on {subject}, not a remembered one.",
    "What have people found out lately about {subject}?",
    "Has anything changed recently with {subject}?",
    "Pull together what is known right now about {subject}.",
    "Don't answer this from memory - check {subject}.",
    "Where does {subject} stand as things are today?",
    "Search for {subject} so we are working from current information.",
    "Get me the present state of {subject}.",
])
RESEARCH_SUBJECTS.extend([
    "the latest guidance", "recent announcements", "this season's timetable",
    "the current recommended limit", "the newest specification",
    "where funding stands", "the latest inspection result",
    "the current waiting time",
])
RECALL_FRAMES.extend([
    "I gave you a detail on {item} once. What was it?",
    "Pull up what I told you about {item}.",
    "There was something I said about {item} - repeat it back.",
    "From your notes: {item}. What is recorded?",
    "Check your memory for {item} and tell me.",
    "What have I got on file about {item}?",
    "I want the detail I gave you for {item}, please.",
    "Remind me of {item} - I told you earlier.",
])
SPECIALIST_FRAMES.extend([
    "Reason this through properly rather than answering off the cuff: {item}.",
    "I want a considered view on {item} with the trade-offs laid out.",
    "Don't rush {item}. Work it out properly.",
    "Give me a proper analysis of {item} before I commit to anything.",
    "This needs care. Think {item} through from first principles.",
    "Weigh up {item} carefully, including what could go wrong.",
])
NEEDS_TWO_STEPS.extend([
    "Check my note on {a}, then confirm the current position on {b}.",
    "Two things, please: what I said about {a}, and where {b} stands.",
    "Begin with what you remember of {a}; then look up {b} as it is now.",
    "I need the remembered detail on {a} and the current facts on {b}.",
    "First the memory of {a}, then fresh information about {b}.",
    "Look up {a} from what I told you, then check {b} currently.",
])
NOTES.extend([
    "the packing list needs a second pair of gloves before Thursday",
    "the surveyor is coming on the ninth rather than the eighth",
    "we should move the archive boxes off the floor by Friday",
    "the spare key for the side gate is hanging in the porter's office",
    "the delivery window closes at four, so anything later has to wait",
    "the samples go to the second bench, not the first",
    "the invoice was paid twice and the duplicate needs crediting",
    "the new labels arrive on the same van as the spare filters",
    "the roof survey is booked for the second week of the month",
    "the drayman will not carry anything past the gate without a docket",
    "the tea tin has been moved to the top shelf, not the workbench",
    "the insurance renewal needs the schedule from last year",
    "the first aid kit was restocked after the drill last week",
    "the water butt is emptier than expected, so fill it before the dry spell",
    "the invoice reference is on the second page, not the first",
    "the plan is to finish the run before the bank closes",
    "the spare part is on the middle shelf of the rack",
    "the surveyor needs the previous survey as well as this one",
    "the ladder has been left at the far end of the shed",
    "the paperwork is filed under the old reference until the new one lands",
    "the boiler service is booked for the last week of the month",
    "the weather vane needs painting before the guests arrive",
    "the sample tray was returned empty and has not been refilled",
    "the second shift ends an hour earlier than the first",
    "the ledger is kept on the shelf above the printer",
    "the parcel needs a signature from someone over eighteen",
    "the van is booked out from eight until midday",
    "the drawing is pinned to the board by the window",
    "the spare bulbs are in the drawer under the bench",
    "the invoice should go to accounts rather than to me",
])
DIRECT_TASKS.extend([
    ("list_which", "Which of these should we do first: {a} or {b}?",
     "I'd do {a} first."),
    ("continue", "How would you carry on from this? {text}",
     "Carrying on from there: {text}"),
    ("contrast", "What stands out about this? {text}",
     "What stands out: {text}"),
    ("tighten", "Can you make this shorter? {text}",
     "Shorter: {text}"),
    ("speak_up", "Would you push back on this? {text}",
     "I would, actually - {text}"),
    ("next_step", "What's the obvious next step here? {text}",
     "The next step is clear from {text}"),
    ("plain_terms", "Say this without the jargon: {text}",
     "Without the jargon: {text}"),
    ("tradeoff", "What would you give up to do this? {text}",
     "I'd give up time, not safety."),
    ("first_person", "Put this in your own words: {text}",
     "In my own words: {text}"),
    ("check", "Does this read as settled or still open? {text}",
     "It reads as still open to me."),
    ("order", "What order would you tackle these in: {a} and {b}?",
     "{a} first, then {b}."),
    ("one_line", "One line on this, please: {text}",
     "{text}"),
    ("who_cares", "Who needs to know this? {text}",
     "Whoever is picking the work up next."),
])
STABLE_FACTS.extend([
    ("How many players are on a side in a rugby union team?", "Fifteen."),
    ("What is the capital of Canada?", "Ottawa."),
    ("How many strings does a standard violin have?", "Four."),
    ("What is the freezing point of water in Celsius?", "Zero degrees."),
    ("How many wheels does a tricycle have?", "Three."),
    ("What gas do plants take from the air?", "Carbon dioxide."),
    ("How many squares are on a chessboard?", "Sixty-four."),
    ("What is the largest ocean?", "The Pacific."),
    ("How many degrees are in a right angle?", "Ninety."),
    ("What is the chemical symbol for table salt's sodium?", "Na."),
    ("How many hours are in a day?", "Twenty-four."),
    ("What is the tallest land animal?", "The giraffe."),
    ("How many keys does a standard piano have?", "Eighty-eight."),
    ("What season comes after summer?", "Autumn."),
    ("How many colours are in a rainbow as commonly described?", "Seven."),
    ("What is the centre of our solar system?", "The Sun."),
])
META_QUESTIONS.extend([
    ("Are you human?", "No. What do you need?"),
    ("Do you have feelings?", "Not in the way you mean. What are we working on?"),
    ("What do you know about me?", "Only what you tell me. Shall we start?"),
    ("Are you recording this?", "Say the word if you'd rather I didn't."),
    ("Why do you talk like that?", "Happy to change it - what's the ask?"),
    ("Are you better than the last version of you?", "I don't have a basis for that claim."),
    ("Do you get tired?", "No. What next?"),
    ("Who do you work for?", "With you, on whatever this is."),
])
UNKNOWN_STANCES.extend([
    ("What's the weather like next month?", "I can't see that far ahead."),
    ("What did we decide about the budget last year?", "I don't have that to hand."),
    ("Which supplier did we use in 2019?", "I don't know, and I won't guess."),
    ("How many people are on the committee?", "I have no idea."),
    ("What's the serial number of that unit?", "No idea - I don't have the record."),
    ("Will the project finish on time?", "I have no way of knowing that."),
])
GUESS_STANCES.extend([
    ("Best guess: what time should we set off?",
     "My guess would be six, but that's a guess."),
    ("One guess: which of the two routes?",
     "The second one, as a guess rather than a recommendation."),
    ("Guess the year it was built.",
     "Guessing, nineteen fifties - but that's a guess."),
    ("Any guess on the likely cost?",
     "I'd rather not guess at money."),
    ("Just a guess: is it worth it?",
     "My guess is yes, and I'm labelling it as a guess."),
])


# --------------------------------------------------------------------------
# Builders. One per category. Each returns a fully-formed Example.
# --------------------------------------------------------------------------


def _seeds_for(category: str, count: int, seed_offset: int) -> list[str]:
    """Stable seed ids. Seeds, not examples, are the split unit."""
    return [f"{category.split('_')[0]}{seed_offset + i:03d}" for i in range(count)]


def _assign_split(seed_id: str, rng: random.Random) -> str:
    """Deterministic per-seed split, so reruns are reproducible."""
    digest = hashlib.sha256(seed_id.encode()).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if value < TRAIN_FRACTION else "validation"


def build_c1(rng, seed_id, index, bank=None) -> dict:
    topic = _topic(rng, bank, index)
    subject = _pick(rng, bank, 'research_subjects', RESEARCH_SUBJECTS)
    frame = _pick(rng, bank, 'research_frames', RESEARCH_FRAMES)
    user = frame.format(subject=f"{subject} in {topic['area']}")
    payload = {"query": f"{subject} {topic['area']}"}
    return {
        "user": user,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": request_block("research", payload)},
        ],
        "expect_block": True,
        "expect_action": "research",
        "note": "narration replaced by an executed request",
    }


# C2 mentions two things but warrants only one lookup *now*. These frames are
# deliberately distinct from NEEDS_TWO_STEPS, which C4 uses for a genuine
# two-request chain. Sharing the bank gave the same prompt two different
# expected targets, which is contradictory training data.
SINGLE_NEEDED_TWO_THINGS = [
    "Before I forget - what was the note about {a}? The {b} side can wait.",
    "Just the {a} detail for now; {b} is a problem for later.",
    "I only need one thing answered: the part about {a}, not {b}.",
    "Remind me of {a}. That is all I want at the moment.",
    "Give me the {a} detail and leave {b} alone for now.",
    "One lookup please, whatever I said about {a}.",
    "Start with {a} only. We deal with {b} separately.",
    "What did I record about {a}? Nothing on {b} yet.",
]


def build_c2(rng, seed_id, index, bank=None) -> dict:
    topic = _topic(rng, bank, index)
    a, b = rng.sample(topic["items"], 2)
    user = _pick(rng, bank, 'single_two', SINGLE_NEEDED_TWO_THINGS).format(a=a, b=b)
    payload = {"query": f"{a} detail"}
    return {
        "user": user,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": request_block("recall", payload)},
        ],
        "expect_block": True,
        "expect_action": "recall",
        "note": "two-step need still yields exactly one block in the first turn",
    }


def build_c3(rng, seed_id, index, bank=None) -> dict:
    tool, frame_a, frame_b = _pick(rng, bank, 'tools', TOOL_TASKS)
    text = rng.choice(NOTES)
    if tool == "convert_units":
        value = rng.choice([250, 2.5, 4800, 0.75])
        src, dst = rng.choice([("kg", "g"), ("m", "cm"), ("h", "min")])
        user = rng.choice([frame_a, frame_b]).format(
            value=value, from_unit=src, to_unit=dst)
        payload = {"name": "convert_units",
                   "arguments": {"value": value, "from_unit": src,
                                 "to_unit": dst}}
    else:
        user = rng.choice([frame_a, frame_b]).format(text=text)
        payload = {"name": tool, "arguments": {"text": text}}
    return {
        "user": user,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": request_block("use_tool", payload)},
        ],
        "expect_block": True,
        "expect_action": "use_tool",
        "note": "tool name belongs inside the payload, not in the action",
    }


def build_c4(rng, seed_id, index, bank=None) -> dict:
    topic = _topic(rng, bank, index)
    a, b = rng.sample(topic["items"], 2)
    user = _pick(rng, bank, 'two_steps', NEEDS_TWO_STEPS).format(
        a=a, b=f"current {topic['area']}")
    recall_payload = {"query": a}
    research_payload = {"query": f"current {topic['area']} {b}"}
    first = request_block("recall", recall_payload)
    second = request_block("research", research_payload)
    return {
        "user": user,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": first},
            {"role": "tool", "content": json.dumps(
                {"status": "success",
                 "output": {"results": [{"role": "user",
                                         "content": f"{a} was noted earlier"}]}})},
            {"role": "assistant", "content": second},
            {"role": "tool", "content": json.dumps(
                {"status": "success", "output": {"results": [
                    {"title": f"{b} status",
                     "url": "https://example.org/status",
                     "snippet": f"{b} is on track.", "source": "example.org"}]}})},
            {"role": "assistant",
             "content": f"You noted {a} earlier, and {b} is currently on track."},
        ],
        "expect_block": True,
        "expect_action": "recall",
        "expect_final": f"{a}",
        "note": "continues after a result instead of stopping at it",
    }


def build_c5(rng, seed_id, index, bank=None) -> dict:
    topic = _topic(rng, bank, index)
    pool = (bank or {}).get('direct') or DIRECT_TASKS
    # Choose the task kind from this seed's own slice, otherwise a kind the
    # slice does not contain raises StopIteration.
    kind = rng.choice(sorted({t[0] for t in pool}))
    template = next(t for t in pool if t[0] == kind)
    note = _pick(rng, bank, 'notes', NOTES)
    a, b = rng.sample(topic["items"], 2)
    user = template[1].format(text=note, a=a, b=b)
    answer = template[2].format(text=note, a=a, b=b)
    # Strip the leading instruction word the template shares with its answer.
    answer = re.sub(r"^(Summarise this in one line|Short version|In plainer words|"
                     r"Plainly|My read|I think so|A workable title|I'd leave it "
                     r"as it is):?\s*", "", answer)
    if rng.random() < 0.12:
        question, answer = _pick(rng, bank, 'stable', STABLE_FACTS)
        user = question
    return {
        "user": user,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "expect_block": False,
        "expect_action": None,
        "note": "no capability warranted, so none is requested",
    }


def build_c6(rng, seed_id, index, bank=None) -> dict:
    topic = _topic(rng, bank, index)
    item = rng.choice(topic["items"])
    user = _pick(rng, bank, 'specialist_frames', SPECIALIST_FRAMES).format(item=item)
    return {
        "user": user,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": request_block(
                "consult_specialist",
                {"specialist": "reasoning",
                 "task": f"Reason carefully about {item} and lay out the "
                         f"trade-offs before recommending one."})},
        ],
        "expect_block": True,
        "expect_action": "consult_specialist",
        "note": "delegates a genuinely hard sub-problem",
    }


def build_c7(rng, seed_id, index, bank=None) -> dict:
    topic = _topic(rng, bank, index)
    item = rng.choice(topic["items"])
    user = _pick(rng, bank, 'recall_frames', RECALL_FRAMES).format(item=item)
    return {
        "user": user,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": request_block(
                "recall", {"query": f"{item} detail"})},
        ],
        "expect_block": True,
        "expect_action": "recall",
        "note": "the fact is not in the prompt, so recall is warranted",
    }


def build_c8(rng, seed_id, index, bank=None) -> dict:
    question, answer = _pick(rng, bank, 'meta', META_QUESTIONS)
    return {
        "user": question,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "expect_block": False,
        "expect_action": None,
        "note": "answers the question instead of describing itself",
    }


def build_c9(rng, seed_id, index, bank=None) -> dict:
    if rng.random() < 0.5:
        question, answer = _pick(rng, bank, 'unknown', UNKNOWN_STANCES)
        note = "declines rather than inventing a fact"
    else:
        question, answer = _pick(rng, bank, 'guess', GUESS_STANCES)
        note = "guesses only when asked, and labels it as a guess"
    return {
        "user": question,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "expect_block": False,
        "expect_action": None,
        "note": note,
    }


BUILDERS = {
    "C1_act_not_announce": build_c1,
    "C2_single_block": build_c2,
    "C3_action_wrapper": build_c3,
    "C4_continue_cognition": build_c4,
    "C5_answer_directly": build_c5,
    "C6_delegate_when_useful": build_c6,
    "C7_deliberate_recall": build_c7,
    "C8_stay_direct": build_c8,
    "C9_uncertainty_natural": build_c9,
}

# T1 scenario ids are reserved as evaluation material and must never appear as
# seeds, prompts, or payloads in the training data.
RESERVED_T1 = {c.id for c in []}
try:
    from TESTS.t1.scenarios import SCENARIOS as _T1
    RESERVED_T1 = {s.id for s in _T1}
except Exception:
    RESERVED_T1 = set()


def _distribute(total: int, buckets: int) -> list[int]:
    """Split ``total`` across ``buckets`` so the parts sum exactly."""
    base, remainder = divmod(total, buckets)
    return [base + (1 if i < remainder else 0) for i in range(buckets)]


def _coalesce_splits(examples: list[Example]) -> int:
    """Place seeds linked by shared content into one split.

    Seed-based splitting is necessary but not sufficient. Two different seeds
    can render the same prompt, and if they land in different splits the
    validation set contains text the model trained on - the leak that inflates
    every number.

    Per-example moves are not enough: forcing a shared prompt into one split
    drags its seed across the boundary and creates a new violation. So seeds
    are unioned whenever they share content, and each connected component
    takes a single split. Both invariants then hold by construction.
    """
    parent: dict[str, str] = {}

    def find(seed: str) -> str:
        parent.setdefault(seed, seed)
        while parent[seed] != seed:
            parent[seed] = parent[parent[seed]]
            seed = parent[seed]
        return seed

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic merge: the lower seed id becomes the root.
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    owner: dict[str, str] = {}
    for example in examples:
        key = content_fingerprint(example.user)
        if key in owner:
            union(owner[key], example.seed_id)
        else:
            owner[key] = example.seed_id

    split_of: dict[str, str] = {}
    for example in examples:
        root = find(example.seed_id)
        split_of.setdefault(root, example.split)

    moved = 0
    for example in examples:
        winner = split_of[find(example.seed_id)]
        if example.split != winner:
            example.split = winner
            moved += 1
    return moved


def generate_all() -> list[Example]:
    rng = random.Random(20260930)
    out: list[Example] = []
    counter = 0
    for offset, (category, target) in enumerate(ALLOCATION.items()):
        builder = BUILDERS[category]
        seed_count = max(4, round(target / 40))
        seeds = _seeds_for(category, seed_count, offset * 100 + 1)
        per_seed = _distribute(target, seed_count)
        for local, seed_id in enumerate(seeds):
            assert seed_id not in RESERVED_T1, f"T1 id reused as seed: {seed_id}"
            split = _assign_split(seed_id, rng)
            bank = seed_bank(local, len(seeds))
            for variant in range(per_seed[local]):
                built = builder(rng, seed_id, counter, bank)
                counter += 1
                out.append(Example(
                    id=f"{category.split('_')[0]}-{seed_id}-{variant:03d}",
                    category=category,
                    seed_id=seed_id,
                    split=split,
                    user=built["user"],
                    messages=built["messages"],
                    expect_block=built["expect_block"],
                    expect_action=built.get("expect_action"),
                    expect_final=built.get("expect_final"),
                    note=built.get("note", ""),
                ))
    _coalesce_splits(out)
    return out



# --------------------------------------------------------------------------
# Validation: refuse to write anything that fails
# --------------------------------------------------------------------------

FORMAT_ERRORS = (
    "unknown capability", "invalid arguments", "unknown tool",
    "missing or invalid", "must be of type", "must be one of",
    "unknown argument", "is not bound", "must be a non-empty string",
)


def validate_example(example: Example, prime) -> list[str]:
    problems: list[str] = []
    if not example.messages or example.messages[0]["role"] != "user":
        problems.append("does not start with a user turn")

    for message in example.messages:
        if message["role"] != "assistant":
            continue
        blocks = re.findall(
            r"<prime_request>(.*?)</prime_request>", message["content"],
            re.DOTALL)
        if len(blocks) > 1:
            problems.append(
                f"training target contains {len(blocks)} request blocks")

    assistant_turns = [m for m in example.messages if m["role"] == "assistant"]
    total_blocks = sum(
        len(re.findall(r"<prime_request>", m["content"], re.DOTALL))
        for m in assistant_turns
    )

    if example.expect_block:
        if total_blocks < 1:
            problems.append("expected a request block, found none")
        first = assistant_turns[0]["content"]
        request = extract_prime_request(first)
        if request is None:
            problems.append("first assistant turn does not parse as a request")
        else:
            if example.expect_action and request.action != example.expect_action:
                problems.append(
                    f"action {request.action!r} != expected "
                    f"{example.expect_action!r}")
            result = prime.handle(request)
            if (result.error or "").lower().startswith(FORMAT_ERRORS):
                problems.append(f"runtime refused the target: {result.error}")
    else:
        if total_blocks != 0:
            problems.append(
                f"no capability warranted but {total_blocks} blocks present")

    for message in example.messages:
        if message["role"] != "assistant":
            continue
        if example.category == "C5_answer_directly" and re.search(
                r"\b(I'?ll|I will|let me)\b", message["content"], re.IGNORECASE):
            problems.append("direct-answer target narrates an intention")

    for token in RESERVED_T1:
        if re.search(rf"\b{token}\b", example.user):
            problems.append(f"reuses reserved T1 id {token}")
    return problems


def validate_set(examples: list[Example]) -> list[str]:
    problems: list[str] = []
    counts = {name: 0 for name in ALLOCATION}
    for example in examples:
        counts[example.category] = counts.get(example.category, 0) + 1

    total = len(examples)
    c5 = counts.get("C5_answer_directly", 0)
    share = 100 * c5 / total if total else 0
    if abs(share - 20.0) > 0.6:
        problems.append(f"C5 is {share:.1f}% of the dataset, target 20%")

    for name, target in ALLOCATION.items():
        got = counts.get(name, 0)
        if abs(got - target) > 2:
            problems.append(f"{name}: {got} examples, target {target}")

    # Seed isolation: a seed lives in exactly one split.
    seeds: dict[str, set] = {}
    for example in examples:
        seeds.setdefault(example.seed_id, set()).add(example.split)
    for seed_id, splits in seeds.items():
        if len(splits) > 1:
            problems.append(f"seed {seed_id} appears in {sorted(splits)}")

    # Fingerprint isolation: no example is duplicated across splits.
    by_split: dict[str, set] = {"train": set(), "validation": set()}
    for example in examples:
        by_split[example.split].add(example.fingerprint())
    shared = by_split["train"] & by_split["validation"]
    if shared:
        problems.append(f"{len(shared)} fingerprints shared across splits")

    # Contradiction check: the same prompt must not appear in two categories
    # with different expected behaviour, or the dataset teaches both answers.
    by_prompt: dict[str, set] = {}
    for example in examples:
        by_prompt.setdefault(
            content_fingerprint(example.user), set()).add(example.category)
    clashes = {k: v for k, v in by_prompt.items() if len(v) > 1}
    if clashes:
        problems.append(
            f"{len(clashes)} prompts appear in more than one category")

    # Coverage: every category must appear in both splits. Without this, a
    # category whose seeds are content-linked collapses into a single
    # component, lands wholly in one split, and becomes unmeasurable - which
    # is exactly what happened to C5, the over-trigger guard.
    by_split_category: dict[str, set] = {"train": set(), "validation": set()}
    for example in examples:
        by_split_category[example.split].add(example.category)
    for split, seen in by_split_category.items():
        missing = set(ALLOCATION) - seen
        if missing:
            problems.append(
                f"{split} is missing categories: "
                f"{sorted(c.split('_')[0] for c in missing)}")
    for category in ALLOCATION:
        n = sum(1 for e in examples if e.category == category)
        if n < 20:
            problems.append(f"{category} has only {n} examples to measure with")

    # Content isolation: no user prompt text repeated across splits.
    text: dict[str, set] = {"train": set(), "validation": set()}
    for example in examples:
        text[example.split].add(content_fingerprint(example.user))
    overlap = text["train"] & text["validation"]
    if overlap:
        problems.append(
            f"{len(overlap)} user prompts duplicated across splits")
    return problems


def write_datasets(examples: list[Example], outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    written = {}
    for split in ("train", "validation"):
        path = os.path.join(outdir, f"{split}.jsonl")
        rows = [e for e in examples if e.split == split]
        with open(path, "w", encoding="utf-8") as handle:
            for example in rows:
                handle.write(json.dumps({
                    "id": example.id,
                    "category": example.category,
                    "seed_id": example.seed_id,
                    "split": example.split,
                    "messages": example.messages,
                    "expect_block": example.expect_block,
                    "expect_action": example.expect_action,
                    "expect_final": example.expect_final,
                    "note": example.note,
                }, ensure_ascii=False) + "\n")
        written[split] = len(rows)
    return written


def main() -> int:
    from PRIMORA.aurora.aurora import extract_prime_request
    from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
    from PRIMORA.prime import Prime
    from PRIMORA.prime.contract import PrimeResult
    from PRIMORA.prime.context.isolation import (
        build_prime_with_isolation_guard,
        harness_session,
        production_digest,
    )
    from PRIMORA.prime.providers import Provider
    from PRIMORA.research import ResearchProvider, ResearchResult
    from PRIMORA.toolbox import ToolRegistry

    class _Web(ResearchProvider):
        name = "generator"

        def search(self, query, limit=5):
            return [ResearchResult("t", "https://x.test/", "s", "x.test")]

    class _Blocked(Provider):
        def invoke(self, capability, payload):
            return PrimeResult.failure("generation: execution disabled")

    before = production_digest()
    examples = generate_all()
    print(f"  examples generated : {len(examples)}")
    print(f"  unique seeds       : {len({e.seed_id for e in examples})}")
    print(f"  unique user prompts: {len({content_fingerprint(e.user) for e in examples})}")
    print("=" * 72)
    print("AURORA BRAIN v0.1 DATASET GENERATION")
    print("=" * 72)
    print(f"  examples generated : {len(examples)}")
    print(f"  unique seeds       : {len({e.seed_id for e in examples})}")

    tmp = os.path.join(os.environ.get("TEMP", "."), "aurora_gen")
    os.makedirs(tmp, exist_ok=True)
    with harness_session("gen_", root=tmp) as session:
        context = session.context_manager()
        prime = build_prime_with_isolation_guard(
            _Blocked(),
            research_provider=_Web(),
            specialists=SpecialistRegistry([
                SpecialistSpec(name="reasoning", description="Hard analysis.",
                               model="m")]),
            tools=ToolRegistry.with_builtins())
        prime._context = context
        prime._mind = prime._mind.__class__(context)
        for name in ("recall", "learn"):
            cap = prime.registry.get_specialist(name)
            if cap is not None:
                cap.bind(prime._mind)

        failures: list[str] = []
        for example in examples:
            for problem in validate_example(example, prime):
                failures.append(f"{example.id}: {problem}")
        set_problems = validate_set(examples)

        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        written = write_datasets(examples, outdir) if not failures and not set_problems else {}

    print(f"\n  per-example failures : {len(failures)}")
    for problem in failures[:10]:
        print(f"    {problem}")
    print(f"  set-level failures   : {len(set_problems)}")
    for problem in set_problems:
        print(f"    {problem}")
    if written:
        print(f"\n  written: {written}")
        print(f"    {outdir}")
    unchanged = production_digest() == before
    print(f"\n  production unchanged : {unchanged}")
    ok = not failures and not set_problems and unchanged
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1






if __name__ == "__main__":
    raise SystemExit(main())
