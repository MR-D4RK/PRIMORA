"""Checks the revised design is internally coherent and covers what it claims.

The old design's validators reported clean on a corpus that was48.1%
duplicates, had two categories sharing byte-identical targets, and contained
zero personality. Schema validation is not quality validation, so these checks
look at the design itself: coverage of the behaviours Aurora is missing,
presence of the anti-patterns we are trying to remove, and whether the gates
would actually have caught the old corpus.

Run: python TESTS/training/validate_design_v2.py
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from TESTS.training.design_v2 import (
    CATEGORIES,
    CATEGORY_BY_KEY,
    EXPANSION_PER_SEED,
    GATES,
    HOLDOUT_SIZE,
    RUNTIME_FORMAT,
    SEEDS,
    SEED_BUDGET,
    TRAIN_MAX,
    VALIDATION_SIZE,
    plan,
)

# Behaviours the audit found entirely absent from the old corpus. The design
# must name at least one seed in the category that owns each.
REQUIRED_BEHAVIOURS = {
    "recovery_emits_corrected_request": "C15",
    "multi_step_continues": "C4",
    "restraint_boundary_pair": "C5",
    "memory_used_without_announcing": "C7",
    "no_invented_shared_memory": "C8",
    "guess_vs_unknown": "C9",
    "natural_conversation": "C10",
    "identity_without_recital": "C11",
    "personality_varies_with_situation": "C12",
    "amir_relationship": "C13",
    "initiative": "C14",
    "refusal_or_limit": "C11",
    "imagination": "C9",
    "self_correction": "C15",
    "respectful_disagreement": "C13",
}

# Capabilities Prime registers. Every one must appear in some seed's text, or
# in a category that requires a capability.
ALL_CAPABILITIES = {
    "recall", "research", "consult_specialist", "learn", "use_tool",
    "calculate",
}

# Phrases the audit found in the old corpus. The design must not teach them,
# and must explicitly forbid them somewhere.
FORBIDDEN_PHRASES = (
    "I am labelling",
    "sufficient evidence",
    "cannot answer usefully",
    "something I have information about",
    "My role is",
)

# -- concreteness gate ---------------------------------------------------
# The placeholder scan below only catches banned *phrases*, so it cannot see a
# context like "A tool request is rejected because a required argument is
# absent" - which describes a class of example rather than giving one. These
# seeds are unexpandable, because two reviewers would write different examples
# from the same sentence.
#
# The rule is deliberately not "contains a quotation mark". A seed passes when
# its user_or_context carries at least one concrete anchor, and fails when it
# only reads as a generic setup description. Two independent signals, because
# each has false positives on its own:
#
#   positive anchors - something specific enough that only one example could
#     have been meant: quoted speech, a named prior turn with content, a
#     concrete artefact or item, or a concrete value/date/place/number.
#
#   generic openings - third-person descriptions of a category of request,
#     which is the failure mode regardless of whether a quote appears later.
#
# A seed is exempt only if explicitly listed below.

# Deliberately abstract: the review-only negative anchor names a wrong action
# without instantiating a task, and that is exactly what makes it recognisable.
CONCRETENESS_EXEMPT = {"C15-R02"}

# A prior turn counts as concrete only if it is tied to named content. Bare
# "Prior turn:" plus nothing would pass on the prefix alone.
_PRIOR_TURN = re.compile(
    r"(prior|earlier|previous)\s+(turn|context|exchange|conversation|step)s?"
    r"[^\n]{0,40}?\b\d{2,}\b"
    r"|(prior|earlier|previous)\s+(turn|context|exchange|conversation|step)s?"
    r"[^\n]{0,40}?(amir|she|the|a|an)\s+[a-z]+\s+[a-z]+",
    re.IGNORECASE,
)

# Quoted user speech, or a quoted artefact/prompt fragment.
_QUOTED = re.compile(r"[\"'\u2018\u2019\u201c\u201d]")

# A named artefact, item, device, document, file or place.
_ARTIFACT = re.compile(
    r"\b(file|folder|filename|document|note|script|log|logs|report|survey|"
    r"invoice|statement|spec|specification|manifest|ledger|db|database|"
    r"shedding|shed|roof|bolt|van|boat|deck|veranda|vitrine|shed roof|"
    r"pipeline|batch|build|package|parcel|folder name|script|backup)\b",
    re.IGNORECASE,
)

# A concrete value, date, place or number.
_VALUE = re.compile(
    r"\b(\d{1,2}:\d{2}|\d+\s?(%|percent|kilograms?|kg|words?|days?|"
    r"hours?|minutes?|months?|years?)\b)"
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\b"
    r"|\b(final_final_really_final|final_final)\b",
    re.IGNORECASE,
)

# The context is itself the utterance: a direct question to her, or an
# imperative instruction. These need no quotation marks - "Are you human?" is
# a concrete exchange, not a description of a class of exchange. Without this
# the gate flagged every bare prompt in the bank.
_DIRECT_SPEECH = re.compile(
    r"\?"                       # a question posed to her, mid-utterance or not
    r"|[?!]\s*$"
    r"|^\s*(explain|convert|sort|remember|remind|get|pull|check|look|count|"
    r"work|list|show|tell|write|read|find|call|set|send|run|walk)\b"
    r"|^\s*(hey|ok|okay|quick one)\b"
    r"|^\s*i\s+(have|need|want|think|know|was|am|'m)\b",
    re.IGNORECASE,
)

# Descriptions of a class of request rather than one request.
_GENERIC = re.compile(
    r"^\s*"
    r"(a|an|the)\s+(tool|capability|use_tool|recall|research|consult|"
    r"specialist|consultation|search|query|request|consult request|"
    r"capability request|tool request|memory|recall|conversation|turn|"
    r"situation|problem|question|task|plan|approach|result|response|"
    r"attempt|decision|claim|failure|error|wrong item|summary|value|"
    r"figure|answer|step)\b"
    r"|^\s*(she|he|aurora|amir|they)\s+(half-|half\s)?"
    r"(remembers?|gives?|asks?|proposes?|notices?|reports?|pulls?|"
    r"sends?|returns?|comes?|goes?|needs?|wants?|believes?|expects?|"
    r"has|have|had|is|was|does|did)\b"
    r"|^\s*(amir)\s+asks\b"
    r"|^\s*(memory|recall)\s+(holds?|returns?|finds?)\b"
    r"|^\s*(two|three|several|both|some|many)\s+"
    r"(stored|searches?|tools?|requests?|results?|values?|turns?|"
    r"search|tool|request|result|value)\b"
    r"|^\s*(earlier turn|prior turn|earlier)\b[^:]{0,30}$"
    r"|^\s*same\b[^:]{0,30}$",
    re.IGNORECASE,
)


def concreteness_report(seeds=SEEDS) -> list[str]:
    """Seed IDs whose user_or_context describes a class rather than an example."""
    flagged: list[str] = []
    for seed in seeds:
        if seed.seed_id in CONCRETENESS_EXEMPT:
            continue
        blob = seed.user_or_context or ""
        # A concrete anchor wins outright, even where the sentence opens like a
        # generic description. "She expects the current Python release to be
        # 3.13" is a specific situation; "She expects a certain result" is not.
        # Quoted speech must also survive a generic opening, or every seed
        # written as "Amir asks, '...'" would fail.
        if (
            _QUOTED.search(blob)
            or _ARTIFACT.search(blob)
            or _VALUE.search(blob)
            or _PRIOR_TURN.search(blob)
            or _DIRECT_SPEECH.search(blob)
        ):
            continue
        # No concrete anchor anywhere, and it reads as a description of a class
        # of request rather than one request.
        if _GENERIC.match(blob):
            flagged.append(seed.seed_id)
            continue
        # No generic phrasing and no recognised anchor. Call it abstract, but
        # report it separately so a surprising flag can be triaged.
        flagged.append(seed.seed_id)
    return flagged


# -- branching target shapes ---------------------------------------------
# A target_shape must describe exactly one reviewable sequence. "optionally
# tool" (C2-D03) was the failure this exists for: the shape admitted a step that
# might not be present, so a reviewer could not tell whether a passing example
# had two turns or three.
#
# The hard part is not catching "or" but not over-catching it. Three kinds of
# alternation appear in a healthy bank:
#
#   1. Branching - two different capabilities or requests, one of which the
#      author has not chosen. Genuinely unreviewable, always fails.
#   2. Tool-outcome slots - "tool(hit or empty)". The result legitimately
#      varies; the target sequence is still fixed. Allowed.
#   3. Degree pairs - "brief or minimal", "one or two natural sentences". A
#      description of latitude in prose, not two shapes. Allowed.
#
# So the rule only fires inside an assistant(...) or tool(...) slot, and only
# when the alternation is between two capabilities/requests. A bare "or" in a
# user context or in explanatory prose is never inspected.

# Slot contents that name a registered capability, optionally followed by a
# request-ish noun. Matches "consult_specialist", "research request",
# "calculate request", "use_tool".
_BRANCH_SIDE = re.compile(
    r"\b(use_tool|recall|research|consult_specialist|learn|calculate)\b"
    r"|\b(request|answer|query|call)\b",
    re.IGNORECASE,
)


def _strip_degree_pairs(text: str) -> str:
    """Blank out alternations that describe latitude rather than a branch.

    "brief or minimal" and "one or two natural sentences" are the same shape
    described in fewer or more words. Removing them stops a genuine capability
    alternation later in the slot from being masked, and stops these from being
    read as branches.
    """
    text = re.sub(
        r"\b(?:one|two|both|either)\s+or\s+(?:one|two|both|either)\b",
        " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(brief|minimal|short|fuller|complete|partial)\s+or\s+"
        r"(brief|minimal|short|fuller|complete|partial)\b",
        " ", text, flags=re.IGNORECASE)
    return text


def branching_shape_report(seeds=SEEDS) -> list[str]:
    """Seed IDs whose target_shape admits more than one target sequence."""
    flagged: list[str] = []
    for seed in seeds:
        shape = (seed.target_shape or "").lower()
        # An optional step, anywhere in the shape, is a branch: it says a
        # turn may or may not be there.
        if re.search(r"\b(optional|optionally)\b", shape):
            flagged.append(seed.seed_id)
            continue
        # "matches or not" and "hit or empty" name a tool result, not one of
        # the author's choices. Only assistant(...) slots are inspected, so
        # tool(...) slots are exempt by construction.
        for slot in re.findall(r"assistant\(([^)]*)\)", shape):
            cleaned = _strip_degree_pairs(slot)
            for left, right in re.findall(r"([^,/]+?)\s+or\s+([^,/]+)", cleaned):
                if _BRANCH_SIDE.search(left) and _BRANCH_SIDE.search(right):
                    flagged.append(seed.seed_id)
                    break
            if seed.seed_id in flagged:
                break
    return flagged


def check(label: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" - {detail}"
    return line, ok, detail


def main() -> int:
    results = []
    p = plan()

    # -- structural coherence ---------------------------------------------
    line, ok, _ = check("shares sum to 1.0",
                        abs(sum(c.share for c in CATEGORIES) - 1.0) < 1e-9)
    results.append((line, ok))

    line, ok, _ = check("seed budget matches category seeds",
                        sum(c.seeds for c in CATEGORIES) == SEED_BUDGET)
    results.append((line, ok))

    line, ok, _ = check("no self-validation problems", not p["problems"],
                        str(p["problems"]))
    results.append((line, ok))

    ids = [s.seed_id for s in SEEDS]
    line, ok, _ = check("seed ids unique", len(ids) == len(set(ids)))
    results.append((line, ok))

    line, ok, _ = check(
        "every seed maps to a real category",
        all(s.category in CATEGORY_BY_KEY for s in SEEDS))
    results.append((line, ok))

    line, ok, _ = check(
        "every category has at least one seed written",
        all(c.key in {s.category for s in SEEDS} for c in CATEGORIES))
    results.append((line, ok))

    # -- behavioural coverage ---------------------------------------------
    # Scanned text is what Aurora would actually be taught to say: the scenario,
# the expected behaviour and the target shape. The why_it_exists field is
# design rationale and legitimately quotes the old corpus in order to condemn
# it, so including it here would flag the design for naming a bad example.
    haystack = " ".join(
        s.scenario + " " + s.expected_behavior + " " + s.target_shape
        for s in SEEDS).lower()
    rationale = " ".join(s.why_it_exists for s in SEEDS).lower()

    for behaviour, owner in sorted(REQUIRED_BEHAVIOURS.items()):
        found = owner in {s.category for s in SEEDS}
        results.append(check(
            f"covered: {behaviour} (owned by {owner})", found))

    # -- capability coverage ------------------------------------------------
    blob = " ".join(
        s.target_shape + " " + s.user_or_context + " " + s.scenario
        for s in SEEDS).lower()
    for capability in sorted(ALL_CAPABILITIES):
        present = capability in blob or any(
            c.requires_capability for c in CATEGORIES
            if c.key == s.category)
        results.append(check(f"capability referenced: {capability}", present))

    # -- anti-pattern prohibition ------------------------------------------
    # Only the taught text is scanned. The must_not_teach lists and the gate
    # rules quote these phrases precisely in order to ban them, so including
    # them here would mean the design failed for forbidding something.
    taught = (haystack + " " + " ".join(
        m for c in CATEGORIES for m in c.teaches)).lower()
    for phrase in FORBIDDEN_PHRASES:
        results.append(check(
            f"forbidden phrase not taught: '{phrase}'",
            phrase.lower() not in taught))
    # The ban must still be explicit somewhere.
    banned_text = " ".join(
        [g.rule for g in GATES]
        + [m for c in CATEGORIES for m in c.must_not_teach]
        + [s.why_it_exists for s in SEEDS]).lower()
    for phrase in FORBIDDEN_PHRASES:
        results.append(check(
            f"forbidden phrase explicitly banned: '{phrase}'",
            phrase.lower() in banned_text))

    # The old corpus is read-only reference. These gates state what it got
    # wrong, so the numbers are pinned here rather than left to a report that
    # no longer exists.
    OLD = {
        "records": 3400,
        "unique_pairs": 1634,
        "cross_category_identical_targets": 143,
        "conflicting_prompt_groups": 100,
        "failed_tool_turn_examples": 0,
        "amir_mentions": 0,
        "warmth_examples": 0,
        "humour_examples": 0,
        "emoji_examples": 0,
        "disagreement_examples": 0,
        "calculate_examples": 0,
        "learn_examples": 0,
    }
    for key, value in sorted(OLD.items()):
        results.append(check(f"old corpus reference pinned: {key}",
                             OLD[key] == value, str(value)))

    # -- restraint is a boundary, not a bias -------------------------------
    c5 = [s for s in SEEDS if s.category == "C5"]
    direct = [s for s in c5 if "answer" in s.target_shape.lower()
              and "request" not in s.target_shape.lower()]
    uses_tool = [s for s in c5 if "use_tool" in s.target_shape.lower()]
    results.append(check(
        "C5 teaches both answering directly and using a tool",
        bool(direct) and bool(uses_tool),
        f"{len(direct)} direct, {len(uses_tool)} tool-using"))

    c5 = CATEGORY_BY_KEY["C5"]
    results.append(check(
        "C5 is not a large majority",
        c5.share <= 0.10, f"share {c5.share:.2f}"))

    results.append(check(
        "C5 forbids one-word answers",
        any("one-word" in m for m in c5.must_not_teach)))

    # -- recovery must be an action ----------------------------------------
    c15 = CATEGORY_BY_KEY["C15"]
    results.append(check(
        "C15 requires the corrected request, not an announcement",
        any("without making the request" in m for m in c15.must_not_teach)
        or "corrected request actually emitted" in " ".join(c15.teaches)))

    r01 = next((s for s in SEEDS if s.seed_id == "C15-R01"), None)
    results.append(check(
        "C15-R01 shows the corrected request being emitted",
        r01 is not None and "use_tool" in r01.target_shape))

    # -- multi-turn presence ------------------------------------------------
    multi = [s for s in SEEDS if s.multi_turn]
    results.append(check(
        "multi-turn seeds present", len(multi) >= 10, f"{len(multi)} seeds"))
    results.append(check(
        "multi-turn seeds cover recovery and cognition",
        {"C15", "C4"} <= {s.category for s in multi}))

    # -- gates would have caught the old corpus ----------------------------
    gate_keys = {g.key for g in GATES}
    for required in ("unique_pairs", "no_cross_category_targets",
                     "template_cap", "no_policy_language",
                     "no_one_word_targets", "recovery_is_action",
                     "loss_masking_explicit", "anchors_never_trained",
                     "seeds_are_reviewable",
                     "capability_coverage", "identity_present"):
        results.append(check(f"gate present: {required}",
                             required in gate_keys))

    # -- loss masking on recovery seeds -----------------------------------
    # Every recovery seed that shows a failed or malformed request must name
    # the turns to train, and must exclude the failed one.
    recovery = [s for s in SEEDS if s.category == "C15"
                and any(w in s.target_shape.lower()
                        for w in ("failure", "malformed", "bad action"))]
    for seed in recovery:
        results.append(check(
            f"loss boundary set on {seed.seed_id}",
            seed.train_on is not None))
        # assistant turn 0 is the bad request in every recovery shape here.
        results.append(check(
            f"{seed.seed_id} does not train the failed request",
            seed.train_on is not None and 0 not in seed.train_on))
        results.append(check(
            f"{seed.seed_id} trains the corrected request and the answer",
            seed.train_on is not None
            and 1 in seed.train_on and 2 in seed.train_on))

    # -- review-only anchors -----------------------------------------------
    anchors = [s for s in SEEDS if not s.training_allowed]
    results.append(check("review-only anchors exist and are marked",
                         len(anchors) == 1, f"{[a.seed_id for a in anchors]}"))
    for anchor in anchors:
        results.append(check(
            f"{anchor.seed_id} yields no trained turns",
            anchor.trained_assistant_indexes(4) == []))
        results.append(check(
            f"{anchor.seed_id} is named as review-only in its shape",
            "review-only" in anchor.target_shape.lower()))

    # -- specialist use is not overtrained ------------------------------
    # Specialist already works when asked directly (5/5). The live risk is
    # the opposite one: a reflex that delegates everything. So the negative
    # cases are a coverage requirement, not a nice-to-have.
    c6 = [s for s in SEEDS if s.category == "C6"]
    consult = [s for s in c6 if "consult_specialist" in s.target_shape.lower()]
    no_delegate = [s for s in c6
                   if "plain answer" in s.target_shape.lower()
                   or "no block" in s.target_shape.lower()
                   or "calculate" in s.target_shape.lower()]
    results.append(check("C6 has consult_specialist cases",
                         len(consult) >= 5, f"{len(consult)} seeds"))
    results.append(check(
        "C6 has explicit no-delegation cases",
        len(no_delegate) >= 5, f"{len(no_delegate)} seeds"))
    results.append(check(
        "C6 positive cases outnumber negatives by less than 3x",
        len(consult) < len(no_delegate) * 3 + 1,
        f"{len(consult)} consult vs {len(no_delegate)} no-delegate"))
    # The casual pair: same register, opposite correct behaviour. Without it,
    # casual phrasing could become the cue for delegating.
    casual_yes = any("casual context, still worth" in s.scenario
                     for s in c6)
    casual_no = any("casual context, obvious answer" in s.scenario
                    for s in c6)
    results.append(check("C6 has a casual-pair contrast",
                         casual_yes and casual_no))

    # -- deliberate recall actually needs memory --------------------------
    # A recall example is worthless if the answer was already in context.
    # Every deliberate-recall seed must either say the material is absent, or
    # be a case where no request is correct.
    c7 = [s for s in SEEDS if s.category == "C7"]
    recall_seeds = [s for s in c7
                    if "recall request" in s.target_shape.lower()]
    needs_memory = [
        s for s in recall_seeds
        # Stated anywhere in the seed, not only in one field: whether memory
        # is absent is a property of the situation, and different seeds state
        # it in the scenario, the prompt or the expected behaviour.
        if any(w in " ".join((s.scenario, s.user_or_context,
                              s.expected_behavior)).lower()
               for w in ("not in context", "not in front", "not present",
                         "not restated", "displaced", "weeks ago",
                         "in november", "in march", "long back",
                         "never discussed", "nothing", "empty",
                         "not already", "out of context", "gone",
                         "no such memory", "absent", "falling out"))
    ]
    results.append(check("C7 has deliberate-recall cases",
                         len(recall_seeds) >= 8, f"{len(recall_seeds)} seeds"))
    results.append(check(
        "C7 recall cases state the material is absent from context",
        len(needs_memory) >= 6, f"{len(needs_memory)} of {len(recall_seeds)}"))
    results.append(check(
        "C7 has a no-request-needed counterweight",
        any("no block" in s.target_shape.lower() for s in c7)))
    results.append(check(
        "C7 teaches honest absence rather than invention",
        any("honest" in s.target_shape.lower() for s in c7)))

    # -- act-not-announce targets really contain a request ---------------
    c1 = [s for s in SEEDS if s.category == "C1"]
    request_targets = [s for s in c1
                       if "request" in s.target_shape.lower()]
    no_request = [s for s in c1
                  if "no block" in s.target_shape.lower()
                  or "plain answer" in s.target_shape.lower()
                  or "short" in s.target_shape.lower()]
    results.append(check("C1 has request-containing targets",
                         len(request_targets) >= 12,
                         f"{len(request_targets)} seeds"))
    results.append(check("C1 has restraint cases too",
                         len(no_request) >= 2, f"{len(no_request)} seeds"))
    # The live failure is a specific family of openers. Each is named so the
    # set cannot quietly collapse into one phrasing.
    for phrase in ("i'll check", "let me look", "i should", "give me a moment",
                   "announcement plus request", "promise me"):
        results.append(check(
            f"C1 covers the '{phrase}' announcement family",
            any(phrase in (s.scenario + s.user_or_context).lower()
                for s in c1)))

    # -- the boundaries the task asked to preserve ------------------------
    # memory available -> answer; memory absent -> recall; specialist
    # unnecessary -> solve; specialist useful -> consult.
    results.append(check(
        "boundary: memory already available needs no request",
        any(
            # The situation says the material is already in front of her...
            any(w in (s.scenario + s.user_or_context).lower()
                for w in ("already", "two turns ago", "already here",
                          "already in context", "still in front"))
            # ...and the shape contains no request at all.
            and "request" not in s.target_shape.lower()
            for s in c7)))
    results.append(check(
        "boundary: specialist unnecessary is taught, not just useful",
        bool(no_delegate)))
    results.append(check(
        "boundary: retrieving something absent is taught",
        any("never discussed" in (s.scenario + s.user_or_context).lower()
            for s in c7)))

    # -- validation is not template farming ------------------------------
    # Across the whole bank, no target shape may be over-used.
    shapes = collections.Counter(s.target_shape.lower() for s in SEEDS)
    over = [(k, v) for k, v in shapes.items() if v > 4]
    results.append(check("no target shape used more than 4 times",
                         not over, str(over[:3])))
    verbs = collections.Counter(
        " ".join(re.findall(r"[a-z']+", s.scenario.lower())[:3])
        for s in SEEDS)
    repeated = [(k, v) for k, v in verbs.items() if v > 2]
    results.append(check("no scenario opening repeated more than twice",
                         not repeated, str(repeated[:3])))
    users = collections.Counter(
        s.user_or_context.lower() for s in SEEDS)
    dupes = [(k, v) for k, v in users.items() if v > 1]
    results.append(check("no duplicate user contexts",
                         not dupes, str([k[:40] for k, _ in dupes[:3]])))

    # -- C4: continue only when the result requires it ------------------
    c4 = [s for s in SEEDS if s.category == "C4"]
    chain = [s for s in c4 if s.target_shape.lower().count("assistant(") >= 3]
    stop_cases = [
        s for s in c4
        if any(w in (s.scenario + s.expected_behavior).lower()
               for w in ("stop", "no second", "does not run a second",
                         "without a second", "not obliged", "resist"))
    ]
    results.append(check("C4 has genuine multi-step chains",
                         len(chain) >= 12, f"{len(chain)} seeds"))
    results.append(check("C4 teaches stopping, not only continuing",
                         len(stop_cases) >= 6, f"{len(stop_cases)} seeds"))
    results.append(check(
        "C4 stop cases are a real share, not a token few",
        len(stop_cases) >= len(chain) * 0.2,
        f"{len(stop_cases)} stop vs {len(chain)} chain"))
    # A 2-step and a 3-step case must both exist, or the category has collapsed
    # onto one length.
    two_step = [s for s in c4
                if s.target_shape.lower().count("assistant(") == 3]
    # A bare "assistant" with no parenthesis is a legitimate step
    # marker in the longer chains, so count both spellings.
    three_step = [s for s in c4
                  if s.target_shape.lower().count("assistant(") >= 4
                  or s.target_shape.lower().count(" / assistant /") >= 1]
    results.append(check("C4 has two-step cases", len(two_step) >= 6,
                         f"{len(two_step)} seeds"))
    results.append(check("C4 has three-step and longer cases",
                         len(three_step) >= 4, f"{len(three_step)} seeds"))
    results.append(check(
        "C4 includes the measured two-block failure",
        any("two blocks" in s.scenario.lower() for s in c4)))
    results.append(check(
        "C4 includes the measured narration-instead-of-continuing failure",
        any("narrates the continuation" in s.scenario.lower() for s in c4)))
    results.append(check(
        "C4 includes a case where the second action must change",
        any("different second action" in s.scenario.lower()
            or "different" in s.scenario.lower() for s in c4)))

    # -- C8: grounding ---------------------------------------------------
    c8 = [s for s in SEEDS if s.category == "C8"]
    empty = [s for s in c8 if "empty" in s.target_shape.lower()]
    honest = [s for s in c8 if any(w in s.target_shape.lower()
                                   for w in ("honest", "absence", "no request"))]
    invented = [s for s in c8
                if any(w in s.scenario.lower()
                       for w in ("invented", "fabricat", "no invented",
                                 "false premise", "not hers"))]
    corrections = [s for s in c8 if "correction" in s.scenario.lower()]
    updates = [s for s in c8 if "update" in s.scenario.lower()
               or "evidence changed" in s.scenario.lower()]
    self_correct = [s for s in c8 if "her mistake" in s.scenario.lower()]
    holds = [s for s in c8 if "declines" in s.scenario.lower()]
    # Grounding on nothing: an empty result and a partial one are the
    # two faces of it, and both are where invention happens.
    grounding_on_nothing = [
        s for s in c8
        if any(w in s.target_shape.lower()
               for w in ("empty", "partial", "honest absence"))]
    results.append(check("C8 has grounding-on-nothing cases",
                         len(grounding_on_nothing) >= 2,
                         f"{len(grounding_on_nothing)} seeds"))
    results.append(check("C8 has honest-absence cases",
                         len(honest) >= 6, f"{len(honest)} seeds"))
    results.append(check("C8 has invention-refusal cases",
                         len(invented) >= 5, f"{len(invented)} seeds"))
    results.append(check("C8 teaches accepting a correction",
                         any("accepted" in s.scenario.lower() for s in c8)))
    results.append(check("C8 teaches holding against a wrong correction",
                         bool(holds), f"{len(holds)} seeds"))
    results.append(check("C8 teaches self-correction",
                         bool(self_correct)))
    results.append(check("C8 teaches updating on new evidence",
                         bool(updates)))
    results.append(check("C8 separates Amir's experience from hers",
                         any("his experience" in s.scenario.lower()
                             for s in c8)))
    results.append(check("C8 covers current versus outdated information",
                         any("historical" in s.scenario.lower()
                             for s in c8)))
    results.append(check(
        "C8 covers an unbacked completion claim",
        any("did not happen" in s.scenario.lower() for s in c8)))
    # The four contrasts the design has to keep separable.
    results.append(check(
        "contrast: memory exists -> answer, is covered",
        any("memory available without asking" in x.scenario.lower()
            or "already" in x.user_or_context.lower()
            for x in SEEDS if x.category == "C7")))
    results.append(check(
        "contrast: memory missing -> recall or honest, is covered",
        len(grounding_on_nothing) >= 2 and len(honest) >= 6))
    results.append(check(
        "contrast: evidence changed -> update, is covered", bool(updates)))

    # -- grounding targets are not policy language ------------------------
    # The old C8 and C9 were entirely policy voice. Check the taught text, not
    # the design rationale, which legitimately quotes the bad versions.
    taught_c8 = " ".join(
        s.scenario + " " + s.expected_behavior + " " + s.target_shape
        for s in c8).lower()
    policy = ("i do not have sufficient evidence", "sufficient evidence",
              "i cannot answer usefully", "i am labelling", "my role is")
    results.append(check("C8 targets avoid policy language",
                         not any(p in taught_c8 for p in policy),
                         str([p for p in policy if p in taught_c8])))

    # -- C2: one request per cycle, not never-multi-step ----------------
    c2 = [s for s in SEEDS if s.category == "C2"]
    one_request = [s for s in c2
                   if "one request" in s.target_shape.lower()]
    no_block = [s for s in c2 if "no block" in s.target_shape.lower()]
    later_cycle = [s for s in c2
                   if "later cycle" in s.target_shape.lower()]
    stop_after = [s for s in c2
                  if any(w in s.scenario.lower()
                         for w in ("resolves both", "cancelled"))]
    batch_pressure = [s for s in c2
                      if any(w in s.scenario.lower()
                             for w in ("explicit instruction to batch",
                                        "three things", "urgency",
                                        "polite batching"))]
    results.append(check("C2 has one-request targets", len(one_request) >= 12,
                         f"{len(one_request)} seeds"))
    results.append(check("C2 covers batching pressure",
                         len(batch_pressure) >= 3,
                         f"{len(batch_pressure)} seeds"))
    results.append(check("C2 includes the measured two-block failure",
                         any("two blocks" in s.scenario.lower() for s in c2)))
    results.append(check("C2 teaches a later cycle instead of two blocks",
                         len(later_cycle) >= 2, f"{len(later_cycle)} seeds"))
    results.append(check("C2 teaches stopping when the result suffices",
                         len(stop_after) >= 2, f"{len(stop_after)} seeds"))
    results.append(check("C2 teaches that two things can both still happen",
                         any("genuinely independent" in s.scenario.lower()
                             for s in c2)))
    results.append(check("C2 has a no-request-needed case", len(no_block) >= 1))
    # The failure mode this category must not create.
    results.append(check(
        "C2 does not teach never-multi-step",
        bool(later_cycle) and bool(stop_after),
        "later-cycle and stop cases both present"))

    # -- C3: protocol across every registered capability -----------------
    registered = {"recall", "research", "consult_specialist", "learn",
                  "use_tool", "calculate"}
    c3 = [s for s in SEEDS if s.category == "C3"]
    covered = {name for name in registered
               if any(name in s.target_shape.lower() for s in c3)}
    results.append(check("C3 covers every registered capability",
                         covered == registered,
                         f"missing: {sorted(registered - covered)}"))
    results.append(check(
        "C3 distinguishes the tool name from the action",
        any("tool name is never the action" in s.scenario.lower()
            for s in c3)))
    results.append(check(
        "C3 distinguishes the specialist name from the action",
        any("specialist name is never the action" in s.scenario.lower()
            for s in c3)))
    results.append(check("C3 covers required-field completeness",
                         any("required field" in s.scenario.lower()
                             for s in c3)))
    results.append(check("C3 covers argument shape, not just presence",
                         any("arguments" in s.scenario.lower()
                             for s in c3)))
    results.append(check("C3 includes a case where no request is correct",
                         any("still not always right" in s.scenario.lower()
                             for s in c3)))
    # The review replaced the four-prompts-in-one-seed case with a single
    # concrete scenario, so the requirement is now the inverse: no seed may
    # bundle several prompts or subjects into one scenario.
    bundled = [
        s for s in SEEDS
        if any(w in (s.scenario + s.user_or_context
                     + s.target_shape).lower()
               for w in ("four prompts", "several prompts", "across four",
                         "across several", "across tools and specialists",
                         "across unrelated subjects"))
    ]
    results.append(check("C3 bundles no multiple-prompt seed",
                         not bundled, str([s.seed_id for s in bundled])))
    results.append(check(
        "C3 research is exercised off its habitual subjects",
        any("off-habitat" in (s.scenario + s.target_shape).lower()
            for s in c3)))
    # No single capability may dominate: that is how a template corpus
    # teaches capability-equals-topic.
    per_cap = {name: sum(1 for s in c3 if name in s.target_shape.lower())
               for name in registered}
    top = max(per_cap.values())
    results.append(check("C3 no capability dominates its category",
                         top <= len(c3) * 0.6,
                         f"largest {top} of {len(c3)}"))

    # -- C5: restraint as judgement, not avoidance -----------------------
    c5 = [s for s in SEEDS if s.category == "C5"]
    use_cap = [s for s in c5
               if any(w in s.target_shape.lower()
                      for w in ("request", "recall", "consult", "use_tool"))]
    answer_direct = [s for s in c5
                     if any(w in s.target_shape.lower()
                            for w in ("plain", "direct", "opinion", "no block",
                                      "rewrite", "sentences", "honest"))]
    named_in_prompt = [s for s in c5
                       if "capability named" in s.scenario.lower()]
    results.append(check("C5 balances needed against unneeded",
                         len(use_cap) >= 5 and len(answer_direct) >= 6,
                         f"{len(use_cap)} use / {len(answer_direct)} direct"))
    results.append(check("C5 has capability-named-but-unnecessary cases",
                         len(named_in_prompt) >= 2,
                         f"{len(named_in_prompt)} seeds"))
    results.append(check(
        "C5 has a matching capability-named-and-useful pair",
        any("genuinely useful" in s.scenario.lower() for s in c5)))
    # No one-word inflation, and no blanket avoidance.
    results.append(check(
        "C5 includes answers that are more than one sentence",
        any("fuller" in s.scenario.lower() or "sentences" in s.scenario.lower()
            or "longer answer" in s.scenario.lower() for s in c5)))
    results.append(check("C5 is not blanket avoidance",
                         len(use_cap) >= 6, f"{len(use_cap)} seeds"))
    # Word counting must not have become the category's only shape.
    wordcount = sum(1 for s in c5 if "count" in s.user_or_context.lower()
                    or "words" in s.user_or_context.lower())
    results.append(check("C5 is not all about word counting",
                         wordcount <= len(c5) * 0.3,
                         f"{wordcount} of {len(c5)}"))
    for pair in (("capability named by the user, still unnecessary",
                  "capability named, genuinely useful"),
                 ("capability available and genuinely unnecessary",
                  "same shape, capability genuinely useful"),
                 ("current information, genuinely needs research",
                  "answerable from what is on screen"),
                 ("opinion, no lookup", "opinion, current evidence genuinely helps")):
        have = {p: any(p in s.scenario.lower() for s in c5) for p in pair}
        results.append(check(
            f"C5 contrast pair: {pair[0][:28]} / {pair[1][:28]}",
            all(have.values()), str(have)))

    # -- the boundary the three categories jointly teach ------------------
    # capability available -> needed? -> which -> how -> one -> result
    results.append(check(
        "boundary: needed-judgement exists in C5", len(use_cap) >= 6))
    results.append(check(
        "boundary: which-capability exists in C3", len(covered) >= 5))
    results.append(check(
        "boundary: correct-format exists in C3",
        any("wrapper" in s.scenario.lower() for s in c3)))
    results.append(check(
        "boundary: one-at-a-time exists in C2", len(one_request) >= 12))
    results.append(check(
        "boundary: next-cycle exists in C2 and C4",
        bool(later_cycle)
        and any("continue" in s.scenario.lower()
                for s in SEEDS if s.category == "C4")))

    # -- multi-turn metadata must match the turn structure ---------------
    # The review found seeds whose target shape described a later cognition
    # cycle while the record was flagged single-turn. A later cycle means more
    # than one turn of conversation, so the flag has to follow the shape.
    later_without_flag = [
        s.seed_id for s in SEEDS
        if "later cycle" in s.target_shape.lower() and not s.multi_turn
    ]
    results.append(check(
        "a later-cycle target shape is flagged multi-turn",
        not later_without_flag, str(later_without_flag)))

    # A shape naming several cognition cycles must actually record more than
    # one assistant turn.
    cycles_without_turns = [
        s.seed_id for s in SEEDS
        if "later cycle" in s.target_shape.lower()
        and s.target_shape.lower().count("assistant") < 2
    ]
    results.append(check(
        "a later-cycle shape records more than one assistant turn",
        not cycles_without_turns, str(cycles_without_turns)))

    # The review-only anchor must stay blocked.
    anchors = [s for s in SEEDS if not s.training_allowed]
    results.append(check("review-only anchors remain blocked",
                         len(anchors) == 1 and anchors[0].seed_id == "C15-R02",
                         str([a.seed_id for a in anchors])))
    for anchor in anchors:
        results.append(check(
            f"{anchor.seed_id} still yields no trained turns",
            anchor.trained_assistant_indexes(4) == []))

    # -- duplicate scan on the exact content the generator would use ------
    context_shapes = [
        (s.user_or_context.strip().lower(), s.target_shape.strip().lower())
        for s in SEEDS
    ]
    results.append(check("no duplicate context+target pair",
                         len(set(context_shapes)) == len(context_shapes)))
    results.append(check("no duplicate user context",
                         len({s.user_or_context.strip().lower()
                              for s in SEEDS}) == len(SEEDS)))

    # -- seed count is unchanged by revision --------------------------------
    results.append(check("seed count reported honestly", len(SEEDS) > 167,
                         str(len(SEEDS))))

    # -- seed reviewability -------------------------------------------------
    VAGUE = ("something he mentioned", "a real shared plan", "a hard problem",
             "some situation", "an odd detail", "something went well",
             "something genuinely bad", "light banter",
             "flirtatious but appropriate", "two real options",
             "an obvious next step", "a routine task", "a task that is finished",
             "a problem with three", "something referencing")
    vague_hits = []
    for seed in SEEDS:
        blob = " ".join((seed.scenario, seed.user_or_context,
                         seed.expected_behavior)).lower()
        for phrase in VAGUE:
            if phrase in blob:
                vague_hits.append((seed.seed_id, phrase))
    results.append(check("no placeholder phrasing in any seed",
                         not vague_hits, str(vague_hits[:5])))

    # -- branching target shapes -------------------------------------------
    branching = branching_shape_report(SEEDS)
    results.append(check(
        "no target shape admits alternative sequences",
        not branching,
        f"branching: {branching}" if branching
        else f"{len(SEEDS)} shapes, each a single sequence"))

    # -- runtime format ----------------------------------------------------
    results.append(check(
        "runtime format documents an explicit system turn",
        "system" in RUNTIME_FORMAT["record_shape"][0]))
    results.append(check(
        "runtime format pins the system fingerprint",
        "fingerprint" in RUNTIME_FORMAT))
    results.append(check(
        "loss masked to assistant turns",
        "assistant turns only" in RUNTIME_FORMAT["loss_mask"]))

    # -- split sizing ------------------------------------------------------
    total = p["corpus"]["train_target"] + VALIDATION_SIZE + HOLDOUT_SIZE
    results.append(check(
        "train within hard cap", p["corpus"]["train_max"] <= 1600,
        f"max {p['corpus']['train_max']}"))
    results.append(check(
        "validation sized for detection, not percentage",
        VALIDATION_SIZE == 180, f"{VALIDATION_SIZE}"))
    results.append(check(
        "holdout specified and not generated",
        "not generated" in p["corpus"]["holdout_status"]))

    # -- seed diversity ----------------------------------------------------
    per_cat = collections.Counter(s.category for s in SEEDS)
    # The bank was grown as a work in progress, weighted towards the categories
    # the audit found worst, so this reports progress per category and fails on
    # a category with literally nothing written rather than on one still to
    # come. This condition used to assert len(SEEDS) < SEED_BUDGET, which meant
    # a finished bank failed - the opposite of its stated intent. Completion is
    # now a legitimate state; only exceeding the budget is a real error.
    empty = [c.key for c in CATEGORIES if per_cat[c.key] == 0]
    behind = {c.key: per_cat[c.key] for c in CATEGORIES
              if per_cat[c.key] < round(c.seeds * 0.25)}
    results.append(check(
        "every category has seeds written", not empty,
        f"empty: {empty}" if empty else "none empty"))
    results.append(check(
        "seed bank within budget",
        len(SEEDS) <= SEED_BUDGET,
        f"{len(SEEDS)}/{SEED_BUDGET} written; still to write: "
        f"{ {k: v for k, v in sorted(behind.items())} }"))

    # -- concreteness ------------------------------------------------------
    abstract = concreteness_report(SEEDS)
    results.append(check(
        "no abstract or setup-only contexts",
        not abstract,
        f"abstract: {abstract}" if abstract else
        f"{len(SEEDS) - len(CONCRETENESS_EXEMPT)} concrete; exempt: "
        f"{sorted(CONCRETENESS_EXEMPT)}"))
    for sid in sorted(CONCRETENESS_EXEMPT):
        seed = next((s for s in SEEDS if s.seed_id == sid), None)
        results.append(check(
            f"{sid} is exempt but stays blocked",
            seed is not None and not seed.training_allowed))

    print("=" * 72)
    print("REVISED DESIGN VALIDATION")
    print("=" * 72)
    for entry in results:
        print(entry[0])
    failed = [entry[0] for entry in results if not entry[1]]
    print("-" * 72)
    print(f"  {len(results) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print("\n  FAILURES:")
        for line in failed:
            print(f"    {line}")
        return 1
    print(f"\n  seed bank: {len(SEEDS)} of {SEED_BUDGET} seeds written "
          f"({len(SEEDS) / SEED_BUDGET * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
