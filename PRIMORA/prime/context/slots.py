"""Slot registry: config-driven personal-fact slots.

The registry lives in slots.json next to this module. Core slot
definitions (name, cardinality, scope, authority phrasings, render
templates, protection) are data, not code. Learned ``favorite <x>``
slots are derived by the normalization rule and exist only as concepts
in states; the registry never stores them and no new table is created.

All slot-routed facts are personal facts: their registry scope is
global, so they are visible from every session. Ordinary corrections
supersede only the same slot concept; protected rule slots refuse
ordinary corrections and retire only on an explicit revocation
phrasing ("stop doing X", "forget that rule").
"""

import json
import os
import re
from typing import Optional

from . import scoring as scoring_mod

_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "slots.json")
_registry_cache = None

_ARTICLE_RE = re.compile(r"^(?:the|my|a|an)\s+", re.IGNORECASE)
_ALWAYS_PREFIX_RE = re.compile(r"^always\s+", re.IGNORECASE)


def _compile_registry(data: dict) -> dict:
    """Validate the config and pre-compile every pattern once."""
    slots = data.get("slots") or []
    by_name = {}
    by_concept_slot = {}
    ordered = []
    fallback = []
    for slot in slots:
        name = slot.get("name")
        if not name or name in by_name:
            raise ValueError("slot registry: bad or duplicate slot name %r"
                             % (name,))
        kind = slot.get("kind")
        if kind not in ("single", "keyed"):
            raise ValueError("slot registry: unknown kind %r for %r"
                             % (kind, name))
        entry = dict(slot)
        entry["phrasings"] = [
            dict(ph, _compiled=re.compile(ph["pattern"],
                                          re.IGNORECASE | re.DOTALL))
            for ph in (slot.get("phrasings") or [])
        ]
        if not entry["phrasings"]:
            raise ValueError("slot registry: slot %r has no phrasings" % (name,))
        entry["retire_phrasings"] = [
            dict(ph, _compiled=re.compile(ph["pattern"],
                                          re.IGNORECASE | re.DOTALL))
            for ph in (slot.get("retire_phrasings") or [])
        ]
        entry.setdefault("concept_slot", name)
        entry.setdefault("mtype", "FACT")
        entry.setdefault("scope", "global")
        entry.setdefault("protected", False)
        by_name[name] = entry
        if kind == "single":
            by_concept_slot[name] = entry
        else:
            by_concept_slot[entry["concept_slot"]] = entry
        if entry.get("fallback_only"):
            fallback.append(entry)
        else:
            ordered.append(entry)
    learned = data.get("learned_favorite") or {}
    learned = dict(learned)
    learned["_compiled"] = re.compile(
        learned.get("pattern", "^$"), re.IGNORECASE | re.DOTALL)
    learned.setdefault("synonyms", {})
    revocations = [
        dict(ph, _compiled=re.compile(ph["pattern"],
                                      re.IGNORECASE | re.DOTALL))
        for ph in (data.get("revocations") or [])
    ]
    return {
        "by_name": by_name,
        "by_concept_slot": by_concept_slot,
        "ordered": ordered,
        "fallback": fallback,
        "learned_favorite": learned,
        "revocations": revocations,
        "notes_cap": int(data.get("notes_cap", 20)),
        "block_heading": data.get(
            "block_heading", "The user's facts:"),
    }


def load_registry() -> dict:
    """Load and cache the slot registry from slots.json."""
    global _registry_cache
    if _registry_cache is None:
        with open(_REGISTRY_PATH, encoding="utf-8") as fh:
            _registry_cache = _compile_registry(json.load(fh))
    return _registry_cache


def notes_cap() -> int:
    return load_registry()["notes_cap"]


def block_heading() -> str:
    return load_registry()["block_heading"]


def normalize_slot_key(text: str) -> str:
    """Canonical key form for a keyed slot concept."""
    out = re.sub(r"\s+", " ", (text or "").strip().lower())
    return _ARTICLE_RE.sub("", out).strip()


def favorite_noun(noun: str, synonyms: Optional[dict] = None) -> str:
    """Normalize a learned favorite noun: synonyms, then singular fold.

    "fav colour" -> "color", "favorite cars" -> "car": the same noun
    always lands in the same slot whatever the phrasing.
    """
    table = synonyms if synonyms is not None else \
        load_registry()["learned_favorite"]["synonyms"]
    words = []
    for word in (noun or "").lower().replace("-", " ").split():
        word = table.get(word, word)
        words.append(scoring_mod.singular(word))
    return " ".join(words)


def _apply_transform(name: str, text: str) -> str:
    if name == "strip_article":
        return _ARTICLE_RE.sub("", (text or "").strip()).strip()
    if name == "strip_always":
        return _ALWAYS_PREFIX_RE.sub("", (text or "").strip()).strip()
    return text


def _finalize(sentence: str) -> str:
    out = re.sub(r"\s+", " ", (sentence or "").strip())
    if out and not out.endswith("."):
        out += "."
    return out


def _route_phrasings(entry: dict, phrasing: dict, groups: dict,
                     text_value: str) -> list[dict]:
    value = text_value.rstrip(".").strip()
    if not value:
        return []
    min_words = phrasing.get("min_words")
    if min_words is not None and len(value.split()) < int(min_words):
        return []
    if phrasing.get("split_name"):
        # "My name is Amir Shahbazi": first word fills the first-name
        # slot, last word the last-name slot; a one-word value fills
        # only first name. "I am tired" never matches these patterns.
        words = value.split()
        first = words[0]
        last = words[-1] if len(words) > 1 else None
        out = [{
            "slot": "first name",
            "concept": "first name",
            "value": first,
            "kind": "establish",
            "mtype": entry.get("mtype", "FACT"),
            "scope": entry.get("scope", "global"),
            "protected": False,
        }]
        if last is not None:
            out.append({
                "slot": "last name",
                "concept": "last name",
                "value": last,
                "kind": "establish",
                "mtype": entry.get("mtype", "FACT"),
                "scope": entry.get("scope", "global"),
                "protected": False,
            })
        return out
    if entry["kind"] == "single":
        concept = entry["name"]
    else:
        # Keyed slots key by the object/topic the phrasing names, not by
        # the rendered value: "I hate pizza" and "I love pizza" share the
        # key "pizza" so the sentiment change supersedes in place.
        key_group = phrasing.get("key_group")
        if key_group and key_group in groups:
            key = normalize_slot_key(groups[key_group])
        else:
            key = normalize_slot_key(value)
        if not key:
            return []
        concept = "%s for %s" % (entry["concept_slot"], key)
    return [{
        "slot": entry["name"],
        "concept": concept,
        "value": value,
        "kind": "establish",
        "mtype": entry.get("mtype", "FACT"),
        "scope": entry.get("scope", "global"),
        "protected": bool(entry.get("protected")),
    }]


def _match_entry(entry: dict, text: str) -> list[dict]:
    for phrasing in entry["retire_phrasings"]:
        match = phrasing["_compiled"].match(text)
        if match:
            groups = {k: v for k, v in match.groupdict().items()
                      if v is not None}
            value = (groups.get("value") or "").strip().rstrip(".").strip()
            if value:
                return [{
                    "slot": entry["name"],
                    "concept": entry["name"],
                    "value": value,
                    "kind": "retire_slot",
                    "mtype": entry.get("mtype", "FACT"),
                    "scope": entry.get("scope", "global"),
                    "protected": bool(entry.get("protected")),
                }]
    for phrasing in entry["phrasings"]:
        match = phrasing["_compiled"].match(text)
        if not match:
            continue
        groups = {k: v for k, v in match.groupdict().items()
                  if v is not None}
        for transform in phrasing.get("transforms", []):
            for key in list(groups):
                groups[key] = _apply_transform(transform, groups[key])
        if phrasing.get("value_template"):
            try:
                value = phrasing["value_template"].format(**groups)
            except (KeyError, IndexError):
                continue
        elif "value" in groups:
            value = groups["value"]
        elif "obj" in groups:
            value = groups["obj"]
        else:
            continue
        routed = _route_phrasings(entry, phrasing, groups, value)
        if routed:
            return routed
    return []


def route_statement(text: str, include_fallback: bool = False):
    """Match one user statement against the slot registry.

    Returns a list of routed dicts: at most one per matched entry, or
    two when a split_name phrasing fills both first and last name.
    Empty list when no slot phrasing matches. Revocations are checked
    first: they are the only way to retire a protected rule.
    """
    work = (text or "").strip()
    if not work:
        return []
    reg = load_registry()
    for phrasing in reg["revocations"]:
        match = phrasing["_compiled"].match(work)
        if match:
            value = (match.groupdict().get("value") or "").strip()
            return [{
                "slot": "rules revoke",
                "concept": "",
                "value": value,
                "kind": "invalidate",
                "mtype": "INSTRUCTION",
                "scope": "global",
                "protected": True,
            }]
    for entry in reg["ordered"]:
        routed = _match_entry(entry, work)
        if routed:
            return routed
    learned = reg["learned_favorite"]
    match = learned["_compiled"].match(work)
    if match:
        noun = favorite_noun(match.group("noun"), learned["synonyms"])
        value = (match.group("value") or "").strip().rstrip(".").strip()
        if noun and value:
            return [{
                "slot": "favorite",
                "concept": "favorite " + noun,
                "value": value,
                "kind": "establish",
                "mtype": "FACT",
                "scope": "global",
                "protected": False,
            }]
    if include_fallback:
        for entry in reg["fallback"]:
            routed = _match_entry(entry, work)
            if routed:
                return routed
    return []


def is_protected_concept(concept: str) -> bool:
    """True for concepts belonging to a protected (rule) slot."""
    if " for " not in (concept or ""):
        return False
    slot_name = concept.split(" for ", 1)[0]
    entry = load_registry()["by_concept_slot"].get(slot_name)
    return bool(entry and entry.get("protected"))


def is_slot_concept(concept: str) -> bool:
    """True when the concept was produced by the slot registry."""
    reg = load_registry()
    concept = concept or ""
    if " for " in concept:
        slot_name = concept.split(" for ", 1)[0]
        if slot_name in reg["by_concept_slot"]:
            return True
        return False
    if concept in reg["by_name"]:
        return True
    return concept.startswith("favorite ")


def render_sentence(concept: str, value: str) -> str:
    """Plain-sentence rendering of one stored fact for the model.

    Slot concepts use their registry template; everything else falls
    back to a generic sentence with an explicit subject. No machine
    format, no provenance tags, no superseded marker.
    """
    concept = (concept or "").strip()
    value = (value or "").strip().rstrip(".")
    if not concept:
        return _finalize(value or "")
    reg = load_registry()
    if " for " in concept:
        slot_name = concept.split(" for ", 1)[0]
        entry = reg["by_concept_slot"].get(slot_name)
        if entry is not None:
            return _finalize(entry["render"].format(value=value))
    entry = reg["by_name"].get(concept)
    if entry is not None:
        return _finalize(entry["render"].format(value=value))
    if concept.startswith("favorite "):
        noun = concept[len("favorite "):]
        return _finalize(reg["learned_favorite"]["render"].format(
            noun=noun, value=value))
    return _finalize("The user's %s is %s." % (concept, value))
