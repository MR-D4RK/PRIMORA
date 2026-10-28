"""M12 preference parsing regression: speech without punctuation or item metadata.

No game/music/car-specific hardcoding and no production memory access.
"""
from __future__ import annotations

import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.state import extract_candidates


@pytest.mark.parametrize(("utterance", "subject", "verb"), [
    (
        "i love cyberpunk i love every aspect of it whether its story "
        "its gameplay or visual all of those are the best",
        "cyberpunk", "loves",
    ),
    (
        "I adore Silver Lanterns I love every part of it, especially the arrangement",
        "silver lanterns", "adores",
    ),
    (
        "I enjoy Northwind Atlas, I love all aspects of it including the maps",
        "northwind atlas", "enjoys",
    ),
    (
        "I like Paper Gardens. I really love every bit of it, particularly the illustrations.",
        "paper gardens", "likes",
    ),
])
def test_repeated_liking_names_one_actual_subject(utterance, subject, verb):
    candidates = extract_candidates(utterance, "user")
    assert len(candidates) == 1, candidates
    candidate = candidates[0]
    assert candidate.concept == f"likes for {subject}"
    assert candidate.value.lower() == f"{verb} {subject}"
    assert candidate.scope == "global"
    assert candidate.auto_confirm
    assert candidate.rationale == ""  # Details remain sourced, not invented.


def test_compound_preference_survives_restart_with_user_source(tmp_path):
    path = tmp_path / "preference.db"
    original = (
        "i love cyberpunk i love every aspect of it whether its story "
        "its gameplay or visual all of those are the best"
    )
    manager = ContextManager(db_path=path, hybrid_retrieval=False)
    manager.reconcile([
        {"role": "user", "content": "What do you remember about my fav video games?"},
        {"role": "assistant", "content": "I don't have a verified entry. Tell me about one."},
        {"role": "user", "content": original},
    ], "session")
    states = manager._get_store().fetch_states(
        scope="global", concept="likes for cyberpunk", status="current",
    )
    assert len(states) == 1
    assert states[0].value == "loves cyberpunk"
    assert len(states[0].source_ids) == 1
    source = manager._get_store().fetch_messages_by_ids(states[0].source_ids)
    assert len(source) == 1 and source[0].content == original
    assert "story" in source[0].content
    assert "gameplay" in source[0].content
    assert "visual" in source[0].content

    restarted = ContextManager(db_path=path, hybrid_retrieval=False)
    again = restarted._get_store().fetch_states(
        scope="global", concept="likes for cyberpunk", status="current",
    )
    assert len(again) == 1
    assert again[0].source_ids == states[0].source_ids
    assert again[0].value == states[0].value


@pytest.mark.parametrize("utterance", [
    "I love this I love every aspect of it.",
    "I love anything I love every part of it.",
    "I love Northern Lights and I love Paper Gardens.",
    "If I loved Northern Lights I would love every aspect of it.",
    "I love every aspect of it.",
])
def test_no_false_named_preference_from_pronouns_or_conditional(utterance):
    candidates = extract_candidates(utterance, "user")
    # No high-confidence structured preference that turns a nonspecific
    # reference/conditional into a current named user preference.
    assert not any(
        c.auto_confirm and c.concept in (
            "likes for this", "likes for anything", "likes for northern lights"
        )
        for c in candidates
    )


def test_core_instruction_respects_answer_category_context():
    from PRIMORA.aurora.configuration import AuroraConfig
    instruction = AuroraConfig.from_env().system_instruction
    assert "interpret X within that category" in instruction
    assert "details already stated" in instruction
