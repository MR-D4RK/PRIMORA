"""Extractive digest builder. No LLM. No invented facts.

A digest is a selection of verbatim source sentences/fragments grouped by
source message, plus full provenance. A future model-assisted summarizer
can replace `build_extractive_digest` without changing store consumers:
it must return the same (content, source_ids) contract.
"""

import re
import time
from typing import Any, Callable, Optional

from . import scoring as scoring_mod
from .models import StoredMessage

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]+|[^.!?\n]+$")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def build_extractive_digest(
    messages: list[StoredMessage],
    query_terms: list[str],
    max_sentences: int = 8,
    digest_type: str = "extractive_v1",
) -> dict[str, Any]:
    """Select verbatim source sentences covering old conversation.

    Returns {"content": str, "source_ids": [...], "seq_start": int,
    "seq_end": int, "digest_type": str, "created_at": float}.
    Content is composed ONLY of source substrings with [src:id] tags.
    """
    if not messages:
        return {
            "content": "",
            "source_ids": [],
            "seq_start": 0,
            "seq_end": 0,
            "digest_type": digest_type,
            "created_at": time.time(),
        }
    now_seq = max(m.seq for m in messages)
    candidates: list[tuple[float, int, str, int]] = []
    for msg in messages:
        for sent in split_sentences(msg.content):
            terms = scoring_mod.tokenize(sent)
            rel = scoring_mod.relevance(terms, query_terms)
            imp = scoring_mod.importance_of(sent, msg.role)
            # Deterministic ordering: score desc, then seq asc.
            candidates.append((rel + 0.5 * imp, msg.seq, sent, msg.message_id))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    picked = candidates[:max(1, max_sentences)]
    # Chronological reassembly for readability.
    picked.sort(key=lambda c: c[1])
    lines = [f"[src:{mid}] {sent}" for _, _, sent, mid in picked]
    source_ids = sorted({mid for _, _, _, mid in picked})
    seqs = [m.seq for m in messages]
    return {
        "content": "\n".join(lines),
        "source_ids": source_ids,
        "seq_start": min(seqs),
        "seq_end": max(seqs),
        "digest_type": digest_type,
        "created_at": time.time(),
    }


def maybe_compact(
    store: Any,
    session_id: str,
    cutoff_seq: int,
    query_terms: Optional[list[str]] = None,
    max_sentences: int = 8,
    token_counter: Optional[Callable[[str], int]] = None,
    max_digest_tokens: int = 300,
    messages: Optional[list[StoredMessage]] = None,
) -> Optional[dict[str, Any]]:
    """Digest old messages. Returns the digest record or None.

    When `messages` is given, digests exactly that material (already
    filtered for uncovered ranges by the caller) instead of re-fetching
    and re-processing the whole window. The recorded seq range always
    describes the input message set, and `source_ids` the rows actually
    selected — never broader. Never touches raw rows. Failures propagate
    to the caller, which must fall back and record a diagnostic.
    """
    old = messages if messages is not None else store.fetch_range(
        session_id, 0, cutoff_seq)
    if not old:
        return None
    digest = build_extractive_digest(
        old, query_terms or [], max_sentences=max_sentences
    )
    if token_counter is not None:
        try:
            if token_counter(digest["content"]) > max_digest_tokens:
                # Trim sentences from the end (newest first) until it fits.
                lines = digest["content"].split("\n")
                while lines and token_counter("\n".join(lines)) > max_digest_tokens:
                    lines.pop()
                digest["content"] = "\n".join(lines)
        except Exception:
            pass
    digest_id = store.save_digest(
        session_id=session_id,
        source_ids=digest["source_ids"],
        seq_start=digest["seq_start"],
        seq_end=digest["seq_end"],
        digest_type=digest["digest_type"],
        content=digest["content"],
    )
    digest["digest_id"] = digest_id
    return digest
