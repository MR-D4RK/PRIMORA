"""Deterministic context scoring. Pure functions only.

No I/O. No model calls. No randomness. Ties break by sequence number.

Weights:
  relevance  — word overlap with the retrieval query
  recency    — 1 / (1 + age in messages)
  importance — stored per-message value (user authority > assistant)
  redundancy — max Jaccard similarity to already-selected texts (penalty)
"""

import re
from typing import Optional

_WORD_RE = re.compile(r"[a-z0-9']+")

IMPORTANCE_SIGNALS = (
    "remember",
    "important",
    "always",
    "never",
    "we decided",
    "agreed",
    "preference",
    "constraint",
)

_STOPWORDS = frozenset(
    "a an the and or but if then so of at by for with on in to is are was were "
    "be been it its this that these those i you he she we they me him her us "
    "them my your his her our their what when where which who how why not no "
    "do does did can could should would will just very really there here".split()
)


def tokenize(text: str) -> list[str]:
    return [
        w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS
    ]


def importance_of(text: str, role: str) -> float:
    """Authority-weighted importance in [0, 1].

    User-authored material carries full authority. Assistant-authored
    claims carry reduced authority: confidence alone is never durable truth.
    """
    lowered = text.lower()
    hits = sum(1 for signal in IMPORTANCE_SIGNALS if signal in lowered)
    base = min(1.0, 0.35 * hits)
    if role == "user":
        return min(1.0, 0.15 + base)
    if role == "system":
        return 0.5
    return min(1.0, 0.4 * base)


def relevance(candidate_terms: list[str], query_terms: list[str]) -> float:
    """Overlap of candidate terms with query terms, in [0, 1]."""
    if not query_terms or not candidate_terms:
        return 0.0
    query_set = set(query_terms)
    hits = sum(1 for t in set(candidate_terms) if t in query_set)
    return hits / max(1, len(query_set))


def recency(seq: int, now_seq: int) -> float:
    age = max(0, now_seq - seq)
    return 1.0 / (1.0 + age)


def redundancy(candidate_terms: list[str], selected_texts: list[str]) -> float:
    """Max Jaccard similarity to already-selected texts, in [0, 1]."""
    if not selected_texts or not candidate_terms:
        return 0.0
    cand = set(candidate_terms)
    best = 0.0
    for text in selected_texts:
        other = set(tokenize(text))
        union = cand | other
        if not union:
            continue
        best = max(best, len(cand & other) / len(union))
    return best


def score(
    text: str,
    role: str,
    seq: int,
    now_seq: int,
    query_terms: list[str],
    selected_texts: Optional[list[str]] = None,
    stored_importance: float = 0.0,
    w_relevance: float = 0.45,
    w_recency: float = 0.25,
    w_importance: float = 0.30,
    w_redundancy: float = 0.50,
) -> dict[str, float]:
    """Score one candidate. Returns the full breakdown for provenance."""
    terms = tokenize(text)
    rel = relevance(terms, query_terms)
    rec = recency(seq, now_seq)
    imp = max(stored_importance, importance_of(text, role))
    red = redundancy(terms, selected_texts or [])
    total = (
        w_relevance * rel + w_recency * rec + w_importance * imp
    ) - w_redundancy * red
    return {
        "score": total,
        "relevance": rel,
        "recency": rec,
        "importance": imp,
        "redundancy": red,
    }


def build_query(*texts: str) -> list[str]:
    """Combine request texts into one query term list."""
    terms: list[str] = []
    for text in texts:
        if text:
            terms.extend(tokenize(text))
    return terms
