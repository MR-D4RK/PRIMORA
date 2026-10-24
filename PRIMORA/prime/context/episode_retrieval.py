"""Read-only, domain-independent retrieval of source-verified episodes.

The language patterns describe a *request for an experience*, not a list of
allowed subjects. Subjects are taken from user evidence, never configured here.
This direct-answer path is intentionally bounded; conversational synthesis and
richer time reasoning are reserved for subsequent integration work.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import time
import re
from typing import Optional

from .models import EpisodeRecord, StoredMessage
from .episodes import extract_experience_claim, experience_subject


@dataclass(frozen=True)
class EpisodeQuery:
    subject_terms: tuple[str, ...]
    all_experiences: bool = False
    needs_subject: bool = False
    event_kind: Optional[str] = None
    action: Optional[str] = None
    needs_single_episode: bool = False
    include_feelings: bool = False
    since: Optional[float] = None


# Words of the *question*, not possible subjects (no project/car/music/etc.).
_QUESTION_WORDS = frozenset({
    "a", "an", "the", "this", "that", "those", "these", "our", "my", "your",
    "i", "we", "you", "me", "us", "it", "of", "to", "for", "with", "about",
    "on", "in", "at", "by", "from", "when", "what", "how", "was", "were",
    "did", "do", "have", "has", "had", "is", "are", "and", "so", "then",
    "happened", "happen", "remember", "recall", "remind", "tell", "please",
    "experience", "experiences", "episode", "episodes", "time", "times",
    "working", "worked", "work", "testing", "tested", "test", "trying", "tried",
    "starting", "started", "start", "finish", "finished", "fix", "fixed",
    "decide", "decided", "choose", "chose", "discuss", "discussed", "doing",
    "done", "after", "before", "progress", "went", "go", "turned", "out",
    "completed", "complete", "resolved", "resolve", "solved", "solve",
    "attempted", "attempt", "began", "begin", "launched", "launch",
    "published", "publish", "failed", "fail", "stopped", "stop",
    "canceled", "cancelled", "cancel", "regarding", "during", "shared",
    "past", "previous", "some", "recent",
    "recently", "ever", "all", "any", "which", "could", "can", "would",
})
_WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)

# Explicit autobiographical *event* requests only. Keep queries asking when
# the user expressed a preference on the separate verified-history route.
_EVENT_QUESTION = re.compile(
    r"^(?:"
    r"(?:do you )?remember when (?:i|we) (?:started|began|finished|completed|"
    r"fixed|resolved|tested|tried|worked|decided|chose|discussed|launched|published|failed|stopped)\b"
    r"|what (?:happened|did (?:i|we) (?:do|start|finish|fix|test|decide|try|discuss)|"
    r"have (?:i|we) (?:done|worked on))\b"
    r"|when did (?:i|we) (?:start|finish|complete|fix|test|decide|try|work on|publish)\b"
    r"|(?:tell|remind) me (?:about|of) (?:our|my|the) (?:experience|time|work)\b"
    r"|how did (?:our|my|the) .+ (?:go|turn out)\b"
    r"|(?:summarize|recap) (?:our|my) (?:experience|work|progress)\b"
    r"|(?:do you )?remember (?:our|my) (?:work|experience) (?:on|with)\b"
    r"|what (?:experiences|activities|milestones) (?:have (?:i|we) (?:shared|reported)|do you remember)\b"
    r")",
    re.IGNORECASE,
)


# Follow-ups are allowed only when the preceding *user question* identified
# an episode. We never resolve them from model responses, arbitrary earlier
# messages, or unrelated stored preferences.
_FOLLOWUP_EVENT = re.compile(
    r"^(?:and\s+)?(?:what happened (?:next|after (?:that|it))|"
    r"what did (?:i|we) do (?:next|after that)|"
    r"did (?:i|we) (?:finish|complete) (?:it|that)|"
    r"when did (?:i|we) (?:finish|complete) (?:it|that)|"
    r"what (?:was|is) the outcome of (?:it|that))\s*[?!.]*$",
    re.I,
)


# Grammatical action patterns, never lists of supported subjects. The same
# parser works for any named film, study, project, destination, or activity.
_ACTIVITY_VERB = r"watched|read|visited|attended|played|listened to"
_ACTIVITY_GERUND = {
    "watching": "watched", "reading": "read", "visiting": "visited",
    "attending": "attended", "playing": "played",
    "listening to": "listened to",
}
_REPORTED_EVENT_DATE = re.compile(
    rf"^when (?:exactly )?did i tell you (?:that )?"
    rf"(?P<event>(?:i|we) (?:(?:recently|just) )?(?:(?:have|had) )?"
    rf"(?:{_ACTIVITY_VERB})\s+.+?)\s*[?!.]*$", re.I,
)
_EXPERIENCE_WITH_ACTION = re.compile(
    r"^(?:what do you remember about|(?:do you )?remember) "
    r"(?:my|our|the) experience (?P<verb>watching|reading|visiting|"
    r"attending|playing|listening to) (?P<subject>.+?)"
    r"(?: and how i (?:felt|feel) about it)?\s*[?!.]*$", re.I,
)
_ACTIVITY_LIST = re.compile(
    rf"^what (?:have|did) i (?P<action>{_ACTIVITY_VERB})"
    r"(?:\s+(?:recently|lately|in the last \d+ days?))?\s*[?!.]*$", re.I,
)
_EVENT_DATE_FOLLOWUP = re.compile(
    r"^when did i tell you about (?:it|that)(?:\s*[?!.]*)$", re.I,
)


def parse_episode_question(
    question: str, *, context: tuple[str, ...] = (),
) -> Optional[EpisodeQuery]:
    if not isinstance(question, str) or not (2 <= len(question.strip()) <= 500):
        return None
    q = re.sub(r"^\s*(?:hey\s+)?aurora\s*[,!:.]?\s*", "",
               question.strip(), flags=re.I)
    q = re.sub(r"^rememeber\b", "remember", q, flags=re.I)
    matched_date = _REPORTED_EVENT_DATE.fullmatch(q)
    if matched_date:
        statement = matched_date.group("event")
        claim = extract_experience_claim(statement)
        subject = experience_subject(statement) if claim else None
        if subject:
            terms = tuple(_WORDS.findall(subject[0].casefold()))
            return EpisodeQuery(terms, action=re.search(_ACTIVITY_VERB, statement, re.I).group().casefold())

    matched_experience = _EXPERIENCE_WITH_ACTION.fullmatch(q)
    if matched_experience:
        action = _ACTIVITY_GERUND[matched_experience.group("verb").casefold()]
        subject = matched_experience.group("subject")
        if not subject.strip():
            return EpisodeQuery((), needs_subject=True)
        words = tuple(dict.fromkeys(w for w in _WORDS.findall(subject.casefold())
                                    if w not in {"a", "an", "the"}))
        return EpisodeQuery(words, action=action,
                            include_feelings=bool(re.search(r"how i (?:felt|feel)", q, re.I)))

    matched_list = _ACTIVITY_LIST.fullmatch(q)
    if matched_list:
        last = re.search(r"\blast (\d+) days?\b", q, re.I)
        since = (time.time() - int(last.group(1)) * 86400) if last else (
            time.time() - 7 * 86400 if re.search(r"\b(?:recently|lately)\b", q, re.I)
            else None)
        return EpisodeQuery((), all_experiences=True,
                            action=matched_list.group("action").casefold(),
                            since=since)

    if _EVENT_DATE_FOLLOWUP.fullmatch(q):
        prior = parse_episode_question(context[-1]) if context else None
        if not prior or prior.needs_subject:
            return None  # This may be a non-episodic personal-history query.
        return EpisodeQuery(prior.subject_terms,
                            all_experiences=prior.all_experiences,
                            needs_single_episode=prior.all_experiences,
                            action=prior.action, since=prior.since)

    followup = _FOLLOWUP_EVENT.fullmatch(q)
    if followup:
        # Only the immediately preceding user question may establish a subject.
        # A chain of vague follow-ups must not be resolved by guessing.
        prior = parse_episode_question(context[-1]) if context else None
        if prior is None or prior.needs_subject or prior.all_experiences or not prior.subject_terms:
            return EpisodeQuery((), needs_subject=True)
        event_kind = ("completion" if re.search(r"\b(?:finish|complete)\b", q, re.I)
                      else None)
        return EpisodeQuery(prior.subject_terms, event_kind=event_kind)

    if not _EVENT_QUESTION.match(q):
        return None
    # A bare demonstrative (without concrete subject) must not be resolved
    # from arbitrary memory. Explicit activity names remain valid.
    if re.search(r"\b(?:that|this|the)\s+"
                 r"(?:activity|experience|thing|task|project)\b"
                 r"(?=\s*(?:[?!.]|$)|\s+(?:we|i)\s+(?:worked|did|finished|tried))",
                 q, re.I):
        return EpisodeQuery((), needs_subject=True)
    words = _WORDS.findall(q.casefold())
    terms = tuple(dict.fromkeys(w for w in words if w not in _QUESTION_WORDS))
    global_request = bool(re.match(
        r"^what (?:experiences|activities|milestones) (?:have|do)\b", q, re.I
    ))
    if not terms and not global_request:
        return None
    event_kind = ("completion" if re.match(
        r"^when did (?:i|we) (?:finish|complete)\b", q, re.I)
        else None)
    return EpisodeQuery(terms, all_experiences=global_request,
                        event_kind=event_kind)


def _subject_matches(episode: EpisodeRecord, terms: tuple[str, ...]) -> bool:
    words = set(_WORDS.findall(episode.subject.casefold()))
    # Require all topic tokens, never partial lexical overlap. This avoids
    # confusing 'task panel' with 'task tracker', even across sessions.
    return bool(terms) and all(term in words for term in terms)


def render_episode_answer(
    question: str, episodes: list[EpisodeRecord], *, max_episodes: int = 4,
    max_events: int = 12, context: tuple[str, ...] = (),
    reactions: Optional[dict[int, StoredMessage]] = None,
) -> Optional[str]:
    """Return verified event reports for a matched episodic question.

    Every datum comes from the already provenance-verified episode view.
    Dates are message report times in UTC, NOT inferred real-world event times.
    Unknown outcomes and missing episodes are never filled with model guesses.
    """
    query = parse_episode_question(question, context=context)
    if query is None:
        return None
    if query.needs_subject:
        return "Which activity or experience do you mean? I can't identify it from that question alone."
    if not 1 <= max_episodes <= 10 or not 1 <= max_events <= 30:
        raise ValueError("invalid episodic answer budget")
    selected = [e for e in episodes if (
        (query.all_experiences or _subject_matches(e, query.subject_terms))
        and (query.event_kind is None or
             any(ev.kind == query.event_kind for ev in e.events))
        and (query.action is None or any(
            re.search(r"\b" + re.escape(query.action) + r"\b", ev.statement, re.I)
            for ev in e.events))
        and (query.since is None or any(ev.reported_at >= query.since
                                        for ev in e.events))
    )]
    if query.needs_single_episode and len(selected) != 1:
        return "Which activity or experience do you mean? I can't identify it from that question alone."
    selected = selected[:max_episodes]
    if not selected:
        return ("I don't have a verified user-reported experience matching "
                "that question. I won't invent what happened.")
    lines = ["Here are the experiences you reported (timestamps show when you "
             "told me, not necessarily when they occurred):"]
    remaining = max_events
    for episode in selected:
        if remaining <= 0:
            break
        lines.append(f"- {episode.subject}:")
        relevant = (tuple(ev for ev in episode.events if ev.kind == query.event_kind)
                    if query.event_kind else episode.events)
        if query.action:
            relevant = tuple(ev for ev in relevant
                             if re.search(r"\b" + re.escape(query.action) +
                                          r"\b", ev.statement, re.I))
        if query.since is not None:
            relevant = tuple(ev for ev in relevant
                             if ev.reported_at >= query.since)
        for event in relevant[:remaining]:
            when = datetime.fromtimestamp(event.reported_at, timezone.utc)
            # Source comes from event + JOIN with original user message.
            lines.append(
                f"  - {when:%Y-%m-%d %H:%M} UTC: "
                f"{event.statement!r} ({event.kind}; source message "
                f"{event.source_message_id})"
            )
            remaining -= 1
            if query.include_feelings and reactions:
                reaction = reactions.get(event.source_message_id)
                if reaction:
                    lines.append(
                        f"    Your next user statement was: {reaction.content!r} "
                        f"(source message {reaction.message_id}); this may "
                        "describe how you felt about the experience."
                    )
        if remaining == 0:
            break
    lines.append("I can only confirm what you reported; I have not independently "
                 "verified that the events occurred.")
    return "\n".join(lines)
