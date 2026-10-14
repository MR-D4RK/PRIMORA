"""Memory retrieval fixes: plural folding, zero-overlap retention, reserve.

Defect under test - three separate gates each dropped a stored fact before
it ever reached the prompt:

1. ``relevance()`` compared raw tokens, so a singular question ("what about
   my fav car") could not match a stored plural ("zr1x the most in cars").
   Folding is deliberately done at comparison time only, because the same
   raw tokens are handed to the FTS index.
2. ``_add_current_state`` skipped any fact with no term overlap, so "what do
   you know about me?" injected nothing at all - the answer was already
   stored and simply never presented.
3. The history fill ran all the way to ``context_target``, leaving zero
   slack for the ``[state ...]`` lines, so on a full turn a stored fact was
   silently dropped.

Read-only against production memory: every test builds its own temp SQLite
file.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import BudgetConfig, ContextManager
from PRIMORA.prime.context import scoring as scoring_mod

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db():
    _SEQ[0] += 1
    return os.path.join(_TEST_TMPDIR.name, "inj%d.db" % _SEQ[0])


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget(**kwargs):
    params = dict(input_target=25, input_ceiling=60, output_reserve=5,
                  total_budget=65, state_reserve=10)
    params.update(kwargs)
    return BudgetConfig(**params)


def _say(mgr, hist, sid, text, budget=None):
    hist.append({"role": "user", "content": text})
    return mgr.build(list(hist), sid, token_counter=_counter,
                     max_new_tokens=5, budget=budget or _budget())


def _joined(view):
    return "\n".join(m["content"] for m in view.messages)


# 1. Plural folding ------------------------------------------------------------
class TestPluralFolding:
    def test_a_singular_question_reaches_a_plural_fact(self):
        """The exact pair the diagnosis found: car vs cars."""
        query = scoring_mod.build_query("what about my fav car")
        candidate = scoring_mod.tokenize("preference = zr1x the most in cars")
        assert scoring_mod.relevance(candidate, query) > 0

    def test_the_pair_did_not_match_before_folding(self):
        """Guards the fix against a silent revert of relevance()."""
        query = scoring_mod.build_query("what about my fav car")
        candidate = scoring_mod.tokenize("preference = zr1x the most in cars")
        raw_overlap = len(set(query) & set(candidate))
        assert raw_overlap == 0, (
            "the fixture no longer discriminates: raw tokens already match, "
            "so this test proves nothing about folding"
        )

    @pytest.mark.parametrize("word,expected", [
        ("cars", "car"),
        ("keys", "key"),
        ("boxes", "box"),
        ("watches", "watch"),
        ("stories", "story"),
        ("car", "car"),
    ])
    def test_a_plural_folds_to_its_singular(self, word, expected):
        assert scoring_mod.singular(word) == expected

    @pytest.mark.parametrize("word", [
        "bus", "news", "series", "species", "class", "this", "gas",
        "means", "plus", "always",
    ])
    def test_words_that_only_look_plural_are_left_alone(self, word):
        """Folding a word the wrong way invents a match that does not exist."""
        assert scoring_mod.singular(word) == word

    def test_tokenize_stays_raw_for_the_fts_index(self):
        """The FTS index stores the word as written; folding must not hide.

        Folding inside tokenize() would trade one missed match for another:
        _fts_phrase queries the index with these same tokens, and a folded
        token finds nothing there.
        """
        assert scoring_mod.tokenize("my favorite cars") == ["favorite", "cars"]
        assert scoring_mod.singular("cars") == "car"

    def test_a_global_plural_fact_reaches_a_singular_question(self):
        """Integration proof: overlap after folding is the only gate open.

        A global state is not this session's own fact, so it can only be
        admitted on term overlap. That makes it the clean discriminator for
        folding: without the fold this fact has zero overlap and never
        reaches the prompt, and the zero-overlap rule cannot rescue it
        because that rule is session-scope only.
        """
        sid = "fold-a"
        mgr = ContextManager(db_path=_db())
        hist = []
        _say(mgr, hist, sid, "Just checking in.")
        mgr._get_store().create_state(
            sid, "global", "preference", "zr1x the most in cars", [1])

        res = _say(mgr, hist, sid, "what about my fav car")
        joined = _joined(res.view)
        assert "preference = zr1x the most in cars" in joined, joined


# 2. Zero-overlap retention ----------------------------------------------------
class TestZeroOverlapRetention:
    def test_a_session_fact_with_no_shared_term_still_injects(self):
        """'what do you know about me?' has no term in common with the fact."""
        sid = "zero-a"
        mgr = ContextManager(db_path=_db())
        hist = []
        _say(mgr, hist, sid, "My name is Amir.")
        mgr._get_store().create_state(
            sid, sid, "preference", "zr1x the most in cars", [1])

        res = _say(mgr, hist, sid, "tell me something about today")
        query_terms = scoring_mod.build_query("tell me something about today")
        state_terms = scoring_mod.tokenize("preference zr1x the most in cars")
        assert scoring_mod.relevance(state_terms, query_terms) == 0.0, (
            "the fixture no longer discriminates: this query is supposed to "
            "share no term with the stored fact"
        )
        joined = _joined(res.view)
        assert "[state current" in joined, joined
        assert "zr1x the most in cars" in joined, joined

    def test_an_unrelated_global_fact_still_has_to_overlap(self):
        """The retention rule is session-scope only, on purpose.

        An unrelated global value must not crowd a turn out of the session's
        own memory: it has no stake in this conversation until the question
        actually names it.
        """
        sid = "zero-b"
        mgr = ContextManager(db_path=_db())
        hist = []
        _say(mgr, hist, sid, "Just checking in.")
        mgr._get_store().create_state(
            sid, "global", "preference", "zr1x the most in cars", [1])

        res = _say(mgr, hist, sid, "tell me something about today")
        joined = _joined(res.view)
        assert "zr1x the most in cars" not in joined, joined

    def test_a_global_fact_the_question_names_still_injects(self):
        """Overlap still admits a global fact at the full ceiling."""
        sid = "zero-c"
        mgr = ContextManager(db_path=_db())
        hist = []
        _say(mgr, hist, sid, "Just checking in.")
        mgr._get_store().create_state(
            sid, "global", "favorite color", "yello", [1])

        res = _say(mgr, hist, sid, "what is my favorite color")
        joined = _joined(res.view)
        assert "favorite color = yello" in joined, joined

    def test_another_session_never_answers_for_this_one(self):
        """The retention rule must not reopen the session-scope hole."""
        other = "other-session"
        mgr = ContextManager(db_path=_db())
        _say(mgr, [], other, "My favorite color is blue.")
        mgr._get_store().create_state(
            other, other, "preference", "zr1x the most in cars", [1])

        hist = [{"role": "user", "content": "tell me something about today"}]
        res = mgr.build(list(hist), "fresh-session", token_counter=_counter,
                        max_new_tokens=5, budget=_budget())
        joined = _joined(res.view)
        assert "zr1x the most in cars" not in joined, joined
        assert "favorite color = blue" not in joined, joined


# 3. The state reserve ---------------------------------------------------------
class TestStateReserveKeepsHistory:
    def test_the_history_fill_leaves_room_for_a_fact(self):
        """On a full turn the fill must stop early enough for a fact to land.

        Before the reserve, the fill ran to ``input_target`` and a later
        ``[state ...]`` line had only the slack between target and ceiling
        to land in - which is zero on a full turn, so the fact was dropped
        on exactly the turns that needed it.
        """
        sid = "reserve-a"
        mgr = ContextManager(db_path=_db())
        # A one-token counter makes the fill land exactly on the limit, so
        # the fixture is deterministic instead of message-length dependent.
        one = lambda messages: 1
        history = [
            {"role": "user", "content": "filler message %02d" % i}
            for i in range(40)
        ]
        mgr._get_store().create_state(sid, sid, "preference",
                                      "zr1x the most in cars", [1])

        res = mgr.build(list(history), sid, token_counter=one,
                        max_new_tokens=5,
                        budget=_budget(input_target=40, input_ceiling=90,
                                       output_reserve=5, total_budget=95,
                                       state_reserve=15))
        joined = _joined(res.view)
        assert "[state current" in joined, joined
        assert "zr1x the most in cars" in joined, joined
        # ...and the conversation must still be there. The reserve buys the
        # fact its room; it must not do so by evicting the turn.
        assert "filler message" in joined, joined

    def test_the_reserve_is_configurable(self):
        assert _budget(state_reserve=7).state_reserve == 7
        assert BudgetConfig().state_reserve == 100
