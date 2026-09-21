"""Integration: a conversation far larger than the active budget.

Verifies raw completeness, ceiling compliance, retrieval of an old
planted fact, newest-turn coherence, dedupe, and bounded prompts.
"""
import os
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import BudgetConfig, ContextManager

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget():
    return BudgetConfig(input_target=120, input_ceiling=150,
                        output_reserve=10, total_budget=160)


def _seeded_conversation():
    msgs = []
    msgs.append({"role": "user",
                 "content": "remember this always: the vault code is BLUE-FALCON-7"})
    for i in range(120):
        if i % 2 == 0:
            msgs.append({"role": "user",
                         "content": f"filler chatter line {i} about errands and weather patterns"})
        else:
            msgs.append({"role": "assistant",
                         "content": f"filler reply {i} acknowledging the small talk warmly"})
    msgs.append({"role": "user", "content": "what is the vault code?"})
    return msgs


def test_long_conversation_stays_bounded_and_complete():
    path = os.path.join(_TEST_TMPDIR.name, "long.db")
    mgr = ContextManager(db_path=path)
    msgs = _seeded_conversation()
    total_tokens = _counter(msgs)
    assert total_tokens > 1000, "seed must dwarf the budget"

    result = mgr.build(msgs, "long-session", token_counter=_counter,
                       max_new_tokens=10, budget=_budget())
    view = result.view

    # 1. raw history remains complete
    assert result.raw_count == len(msgs)
    assert mgr._get_store().count("long-session") == len(msgs)

    # 2. active context stays below the hard ceiling
    assert view.input_tokens <= 150

    # 3. newest request present (it is the current user turn, last)
    assert view.messages[-1] == {"role": "user",
                                 "content": "what is the vault code?"}

    # 4. old relevant fact retrievable
    texts = " ".join(m["content"] for m in view.messages)
    assert "BLUE-FALCON-7" in texts

    # 5. irrelevant old material mostly excluded (budget forces choice;
    # relevance wins a slot for the vault fact over old filler)
    assert texts.count("filler chatter") <= 10
    assert "filler chatter line 0 " not in texts

    # 6. no duplicates
    fps = [(m.get("role"), m.get("content")) for m in view.messages]
    assert len(fps) == len(set(fps))

    # 7. output budget respected
    assert view.input_tokens + view.allowed_output <= 160
    assert view.allowed_output == min(10, 160 - view.input_tokens)


def test_compaction_kicks_in_before_ceiling():
    path = os.path.join(_TEST_TMPDIR.name, "compact.db")
    mgr = ContextManager(db_path=path)
    msgs = ([{"role": "user", "content": f"old discussion point {i} with substance"} for i in range(70)]
            + [{"role": "user", "content": "newest question here"}])
    view = mgr.build(msgs, "compact-session", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    assert view.input_tokens <= 150
    digests = mgr._get_store().fetch_digests("compact-session")
    assert len(digests) >= 1
    assert digests[0]["source_ids"]
    # raw intact despite compaction
    assert mgr._get_store().count("compact-session") == 71


if __name__ == "__main__":
    test_long_conversation_stays_bounded_and_complete()
    test_compaction_kicks_in_before_ceiling()
    print("All context integration tests passed!")
