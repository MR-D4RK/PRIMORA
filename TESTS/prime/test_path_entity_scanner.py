"""C1 regression: linear path-token scanner (MEMORY_AUDIT C1).

``_scan_path_tokens`` must stay exactly equivalent to
``_PATH_TOKEN.finditer`` -- the original pattern is kept in
``entity_formation`` as the oracle -- while never taking quadratic time on
long separator-free or separator-dense text. The former hang case
(``test_recall_output_is_bounded_by_construction``) is the acceptance
test at the system level; the timing tests here pin the scanner itself.
"""

import random
import sys
import time

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import entity_formation as ef


def _oracle(text):
    return [m.group(0) for m in ef._PATH_TOKEN.finditer(text)]


_EDGE_CASES = [
    "",
    " ",
    "a",
    "a/b",
    "a/b.txt",
    "dir/file.txt/extra",
    "C:\\x",
    "C:\\x\\y.py",
    "C:/x.py",
    "c:/",
    "C:",
    "C\\x",
    "ab:/c",
    "ab:cde",
    "a:a:a:a",
    "a/a/a",
    "a/a/a.txt",
    "./a/b/hybrid.py",
    "./c/store.py",
    "x.txt/a/a/a",
    "manager.py",
    "a.abcdefghij",
    "a.b.cdefghijkl",
    "a..b",
    "no-sep-here.txt",
    "trailing.",
    "end.txt.",
    "a.txt,b.txt",
    'quote "C:\\x\\y.py" done',
    "a\tb/c.py",
    "a\nb/c.py",
    "<C:\\x.py>",
    "pipe|a/b.py",
    "o'brien/a.py",
    "a/b.c/d",
    "A:\\",
    "A:/",
    "A:\\.",
    "9:/x",
    ":/x",
    "a/b.C",
    "a/b.12345678",
    "a/b.123456789",
    "dir\\",
    "a\\b.txt",
    "a\\\\b.txt",
    "The path C:\\PRIMORA PROJECT\\file.py and ./x/y.py ok",
    "a/b.txt/extra/deep.py",
    "x/./y.py",
    "x/../y.py",
    "..",
    ".",
    "/",
    "\\",
    "a/",
    "/a.py",
    "//a.py",
    "file.tar.gz",
    "v1.2.3/x.py",
    "-d/x.py",
    "_a/b_c.py",
    "emoji\U0001F600/a.py",
    "café/naïve.py",
    "a/b.Ä",
    "z:/|",
    "Z:\\\trest",
]


def test_scanner_matches_oracle_on_edge_cases():
    for text in _EDGE_CASES:
        assert ef._scan_path_tokens(text) == _oracle(text), repr(text)


def test_scanner_matches_oracle_on_seeded_fuzz():
    alphabet = "abzAZ09.:-_/\\ \t\"'<>|xyéÄ中\U0001F600"
    rng = random.Random(20261013)
    for _ in range(4000):
        text = "".join(
            rng.choice(alphabet) for _ in range(rng.randint(0, 80))
        )
        assert ef._scan_path_tokens(text) == _oracle(text), repr(text)


def test_bare_filename_is_not_a_path():
    """Alt 2 requires at least one (segment + separator); no separator,
    no match -- "manager.py" alone must never form a FILE entity."""
    assert ef._scan_path_tokens("manager.py") == []
    assert ef._path_entities("manager.py") == []


def test_nested_path_keeps_engine_semantics():
    """"dir/file.txt/extra": the engine's chain backtracking ends the
    match at the first usable extension, never spanning "/extra"."""
    assert ef._scan_path_tokens("dir/file.txt/extra") == ["dir/file.txt"]
    assert ef._path_entities("dir/file.txt/extra") == ["file.txt"]


def test_colon_inside_separator_free_run_prefers_alt1():
    assert ef._scan_path_tokens("ab:/c") == ["b:/c"]
    assert ef._scan_path_tokens("ab:cde") == []
    assert ef._scan_path_tokens("a:a:a:a") == []


def test_path_entities_deduplicates_and_strips():
    assert ef._path_entities("see ./x/a.txt and ./y/a.txt") == ["a.txt"]
    assert ef._path_entities("end.txt.") == []
    assert ef._path_entities("./x/end.txt.") == ["end.txt"]
    assert ef._path_entities(
        "check ./a/b/hybrid.py and ./c/store.py"
    ) == ["hybrid.py", "store.py"]
    assert ef._path_entities(
        r"look at C:\PRIMORA PROJECT\PRIMORA\prime\context\manager.py"
    ) == ["manager.py"]


def _timed(text):
    start = time.perf_counter()
    result = ef._scan_path_tokens(text)
    return result, time.perf_counter() - start


def test_long_separator_free_run_is_linear():
    """Former hang: every start position used to rescan the run."""
    result, elapsed = _timed("ab" * 200_000)
    assert result == []
    assert elapsed < 5.0, f"{elapsed:.3f}s"


def test_long_separator_chain_without_extension_is_linear():
    result, elapsed = _timed("a/" * 100_000 + "a")
    assert result == []
    assert elapsed < 5.0, f"{elapsed:.3f}s"


def test_long_separator_chain_with_extension_is_linear():
    result, elapsed = _timed("a/" * 100_000 + "a.txt")
    assert result == ["a/" * 100_000 + "a.txt"]
    assert elapsed < 5.0, f"{elapsed:.3f}s"


def test_oracle_pattern_is_preserved():
    """The regex stays in the module as the behavioural oracle."""
    assert ef._PATH_TOKEN.search("a/b.txt").group(0) == "a/b.txt"
