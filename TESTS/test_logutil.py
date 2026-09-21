"""Concurrency test for atomic diagnostic-block emission.

Conventions follow the repo: stdlib only, plain test functions.
"""
import io
import sys
import threading
from contextlib import redirect_stdout

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.logutil import emit_block


def test_blocks_stay_contiguous_under_threads():
    out = io.StringIO()

    def worker(i):
        for _ in range(20):
            emit_block([f"[{i}] line A", f"[{i}] line B", f"[{i}] line C"])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    with redirect_stdout(out):
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    lines = out.getvalue().splitlines()
    assert len(lines) == 8 * 20 * 3
    for k in range(0, len(lines), 3):
        owners = {line.split("]")[0] for line in lines[k:k + 3]}
        assert len(owners) == 1, f"interleaved block at line {k}"


def test_emit_block_single_call_shape():
    out = io.StringIO()
    with redirect_stdout(out):
        emit_block(["one", "two"])
    assert out.getvalue().splitlines() == ["one", "two"]


if __name__ == "__main__":
    test_blocks_stay_contiguous_under_threads()
    test_emit_block_single_call_shape()
    print("All logutil tests passed!")
