"""Atomic diagnostic-block emission for concurrent runtimes.

CLI and web requests generate on different threads; multi-print diagnostic
blocks would otherwise interleave line-by-line. Each block is composed into
one string and emitted under a short lock. The lock covers only the emit,
never any inference workload.
"""

import threading

_lock = threading.Lock()


def emit_block(lines) -> None:
    """Print a multi-line diagnostic block atomically."""
    text = "\n".join(str(line) for line in lines)
    with _lock:
        print(text, flush=True)
