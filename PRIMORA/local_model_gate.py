"""Serialize in-process requests to a shared local inference server.

Only localhost/loopback endpoints are gated. Streaming responses hold the
gate until complete, preventing the deferred memory worker from launching
a concurrent Qwen generation against the interactive brain on a small GPU.
Remote providers and non-model work remain independent.

This is intentionally process-local; a separate OS process using the same
server needs an external scheduler or server-side concurrency limit.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from urllib.parse import urlsplit

_LOCAL_REQUEST_LOCK = threading.RLock()
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@contextmanager
def local_model_request(url: str):
    enabled = str(os.environ.get(
        "PRIMORA_LOCAL_MODEL_SERIALIZE", "1"
    )).strip().lower() not in {"0", "false", "off", "no"}
    try:
        hostname = urlsplit(url).hostname
    except (TypeError, ValueError):
        hostname = None
    if enabled and hostname in _LOOPBACK_HOSTS:
        with _LOCAL_REQUEST_LOCK:
            yield
    else:
        yield
