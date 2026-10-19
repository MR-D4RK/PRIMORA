"""Streaming tests for Aurora's OpenAI-compatible brain path.

Verifies SSE parsing, progressive delta delivery, answer-cleaning
consistency, and safety (reasoning / internal-request markup never reaching
the user). Uses mocked HTTP servers and isolated stores; never touches the
production database.
"""

import json
import os
import sys
import tempfile
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import patch

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora.brain import OpenAIBrain
from PRIMORA.aurora.request_shape import (
    displayable_text,
    displayable_len,
)
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.providers import Provider


class MockProvider(Provider):
    def invoke(self, capability: str, payload: dict) -> PrimeResult:
        return PrimeResult.failure(f"Unknown capability: {capability}")


def _isolated_prime() -> Prime:
    db = os.path.join(tempfile.mkdtemp(), "prime_stream_test.db")
    with patch(
        "PRIMORA.prime.prime.ContextManager",
        lambda: __import__(
            "PRIMORA.prime.context.manager", fromlist=["ContextManager"]
        ).ContextManager(db_path=db),
    ):
        return Prime(MockProvider())


def _config(**overrides) -> AuroraConfig:
    defaults = dict(
        model_path="/fake/model",
        adapter_path="/fake/adapter",
        max_new_tokens=128,
        temperature=0.5,
        top_p=0.8,
        repetition_penalty=1.1,
    )
    defaults.update(overrides)
    return AuroraConfig(**defaults)


class SseHandler(BaseHTTPRequestHandler):
    """Mock server that streams SSE content; events may be split across
    tiny writes to simulate arbitrary network chunk boundaries."""

    events = []   # list of (data_str, split_n_bytes or None)
    status_code = 200
    captured_body = None

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.captured_body = self.rfile.read(length) if length else b""
        self.send_response(self.status_code)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for data, split in self.events:
            payload = f"data: {data}\n\n".encode("utf-8")
            if split:
                for i in range(0, len(payload), split):
                    self.wfile.write(payload[i:i + split])
                    self.wfile.flush()
                    time.sleep(0.001)
            else:
                self.wfile.write(payload)
                self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


class RawHandler(BaseHTTPRequestHandler):
    """Mock server that sends a single raw JSON body (non-streaming shape)."""

    status_code = 200
    body = {"choices": [{"message": {"role": "assistant", "content": "Hello there"}}]}
    captured_body = None

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        RawHandler.captured_body = self.rfile.read(length) if length else b""
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.body).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def _start_server(handler_cls):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    return server, port


def _brain(port, **kwargs):
    return OpenAIBrain(
        config=_config(),
        base_url=f"http://127.0.0.1:{port}/v1",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# SSE parsing across arbitrary network chunk boundaries
# ---------------------------------------------------------------------------

def test_iter_sse_reassembles_tiny_chunks():
    """_iter_sse reassembles SSE events split across arbitrarily tiny reads
    (simulating arbitrary network chunk boundaries)."""

    class FakeResp:
        def __init__(self, chunks):
            self._chunks = [c if isinstance(c, bytes) else c.encode()
                            for c in chunks]
            self._pos = 0
            self._buf = b""
            self.status = 200

        def __iter__(self):
            return self

        def __next__(self):
            line = self._readline()
            if line == b"":
                raise StopIteration
            return line

        def _readline(self):
            while b"\n" not in self._buf:
                if self._pos >= len(self._chunks):
                    if self._buf:
                        out = self._buf
                        self._buf = b""
                        return out
                    raise StopIteration
                self._buf += self._chunks[self._pos]
                self._pos += 1
            i = self._buf.index(b"\n")
            out = self._buf[:i + 1]
            self._buf = self._buf[i + 1:]
            return out

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    full = (
        'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
        'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n'
        'data: [DONE]\n\n'
    )
    data = full.encode("utf-8")
    chunks = []
    i = 0
    while i < len(data):
        n = min(3, len(data) - i)
        chunks.append(data[i:i + n])
        i += n

    brain = OpenAIBrain(config=_config())
    events = list(brain._iter_sse(FakeResp(chunks)))
    assert len(events) == 3
    assert events[0]["data"].startswith('{"choices"')
    assert '"Hello"' in events[0]["data"]
    assert '" world"' in events[1]["data"]
    assert events[2]["data"] == "[DONE]"


def test_sse_done_terminates_stream():
    class DoneHandler(SseHandler):
        events = [
            ('{"choices":[{"delta":{"content":"abc"},"index":0}]}', None),
        ]

    server, port = _start_server(DoneHandler)
    try:
        brain = _brain(port)
        deltas = []
        result = brain.generate(
            [{"role": "user", "content": "Hi"}],
            on_delta=lambda d: deltas.append(d),
        )
        assert result.success
        assert result.content == "abc"
        assert deltas == ["abc"]
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Progressive delivery before the complete response arrives
# ---------------------------------------------------------------------------

def test_on_delta_fires_multiple_times_before_completion():
    class MultiHandler(SseHandler):
        events = [
            ('{"choices":[{"delta":{"content":"one"},"index":0}]}', None),
            ('{"choices":[{"delta":{"content":"two"},"index":0}]}', None),
            ('{"choices":[{"delta":{"content":"three"},"index":0}]}', None),
        ]

    server, port = _start_server(MultiHandler)
    try:
        brain = _brain(port)
        seen = []
        result = brain.generate(
            [{"role": "user", "content": "Hi"}],
            on_delta=lambda d: seen.append(d),
        )
        assert result.success
        assert result.content == "onetwothree"
        assert seen == ["one", "two", "three"]
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Final streamed output matches existing answer-cleaning behavior
# ---------------------------------------------------------------------------

def test_streamed_thinking_fence_matches_non_streamed():
    class ThinkHandler(SseHandler):
        events = [
            ('{"choices":[{"delta":{"content":"```think\\nhidden\\n```\\nThe answer is 4."},"index":0}]}', None),
        ]

    server, port = _start_server(ThinkHandler)
    try:
        brain = _brain(port)
        deltas = []
        result = brain.generate(
            [{"role": "user", "content": "2+2?"}],
            on_delta=lambda d: deltas.append(d),
        )
        assert result.success
        assert result.content == "The answer is 4."
        assert "```" not in result.content
        assert "hidden" not in result.content
        assert "```" not in "".join(deltas)
        assert "hidden" not in "".join(deltas)
    finally:
        server.shutdown()


def test_streamed_only_thinking_is_empty_failure():
    class OnlyThink(SseHandler):
        events = [
            ('{"choices":[{"delta":{"content":"```think\\nall reasoning\\n```"},"index":0}]}', None),
        ]

    server, port = _start_server(OnlyThink)
    try:
        brain = _brain(port)
        result = brain.generate(
            [{"role": "user", "content": "Hi"}],
            on_delta=lambda d: None,
        )
        assert not result.success
        assert "empty" in result.error.lower()
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Hidden reasoning and internal-request markup never shown
# ---------------------------------------------------------------------------

def test_reasoning_content_never_forwarded():
    class ReasoningHandler(SseHandler):
        events = [
            ('{"choices":[{"delta":{"content":"Visible","reasoning_content":"SECRET REASONING"},"index":0}]}', None),
            ('{"choices":[{"delta":{"content":" answer"},"index":0}]}', None),
        ]

    server, port = _start_server(ReasoningHandler)
    try:
        brain = _brain(port)
        deltas = []
        result = brain.generate(
            [{"role": "user", "content": "Hi"}],
            on_delta=lambda d: deltas.append(d),
        )
        assert result.success
        assert result.content == "Visible answer"
        joined = "".join(deltas)
        assert "SECRET REASONING" not in joined
    finally:
        server.shutdown()


def test_partial_prime_request_withheld_until_closed():
    buf = "Let me check <prime_request>{\"action\":\"recall\""
    assert displayable_text(buf) == "Let me check "
    assert displayable_len(buf) == len("Let me check ")


def test_closed_prime_request_skipped_then_text_resumes():
    buf = "Intro <prime_request>{\"action\":\"x\"}</prime_request> outro"
    assert displayable_text(buf) == "Intro  outro"


# ---------------------------------------------------------------------------
# Existing non-streaming callers retain prior behavior
# ---------------------------------------------------------------------------

def test_non_streaming_path_unchanged():
    server, port = _start_server(RawHandler)
    try:
        brain = _brain(port)
        result = brain.generate([{"role": "user", "content": "Hi"}])
        assert result.success
        assert result.content == "Hello there"
        body = json.loads(RawHandler.captured_body.decode("utf-8"))
        assert body["stream"] is False
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# HTTP errors, malformed/incomplete streams clean up safely
# ---------------------------------------------------------------------------

def test_streaming_http_error():
    class ErrorHandler(SseHandler):
        status_code = 500
        events = []

    server, port = _start_server(ErrorHandler)
    try:
        brain = _brain(port)
        result = brain.generate(
            [{"role": "user", "content": "Hi"}],
            on_delta=lambda d: None,
        )
        assert not result.success
        assert "HTTP error 500" in result.error
    finally:
        server.shutdown()


def test_malformed_sse_events_skipped():
    class MalformedHandler(SseHandler):
        events = [
            ('not json at all', None),
            ('{"choices":[{"delta":{"content":"ok"},"index":0}]}', None),
        ]

    server, port = _start_server(MalformedHandler)
    try:
        brain = _brain(port)
        deltas = []
        result = brain.generate(
            [{"role": "user", "content": "Hi"}],
            on_delta=lambda d: deltas.append(d),
        )
        assert result.success
        assert result.content == "ok"
        assert deltas == ["ok"]
    finally:
        server.shutdown()


def test_cancellation_stops_stream():
    """A cancellation request stops reading between chunks."""

    class SlowHandler(SseHandler):
        events = [
            ('{"choices":[{"delta":{"content":"a"},"index":0}]}', None),
            ('{"choices":[{"delta":{"content":"b"},"index":0}]}', None),
            ('{"choices":[{"delta":{"content":"c"},"index":0}]}', None),
        ]

    cancel = {"flag": False}

    server, port = _start_server(SlowHandler)
    try:
        brain = _brain(port)
        deltas = []

        def on_delta(d):
            deltas.append(d)
            if len(deltas) >= 1:
                cancel["flag"] = True

        result = brain.generate(
            [{"role": "user", "content": "Hi"}],
            on_delta=on_delta,
            cancelled=lambda: cancel["flag"],
        )
        assert result.success
        # Cancelled after the first delta: only "a" was delivered.
        assert deltas == ["a"]
        assert result.content == "a"
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# request_shape helpers
# ---------------------------------------------------------------------------

def test_displayable_len_incomplete_think_withheld():
    buf = "Safe text ```think incomplete"
    assert displayable_text(buf) == "Safe text "
    assert displayable_len(buf) == len("Safe text ")


def test_displayable_text_monotonic_prefix():
    """Earlier displayable output is always a prefix of later output."""
    stages = [
        "Hello",
        "Hello <prime_request>",
        "Hello <prime_request>{\"a\":1}",
        "Hello <prime_request>{\"a\":1}</prime_request> world",
    ]
    outs = [displayable_text(s) for s in stages]
    for i in range(1, len(outs)):
        assert outs[i].startswith(outs[i - 1]), (outs[i - 1], outs[i])


# ---------------------------------------------------------------------------
# Multi-cycle: intermediate segment text must never leak to the answer
# ---------------------------------------------------------------------------

def _scripted_prime():
    from PRIMORA.prime import Prime
    from PRIMORA.prime.contract import PrimeResult
    from PRIMORA.prime.providers import Provider

    class _Provider(Provider):
        def invoke(self, capability, payload):
            return PrimeResult.success({"value": 4})

    prime = Prime(_Provider())
    db = os.path.join(tempfile.mkdtemp(prefix="leak_"), "leak.db")
    prime._context = prime._context.__class__(db_path=db)
    prime._mind = prime._mind.__class__(prime._context)
    for name in ("recall", "learn"):
        cap = prime.registry.get_specialist(name)
        if cap is not None:
            cap.bind(prime._mind)
    return prime


def test_multi_cycle_intermediate_text_never_returned():
    """Segment 1 is intermediate (intro prose + a capability request);
    segment 2 is the final answer. respond() must return only segment 2."""
    from PRIMORA.aurora import Aurora
    from PRIMORA.aurora.brain import Brain, BrainResult
    from PRIMORA.aurora.configuration import AuroraConfig

    class _Scripted(Brain):
        def __init__(self, *replies):
            self.replies = list(replies)
            self.calls = 0

        def generate(self, messages, max_output_tokens=None,
                     on_delta=None, cancelled=None):
            self.calls += 1
            r = self.replies[min(self.calls - 1, len(self.replies) - 1)]
            return BrainResult.success(r, "scripted")

        def get_model_name(self):
            return "scripted"

    intro = ("Let me look that up. " +
             "<prime_request>\n" +
             json.dumps({"action": "recall", "payload": {"query": "cargo"},
                         "context": None, "metadata": None}) +
             "\n</prime_request>")
    final = "The last cargo was kestrel seventeen."

    prime = _scripted_prime()
    a = Aurora(prime, brain=_Scripted(intro, final),
               config=AuroraConfig(model_path="m", adapter_path="a"))
    result = a.respond([{"role": "user", "content": "What was the cargo?"}],
                       "s1")
    assert result.success
    # Only the final user-facing answer is returned.
    assert result.content == final
    # Intermediate prose and the raw request markup never reach the answer.
    assert intro not in result.content
    assert "Let me look that up" not in result.content
    assert "<prime_request>" not in result.content
    # The loop really did multiple cycles and executed the recall.
    assert a.loop.executor.used == 1
    assert a.loop.executor.succeeded == ["recall"]
    assert a.loop.last_trace.count == 2


if __name__ == "__main__":
    test_iter_sse_reassembles_tiny_chunks()
    test_sse_done_terminates_stream()
    test_on_delta_fires_multiple_times_before_completion()
    test_streamed_thinking_fence_matches_non_streamed()
    test_streamed_only_thinking_is_empty_failure()
    test_reasoning_content_never_forwarded()
    test_partial_prime_request_withheld_until_closed()
    test_closed_prime_request_skipped_then_text_resumes()
    test_non_streaming_path_unchanged()
    test_streaming_http_error()
    test_malformed_sse_events_skipped()
    test_cancellation_stops_stream()
    test_displayable_len_incomplete_think_withheld()
    test_displayable_text_monotonic_prefix()
    test_multi_cycle_intermediate_text_never_returned()
    print("All streaming tests passed!")
