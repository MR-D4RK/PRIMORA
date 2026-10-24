"""
Minimal local web server for the Aurora V1 interface.

HTTP-only: it serves the browser-based UI in static/index.html and exposes
the caller-provided Aurora instance over a single HTTP endpoint
(/api/chat -> Aurora.respond(messages)).

Aurora itself is created, warmed up, and owned by PRIMORA/main.py, which is
the single application entry point:

    export PRIMORA_OLLAMA_MODEL=qwen3-8b-16k:latest
    python -m PRIMORA.main

Then open http://127.0.0.1:8420
"""

import sys
import os
import json
import mimetypes
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PRIMORA.aurora import Aurora

STATIC_DIR = Path(__file__).resolve().parent / "static"
HOST = os.environ.get("PRIMORA_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("PRIMORA_SERVER_PORT", "8420"))

aurora: Aurora | None = None


def serve(aurora_instance: Aurora, host: str = HOST, port: int = PORT) -> None:
    """Serve the web UI using an already-initialized Aurora instance."""
    global aurora
    aurora = aurora_instance
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Aurora V1 running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        server.shutdown()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.is_file():
            self._send_json(404, {"error": "Not found"})
            return
        content_type, _ = mimetypes.guess_type(str(path))
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path == "":
            self._send_file(STATIC_DIR / "index.html")
            return
        # serve any other static asset under /static, e.g. /app.js
        rel = self.path.lstrip("/")
        candidate = (STATIC_DIR / rel).resolve()
        if STATIC_DIR in candidate.parents or candidate == STATIC_DIR:
            self._send_file(candidate)
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""

        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Malformed JSON"})
            return

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_json(400, {"error": "Missing or invalid 'messages'"})
            return

        session_id = body.get("session_id") or body.get("conversation_id") or "default"
        if not isinstance(session_id, str) or not session_id:
            session_id = "default"

        try:
            result = aurora.respond(messages, session_id=session_id)
        except Exception as e:
            self._send_json(500, {"error": f"Aurora error: {e}"})
            return

        if result.success:
            self._send_json(200, {"success": True, "content": result.content,
                                  "model": result.model, "session_id": session_id})
        else:
            self._send_json(200, {"success": False, "error": result.error,
                                  "model": result.model, "session_id": session_id})


if __name__ == "__main__":
    print("PRIMORA server is started from the single entry point: python -m PRIMORA.main")
    sys.exit(2)
