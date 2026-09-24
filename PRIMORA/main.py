import sys
import os

# Disable Transformers' parallel tensor materialization before anything else.
# Concurrent slicing of the memory-mapped safetensors weights crashes with a
# Windows access violation on this CUDA stack (transformers 5.15.1
# core_model_loading + torch storage __getitem__). Transformers reads this
# variable at model-load time; setting it here covers CLI and GUI paths.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "true")

import threading
from PRIMORA.prime import Prime
from PRIMORA.prime.providers import OllamaProvider
from PRIMORA.aurora import Aurora
from PRIMORA.aurora.interface.server import serve, HOST, PORT

# Fallback model name used only when PRIMORA_OLLAMA_MODEL is unset.
# The provider is Prime's fallback for non-specialist capabilities and is
# currently unreachable in the live path (the router rejects unregistered
# actions before execution), so this default is inert unless a future
# capability actually invokes the provider. Construction itself performs
# no network I/O.
DEFAULT_OLLAMA_MODEL = "qwen3-8b-16k:latest"


def main():
    print("Starting PRIMORA...")
    print("Type 'exit' or 'quit' to quit.\n")

    cli_mode = "--cli" in sys.argv
    ollama_model = os.environ.get("PRIMORA_OLLAMA_MODEL")
    ollama_url = os.environ.get("PRIMORA_OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_timeout = int(os.environ.get("PRIMORA_OLLAMA_TIMEOUT", "120"))

    if cli_mode and not ollama_model:
        print("ERROR: PRIMORA_OLLAMA_MODEL environment variable must be set")
        print("Example: export PRIMORA_OLLAMA_MODEL=qwen3-8b-16k:latest")
        sys.exit(1)
    if not ollama_model:
        ollama_model = DEFAULT_OLLAMA_MODEL

    try:
        provider = OllamaProvider(
            model=ollama_model,
            base_url=ollama_url,
            timeout=ollama_timeout
        )
        prime = Prime(provider)
        aurora = Aurora(prime)
    except Exception as e:
        print(f"ERROR: Failed to initialize PRIMORA: {e}")
        sys.exit(1)

    if cli_mode:
        print("Loading Aurora model weights (this may take a while on first start)...")
        warmup_error = aurora.warmup()
        if warmup_error:
            print(f"ERROR: Failed to load Aurora model: {warmup_error}")
            sys.exit(1)

    print(f"PRIMORA ready. Aurora model: {aurora.get_model_name()}")
    print(f"Prime provider: {ollama_model} @ {ollama_url}")

    server_thread = threading.Thread(target=serve, args=(aurora,), kwargs={"host": HOST, "port": PORT}, daemon=True)
    server_thread.start()
    print(f"Web UI running at http://{HOST}:{PORT}\n")

    if cli_mode:
        run_cli(aurora)
        return 0
    return run_gui(aurora)


def run_cli(aurora):
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            result = aurora.respond(
                [{"role": "user", "content": user_input}],
                session_id=os.environ.get("PRIMORA_SESSION_ID", "default"),
            )

            if result.success:
                print(f"Aurora: {result.content}\n")
            else:
                print(f"ERROR: {result.error}\n")

        except Exception as e:
            print(f"ERROR: {e}\n")


def run_gui(aurora):
    from PySide6.QtGui import QGuiApplication

    from PRIMORA.verse.app import VerseApplication
    from PRIMORA.verse.state import PresentationMode

    app = QGuiApplication(sys.argv)
    verse = VerseApplication(aurora, mode=PresentationMode.IMMERSIVE)
    # Single warmup path: VerseApplication.start() spawns the WarmupWorker,
    # which performs the one and only aurora.warmup() and reports back via
    # signal (AWAKENING -> AWAKE/DEGRADED). No synchronous warmup here.
    verse.start()
    app.aboutToQuit.connect(verse.shutdown)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())