import sys
import os
import threading
from PRIMORA.prime import Prime
from PRIMORA.prime.providers import OllamaProvider
from PRIMORA.aurora import Aurora
from PRIMORA.aurora.interface.server import serve, HOST, PORT


def main():
    print("Starting PRIMORA...")
    print("Type 'exit' or 'quit' to quit.\n")

    ollama_model = os.environ.get("PRIMORA_OLLAMA_MODEL")
    ollama_url = os.environ.get("PRIMORA_OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_timeout = int(os.environ.get("PRIMORA_OLLAMA_TIMEOUT", "120"))

    if not ollama_model:
        print("ERROR: PRIMORA_OLLAMA_MODEL environment variable must be set")
        print("Example: export PRIMORA_OLLAMA_MODEL=qwen3-8b-16k:latest")
        sys.exit(1)

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


if __name__ == "__main__":
    main()