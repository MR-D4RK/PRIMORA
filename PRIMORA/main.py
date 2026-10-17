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
from PRIMORA.prime.providers import OllamaProvider, OpenAICompatibleProvider
from PRIMORA.research import (
    SEARXNG_ENDPOINT,
    DuckDuckGoResearchProvider,
    SearXNGResearchProvider,
    SearXNGService,
)
from PRIMORA.aurora import Aurora
from PRIMORA.aurora.interface.server import serve, HOST, PORT
from PRIMORA.model_server import (
    ModelServerError,
    ModelServerService,
    should_manage,
)

# Fallback model name used only when PRIMORA_OLLAMA_MODEL is unset.
# The provider is Prime's fallback for non-specialist capabilities and is
# currently unreachable in the live path (Prime.handle rejects unregistered
# actions before execution), so this default is inert unless a future
# capability actually invokes the provider. Construction itself performs
# no network I/O.
DEFAULT_OLLAMA_MODEL = "qwen3-8b-16k:latest"

# Fallback model name used only when PRIMORA_PROVIDER selects the
# OpenAI-compatible backend and PRIMORA_OPENAI_MODEL is unset. llama-server
# echoes whatever name is sent (the served GGUF defines the real model), so
# this is a placeholder, not a weight identifier. Construction still
# performs no network I/O.
DEFAULT_OPENAI_MODEL = "llama-server"

# Timeout for the OpenAI-compatible endpoints, matching the proven
# launcher configuration that PRIMORA/model_server.py now owns: a cold
# generation right after the model finished loading can exceed the 120 s
# library default on this machine.
DEFAULT_OPENAI_TIMEOUT = "300"


def configure_runtime_defaults(service):
    """Apply this entry point's default configuration, never overriding.

    ``python -m PRIMORA.main`` defaults to the live configuration: the
    OpenAI-compatible llama-server backend for both Aurora's brain and
    Prime's provider (per DOCS/02_audits/CLEANUP_INVENTORY.md - one entry,
    one backend). Every value uses ``setdefault``, so anything the operator
    exported explicitly - transformers, ollama, a remote endpoint - stays
    exactly as set and remains the escape hatch.
    """
    os.environ.setdefault("PRIMORA_AURORA_BACKEND", "openai")
    os.environ.setdefault("PRIMORA_PROVIDER", "openai")
    os.environ.setdefault("PRIMORA_OPENAI_URL", service.base_url)
    os.environ.setdefault("PRIMORA_OPENAI_MODEL", service.alias)
    os.environ.setdefault("PRIMORA_OPENAI_TIMEOUT", DEFAULT_OPENAI_TIMEOUT)


def research_provider_from_env(env=None):
    """Select the research backend from PRIMORA_RESEARCH_PROVIDER.

    Composition-layer wiring. DuckDuckGo is the default backend: an absent
    or empty ``PRIMORA_RESEARCH_PROVIDER`` selects it, because a research
    capability with no backend fails closed on every request and leaves the
    base model to answer a "current information" question from its own
    weights - which is exactly how a fabricated Python release date and a
    made-up source number reached the transcript. An explicit ``ddg`` or
    ``duckduckgo`` is the same selection. Constructing a provider performs
    no I/O; a request happens only if the capability later runs.

    ``searxng`` additionally requires PRIMORA_RESEARCH_ENDPOINT to be
    exactly the fixed local address; a missing or different endpoint warns
    and disables the provider rather than falling back to any other backend
    or broadening the target beyond localhost. An explicit but unsupported
    value still fails closed: an operator who named a backend they do not
    have gets no silent substitution.
    """
    source = os.environ if env is None else env
    choice = str(source.get("PRIMORA_RESEARCH_PROVIDER") or "").strip().lower()
    if choice in ("", "ddg", "duckduckgo"):
        return DuckDuckGoResearchProvider()
    if choice == "searxng":
        endpoint = str(source.get("PRIMORA_RESEARCH_ENDPOINT") or "").strip()
        if endpoint != SEARXNG_ENDPOINT:
            print(
                "WARNING: PRIMORA_RESEARCH_PROVIDER='searxng' requires "
                f"PRIMORA_RESEARCH_ENDPOINT={SEARXNG_ENDPOINT} "
                f"(got {endpoint or 'unset'}); research stays disabled "
                "(fail-closed)."
            )
            return None
        return SearXNGResearchProvider(endpoint=endpoint)
    if choice:
        print(
            f"WARNING: unsupported PRIMORA_RESEARCH_PROVIDER={choice!r}; "
            "research stays disabled (fail-closed)."
        )
    return None


def prime_provider_from_env(env=None, model=None):
    """Select the Prime provider backend from PRIMORA_PROVIDER.

    ``ollama`` (the default, for an absent or empty value) speaks Ollama's
    native ``/api/chat`` protocol against PRIMORA_OLLAMA_URL.
    ``openai`` (alias ``llamacpp``) speaks the OpenAI-compatible
    ``/chat/completions`` protocol served by llama.cpp llama-server, vLLM,
    LM Studio and similar local servers; the endpoint defaults to
    http://localhost:8080/v1 and is overridden with PRIMORA_OPENAI_URL.
    Timeouts come from PRIMORA_OLLAMA_TIMEOUT / PRIMORA_OPENAI_TIMEOUT.
    Constructing a provider performs no I/O. An explicit but unsupported
    value fails closed (ValueError) rather than silently substituting a
    backend the operator did not ask for.
    """
    source = os.environ if env is None else env
    choice = str(source.get("PRIMORA_PROVIDER") or "").strip().lower()
    if choice in ("", "ollama"):
        return OllamaProvider(
            model=model or source.get("PRIMORA_OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
            base_url=source.get("PRIMORA_OLLAMA_URL") or "http://127.0.0.1:11434",
            timeout=int(source.get("PRIMORA_OLLAMA_TIMEOUT") or "120")
        )
    if choice in ("openai", "llamacpp"):
        return OpenAICompatibleProvider(
            model=model or source.get("PRIMORA_OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
            base_url=source.get("PRIMORA_OPENAI_URL") or "http://localhost:8080/v1",
            timeout=int(source.get("PRIMORA_OPENAI_TIMEOUT") or "120")
        )
    raise ValueError(
        f"unsupported PRIMORA_PROVIDER={choice!r}; expected 'ollama' or 'openai'"
    )


def main():
    print("Starting PRIMORA...")
    print("Type 'exit' or 'quit' to quit.\n")

    cli_mode = "--cli" in sys.argv

    # Endpoint decisions must see the operator's *explicit* configuration,
    # captured before configure_runtime_defaults() fills in the defaults.
    explicit_backend = os.environ.get("PRIMORA_AURORA_BACKEND")
    if explicit_backend is None:
        effective_backend = "openai"  # this entry point's default
    else:
        choice = explicit_backend.strip().lower()
        # Mirrors aurora_backend_from_env: "" means transformers, and an
        # unsupported value is left to Aurora's own fail-closed check.
        effective_backend = (
            "openai" if choice in ("openai", "llamacpp") else choice
        )
    explicit_url = str(os.environ.get("PRIMORA_OPENAI_URL") or "")

    try:
        service = ModelServerService.from_env()
    except ModelServerError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)
    configure_runtime_defaults(service)

    provider_choice = str(os.environ.get("PRIMORA_PROVIDER") or "").strip().lower()

    if cli_mode and provider_choice in ("", "ollama") and not os.environ.get("PRIMORA_OLLAMA_MODEL"):
        print("ERROR: PRIMORA_OLLAMA_MODEL environment variable must be set")
        print("Example: export PRIMORA_OLLAMA_MODEL=qwen3-8b-16k:latest")
        sys.exit(1)
    if cli_mode and provider_choice in ("openai", "llamacpp") and not os.environ.get("PRIMORA_OPENAI_MODEL"):
        print("ERROR: PRIMORA_OPENAI_MODEL environment variable must be set")
        print("Example: export PRIMORA_OPENAI_MODEL=llama-server")
        sys.exit(1)

    # Model-server lifecycle, before anything constructs a brain that
    # talks to the endpoint: adopt an already-healthy llama-server or
    # start one (never downloading anything). Failures are actionable and
    # exit with a distinct code, the way the launcher did; ensure()
    # leaves nothing of ours running when it raises, and a server it did
    # start carries an atexit backstop for exits that never reach a
    # finally below - the same pattern the SearXNG service uses.
    manage = should_manage(
        service.mode, effective_backend, explicit_url,
        service.host, service.port,
    )
    if manage:
        try:
            status = service.ensure()
        except ModelServerError as exc:
            print(f"ERROR: {exc}")
            service.stop()
            sys.exit(2)
        pid_note = f" (PID {status.pid})" if status.pid else ""
        print(f"Model server: {status.state} at {status.base_url}{pid_note}")
    elif effective_backend == "openai":
        print(
            f"Model server: not managed (PRIMORA_MODEL_SERVER="
            f"{service.mode}); expecting an endpoint at {service.base_url}"
        )

    research_service = None
    try:
        provider = prime_provider_from_env()
        # Research selects its backend here, in the composition layer:
        # DuckDuckGo by default, SearXNG only on an explicit, correctly
        # addressed opt-in, and no backend at all for a value this does
        # not understand. Prime keeps its fail-closed default whenever
        # this returns None.
        research_provider = research_provider_from_env()
        # SearXNG is a local *service*, not just a client endpoint, so
        # its lifecycle is resolved here in the composition layer
        # (covering both the CLI and GUI paths): probe and adopt a
        # healthy instance, or auto-start one, and never let a startup
        # problem stop Aurora.
        if isinstance(research_provider, SearXNGResearchProvider):
            research_service = SearXNGService.from_env()
            research_service.start()
        prime = Prime(
            provider, research_provider=research_provider
        )
        aurora = Aurora(prime)
    except Exception as e:
        print(f"ERROR: Failed to initialize PRIMORA: {e}")
        if research_service is not None:
            research_service.stop()
        service.stop()
        sys.exit(1)

    # Warmup deliberately sits outside the run scope below: on failure
    # both services are released by their own atexit backstops (the
    # SearXNG tests pin that behaviour).
    if cli_mode:
        print("Warming up Aurora (brain and local embedding model)...")
        warmup_error = aurora.warmup()
        if warmup_error:
            print(f"ERROR: Failed to warm up Aurora: {warmup_error}")
            sys.exit(1)

    print(f"PRIMORA ready. Aurora model: {aurora.get_model_name()}")
    _prov_model = getattr(provider, "_model", None) or getattr(provider, "model", None) or "?"
    _prov_url = getattr(provider, "_base_url", None) or getattr(provider, "base_url", None) or "?"
    print(f"Prime provider: {provider.__class__.__name__} "
          f"({_prov_model} @ {_prov_url})")

    # Web UI is a separate interface from the CLI stdin loop and the Verse
    # GUI. CLI operation never needs it; starting a daemon thread there
    # only opens an unnecessary listening socket for the life of the
    # process. GUI (Verse) keeps the historical behaviour of also exposing
    # the web interface alongside the Qt surface.
    if not cli_mode:
        server_thread = threading.Thread(target=serve, args=(aurora,), kwargs={"host": HOST, "port": PORT}, daemon=True)
        server_thread.start()
        print(f"Web UI running at http://{HOST}:{PORT}\n")

    try:
        if cli_mode:
            run_cli(aurora)
            return 0
        return run_gui(aurora)
    finally:
        # Stops only what this process started: the SearXNG instance
        # (adopted or disabled services are no-ops) and a llama-server
        # this process started (an adopted server is never touched).
        # Verse's own shutdown still runs first via aboutToQuit.
        if research_service is not None:
            research_service.stop()
        service.stop()


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