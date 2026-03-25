"""CLI — argparse setup and subcommand dispatch."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

from council.settings import get_settings
from council.db import (
    add_message,
    get_db,
    get_messages,
    get_or_create_conversation,
    get_system_instruction,
)
from council.formatting import print_response, process_files
from council.providers import ProviderRegistry

log = logging.getLogger("council")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.setLevel(level)
    log.addHandler(handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAST_ASK = "__last_ask"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _default_provider() -> str:
    """Return the first available provider name, or 'gemini' as last resort."""
    avail = ProviderRegistry.available()
    if avail:
        return avail[0]
    print(
        "Error: No providers configured.\n"
        "  Run 'council init' to set up API keys, or edit ~/.council/config.toml",
        file=sys.stderr,
    )
    sys.exit(1)


def _read_prompt_from_editor() -> tuple[str, list[str]]:
    """Open $EDITOR for composing a prompt. Returns (message, file_paths)."""
    import tempfile

    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))

    template = (
        "\n"
        "# Write your prompt above.\n"
        "# Lines starting with @file: attach files, e.g.:\n"
        "#   @file:/path/to/document.pdf\n"
        "#   @file:~/screenshots/diagram.png\n"
        "# Lines starting with # are comments and will be removed.\n"
    )

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, prefix="council-") as f:
            f.write(template)
            tmp_path = f.name

        os.system(f'{editor} "{tmp_path}"')

        with open(tmp_path) as f:
            raw = f.read()

        os.unlink(tmp_path)
    except Exception as exc:
        log.warning("Editor failed (%s), falling back to stdin", exc)
        print("Enter your message (empty line to finish):", file=sys.stderr)
        lines = []
        try:
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            pass
        return "\n".join(lines), []

    # Parse content: extract file attachments and strip comments
    message_lines = []
    file_paths = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("@file:"):
            path = stripped[6:].strip()
            if path:
                file_paths.append(os.path.expanduser(path))
        else:
            message_lines.append(line)

    message = "\n".join(message_lines).strip()
    return message, file_paths


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_ask(args: argparse.Namespace) -> None:
    """Single-shot query. Stashes result so it can be promoted via `council promote`."""
    provider = ProviderRegistry.get(getattr(args, "model", None) or _default_provider())
    system = getattr(args, "system", None)
    attachments = process_files(getattr(args, "file", None))
    if attachments:
        file_note = " [attached: " + ", ".join(a["name"] for a in attachments) + "]"
    else:
        file_note = ""
    messages = [{"participant": "user", "content": args.query + file_note, "created_at": _now(), "attachments": attachments}]

    # Show a spinner while waiting
    from rich.console import Console
    console = Console(stderr=True)
    hint = " (first call may be slow while model loads)" if provider.name == "ollama" else ""
    with console.status(f"[dim]{provider.name} is thinking...{hint}[/dim]", spinner="dots"):
        response = provider.generate_sync(messages, system_instruction=system)
    print_response(response.text, response.participant)

    db = get_db()
    session_id = getattr(args, "session", None)
    row = db.execute(
        "SELECT id FROM conversations WHERE name = ? AND session_id IS ?",
        (_LAST_ASK, session_id),
    ).fetchone()
    if row:
        db.execute("DELETE FROM messages WHERE conversation_id = ?", (row["id"],))
        db.execute("DELETE FROM conversations WHERE id = ?", (row["id"],))
        db.commit()
    conv_id = get_or_create_conversation(db, _LAST_ASK, session_id, system)
    add_message(db, conv_id, "user", args.query + file_note, attachments)
    add_message(db, conv_id, response.participant, response.text)
    log.debug("Stashed ask (session=%s) for potential promotion", session_id)


def cmd_promote(args: argparse.Namespace) -> None:
    """Promote the last ask into a named conversation."""
    db = get_db()
    session_id = getattr(args, "session", None)
    row = db.execute(
        "SELECT id FROM conversations WHERE name = ? AND session_id IS ?",
        (_LAST_ASK, session_id),
    ).fetchone()
    if not row:
        print("Nothing to promote — run `council ask` first.", file=sys.stderr)
        sys.exit(1)

    db.execute(
        "UPDATE conversations SET name = ?, updated_at = datetime('now') WHERE id = ?",
        (args.name, row["id"]),
    )
    db.commit()
    print(f"Promoted to '{args.name}' — continue with: council chat {args.name} \"...\"")


def cmd_chat(args: argparse.Namespace) -> None:
    """Send a message in a named conversation. Supports multiple models."""
    # Determine providers: --models takes precedence, then --model, then all available
    models_str = getattr(args, "models", None)
    single_model = getattr(args, "model", None)
    if models_str:
        model_names = [m.strip() for m in models_str.split(",")]
    elif single_model:
        model_names = [single_model]
    else:
        model_names = ProviderRegistry.available()

    providers = [ProviderRegistry.get(name) for name in model_names]

    db = get_db()
    system = getattr(args, "system", None)
    conv_id = get_or_create_conversation(db, args.name, args.session, system)

    stored_system = get_system_instruction(db, conv_id)
    effective_system = system or stored_system

    attachments = process_files(getattr(args, "file", None))
    if attachments:
        file_note = " [attached: " + ", ".join(a["name"] for a in attachments) + "]"
    else:
        file_note = ""

    add_message(db, conv_id, "user", args.message + file_note, attachments)
    messages = get_messages(db, conv_id)

    if len(providers) == 1:
        # Single model — sync, simple
        from rich.console import Console
        console = Console(stderr=True)
        hint = " (first call may be slow while model loads)" if providers[0].name == "ollama" else ""
        with console.status(f"[dim]{providers[0].name} is thinking...{hint}[/dim]", spinner="dots"):
            response = providers[0].generate_sync(messages, system_instruction=effective_system)
        add_message(db, conv_id, response.participant, response.text)
        print_response(response.text, response.participant)
    else:
        # Multiple models — parallel async
        async def _run_parallel():
            async def _gen(p):
                try:
                    return await p.generate(messages, system_instruction=effective_system)
                except Exception as exc:
                    log.error("Provider %s failed: %s", p.name, exc)
                    return None

            results = await asyncio.gather(*[_gen(p) for p in providers])
            for resp in results:
                if resp is None:
                    continue
                add_message(db, conv_id, resp.participant, resp.text)
                print_response(resp.text, resp.participant)

        asyncio.run(_run_parallel())


def cmd_list(args: argparse.Namespace) -> None:
    """List conversations."""
    db = get_db()
    if args.session:
        rows = db.execute(
            "SELECT c.name, c.session_id, c.created_at, c.updated_at, "
            "COUNT(m.id) as msg_count "
            "FROM conversations c LEFT JOIN messages m ON c.id = m.conversation_id "
            "WHERE c.session_id = ? AND c.name != ? "
            "GROUP BY c.id ORDER BY c.updated_at DESC",
            (args.session, _LAST_ASK),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT c.name, c.session_id, c.created_at, c.updated_at, "
            "COUNT(m.id) as msg_count "
            "FROM conversations c LEFT JOIN messages m ON c.id = m.conversation_id "
            "WHERE c.name != ? "
            "GROUP BY c.id ORDER BY c.updated_at DESC",
            (_LAST_ASK,),
        ).fetchall()

    if not rows:
        print("No conversations found.")
        return

    for r in rows:
        session_tag = f" [session: {r['session_id']}]" if r["session_id"] else ""
        print(f"  {r['name']}{session_tag} — {r['msg_count']} messages (updated {r['updated_at']})")


def cmd_history(args: argparse.Namespace) -> None:
    """Show full conversation history."""
    db = get_db()

    row = None
    if args.session:
        row = db.execute(
            "SELECT id FROM conversations WHERE name = ? AND session_id = ?",
            (args.name, args.session),
        ).fetchone()
    if not row:
        row = db.execute(
            "SELECT id FROM conversations WHERE name = ? ORDER BY updated_at DESC LIMIT 1",
            (args.name,),
        ).fetchone()

    if not row:
        print(f"No conversation '{args.name}' found.")
        return

    messages = get_messages(db, row["id"])
    if not messages:
        print("No messages in this conversation.")
        return

    for msg in messages:
        print(f"[{msg['created_at']}] {msg['participant']}:")
        print(f"  {msg['content']}")
        if msg.get("attachments"):
            for att in msg["attachments"]:
                print(f"  📎 {att['name']} ({att['mime_type']})")
        print()


def cmd_delete(args: argparse.Namespace) -> None:
    """Delete a conversation."""
    db = get_db()

    row = None
    if args.session:
        row = db.execute(
            "SELECT id, name FROM conversations WHERE name = ? AND session_id = ?",
            (args.name, args.session),
        ).fetchone()
    if not row:
        row = db.execute(
            "SELECT id, name FROM conversations WHERE name = ? ORDER BY updated_at DESC LIMIT 1",
            (args.name,),
        ).fetchone()

    if not row:
        print(f"No conversation '{args.name}' found.")
        return

    db.execute("DELETE FROM conversations WHERE id = ?", (row["id"],))
    db.commit()
    print(f"Deleted conversation '{row['name']}'.")


def cmd_discuss(args: argparse.Namespace) -> None:
    """Launch a bounded 3-stage council discussion (proposals → critique → synthesis)."""
    from council.orchestrator import run_discuss
    from council.renderers import create_renderer

    models_str = getattr(args, "models", None) or os.environ.get("COUNCIL_MODELS", None)
    if models_str:
        model_names = [m.strip() for m in models_str.split(",")]
    else:
        model_names = ProviderRegistry.available()
    providers = [ProviderRegistry.get(name) for name in model_names]

    # Get prompt — from args, or open editor interactively
    prompt = getattr(args, "prompt", None)
    extra_files = getattr(args, "file", None) or []
    if not prompt and sys.stdin.isatty():
        prompt, editor_files = _read_prompt_from_editor()
        extra_files = extra_files + editor_files
    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        return

    renderer = create_renderer()
    db = get_db()
    is_interactive = sys.stdout.isatty()
    session_id = getattr(args, "session", None)
    system = getattr(args, "system", None)
    conversation_name = getattr(args, "name", None)

    def _run_async(coro):
        import signal
        loop = asyncio.new_event_loop()
        main_task = loop.create_task(coro)
        interrupt_count = 0

        def _cancel_on_sigint():
            nonlocal interrupt_count
            interrupt_count += 1
            if interrupt_count == 1:
                main_task.cancel()
            else:
                raise KeyboardInterrupt

        try:
            loop.add_signal_handler(signal.SIGINT, _cancel_on_sigint)
            return loop.run_until_complete(main_task)
        except asyncio.CancelledError:
            if hasattr(renderer, "cleanup"):
                renderer.cleanup()
            renderer.emit(UiEvent(type="status", text="Interrupted."))
            return None
        except KeyboardInterrupt:
            if hasattr(renderer, "cleanup"):
                renderer.cleanup()
            renderer.emit(UiEvent(type="status", text="Force interrupted."))
            return None
        finally:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):
                pass
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    _run_async(run_discuss(
        db=db,
        prompt=prompt,
        providers=providers,
        session_id=session_id,
        system=system,
        files=extra_files or None,
        conversation_name=conversation_name,
        sink=renderer,
        interactive=is_interactive,
    ))

    # Hint: use the REPL for interactive follow-up
    if is_interactive and len(providers) > 1:
        print(
            "\n  To follow up interactively, launch the REPL: council",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Init — validation helpers
# ---------------------------------------------------------------------------

def _test_with_retry(test_fn, settings, provider_name: str) -> bool:
    """Run a test function in a retry loop. Returns True if test passes or user chooses to keep."""
    while True:
        print(f"\n  Testing {provider_name} connection...")
        if test_fn(settings):
            print(f"  ✓ {provider_name} is working")
            return True
        print()
        print(f"    [r] Retry — fix the issue and test again")
        print(f"    [k] Keep — save this config anyway (fix later)")
        print(f"    [s] Skip — remove this provider from config")
        choice = input(f"  Choose [r/k/s]: ").strip().lower()
        if choice == "r":
            continue
        elif choice == "k":
            return True
        else:
            return False

def _test_gemini(settings) -> bool:
    """Quick validation of Gemini credentials."""
    try:
        from google import genai
        if settings.gemini.api_key and not settings.gemini.vertex_ai:
            client = genai.Client(api_key=settings.gemini.api_key)
        else:
            client = genai.Client(
                vertexai=True,
                project=settings.gemini.project,
                location=settings.gemini.location or "global",
            )
        resp = client.models.generate_content(
            model=settings.gemini.model,
            contents="Say 'ok' and nothing else.",
        )
        return bool(resp.text)
    except Exception as exc:
        err = str(exc).lower()
        if "quota" in err or "billing" in err or "429" in err:
            print(f"  ⚠ Gemini: API quota exceeded or billing not enabled.")
            print(f"    Check your billing at https://console.cloud.google.com/billing")
        elif "403" in err or "permission" in err or "forbidden" in err:
            print(f"  ⚠ Gemini: Permission denied. The API key may not have access to this model,")
            print(f"    or the Generative Language API may not be enabled for your project.")
        elif "401" in err or "invalid" in err or "unauthenticated" in err:
            print(f"  ⚠ Gemini: Invalid API key or credentials.")
        elif "404" in err or "not found" in err:
            print(f"  ⚠ Gemini: Model '{settings.gemini.model}' not found.")
            print(f"    Check available models at https://ai.google.dev/gemini-api/docs/models")
        else:
            print(f"  ⚠ Gemini: Connection failed — {exc}")
        return False


def _test_openai(settings) -> bool:
    """Quick validation of OpenAI credentials."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.chatgpt.api_key)
        resp = client.responses.create(
            model=settings.chatgpt.model,
            input="Say 'ok' and nothing else.",
        )
        return bool(resp.output_text)
    except Exception as exc:
        err = str(exc).lower()
        if "quota" in err or "billing" in err or "429" in err or "rate" in err:
            print(f"  ⚠ ChatGPT: Rate limit or billing issue.")
            print(f"    Check your usage at https://platform.openai.com/usage")
        elif "401" in err or "invalid" in err or "incorrect" in err:
            print(f"  ⚠ ChatGPT: Invalid API key.")
        elif "404" in err or "not found" in err:
            print(f"  ⚠ ChatGPT: Model '{settings.chatgpt.model}' not found.")
        elif "insufficient_quota" in err:
            print(f"  ⚠ ChatGPT: Insufficient quota. Add credits at https://platform.openai.com/settings/organization/billing")
        else:
            print(f"  ⚠ ChatGPT: Connection failed — {exc}")
        return False


def _test_claude(settings) -> bool:
    """Quick validation of Anthropic credentials."""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=settings.claude.api_key)
        resp = client.messages.create(
            model=settings.claude.model,
            max_tokens=32,
            messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
        )
        return bool(resp.content)
    except Exception as exc:
        err = str(exc).lower()
        if "credit" in err or "billing" in err or "429" in err or "rate" in err:
            print(f"  ⚠ Claude: Rate limit or billing issue.")
            print(f"    Check your plan at https://console.anthropic.com/settings/plans")
        elif "401" in err or "invalid" in err or "authentication" in err:
            print(f"  ⚠ Claude: Invalid API key.")
        elif "404" in err or "not found" in err:
            print(f"  ⚠ Claude: Model '{settings.claude.model}' not found.")
            print(f"    Check available models at https://docs.anthropic.com/en/docs/about-claude/models")
        elif "overloaded" in err:
            print(f"  ⚠ Claude: API is overloaded. Try again in a moment.")
        else:
            print(f"  ⚠ Claude: Connection failed — {exc}")
        return False


def _test_ollama(settings) -> bool:
    """Quick validation that Ollama is reachable and the model exists."""
    import httpx
    host = settings.ollama.host.rstrip("/")

    # Check if server is reachable
    try:
        resp = httpx.get(f"{host}/api/version", timeout=5.0)
        resp.raise_for_status()
        version = resp.json().get("version", "unknown")
        print(f"  ✓ Ollama server reachable (v{version})")
    except httpx.ConnectError:
        print(f"  ⚠ Ollama: Cannot connect to {host}")
        print(f"    Make sure Ollama is running: ollama serve")
        return False
    except Exception as exc:
        print(f"  ⚠ Ollama: Cannot connect to {host} — {exc}")
        return False

    # Check if model is available
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        target = settings.ollama.model
        # Ollama model names can include tags (e.g., "llama3.1:8b" or just "llama3.1")
        if any(target in m or m.startswith(target.split(":")[0]) for m in models):
            print(f"  ✓ Model '{target}' is available")
            return True
        else:
            print(f"  ⚠ Ollama: Model '{target}' not found locally.")
            if models:
                print(f"    Installed models: {', '.join(models[:5])}")
            print(f"    Pull it with: ollama pull {target}")
            print(f"    Browse models:   https://ollama.com/library")
            return False
    except Exception as exc:
        print(f"  ⚠ Ollama: Could not list models — {exc}")
        return False


# ---------------------------------------------------------------------------
# Init command
# ---------------------------------------------------------------------------

def cmd_init(_args: argparse.Namespace) -> None:
    """Guided setup — create ~/.council/config.toml."""
    from council.settings import CONFIG_PATH, CONFIG_DIR, Settings, write_config

    print("\n  Council Engine — Setup\n")

    if CONFIG_PATH.exists():
        print(f"  A config file already exists at {CONFIG_PATH}\n")
        print("  Options:")
        print("    [o] Overwrite — start fresh (existing config will be replaced)")
        print("    [e] Edit — open the config file in your editor")
        print("    [q] Quit — keep existing config\n")
        choice = input("  Choose [o/e/q]: ").strip().lower()
        if choice == "e":
            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR", "vi")
            os.execvp(editor, [editor, str(CONFIG_PATH)])
            return
        elif choice != "o":
            print("  Keeping existing config.")
            return
        print()

    settings = Settings()

    print("  Council uses multiple AI models to deliberate on your questions.")
    print("  Configure at least one provider below. You can always add more later")
    print(f"  by editing {CONFIG_PATH}\n")
    print("  Press Enter to skip any provider you don't want to set up now.\n")
    print("  " + "─" * 60)

    # --- Gemini ---
    print("\n  Gemini (Google)")
    print("  Three ways to authenticate:")
    print("    [1] API key from AI Studio  — simplest, get one at https://aistudio.google.com/apikey")
    print("    [2] API key from Vertex AI  — for Vertex AI features, https://console.cloud.google.com/apis/credentials")
    print("    [3] Application Default Credentials (ADC) — if you already have gcloud configured")
    print("    [s] Skip")
    print()
    gemini_choice = input("  Choose [1/2/3/s]: ").strip().lower()
    gemini_set = False
    if gemini_choice == "1":
        key = input("  AI Studio API key: ").strip()
        if key:
            settings.gemini.api_key = key
            gemini_set = True
    elif gemini_choice == "2":
        key = input("  Vertex AI API key: ").strip()
        if key:
            settings.gemini.api_key = key
            settings.gemini.vertex_ai = True
            project = input("  GCP project ID: ").strip()
            if project:
                settings.gemini.project = project
            location = input("  GCP location (default: global): ").strip()
            if location:
                settings.gemini.location = location
            gemini_set = True
    elif gemini_choice == "3":
        print("  Using ADC — ensure you've run: gcloud auth application-default login")
        settings.gemini.vertex_ai = True
        project = input("  GCP project ID (required for ADC): ").strip()
        if project:
            settings.gemini.project = project
            gemini_set = True
        else:
            print("  Warning: GCP project ID is required for ADC. Skipping Gemini.")
            settings.gemini.vertex_ai = False
        location = input("  GCP location (default: global): ").strip()
        if location:
            settings.gemini.location = location

    if gemini_set:
        if not _test_with_retry(_test_gemini, settings, "Gemini"):
            settings.gemini.api_key = ""
            settings.gemini.vertex_ai = False
            gemini_set = False

    print("\n  " + "─" * 60)

    # --- ChatGPT ---
    print("\n  ChatGPT (OpenAI)")
    print("  Get an API key from: https://platform.openai.com/api-keys")
    print()
    key = input("  OpenAI API key (Enter to skip): ").strip()
    chatgpt_set = False
    if key:
        settings.chatgpt.api_key = key
        if _test_with_retry(_test_openai, settings, "ChatGPT"):
            chatgpt_set = True
        else:
            settings.chatgpt.api_key = ""

    print("\n  " + "─" * 60)

    # --- Claude ---
    print("\n  Claude (Anthropic)")
    print("  Get an API key from: https://console.anthropic.com/settings/keys")
    print()
    key = input("  Anthropic API key (Enter to skip): ").strip()
    claude_set = False
    if key:
        settings.claude.api_key = key
        if _test_with_retry(_test_claude, settings, "Claude"):
            claude_set = True
        else:
            settings.claude.api_key = ""

    print("\n  " + "─" * 60)

    # --- Ollama ---
    print("\n  Ollama (local models)")
    print("  Runs models locally — no API key needed, just a running Ollama server.")
    print("  Install:      brew install ollama (macOS) or https://ollama.com/download")
    print("  Start:        ollama serve")
    print("  Pull a model: ollama pull <model_name>")
    print("  Browse models: https://ollama.com/library")
    print()
    print("  No special configuration needed if Ollama runs on this machine.")
    print("  For Docker or remote servers, set OLLAMA_HOST=0.0.0.0 on the Ollama side.")
    print()
    use_ollama = input("  Configure Ollama? [y/N]: ").strip().lower()
    ollama_configured = False
    if use_ollama in ("y", "yes"):
        host = input("  Ollama host (default: http://localhost:11434): ").strip()
        if host:
            settings.ollama.host = host
        model = input("  Ollama model (default: llama3.1:8b): ").strip()
        if model:
            settings.ollama.model = model

        if _test_with_retry(_test_ollama, settings, "Ollama"):
            ollama_configured = True

    print("\n  " + "─" * 60)

    # --- Write config ---
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = write_config(settings)
    print(f"\n  Config written to {path}")

    # Show what's available
    configured = []
    if settings.gemini.api_key:
        configured.append("gemini")
    if settings.chatgpt.api_key:
        configured.append("chatgpt")
    if settings.claude.api_key:
        configured.append("claude")
    if ollama_configured:
        configured.append("ollama")

    if configured:
        first = configured[0]
        print(f"  Configured providers: {', '.join(configured)}\n")
        print(f"  Get started:\n")
        if len(configured) >= 2:
            models_str = ",".join(configured[:2])
            print(f"    # Run a multi-model deliberation (proposals → critique → resolution)")
            print(f"    council discuss --models {models_str} 'Should we use a message queue or direct API calls?'\n")
        print(f"    # Quick single-model query")
        print(f"    council ask --model {first} 'What are the tradeoffs of gRPC vs REST?'\n")
        print(f"    # Start a named conversation")
        print(f"    council chat --model {first} my-project 'Explain the retry logic in our codebase'\n")
        print(f"    # Interactive REPL session")
        print(f"    council\n")
        print(f"  Reconfigure anytime: council init")
        print(f"  Edit config directly: {CONFIG_PATH}\n")
    else:
        print(f"\n  No providers configured.")
        print(f"  Edit {CONFIG_PATH} to add API keys when you're ready.\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="council",
        description="Multi-model brainstorm CLI.\n\n"
            "Two modes of use:\n"
            "  council              Interactive REPL — follow-ups, conversations, attachments\n"
            "  council <command>    One-shot CLI — scripting, pipelines, quick queries\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging (written to stderr)",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # ask
    p_ask = sub.add_parser("ask", help="Single-shot query (promote to conversation later with `promote`)")
    p_ask.add_argument("query", help="The question to ask")
    p_ask.add_argument("--session", default=None, help="Session tag for stash isolation")
    p_ask.add_argument("--system", default=None, help="System prompt (overrides default)")
    p_ask.add_argument("--file", action="append", help="Attach a file (image, PDF, etc.). Repeatable.")
    p_ask.add_argument("--model", default=None, help="Provider to use (default: first available). Options: gemini, chatgpt, claude, ollama")

    # promote
    p_promote = sub.add_parser("promote", help="Promote the last ask into a named conversation")
    p_promote.add_argument("name", help="Conversation name")
    p_promote.add_argument("--session", default=None, help="Session tag to find the right stash")

    # chat
    p_chat = sub.add_parser("chat", help="Send a message in a named conversation")
    p_chat.add_argument("name", help="Conversation name")
    p_chat.add_argument("message", help="Message to send")
    p_chat.add_argument("--session", default=None, help="Optional session tag for grouping")
    p_chat.add_argument("--system", default=None, help="System prompt (set on creation, override on existing)")
    p_chat.add_argument("--file", action="append", help="Attach a file (image, PDF, etc.). Repeatable.")
    p_chat.add_argument("--model", default=None, help="Single provider (default: first available)")
    p_chat.add_argument("--models", default=None, help="Multiple providers, comma-separated (e.g. gemini,chatgpt,claude)")

    # list
    p_list = sub.add_parser("list", help="List conversations")
    p_list.add_argument("--session", default=None, help="Filter by session tag")

    # history
    p_hist = sub.add_parser("history", help="Show conversation history")
    p_hist.add_argument("name", help="Conversation name")
    p_hist.add_argument("--session", default=None, help="Session tag to disambiguate")

    # delete
    p_del = sub.add_parser("delete", help="Delete a conversation")
    p_del.add_argument("name", help="Conversation name")
    p_del.add_argument("--session", default=None, help="Session tag to disambiguate")

    # discuss
    p_discuss = sub.add_parser("discuss", help="One-shot council discussion (proposals → critique → synthesis). Use the REPL for follow-ups")
    p_discuss.add_argument("prompt", nargs="?", default=None, help="The topic or question to discuss (prompted if omitted)")
    p_discuss.add_argument("--models", default=None, help="Comma-separated provider names (default: all available)")
    p_discuss.add_argument("--name", default=None, help="Conversation name (auto-generated if omitted)")
    p_discuss.add_argument("--session", default=None, help="Session tag for grouping")
    p_discuss.add_argument("--system", default=None, help="System prompt for all participants")
    p_discuss.add_argument("--file", action="append", help="Attach a file. Repeatable.")

    sub.add_parser("init", help="Set up Council — create config file with API keys")

    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))

    if args.command is None:
        # No subcommand — launch REPL if TTY, otherwise show help
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                from council.repl import CouncilRepl
                CouncilRepl().run()
            except ImportError:
                # REPL not yet implemented — show help
                parser.print_help()
                sys.exit(0)
        else:
            parser.print_help()
            sys.exit(2)
        return

    log.debug("council v0.3.0 — command=%s, available=%s", args.command, ProviderRegistry.available())

    commands = {
        "ask": cmd_ask,
        "promote": cmd_promote,
        "chat": cmd_chat,
        "list": cmd_list,
        "history": cmd_history,
        "delete": cmd_delete,
        "discuss": cmd_discuss,
        "init": cmd_init,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
