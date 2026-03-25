"""Interactive REPL — prompt_toolkit-based council session with persistent event loop."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

from council.db import get_db, get_or_create_conversation, add_message, get_messages
from council.events import UiEvent
from council.providers import ProviderRegistry
from council.renderers import RichTranscriptRenderer, PARTICIPANT_STYLES

log = logging.getLogger("council")


_SLASH_COMMANDS = [
    "/help", "/models", "/attach", "/files", "/new", "/resume", "/rename",
    "/history", "/discuss", "/save", "/exit", "/quit",
]


class _CouncilCompleter(Completer):
    """Context-aware completer: slash commands, @model, and file paths after /attach."""

    def __init__(self, commands: list[str], model_names: list[str]) -> None:
        self.commands = commands
        self.model_names = model_names
        self._path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        if text.startswith("/attach "):
            path_text = text[len("/attach "):]
            sub_doc = Document(path_text, len(path_text))
            yield from self._path_completer.get_completions(sub_doc, complete_event)
            return

        if text.startswith("/resume ") or text.startswith("/history "):
            prefix_len = text.index(" ") + 1
            partial = text[prefix_len:].lower()
            try:
                from council.db import get_db
                db = get_db()
                rows = db.execute(
                    "SELECT DISTINCT name FROM conversations "
                    "WHERE name != '__last_ask' ORDER BY updated_at DESC LIMIT 20",
                ).fetchall()
                for row in rows:
                    name = row["name"]
                    if partial in name.lower():
                        yield Completion(name, start_position=-len(text[prefix_len:]))
            except Exception:
                pass
            return

        if text.startswith("/"):
            for cmd in self.commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
            return

        if text.startswith("@"):
            for name in self.model_names:
                full = f"@{name}"
                if full.startswith(text):
                    yield Completion(full + " ", start_position=-len(text))
            return


_REPL_STYLE = Style.from_dict({
    "prompt": "#b48ead bold",
    "prompt.lead": "#a3be8c",
    "prompt.model": "#a3be8c bold",
    "bottom-toolbar": "bg:#2e3440 #d8dee9",
})


class CouncilRepl:
    """Interactive council REPL with persistent event loop."""

    def __init__(self) -> None:
        self.db = get_db()
        available = ProviderRegistry.available()
        self.providers = [ProviderRegistry.get(n) for n in available]
        self.provider_names = [p.name for p in self.providers]

        self.conversation_name: str | None = None
        self.conv_id: int | None = None
        self.attachments: list[str] = []
        self.system: str | None = None
        self._last_response: str | None = None

        from council.chair import Chair
        self._chair = Chair(Path.cwd(), interactive=True)

        self._post_synthesis = False
        self._lead_provider = None
        self._current_task: asyncio.Task | None = None

        self._completer = _CouncilCompleter(_SLASH_COMMANDS, self.provider_names)
        self.session = PromptSession(style=_REPL_STYLE, completer=self._completer)
        self._inner_renderer = RichTranscriptRenderer()
        self.renderer = self  # proxy EventSink

    def emit(self, event: UiEvent) -> None:
        """Proxy EventSink — intercepts responses for /save."""
        if event.type == "response" and event.text:
            self._last_response = event.text
        self._inner_renderer.emit(event)

    def cleanup(self) -> None:
        self._inner_renderer.cleanup()

    # -------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------

    def run(self) -> None:
        """Sync entry point — launches the async REPL on a single event loop."""
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            pass

    async def _run_async(self) -> None:
        """Main async REPL loop — one persistent event loop for the entire session."""
        self._print_welcome()

        loop = asyncio.get_running_loop()

        # Wire Ctrl+C: cancel active generation task, or let prompt_toolkit handle it
        def _on_sigint():
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
            # If no active task, prompt_async's handle_sigint=True raises KeyboardInterrupt

        loop.add_signal_handler(signal.SIGINT, _on_sigint)

        try:
            while True:
                try:
                    prompt_str = self._get_prompt()
                    user_input = await self.session.prompt_async(
                        prompt_str,
                        bottom_toolbar=self._toolbar,
                        handle_sigint=True,
                    )
                    text = user_input.strip()
                    if not text:
                        continue
                    await self._handle_async(text)
                except KeyboardInterrupt:
                    continue  # Ctrl+C at prompt clears input
                except EOFError:
                    break  # Ctrl+D exits
                except Exception as exc:
                    self._inner_renderer.cleanup()
                    from rich.console import Console
                    Console(stderr=True).print(f"[bold red]Error:[/] {exc}")
                    log.debug("Unhandled REPL error", exc_info=True)
        finally:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):
                pass

    def _print_welcome(self) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        cwd = os.getcwd()
        home = os.path.expanduser("~")
        display_cwd = cwd.replace(home, "~", 1) if cwd.startswith(home) else cwd

        models_text = " · ".join(self.provider_names)
        welcome = Text()
        welcome.append("council", style="bold #b48ead")
        welcome.append("  multi-model brainstorm\n\n", style="#d8dee9")
        welcome.append("  models  ", style="#88c0d0")
        welcome.append(f"{models_text}\n", style="#a3be8c")
        welcome.append("  cwd     ", style="#88c0d0")
        welcome.append(f"{display_cwd}\n", style="#d8dee9")
        welcome.append("  input   ", style="#88c0d0")
        welcome.append("message", style="dim")
        welcome.append(" → council  ", style="#616e88")
        welcome.append("@model msg", style="dim")
        welcome.append(" → direct  ", style="#616e88")
        welcome.append("/help", style="dim")
        welcome.append(" → commands", style="#616e88")
        console.print(Panel(welcome, border_style="#4c566a", padding=(1, 2)))
        console.print()

    def _get_prompt(self):
        from prompt_toolkit.formatted_text import FormattedText
        try:
            if self._post_synthesis and self._lead_provider:
                lead = self._lead_provider.name
                return FormattedText([
                    ("#4c566a", "/discuss  /save  @model  /help\n"),
                    ("class:prompt.lead", "lead "),
                    ("class:prompt.model", lead),
                    ("class:prompt.lead", " ❯ "),
                ])
            return FormattedText([
                ("#4c566a", "@model  /help\n"),
                ("class:prompt", "council ❯ "),
            ])
        except Exception:
            return "> "

    def _toolbar(self):
        from prompt_toolkit.formatted_text import FormattedText
        try:
            cwd = os.getcwd()
            home = os.path.expanduser("~")
            display_cwd = cwd.replace(home, "~", 1) if cwd.startswith(home) else cwd
            conv = self.conversation_name or "scratch"
            models = ", ".join(self.provider_names)

            sep = ("bg:#1a1a2e #555555", "  \u2022  ")
            parts: list[tuple[str, str]] = []
            parts.append(("bg:#1a1a2e #e0e0e0 bold", f" {models} "))
            parts.append(sep)
            parts.append(("bg:#1a1a2e #888888", f"{conv}"))
            parts.append(sep)
            parts.append(("bg:#1a1a2e #666666", f"{display_cwd}"))
            if self.attachments:
                parts.append(sep)
                parts.append(("bg:#1a1a2e #ccaa44", f"{len(self.attachments)} file(s)"))
            if self._post_synthesis:
                parts.append(sep)
                parts.append(("bg:#1a1a2e #666666 italic", "/discuss to reconvene"))
            parts.append(("bg:#1a1a2e", " "))
            return FormattedText(parts)
        except Exception:
            return " [council] "

    # -------------------------------------------------------------------
    # Async dispatch
    # -------------------------------------------------------------------

    async def _handle_async(self, text: str) -> None:
        if text.startswith("/"):
            self._slash_command(text)  # slash commands stay sync
        elif text.startswith("@"):
            await self._direct_message(text)
        elif self._post_synthesis:
            await self._lead_followup(text)
        else:
            await self._run_council(text)

    def _ensure_conversation(self) -> None:
        if self.conversation_name is None:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            self.conversation_name = f"repl-{ts}"

    # -------------------------------------------------------------------
    # Council protocol (default Enter)
    # -------------------------------------------------------------------

    async def _run_council(self, prompt: str) -> None:
        """Run the full council protocol as an awaitable task."""
        from council.orchestrator import run_discuss

        self._ensure_conversation()
        extra_files = self.attachments.copy()
        self.attachments.clear()

        self._current_task = asyncio.current_task()
        try:
            await run_discuss(
                db=self.db,
                prompt=prompt,
                providers=self.providers,
                conversation_name=self.conversation_name,
                system=self.system,
                sink=self.renderer,
                interactive=True,
                chair=self._chair,
                files=extra_files or None,
            )
            if len(self.providers) > 1:
                self._post_synthesis = True
                self._lead_provider = self.providers[0]
        except asyncio.CancelledError:
            self.renderer.cleanup()
            self.renderer.emit(UiEvent(type="status", text="Interrupted."))
        finally:
            self._current_task = None

    # -------------------------------------------------------------------
    # Lead follow-up (post-synthesis)
    # -------------------------------------------------------------------

    async def _lead_followup(self, prompt: str) -> None:
        from council.orchestrator import run_lead_followup

        self._ensure_conversation()
        conv_id = get_or_create_conversation(self.db, self.conversation_name, None, None)

        self._current_task = asyncio.current_task()
        try:
            result = await run_lead_followup(
                lead=self._lead_provider,
                prompt=prompt,
                db=self.db,
                conv_id=conv_id,
                system=self.system,
                sink=self.renderer,
                chair=self._chair,
            )
            if result == "reconvene":
                self.renderer.emit(UiEvent(type="status", text="Reconvening the full council..."))
                self._post_synthesis = False
                await self._run_council(prompt)
        except asyncio.CancelledError:
            self.renderer.cleanup()
            self.renderer.emit(UiEvent(type="status", text="Interrupted."))
        finally:
            self._current_task = None

    # -------------------------------------------------------------------
    # Direct message (@model) — now async
    # -------------------------------------------------------------------

    async def _direct_message(self, text: str) -> None:
        parts = text[1:].split(None, 1)
        if not parts:
            self.renderer.emit(UiEvent(type="error", text="Usage: @model <message>"))
            return

        model_name = parts[0].lower()
        message = parts[1] if len(parts) > 1 else ""
        if not message:
            self.renderer.emit(UiEvent(type="error", text=f"Usage: @{model_name} <message>"))
            return

        try:
            provider = ProviderRegistry.get(model_name)
        except SystemExit:
            self.renderer.emit(UiEvent(
                type="error",
                text=f"Unknown model: {model_name}. Available: {', '.join(ProviderRegistry.available())}",
            ))
            return

        self._ensure_conversation()
        conv_id = get_or_create_conversation(self.db, self.conversation_name, None, None)
        add_message(self.db, conv_id, "user", f"@{model_name} {message}")
        messages = get_messages(self.db, conv_id)

        self.renderer.emit(UiEvent(type="generation_start", participant=model_name))
        self._current_task = asyncio.current_task()
        try:
            response = await provider.generate(messages)
            self.renderer.emit(UiEvent(type="generation_end", participant=model_name))
            self.renderer.emit(UiEvent(
                type="response", participant=response.participant,
                text=response.text,
            ))
            add_message(self.db, conv_id, response.participant, response.text)
        except asyncio.CancelledError:
            self.renderer.cleanup()
            self.renderer.emit(UiEvent(type="status", text="Interrupted."))
        except Exception as exc:
            self.renderer.emit(UiEvent(type="generation_end", participant=model_name))
            self.renderer.emit(UiEvent(type="error", text=f"{model_name} failed: {exc}"))
        finally:
            self._current_task = None

    # -------------------------------------------------------------------
    # Slash commands (sync — fast, no async needed)
    # -------------------------------------------------------------------

    def _slash_command(self, text: str) -> None:
        parts = text[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1] if len(parts) > 1 else ""

        commands = {
            "help": self._cmd_help,
            "models": self._cmd_models,
            "attach": self._cmd_attach,
            "files": self._cmd_files,
            "new": self._cmd_new,
            "resume": self._cmd_resume,
            "history": self._cmd_history,
            "discuss": self._cmd_discuss,
            "rename": self._cmd_rename,
            "save": self._cmd_save,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
        }

        handler = commands.get(cmd)
        if handler:
            handler(arg)
        else:
            self.renderer.emit(UiEvent(
                type="error",
                text=f"Unknown command: /{cmd}. Type /help for available commands.",
            ))

    def _cmd_help(self, _arg: str) -> None:
        from rich.console import Console
        Console().print(
            "\n[bold]Commands:[/]\n"
            "  [cyan]/help[/]              Show this help\n"
            "  [cyan]/models[/] [names]    Show or set active models\n"
            "  [cyan]/attach[/] <path>     Attach a file for the next message\n"
            "  [cyan]/files[/]             List queued attachments\n"
            "  [cyan]/new[/] [name]        Start a new conversation\n"
            "  [cyan]/resume[/] <name>       Resume a previous conversation\n"
            "  [cyan]/rename[/] <name>     Rename the current conversation\n"
            "  [cyan]/discuss[/] [msg]     Reconvene the full council\n"
            "  [cyan]/save[/] [filename]   Save last response to file (default: markdown)\n"
            "  [cyan]/history[/] [name]    Show conversation history (or list all)\n"
            "  [cyan]/exit[/]              Exit the REPL\n"
            "\n[bold]Input:[/]\n"
            "  [dim]message[/]            Council protocol (or lead follow-up after synthesis)\n"
            "  [dim]@model message[/]     Send to a single model directly\n"
        )

    def _cmd_models(self, arg: str) -> None:
        if not arg:
            available = ProviderRegistry.available()
            active = set(self.provider_names)
            from rich.console import Console
            console = Console()
            console.print("\n[bold]Models:[/]")
            for name in available:
                status = "[green]active[/]" if name in active else "[dim]inactive[/]"
                console.print(f"  {name}: {status}")
            console.print(f"\n[dim]Usage: /models gemini,claude,chatgpt[/]")
            return

        names = [n.strip() for n in arg.replace(",", " ").split()]
        try:
            self.providers = [ProviderRegistry.get(n) for n in names]
            self.provider_names = [p.name for p in self.providers]
            self.renderer.emit(UiEvent(
                type="status",
                text=f"Active models: {', '.join(self.provider_names)}",
            ))
        except SystemExit:
            self.renderer.emit(UiEvent(
                type="error",
                text=f"Failed to set models. Available: {', '.join(ProviderRegistry.available())}",
            ))

    def _cmd_attach(self, arg: str) -> None:
        if not arg:
            self.renderer.emit(UiEvent(type="error", text="Usage: /attach <path>"))
            return
        path = os.path.expanduser(arg.strip())
        if not os.path.exists(path):
            self.renderer.emit(UiEvent(type="error", text=f"File not found: {path}"))
            return
        self.attachments.append(path)
        self.renderer.emit(UiEvent(
            type="status",
            text=f"Attached: {os.path.basename(path)} ({len(self.attachments)} file(s) queued)",
        ))

    def _cmd_files(self, _arg: str) -> None:
        if not self.attachments:
            self.renderer.emit(UiEvent(type="status", text="No files attached."))
            return
        from rich.console import Console
        console = Console()
        console.print("\n[bold]Queued attachments:[/]")
        for i, path in enumerate(self.attachments, 1):
            console.print(f"  {i}. {path}")
        console.print()

    def _cmd_new(self, arg: str) -> None:
        from datetime import datetime, timezone
        if arg:
            self.conversation_name = arg.strip()
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            self.conversation_name = f"repl-{ts}"
        self.attachments.clear()
        self._post_synthesis = False
        self._lead_provider = None
        self.renderer.emit(UiEvent(
            type="status",
            text=f"New conversation: {self.conversation_name}",
        ))

    def _cmd_resume(self, arg: str) -> None:
        if not arg:
            rows = self.db.execute(
                "SELECT c.name, c.updated_at, COUNT(m.id) as msg_count "
                "FROM conversations c LEFT JOIN messages m ON c.id = m.conversation_id "
                "WHERE c.name != '__last_ask' "
                "GROUP BY c.id ORDER BY c.updated_at DESC LIMIT 15",
            ).fetchall()
            if not rows:
                self.renderer.emit(UiEvent(type="status", text="No conversations to resume."))
                return
            from rich.console import Console
            console = Console()
            console.print("\n[bold]Recent conversations:[/]")
            for r in rows:
                console.print(f"  [cyan]{r['name']}[/]  [dim]{r['msg_count']} msgs · {r['updated_at']}[/]")
            console.print(f"\n[dim]Use /resume <name> to resume[/]\n")
            return

        name = arg.strip()
        row = self.db.execute(
            "SELECT id, name FROM conversations WHERE name = ? ORDER BY updated_at DESC LIMIT 1",
            (name,),
        ).fetchone()
        if not row:
            self.renderer.emit(UiEvent(type="error", text=f"No conversation '{name}' found."))
            return

        self.conversation_name = row["name"]
        self._post_synthesis = False
        self._lead_provider = None

        messages = get_messages(self.db, row["id"])
        if messages:
            last_meta = messages[-1].get("metadata") or {}
            if last_meta.get("stage") in ("synthesis", "resolution"):
                self._post_synthesis = True
                self._lead_provider = self.providers[0]

        self.renderer.emit(UiEvent(
            type="status",
            text=f"Resumed '{self.conversation_name}' ({len(messages)} messages)",
        ))

    def _cmd_save(self, arg: str) -> None:
        if not self._last_response:
            self.renderer.emit(UiEvent(type="error", text="Nothing to save — no responses yet."))
            return

        filename = arg.strip() if arg else None
        if not filename:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            base = self.conversation_name or "council"
            filename = f"{base}-{ts}.md"

        if "." not in Path(filename).name:
            filename += ".md"

        out_path = Path.cwd() / Path(filename).name
        try:
            out_path.write_text(self._last_response, encoding="utf-8")
            self.renderer.emit(UiEvent(
                type="status",
                text=f"Saved to {out_path} (markdown, {len(self._last_response)} bytes)",
            ))
        except Exception as exc:
            self.renderer.emit(UiEvent(type="error", text=f"Failed to save: {exc}"))

    def _cmd_rename(self, arg: str) -> None:
        new_name = arg.strip()
        if not new_name:
            self.renderer.emit(UiEvent(type="error", text="Usage: /rename <new-name>"))
            return
        if not self.conversation_name:
            self.renderer.emit(UiEvent(type="error", text="No active conversation to rename."))
            return
        old_name = self.conversation_name
        try:
            self.db.execute(
                "UPDATE conversations SET name = ?, updated_at = datetime('now') "
                "WHERE name = ?",
                (new_name, old_name),
            )
            self.db.commit()
            self.conversation_name = new_name
            self.renderer.emit(UiEvent(
                type="status",
                text=f"Renamed '{old_name}' → '{new_name}'",
            ))
        except Exception as exc:
            self.renderer.emit(UiEvent(type="error", text=f"Rename failed: {exc}"))

    def _cmd_discuss(self, arg: str) -> None:
        self._post_synthesis = False
        self._lead_provider = None
        if arg:
            # Schedule the council run on the event loop
            asyncio.ensure_future(self._run_council(arg))
        else:
            self.renderer.emit(UiEvent(
                type="status",
                text="Council mode. Type your message to convene the full council.",
            ))

    def _cmd_history(self, arg: str) -> None:
        from rich.console import Console
        console = Console()

        name = arg.strip() if arg else self.conversation_name
        if name:
            row = self.db.execute(
                "SELECT id, name FROM conversations WHERE name = ? ORDER BY updated_at DESC LIMIT 1",
                (name,),
            ).fetchone()
            if not row:
                self.renderer.emit(UiEvent(type="status", text=f"No conversation '{name}' found."))
                return
            messages = get_messages(self.db, row["id"])
            if not messages:
                self.renderer.emit(UiEvent(type="status", text=f"No messages in '{name}'."))
                return

            console.print(f"\n[bold dim]── History: {row['name']} ──[/]\n")
            current_stage = None
            for msg in messages:
                meta = msg.get("metadata") or {}
                participant = msg["participant"]
                content = msg["content"]
                stage = meta.get("stage")

                if stage and stage != current_stage and stage != "prompt":
                    current_stage = stage
                    stage_participant = None
                    if stage in ("synthesis", "resolution"):
                        stage_participant = participant
                    self._inner_renderer.emit(UiEvent(
                        type="stage_start", stage=stage,
                        participant=stage_participant,
                    ))

                if content == "(passed)":
                    self._inner_renderer.emit(UiEvent(
                        type="pass", participant=participant,
                        text=meta.get("note"),
                    ))
                elif stage == "critique" and meta.get("kind"):
                    self._inner_renderer.emit(UiEvent(
                        type="critique", participant=participant,
                        text=content, kind=meta.get("kind"),
                        target=meta.get("target"),
                    ))
                elif participant == "user":
                    console.print(f"[cyan bold]you[/]")
                    console.print(f"  {content}\n")
                else:
                    role = ""
                    if stage in ("synthesis", "resolution"):
                        role = "resolver"
                    elif stage == "followup":
                        role = "lead"
                    self._inner_renderer.emit(UiEvent(
                        type="response", participant=participant,
                        text=content, stage=stage,
                        metadata={"role": role} if role else None,
                    ))
            console.print(f"[dim]── End of history ──[/]\n")
        else:
            rows = self.db.execute(
                "SELECT c.name, c.updated_at, COUNT(m.id) as msg_count "
                "FROM conversations c LEFT JOIN messages m ON c.id = m.conversation_id "
                "WHERE c.name != '__last_ask' "
                "GROUP BY c.id ORDER BY c.updated_at DESC LIMIT 20",
            ).fetchall()
            if not rows:
                self.renderer.emit(UiEvent(type="status", text="No conversations found."))
                return
            console.print("\n[bold]Recent conversations:[/]")
            for r in rows:
                console.print(f"  [cyan]{r['name']}[/]  [dim]{r['msg_count']} msgs · {r['updated_at']}[/]")
            console.print(f"\n[dim]Use /history <name> to view, /resume <name> to continue[/]")
            console.print()

    def _cmd_exit(self, _arg: str) -> None:
        raise EOFError
