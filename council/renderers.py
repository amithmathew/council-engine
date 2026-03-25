"""Renderers — event sinks that produce terminal output."""

from __future__ import annotations

import sys

from council.events import EventType, UiEvent


# Participant colors — each model gets a distinct style
PARTICIPANT_STYLES = {
    "gemini": "green",
    "chatgpt": "blue",
    "claude": "#d4a574",
    "ollama": "#88c0d0",
    "operator": "yellow",
    "user": "cyan",
    "synthesizer": "magenta",
}


class PlainTextRenderer:
    """Plain text renderer for non-TTY output (Claude Code, pipes)."""

    def emit(self, event: UiEvent) -> None:
        if event.type == "stage_start":
            label = (event.stage or "").capitalize()
            if event.participant:
                label += f" ({event.participant})"
            print(f"\n── {label} {'─' * max(0, 50 - len(label))}")

        elif event.type == "stage_end":
            pass  # no output needed

        elif event.type == "response":
            participant = event.participant or "unknown"
            text = event.text or ""
            indent = "  "
            indented = "\n".join(indent + line for line in text.splitlines())
            elapsed_str = f"  {event.elapsed:.1f}s" if event.elapsed else ""
            role = ""
            if event.metadata and event.metadata.get("role"):
                role = f"  ({event.metadata['role']})"
            elif event.stage == "synthesis":
                role = "  (synthesizer)"
            elif event.stage == "followup":
                role = "  (lead)"
            print(f"\n{participant}{role}:{elapsed_str}\n{indented}")

        elif event.type == "pass":
            participant = event.participant or "unknown"
            reason = f"  ({event.text})" if event.text else ""
            elapsed_str = f"  {event.elapsed:.1f}s" if event.elapsed else ""
            print(f"{participant}  pass{reason}{elapsed_str}")

        elif event.type == "critique":
            participant = event.participant or "unknown"
            kind = event.kind or "contribute"
            target_str = f" → {event.target}" if event.target else ""
            text = event.text or ""
            indent = "  "
            indented = "\n".join(indent + line for line in text.splitlines())
            elapsed_str = f"  {event.elapsed:.1f}s" if event.elapsed else ""
            print(f"\n{participant}  {kind}{target_str}{elapsed_str}\n{indented}")

        elif event.type == "status":
            print(f"· {event.text or ''}")

        elif event.type == "error":
            print(f"! {event.text or ''}", file=sys.stderr)

        elif event.type == "generation_start":
            participant = event.participant or "unknown"
            print(f"· {participant} is thinking...", end="", flush=True)

        elif event.type == "generation_end":
            print()  # newline after "thinking..."

        elif event.type == "operator_request":
            participant = event.participant or "unknown"
            print(f"\n⚡ {participant} requests: {event.text or ''}")

        elif event.type == "chair_request":
            participant = event.participant or "unknown"
            print(f"· [{participant} asks chair]: {event.text or ''}")

        elif event.type == "chair_response":
            source = (event.metadata or {}).get("source", "")
            source_label = f" ({source})" if source else ""
            # Truncate long responses for plain text
            text = event.text or ""
            if len(text) > 300:
                text = text[:300] + "..."
            print(f"· [chair{source_label}]: {text}")

        elif event.type == "write_proposed":
            participant = event.participant or "unknown"
            print(f"· {participant} proposes writing: {event.text or ''}")

        elif event.type == "write_approved":
            print(f"· Saved: {event.text or ''}")

        elif event.type == "write_denied":
            print(f"· Write denied: {event.text or ''}")


class RichTranscriptRenderer:
    """Rich-based renderer for TTY output."""

    def __init__(self) -> None:
        from rich.console import Console
        self._console = Console(stderr=True)
        self._out = Console()

    def _pause_live(self) -> bool:
        """Temporarily stop the Live spinner so stdout output doesn't collide.

        Returns True if the spinner was active and was paused.
        """
        if hasattr(self, "_live") and self._live is not None:
            self._live.stop()
            return True
        return False

    def _resume_live(self) -> None:
        """Restart the Live spinner after a pause."""
        if hasattr(self, "_live") and self._live is not None:
            self._live.start()

    def emit(self, event: UiEvent) -> None:
        if event.type == "stage_start":
            from rich.rule import Rule
            label = (event.stage or "").capitalize()
            if event.participant:
                label += f" ({event.participant})"
            paused = self._pause_live()
            self._out.print()
            self._out.print(Rule(f"[bold]{label}[/]", style="dim"))
            if paused:
                self._resume_live()

        elif event.type == "stage_end":
            pass

        elif event.type == "response":
            from rich.markdown import Markdown
            participant = event.participant or "unknown"
            style = PARTICIPANT_STYLES.get(participant, "white")
            elapsed_str = f"  [dim]{event.elapsed:.1f}s[/]" if event.elapsed else ""
            # Show role label if present (synthesizer, lead, etc.)
            role = ""
            if event.metadata and event.metadata.get("role"):
                role = f"  [dim italic]{event.metadata['role']}[/]"
            elif event.stage == "synthesis":
                role = "  [dim italic]synthesizer[/]"
            elif event.stage == "followup":
                role = "  [dim italic]lead[/]"
            paused = self._pause_live()
            self._out.print()
            self._out.print(f"[bold {style}]{participant}[/]{role}{elapsed_str}")
            if event.text:
                md = Markdown(event.text)
                self._out.print(md, width=self._out.width - 2)
            self._out.print("[dim]─[/]" * min(40, self._out.width // 2))
            if paused:
                self._resume_live()

        elif event.type == "pass":
            participant = event.participant or "unknown"
            style = PARTICIPANT_STYLES.get(participant, "white")
            reason = f"  [dim]({event.text})[/]" if event.text else ""
            elapsed_str = f"  [dim]{event.elapsed:.1f}s[/]" if event.elapsed else ""
            paused = self._pause_live()
            self._out.print(f"[{style}]{participant}[/]  [dim]pass[/]{reason}{elapsed_str}")
            if paused:
                self._resume_live()

        elif event.type == "critique":
            from rich.markdown import Markdown
            participant = event.participant or "unknown"
            style = PARTICIPANT_STYLES.get(participant, "white")
            kind = event.kind or "contribute"
            target_str = f" → {event.target}" if event.target else ""
            elapsed_str = f"  [dim]{event.elapsed:.1f}s[/]" if event.elapsed else ""
            paused = self._pause_live()
            self._out.print()
            self._out.print(f"[bold {style}]{participant}[/]  [italic]{kind}[/]{target_str}{elapsed_str}")
            if event.text:
                md = Markdown(event.text)
                self._out.print(md, width=self._out.width - 2)
            if paused:
                self._resume_live()

        elif event.type == "status":
            self._console.print(f"[dim]· {event.text or ''}[/]")

        elif event.type == "error":
            self._console.print(f"[bold red]! {event.text or ''}[/]")

        elif event.type == "generation_start":
            participant = event.participant or "unknown"
            style = PARTICIPANT_STYLES.get(participant, "white")
            # Track which participants are generating
            if not hasattr(self, "_generating"):
                self._generating = set()
            self._generating.add(participant)

            # Only one Live display at a time — show all active participants
            if not hasattr(self, "_live") or self._live is None:
                from rich.spinner import Spinner
                from rich.live import Live
                from rich.text import Text
                label = ", ".join(sorted(self._generating))
                spinner = Spinner("dots", text=Text.from_markup(f"[cyan]{label}[/] thinking..."), style="cyan")
                self._spinner = spinner
                self._live = Live(spinner, console=self._console, transient=True)
                self._live.start()
            else:
                # Update existing spinner text to include new participant
                from rich.text import Text
                label = ", ".join(sorted(self._generating))
                self._spinner.text = Text.from_markup(f"[cyan]{label}[/] thinking...")

        elif event.type == "generation_end":
            participant = event.participant or "unknown"
            if hasattr(self, "_generating"):
                self._generating.discard(participant)
                if not self._generating and hasattr(self, "_live") and self._live is not None:
                    self._live.stop()
                    self._live = None
                elif self._generating and hasattr(self, "_spinner"):
                    from rich.text import Text
                    label = ", ".join(sorted(self._generating))
                    self._spinner.text = Text.from_markup(f"[cyan]{label}[/] thinking...")

        elif event.type == "operator_request":
            participant = event.participant or "unknown"
            style = PARTICIPANT_STYLES.get(participant, "white")
            self._out.print()
            self._out.print(f"[bold yellow]⚡[/] [{style}]{participant}[/] requests: {event.text or ''}")

        elif event.type == "chair_request":
            participant = event.participant or "unknown"
            style = PARTICIPANT_STYLES.get(participant, "white")
            self._console.print(f"[dim]· [{style}]{participant}[/] asks chair:[/] [dim italic]{event.text or ''}[/]")

        elif event.type == "chair_response":
            source = (event.metadata or {}).get("source", "")
            source_label = f" [dim]({source})[/]" if source else ""
            text = event.text or ""
            # Truncate long file contents in the status display
            if len(text) > 300:
                text = text[:300] + "..."
            self._console.print(f"[dim]· chair{source_label}: {text}[/]")

        elif event.type == "write_proposed":
            participant = event.participant or "unknown"
            style = PARTICIPANT_STYLES.get(participant, "white")
            self._console.print(f"[dim]· [{style}]{participant}[/] proposes writing:[/] [dim]{event.text or ''}[/]")

        elif event.type == "write_approved":
            self._console.print(f"[dim]· [green]Saved:[/] {event.text or ''}[/]")

        elif event.type == "write_denied":
            self._console.print(f"[dim]· [yellow]Write denied:[/] {event.text or ''}[/]")

    def cleanup(self) -> None:
        """Force-stop any active Live displays. Call on interrupt."""
        if hasattr(self, "_live") and self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        if hasattr(self, "_generating"):
            self._generating.clear()


def create_renderer() -> PlainTextRenderer | RichTranscriptRenderer:
    """Factory: pick renderer based on TTY detection."""
    from rich.console import Console
    if Console().is_terminal:
        return RichTranscriptRenderer()
    return PlainTextRenderer()
