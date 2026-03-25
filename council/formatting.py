"""File handling utilities and legacy display bridge."""

from __future__ import annotations

import logging
import mimetypes
import sys
from pathlib import Path

from google.genai.types import Part

log = logging.getLogger("council")

# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

_MIME_OVERRIDES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".log": "text/plain",
    ".sh": "text/x-shellscript",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/javascript",
    ".jsx": "text/javascript",
    ".tsx": "text/javascript",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "text/plain",
    ".sql": "text/plain",
    ".rs": "text/plain",
    ".go": "text/plain",
    ".java": "text/plain",
    ".rb": "text/plain",
    ".swift": "text/plain",
    ".kt": "text/plain",
}


def guess_mime(path: Path) -> str:
    """Guess mime type with overrides for common dev file types."""
    override = _MIME_OVERRIDES.get(path.suffix.lower())
    if override:
        return override
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def is_text_mime(mime: str) -> bool:
    return mime.startswith("text/") or mime in ("application/json",)


def resolve_file(file_path: str) -> dict:
    """Read a file and return metadata."""
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        print(f"Error: File not found: {p}", file=sys.stderr)
        sys.exit(1)
    mime = guess_mime(p)
    log.debug("Resolved file %s (%s, %d bytes)", p.name, mime, p.stat().st_size)
    return {"path": str(p), "name": p.name, "mime_type": mime}


def file_parts(attachments: list[dict] | None) -> list[Part]:
    """Build Gemini Part objects from stored attachment metadata."""
    if not attachments:
        return []
    parts = []
    for att in attachments:
        p = Path(att["path"])
        if not p.exists():
            log.warning("Attachment no longer exists: %s (skipping)", att["path"])
            parts.append(Part(text=f"[Attachment unavailable: {att['name']}]"))
            continue
        if is_text_mime(att["mime_type"]):
            text = p.read_text(errors="replace")
            parts.append(Part(text=f"--- File: {att['name']} ---\n{text}\n--- End: {att['name']} ---"))
        else:
            data = p.read_bytes()
            parts.append(Part.from_bytes(data=data, mime_type=att["mime_type"]))
    return parts


def process_files(file_paths: list[str] | None) -> list[dict] | None:
    """Resolve file paths and return attachment metadata for storage."""
    if not file_paths:
        return None
    return [
        {"path": r["path"], "name": r["name"], "mime_type": r["mime_type"]}
        for r in (resolve_file(fp) for fp in file_paths)
    ]


# ---------------------------------------------------------------------------
# Legacy display bridge — used by cmd_ask and cmd_chat
# ---------------------------------------------------------------------------

def print_response(text: str, participant: str = "gemini") -> None:
    """Print a model response via the event-based renderer."""
    from council.events import UiEvent
    from council.renderers import create_renderer
    renderer = create_renderer()
    renderer.emit(UiEvent(type="response", participant=participant, text=text))


def print_status(message: str) -> None:
    """Print a status message via the event-based renderer."""
    from council.events import UiEvent
    from council.renderers import create_renderer
    renderer = create_renderer()
    renderer.emit(UiEvent(type="status", text=message))
