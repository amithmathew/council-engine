"""Chair subsystem — mediates between council participants, workspace, and operator."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from council.events import EventSink, UiEvent

log = logging.getLogger("council")

# Max file size the chair will read (50KB)
_MAX_READ_SIZE = 50_000

# Output directory name
OUTPUT_DIR_NAME = "council-output"

# File extensions the chair will read (text-like only)
_TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java", ".rb", ".swift",
    ".kt", ".c", ".cpp", ".h", ".hpp", ".cs", ".sh", ".bash", ".zsh",
    ".md", ".txt", ".rst", ".csv", ".tsv", ".log",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".css", ".scss",
    ".sql", ".graphql", ".proto", ".dockerfile", ".env", ".gitignore",
    ".cfg", ".ini", ".conf",
}

# Directories to skip when listing files
_SKIP_DIRS = {
    "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".git", ".hg", ".svn", "target", ".tox", ".mypy_cache", ".pytest_cache",
}

# Patterns that look like file paths in a question
_PATH_PATTERNS = [
    r'(?:read|show|check|look at|contents? of|what\'?s in)\s+[`"]?([^\s`"]+\.\w+)',
    r'[`"]([^\s`"]+/[^\s`"]+)[`"]',
    r'[`"]([^\s`"]+\.\w{1,5})[`"]',
]


def _emit(sink: EventSink | None, event: UiEvent) -> None:
    if sink is not None:
        sink.emit(event)


class Chair:
    """Chair/Secretary — mediates file operations, info requests, and operator interaction.

    All file operations are brokered through this subsystem:
    - Reads: workspace-wide, text-only, size-limited
    - Writes/edits: restricted to ./council-output/ directory
    - Info requests: file reads → operator fallback
    """

    def __init__(
        self,
        workspace_root: Path | None = None,
        interactive: bool = False,
    ) -> None:
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.output_dir = self.workspace_root / OUTPUT_DIR_NAME
        self.interactive = interactive
        self.clarifications: list[dict] = []

    # -------------------------------------------------------------------
    # File reading (workspace-wide)
    # -------------------------------------------------------------------

    def read_file(
        self,
        path: str,
        participant: str = "unknown",
        sink: EventSink | None = None,
    ) -> str:
        """Read a text file from the workspace. Returns content or error message."""
        p = (self.workspace_root / path).resolve()

        # Security: must be within workspace
        try:
            p.relative_to(self.workspace_root)
        except ValueError:
            return f"Error: path '{path}' is outside the workspace."

        if not p.exists():
            return f"Error: file not found: {path}"

        if not p.is_file():
            return f"Error: '{path}' is not a file."

        # Follow symlinks but verify target is still in workspace
        if p.is_symlink():
            real = p.resolve()
            try:
                real.relative_to(self.workspace_root)
            except ValueError:
                return f"Error: symlink '{path}' points outside the workspace."

        # Text-only check
        if p.suffix.lower() not in _TEXT_EXTENSIONS and p.suffix:
            return f"Error: '{path}' appears to be a binary file ({p.suffix}). Only text files can be read."

        # Size check
        size = p.stat().st_size
        _emit(sink, UiEvent(
            type="chair_request", participant=participant,
            text=f"read {path} ({size} bytes)",
        ))

        try:
            text = p.read_text(errors="replace")
            if len(text) > _MAX_READ_SIZE:
                text = text[:_MAX_READ_SIZE] + f"\n\n[Truncated — showing first {_MAX_READ_SIZE} of {size} bytes]"

            _emit(sink, UiEvent(
                type="chair_response",
                text=f"read {path} ({len(text)} bytes)",
                metadata={"source": "file", "path": path},
            ))
            return text
        except Exception as exc:
            return f"Error reading '{path}': {exc}"

    # -------------------------------------------------------------------
    # File writing (output directory only)
    # -------------------------------------------------------------------

    def write_file(
        self,
        path: str,
        content: str,
        participant: str = "unknown",
        sink: EventSink | None = None,
    ) -> str:
        """Write a file to the output directory. Returns result message."""
        out_path = self._resolve_output_path(path)
        if isinstance(out_path, str):
            return out_path  # error message

        is_overwrite = out_path.exists()

        _emit(sink, UiEvent(
            type="write_proposed", participant=participant,
            text=f"{out_path.relative_to(self.workspace_root)} ({len(content)} bytes)"
                 + (" [overwrite]" if is_overwrite else " [new]"),
        ))

        # Confirmation for overwrites in interactive mode
        if is_overwrite and self.interactive and sys.stdin.isatty():
            print(f"\n  [{participant}] wants to overwrite: {out_path.relative_to(self.workspace_root)}")
            try:
                confirm = input("  Allow overwrite? [y/N] > ").strip().lower()
                if confirm not in ("y", "yes"):
                    _emit(sink, UiEvent(
                        type="write_denied", participant=participant,
                        text=str(out_path.relative_to(self.workspace_root)),
                    ))
                    return f"Overwrite denied: {path}"
            except (EOFError, KeyboardInterrupt):
                return f"Write cancelled: {path}"

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            out_path.write_text(content, encoding="utf-8")
            rel_path = out_path.relative_to(self.workspace_root)
            log.debug("Chair write: %s (%d bytes)", out_path, len(content))
            _emit(sink, UiEvent(
                type="write_approved", participant=participant,
                text=f"{rel_path} ({len(content)} bytes)",
            ))
            return f"Wrote {rel_path} ({len(content)} bytes)"
        except Exception as exc:
            return f"Error writing '{path}': {exc}"

    # -------------------------------------------------------------------
    # File editing (output directory only, unique match)
    # -------------------------------------------------------------------

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        participant: str = "unknown",
        sink: EventSink | None = None,
    ) -> str:
        """Edit a file by replacing a unique text match. Output directory only."""
        out_path = self._resolve_output_path(path)
        if isinstance(out_path, str):
            return out_path  # error message

        if not out_path.exists():
            return f"Error: file not found: {path}. Use write_file to create new files."

        try:
            current = out_path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading '{path}' for edit: {exc}"

        # Validate unique match
        count = current.count(old_text)
        if count == 0:
            # Show a snippet of the file to help the model
            preview = current[:500] + "..." if len(current) > 500 else current
            return f"Error: old_text not found in '{path}'. File preview:\n{preview}"
        if count > 1:
            return f"Error: old_text matches {count} times in '{path}'. It must be unique (match exactly once)."

        # Apply the edit
        new_content = current.replace(old_text, new_text, 1)
        rel_path = out_path.relative_to(self.workspace_root)

        _emit(sink, UiEvent(
            type="write_proposed", participant=participant,
            text=f"edit {rel_path} (replace {len(old_text)} → {len(new_text)} bytes)",
        ))

        try:
            out_path.write_text(new_content, encoding="utf-8")
            log.debug("Chair edit: %s (%d → %d bytes)", out_path, len(old_text), len(new_text))
            _emit(sink, UiEvent(
                type="write_approved", participant=participant,
                text=f"edited {rel_path}",
            ))
            return f"Edited {rel_path} (replaced {len(old_text)} bytes with {len(new_text)} bytes)"
        except Exception as exc:
            return f"Error writing edit to '{path}': {exc}"

    # -------------------------------------------------------------------
    # File discovery (workspace-wide)
    # -------------------------------------------------------------------

    def find_files(
        self,
        pattern: str | None = None,
        directory: str | None = None,
        sink: EventSink | None = None,
    ) -> str:
        """Find files in the workspace by glob pattern."""
        base = self.workspace_root
        if directory:
            base = (self.workspace_root / directory).resolve()
            try:
                base.relative_to(self.workspace_root)
            except ValueError:
                return f"Error: directory '{directory}' is outside the workspace."
            if not base.exists():
                return f"Error: directory '{directory}' not found."

        glob_pattern = pattern or "**/*"
        results = []
        try:
            for p in sorted(base.glob(glob_pattern)):
                if not p.is_file():
                    continue
                rel = p.relative_to(self.workspace_root)
                parts = rel.parts
                if any(part in _SKIP_DIRS for part in parts):
                    continue
                if any(part.startswith(".") for part in parts):
                    continue
                results.append(str(rel))
        except Exception as exc:
            return f"Error searching files: {exc}"

        if not results:
            return f"No files found matching '{glob_pattern}'" + (f" in {directory}" if directory else "")

        # Limit output
        total = len(results)
        display = results[:100]
        output = "\n".join(display)
        if total > 100:
            output += f"\n... and {total - 100} more files"

        _emit(sink, UiEvent(
            type="chair_response",
            text=f"found {total} files" + (f" matching {pattern}" if pattern else ""),
            metadata={"source": "workspace"},
        ))
        return output

    # -------------------------------------------------------------------
    # Info requests (question-answering fallback)
    # -------------------------------------------------------------------

    def fulfill_request(
        self,
        question: str,
        participant: str,
        sink: EventSink | None = None,
    ) -> str:
        """Fulfill a request_info call — for non-file questions."""
        _emit(sink, UiEvent(
            type="chair_request", participant=participant, text=question,
        ))

        # Try file read heuristic
        file_path = self._extract_file_path(question)
        if file_path:
            content = self.read_file(file_path, participant, sink)
            if not content.startswith("Error:"):
                answer = f"[from file: {file_path}]\n{content}"
                self._record(question, answer, participant, "file")
                return answer

        # Structure question
        if self._is_structure_question(question):
            listing = self.find_files(sink=sink)
            if not listing.startswith("No files"):
                answer = f"[from workspace listing]\n{listing}"
                self._record(question, answer, participant, "workspace")
                return answer

        # Ask the human (interactive only)
        if self.interactive and sys.stdin.isatty():
            try:
                human_answer = input(f"\n  [{participant} asks]: {question}\n  Your answer: ").strip()
                if human_answer:
                    answer = f"[from operator] {human_answer}"
                    self._record(question, answer, participant, "human")
                    _emit(sink, UiEvent(
                        type="chair_response", text=answer,
                        metadata={"source": "human"},
                    ))
                    return answer
            except (EOFError, KeyboardInterrupt):
                pass

        # Unavailable
        answer = (
            "OPERATOR_UNAVAILABLE: No human response available. "
            "Proceed with best judgment and state assumptions explicitly."
        )
        self._record(question, answer, participant, "unavailable")
        _emit(sink, UiEvent(
            type="chair_response", text=answer,
            metadata={"source": "unavailable"},
        ))
        return answer

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _resolve_output_path(self, path: str) -> Path | str:
        """Resolve a path within the output directory. Returns Path or error string."""
        # Strip leading council-output/ if the model includes it
        clean = path
        if clean.startswith(f"{OUTPUT_DIR_NAME}/"):
            clean = clean[len(f"{OUTPUT_DIR_NAME}/"):]

        out_path = (self.output_dir / clean).resolve()

        # Security: must be within output directory
        try:
            out_path.relative_to(self.output_dir.resolve())
        except ValueError:
            return f"Error: path '{path}' is outside the output directory ({OUTPUT_DIR_NAME}/)."

        return out_path

    def _extract_file_path(self, question: str) -> str | None:
        """Try to extract a file path from a question."""
        for pattern in _PATH_PATTERNS:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                candidate = match.group(1)
                p = self.workspace_root / candidate
                if p.exists():
                    return candidate
        return None

    def _is_structure_question(self, question: str) -> bool:
        indicators = [
            "project structure", "file structure", "directory",
            "what files", "list files", "folder structure",
            "codebase", "repo structure", "tree",
        ]
        return any(ind in question.lower() for ind in indicators)

    def _record(self, question: str, answer: str, participant: str, source: str) -> None:
        self.clarifications.append({
            "participant": participant,
            "question": question,
            "answer": answer,
            "source": source,
        })
