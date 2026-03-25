"""Tool definitions and types for council model interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    """Internal tool definition — provider-neutral."""
    name: str
    description: str
    parameters: dict          # JSON Schema
    kind: Literal["informational", "context_loading", "terminal"]


@dataclass
class ToolCall:
    """A tool call extracted from a model response."""
    call_id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    call_id: str
    name: str
    content: str | None = None               # text result (informational tools)
    native_parts: list[Any] | None = None    # multimodal parts (context-loading tools)
    is_terminal: bool = False
    terminal_action: str | None = None       # "pass", "contribute", or "operator_request"
    terminal_payload: str | None = None      # message for call_contribute / call_operator
    critique_kind: str | None = None         # "challenge", "alternative", etc.
    critique_target: str | None = None       # target participant name


# ---------------------------------------------------------------------------
# Council tool definitions
# ---------------------------------------------------------------------------

LIST_ATTACHMENTS = ToolSpec(
    name="list_attachments",
    description=(
        "List all files attached to this conversation. Returns metadata "
        "including ID, name, type, and size. Use this to see what files "
        "are available before reading them."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    kind="informational",
)

READ_ATTACHMENT = ToolSpec(
    name="read_attachment",
    description=(
        "Load a file attachment into your working context for inspection. "
        "The file will be provided in its native format (text for code/documents, "
        "visual for images/PDFs). Use list_attachments first to see available files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The attachment ID from list_attachments.",
            },
        },
        "required": ["id"],
    },
    kind="context_loading",
)

# --- v1 tools (kept for backward compat with existing cascade) ---

CALL_PASS_V1 = ToolSpec(
    name="call_pass",
    description=(
        "End your turn without contributing. Use this when you have nothing "
        "new or valuable to add beyond what has already been said."
    ),
    parameters={
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "Brief reason for passing.",
            },
        },
        "required": ["note"],
    },
    kind="terminal",
)

CALL_CONTRIBUTE_V1 = ToolSpec(
    name="call_contribute",
    description=(
        "End your turn with a contribution to the discussion. Use this "
        "when you have something valuable to add — a new perspective, "
        "a challenge, a correction, or a concrete suggestion."
    ),
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Your contribution to the discussion.",
            },
        },
        "required": ["message"],
    },
    kind="terminal",
)

# --- v2 tools (3-stage protocol) ---

CALL_PASS = ToolSpec(
    name="call_pass",
    description=(
        "Yield the floor. Use this if you have no critical flaws to point out. "
        "Provide a brief reason so the council understands your assessment."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Brief reason for passing (e.g. 'positions are sound', 'no material gaps').",
            },
        },
        "required": ["reason"],
    },
    kind="terminal",
)

CALL_CONTRIBUTE = ToolSpec(
    name="call_contribute",
    description=(
        "End your turn with a structured contribution. Only use this if you "
        "have a genuinely novel point — a concrete flaw, a materially better "
        "alternative, a missing edge case, or a critical question."
    ),
    parameters={
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["challenge", "alternative", "refinement", "question"],
                "description": (
                    "Type of contribution: 'challenge' (identify a flaw), "
                    "'alternative' (propose a different approach), "
                    "'refinement' (improve an existing proposal), "
                    "'question' (surface a critical ambiguity)."
                ),
            },
            "message": {
                "type": "string",
                "description": "Your contribution. Keep it short and specific.",
            },
            "target": {
                "type": "string",
                "description": "Name of the participant this critique targets (optional).",
            },
        },
        "required": ["kind", "message"],
    },
    kind="terminal",
)

CALL_OPERATOR = ToolSpec(
    name="call_operator",
    description=(
        "Request the operator (human or automation agent) to perform a task. "
        "Use this when you need external information — code inspection, "
        "test execution, or a decision only the operator can make."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "What you need the operator to do.",
            },
        },
        "required": ["task_description"],
    },
    kind="terminal",
)

READ_FILE = ToolSpec(
    name="read_file",
    description=(
        "Read a text file from the workspace. Returns the file contents. "
        "Use this to inspect project files, prior deliverables, or any text file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the workspace root.",
            },
        },
        "required": ["path"],
    },
    kind="informational",
)

WRITE_FILE = ToolSpec(
    name="write_file",
    description=(
        "Create a new file or overwrite an existing one in the output directory. "
        "Use this for deliverables — reports, playbooks, documents, scripts. "
        "Files are written to the council-output/ directory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Filename or relative path within the output directory (e.g., 'playbook.md').",
            },
            "content": {
                "type": "string",
                "description": "The full file content to write.",
            },
        },
        "required": ["path", "content"],
    },
    kind="informational",
)

EDIT_FILE = ToolSpec(
    name="edit_file",
    description=(
        "Edit a file by replacing a unique text match. The old_text must appear "
        "exactly once in the file. Use this for targeted edits to existing "
        "deliverables instead of rewriting the entire file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file within the output directory.",
            },
            "old_text": {
                "type": "string",
                "description": "The exact text to find and replace. Must be unique in the file.",
            },
            "new_text": {
                "type": "string",
                "description": "The replacement text.",
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
    kind="informational",
)

FIND_FILES = ToolSpec(
    name="find_files",
    description=(
        "Search for files in the workspace by glob pattern or list a directory. "
        "Use this to discover project structure, find relevant files, or check "
        "what deliverables have been created."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g., '*.py', 'src/**/*.ts'). Defaults to all files.",
            },
            "directory": {
                "type": "string",
                "description": "Directory to search in, relative to workspace root. Defaults to workspace root.",
            },
        },
        "required": [],
    },
    kind="informational",
)

REQUEST_INFO = ToolSpec(
    name="request_info",
    description=(
        "Ask the council chair for information you need. The chair can read "
        "workspace files, check project structure, or ask the human operator. "
        "Use this when you need facts you don't have — file contents, "
        "project layout, clarification on requirements, etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What you need to know. Be specific — e.g., 'read src/config.py' or 'what testing framework does this project use?'",
            },
        },
        "required": ["question"],
    },
    kind="informational",  # non-terminal — model continues after getting the answer
)

CALL_COUNCIL = ToolSpec(
    name="call_council",
    description=(
        "Reconvene the full council for a new deliberation round. Use this when "
        "a follow-up question requires diverse perspectives, changed assumptions, "
        "or re-evaluation — not just elaboration of the existing answer. "
        "This triggers a full proposals → critique → synthesis cycle."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why you believe the full council should reconvene.",
            },
        },
        "required": ["reason"],
    },
    kind="terminal",
)

# --- Tool sets ---

# v1 cascade (kept for backward compat)
CASCADE_TOOLS = [LIST_ATTACHMENTS, READ_ATTACHMENT, CALL_PASS_V1, CALL_CONTRIBUTE_V1]

# Stage 1: Proposals (read-only + discovery + info requests)
PROPOSAL_TOOLS = [LIST_ATTACHMENTS, READ_ATTACHMENT, READ_FILE, FIND_FILES, REQUEST_INFO]

# Stage 2: Critique (structured actions + read + info requests)
CRITIQUE_TOOLS = [LIST_ATTACHMENTS, READ_ATTACHMENT, READ_FILE, FIND_FILES, CALL_PASS, CALL_CONTRIBUTE, REQUEST_INFO]

# Stage 3: Resolution (full file ops + info requests)
RESOLUTION_TOOLS = [LIST_ATTACHMENTS, READ_ATTACHMENT, READ_FILE, WRITE_FILE, EDIT_FILE, FIND_FILES, REQUEST_INFO]

# Lead follow-up (full capabilities)
LEAD_FOLLOWUP_TOOLS = [LIST_ATTACHMENTS, READ_ATTACHMENT, READ_FILE, WRITE_FILE, EDIT_FILE, FIND_FILES, CALL_COUNCIL, REQUEST_INFO]

# Tools available during chat (no terminal actions needed)
CHAT_TOOLS = [LIST_ATTACHMENTS, READ_ATTACHMENT]
