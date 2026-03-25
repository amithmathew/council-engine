"""Event bus — typed events for decoupling orchestration from rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


EventType = Literal[
    "stage_start",
    "stage_end",
    "response",           # model text output
    "pass",               # compact pass
    "critique",           # structured critique (kind + target)
    "status",             # informational
    "error",
    "generation_start",   # model is thinking
    "generation_end",
    "operator_request",   # call_operator
    "chair_request",      # participant asked chair for info
    "chair_response",     # chair answered (with source in metadata)
    "write_proposed",     # chair received a write request
    "write_approved",     # write executed
    "write_denied",       # write denied by operator
]


@dataclass
class UiEvent:
    """A single UI event emitted by the orchestrator."""
    type: EventType
    participant: str | None = None
    text: str | None = None
    stage: str | None = None        # "proposals", "critique", "synthesis"
    kind: str | None = None         # challenge, alternative, refinement, question
    target: str | None = None       # target participant
    elapsed: float | None = None
    metadata: dict[str, Any] | None = None


class EventSink(Protocol):
    """Protocol for event consumers (renderers)."""
    def emit(self, event: UiEvent) -> None: ...
