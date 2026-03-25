"""Provider protocol and shared response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from council.tools import ToolCall, ToolResult, ToolSpec


@dataclass
class ModelResponse:
    """Response from a provider's generate() call."""
    text: str
    participant: str          # e.g. "gemini", "chatgpt"
    model_id: str             # e.g. "gemini-3.1-pro-preview"
    sources: list[str] = field(default_factory=list)  # grounding URLs
    tool_calls: list[ToolCall] | None = None  # if model wants to call tools
    raw: Any = None           # provider-specific response object


@dataclass
class StructuredResponse:
    """Response from a provider's generate_structured() call (PASS/CONTRIBUTE).
    Kept for backward compat — will be replaced by terminal tools in v3."""
    internal_reasoning: str
    action: str               # "CONTRIBUTE" or "PASS"
    payload: str
    participant: str
    model_id: str


class Provider(Protocol):
    """Interface that all model providers must implement."""

    @property
    def name(self) -> str:
        """Provider name used as participant tag (e.g. 'gemini', 'chatgpt')."""
        ...

    @property
    def model_id(self) -> str:
        """Specific model identifier (e.g. 'gemini-3.1-pro-preview')."""
        ...

    def generate_sync(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> ModelResponse:
        """Synchronous generation — used by ask/chat subcommands."""
        ...

    async def generate(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> ModelResponse:
        """Async generation — used by discuss orchestrator."""
        ...

    async def generate_structured(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> StructuredResponse:
        """Async structured generation (PASS/CONTRIBUTE JSON schema)."""
        ...

    def format_tools(self, specs: list[ToolSpec]) -> Any:
        """Convert internal ToolSpecs to provider-native tool declarations."""
        ...

    def extract_tool_calls(self, response: Any) -> list[ToolCall] | None:
        """Extract tool calls from a raw provider response. Returns None if no tool calls."""
        ...

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        system_instruction: str | None = None,
        extra_native_parts: list[Any] | None = None,
    ) -> ModelResponse:
        """Async generation with tool declarations. May return tool_calls in response."""
        ...
