"""Claude provider — Anthropic API with tool use support."""

from __future__ import annotations

import asyncio
import json
import logging
import sys

try:
    from anthropic import Anthropic, AsyncAnthropic
except ImportError:
    Anthropic = None  # type: ignore[assignment,misc]
    AsyncAnthropic = None  # type: ignore[assignment,misc]

from council.config import DEFAULT_SYSTEM_PROMPT, format_system_prompt
from council.formatting import file_parts
from council.providers.base import ModelResponse, StructuredResponse
from council.settings import get_settings
from council.tools import ToolCall, ToolSpec

log = logging.getLogger("council")


def _handle_api_error(exc: Exception) -> None:
    """Print a human-readable error for common Anthropic API failures."""
    err = str(exc).lower()
    msg = str(exc)

    if "credit balance" in err or "billing" in err:
        print(
            f"Error: Anthropic API — insufficient credits.\n"
            f"  Purchase credits at https://console.anthropic.com/settings/plans\n"
            f"  Detail: {msg}",
            file=sys.stderr,
        )
    elif "rate" in err or "429" in err:
        print(
            f"Error: Anthropic API — rate limited.\n"
            f"  Wait a moment and try again, or check your plan limits.\n"
            f"  Detail: {msg}",
            file=sys.stderr,
        )
    elif "authentication" in err or "401" in err or "invalid.*key" in err:
        print(
            f"Error: Anthropic API — invalid API key.\n"
            f"  Check your key in ~/.council/config.toml or COUNCIL_CLAUDE_API_KEY\n"
            f"  Detail: {msg}",
            file=sys.stderr,
        )
    elif "not_found" in err or "404" in err:
        print(
            f"Error: Anthropic API — model not found.\n"
            f"  Check available models at https://docs.anthropic.com/en/docs/about-claude/models\n"
            f"  Detail: {msg}",
            file=sys.stderr,
        )
    elif "overloaded" in err or "529" in err:
        print(
            f"Error: Anthropic API — overloaded. Try again in a moment.\n"
            f"  Detail: {msg}",
            file=sys.stderr,
        )
    else:
        print(
            f"Error: Anthropic API call failed.\n"
            f"  Detail: {msg}",
            file=sys.stderr,
        )

# ---------------------------------------------------------------------------
# Structured output schema for PASS/CONTRIBUTE (used as tool definition)
# ---------------------------------------------------------------------------

_STRUCTURED_TOOL = {
    "name": "council_response",
    "description": "Submit your structured PASS or CONTRIBUTE response.",
    "input_schema": {
        "type": "object",
        "properties": {
            "internal_reasoning": {
                "type": "string",
                "description": "Your private reasoning about whether to contribute or pass.",
            },
            "action": {
                "type": "string",
                "enum": ["CONTRIBUTE", "PASS"],
                "description": "CONTRIBUTE if you have something valuable to add, PASS if not.",
            },
            "payload": {
                "type": "string",
                "description": "Your response text (ignored if action is PASS).",
            },
        },
        "required": ["internal_reasoning", "action", "payload"],
    },
}


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def _build_messages(messages: list[dict]) -> list[dict]:
    """Convert council messages to Anthropic format with strict alternation."""
    result: list[dict] = []

    for msg in messages:
        participant = msg.get("participant", "user")
        content = msg.get("content", "")
        ts = msg.get("created_at", "")

        role = "user" if participant in ("user", "operator") else "assistant"
        text = f"[{ts}] [{participant}]: {content}" if ts else f"[{participant}]: {content}"

        # Handle file attachments — inline text files
        attachments = msg.get("attachments") or []
        for att in attachments:
            if att.get("mime_type", "").startswith("text/") or att.get("name", "").endswith(
                (".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml", ".toml", ".sql")
            ):
                text += f"\n\n--- {att.get('name', 'attachment')} ---\n{att.get('content', '')}"

        # Merge consecutive same-role messages (Claude requires strict alternation)
        if result and result[-1]["role"] == role:
            prev = result[-1]["content"]
            result[-1]["content"] = f"{prev}\n\n{text}"
        else:
            result.append({"role": role, "content": text})

    return result


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class ClaudeProvider:
    """Claude provider using the Anthropic API."""

    def __init__(self) -> None:
        if AsyncAnthropic is None:
            print(
                "Error: Claude provider requires the 'anthropic' package.\n"
                "  Install with: pip install council-engine",
                file=sys.stderr,
            )
            sys.exit(1)

        cfg = get_settings().claude
        if not cfg.api_key:
            print(
                "Error: Claude API key not configured.\n"
                "  Set it in ~/.council/config.toml or COUNCIL_CLAUDE_API_KEY env var.",
                file=sys.stderr,
            )
            sys.exit(1)

        self._async_client = AsyncAnthropic(api_key=cfg.api_key)
        self._sync_client = Anthropic(api_key=cfg.api_key)
        self._name = "claude"
        self._model_id = cfg.model

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_id(self) -> str:
        return self._model_id

    # -------------------------------------------------------------------
    # Sync generation
    # -------------------------------------------------------------------

    def generate_sync(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> ModelResponse:
        system = format_system_prompt(system_instruction, self._name)
        claude_messages = _build_messages(messages)

        try:
            resp = self._sync_client.messages.create(
                model=self._model_id,
                max_tokens=4096,
                system=system,
                messages=claude_messages,
            )
        except Exception as exc:
            _handle_api_error(exc)
            raise

        text = ""
        for block in resp.content:
            if block.type == "text":
                text += block.text

        return ModelResponse(
            text=text,
            participant=self._name,
            model_id=self._model_id,
            sources=[],
            raw=resp,
        )

    # -------------------------------------------------------------------
    # Async generation
    # -------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> ModelResponse:
        system = format_system_prompt(system_instruction, self._name)
        claude_messages = _build_messages(messages)

        try:
            resp = await self._async_client.messages.create(
                model=self._model_id,
                max_tokens=4096,
                system=system,
                messages=claude_messages,
            )
        except Exception as exc:
            _handle_api_error(exc)
            raise

        text = ""
        for block in resp.content:
            if block.type == "text":
                text += block.text

        return ModelResponse(
            text=text,
            participant=self._name,
            model_id=self._model_id,
            sources=[],
            raw=resp,
        )

    # -------------------------------------------------------------------
    # Structured generation (PASS/CONTRIBUTE)
    # -------------------------------------------------------------------

    async def generate_structured(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> StructuredResponse:
        system = format_system_prompt(system_instruction, self._name)
        claude_messages = _build_messages(messages)

        try:
            resp = await self._async_client.messages.create(
                model=self._model_id,
                max_tokens=2048,
                system=system,
                messages=claude_messages,
                tools=[_STRUCTURED_TOOL],
                tool_choice={"type": "tool", "name": "council_response", "disable_parallel_tool_use": True},
            )
        except Exception as exc:
            _handle_api_error(exc)
            raise

        # Extract the tool use input
        for block in resp.content:
            if block.type == "tool_use" and block.name == "council_response":
                data = block.input
                return StructuredResponse(
                    internal_reasoning=data.get("internal_reasoning", ""),
                    action=data.get("action", "PASS"),
                    payload=data.get("payload", ""),
                    participant=self._name,
                    model_id=self._model_id,
                )

        # Fallback: treat as PASS
        return StructuredResponse(
            rationale="Failed to parse structured response",
            action="PASS",
            payload="",
            participant=self._name,
            model_id=self._model_id,
        )

    # -------------------------------------------------------------------
    # Tool support
    # -------------------------------------------------------------------

    def format_tools(self, specs: list[ToolSpec]) -> list[dict]:
        """Convert ToolSpecs to Anthropic tool format."""
        tools = []
        for spec in specs:
            tools.append({
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.parameters,
            })
        return tools

    def extract_tool_calls(self, response) -> list[ToolCall] | None:
        """Extract tool calls from an Anthropic response."""
        calls = []
        for block in response.content:
            if block.type == "tool_use":
                calls.append(ToolCall(
                    call_id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))
        return calls if calls else None

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        system_instruction: str | None = None,
        extra_native_parts: list | None = None,
    ) -> ModelResponse:
        system = format_system_prompt(system_instruction, self._name)
        claude_messages = _build_messages(messages)

        # Inject extra native parts as a user message if provided
        if extra_native_parts:
            parts_text = []
            for part in extra_native_parts:
                if isinstance(part, dict) and "data" in part:
                    parts_text.append(f"[file: {part.get('name', 'attachment')}]")
                elif isinstance(part, dict) and "content" in part:
                    parts_text.append(part["content"])
            if parts_text:
                extra_msg = {"role": "user", "content": "\n".join(parts_text)}
                # Merge if last message is also user
                if claude_messages and claude_messages[-1]["role"] == "user":
                    claude_messages[-1]["content"] += "\n\n" + extra_msg["content"]
                else:
                    claude_messages.append(extra_msg)

        formatted_tools = self.format_tools(tools)

        try:
            resp = await self._async_client.messages.create(
                model=self._model_id,
                max_tokens=4096,
                system=system,
                messages=claude_messages,
                tools=formatted_tools,
            )
        except Exception as exc:
            _handle_api_error(exc)
            raise

        # Extract text and tool calls
        text = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    call_id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return ModelResponse(
            text=text,
            participant=self._name,
            model_id=self._model_id,
            sources=[],
            tool_calls=tool_calls if tool_calls else None,
            raw=resp,
        )
