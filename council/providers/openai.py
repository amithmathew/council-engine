"""OpenAI provider — using the Responses API with GPT-5.4."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path

from council.config import DEFAULT_SYSTEM_PROMPT, format_system_prompt
from council.settings import get_settings
from council.formatting import is_text_mime
from council.providers.base import ModelResponse, StructuredResponse
from council.tools import ToolCall, ToolSpec

log = logging.getLogger("council")

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# ---------------------------------------------------------------------------
# Structured output schema for PASS/CONTRIBUTE
# ---------------------------------------------------------------------------

_STRUCTURED_SCHEMA = {
    "type": "json_schema",
    "name": "council_response",
    "strict": True,
    "schema": {
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
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Content building
# ---------------------------------------------------------------------------

def _build_input(
    messages: list[dict],
    system_instruction: str | None = None,
    model_name: str = "ChatGPT",
) -> tuple[str, list[dict]]:
    """Convert stored messages to OpenAI Responses API format.

    Returns (instructions, input_messages) — system prompt goes to the
    top-level `instructions` parameter, not in the input array.

    Binary files (PDFs, images) are only sent on the MOST RECENT user message
    to avoid re-encoding the same file on every turn in the conversation.
    Older messages get a text placeholder instead.
    """
    instructions = format_system_prompt(system_instruction, model_name)
    oai_input: list[dict] = []

    # Find the index of the last user message (for file attachment decisions)
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg["participant"] == "user":
            last_user_idx = i

    for i, msg in enumerate(messages):
        participant = msg["participant"]
        timestamp = msg["created_at"]
        text = f"[{timestamp}] [{participant}]: {msg['content']}"
        role = "user" if participant == "user" else "assistant"

        content_parts: list[dict] = []
        file_text_additions = ""

        if msg.get("attachments"):
            is_latest_user = (i == last_user_idx)
            for att in msg["attachments"]:
                p = Path(att["path"])
                if not p.exists():
                    file_text_additions += f"\n\n[Attachment unavailable: {att['name']}]"
                elif is_text_mime(att["mime_type"]):
                    # Text files: always inline (small)
                    file_content = p.read_text(errors="replace")
                    file_text_additions += f"\n\n--- File: {att['name']} ---\n{file_content}\n--- End: {att['name']} ---"
                elif is_latest_user and role == "user":
                    # Binary files: only send on the latest user message
                    data = p.read_bytes()
                    b64 = base64.b64encode(data).decode("utf-8")
                    content_parts.append({
                        "type": "input_file",
                        "filename": att["name"],
                        "file_data": f"data:{att['mime_type']};base64,{b64}",
                    })
                    log.debug("Attaching binary file %s (%d bytes) to latest user message",
                              att["name"], len(data))
                else:
                    # Binary files on older messages: text placeholder
                    file_text_additions += f"\n\n[Previously attached file: {att['name']}]"

        if content_parts and role == "user":
            # Mixed content: text + binary files (only valid on user messages)
            content_parts.insert(0, {
                "type": "input_text",
                "text": text + file_text_additions,
            })
            oai_input.append({"role": role, "content": content_parts})
        else:
            oai_input.append({"role": role, "content": text + file_text_additions})

    return instructions, oai_input


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

class OpenAIProvider:
    """OpenAI provider using the Responses API (GPT-5.4)."""

    def __init__(self) -> None:
        if AsyncOpenAI is None:
            print(
                "Error: OpenAI provider requires the 'openai' package.\n"
                "  Install with: pip install council-engine",
                file=sys.stderr,
            )
            sys.exit(1)

        cfg = get_settings().chatgpt
        if not cfg.api_key:
            print(
                "Error: ChatGPT API key not configured.\n"
                "  Set it in ~/.council/config.toml or COUNCIL_OPENAI_API_KEY env var.",
                file=sys.stderr,
            )
            sys.exit(1)

        self._client = AsyncOpenAI(api_key=cfg.api_key)
        self._name = "chatgpt"
        self._model_id = cfg.model

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate_sync(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> ModelResponse:
        """Sync fallback — wraps async."""
        import asyncio
        return asyncio.run(self.generate(messages, system_instruction))

    async def generate(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> ModelResponse:
        """Async generation using the Responses API."""
        instructions, oai_input = _build_input(messages, system_instruction, self._name)
        log.debug("[async] Querying OpenAI model=%s with %d input items", self._model_id, len(oai_input))

        try:
            # web_search_preview with reasoning=none to avoid search loops.
            # GPT-5.4 + web_search + medium reasoning spirals into repeated
            # searches on broad queries. Reasoning stays at medium for
            # structured output (cascade rounds) which don't use search.
            response = await self._client.responses.create(
                model=self._model_id,
                instructions=instructions,
                input=oai_input,
                tools=[{"type": "web_search_preview"}],
                reasoning={"effort": "none"},
                text={"format": {"type": "text"}, "verbosity": "medium"},
            )
        except Exception as exc:
            log.debug("OpenAI API error", exc_info=True)
            print(f"Error: OpenAI API error: {exc}", file=sys.stderr)
            raise

        text = response.output_text or ""
        log.debug("[async] OpenAI response received (%d chars)", len(text))

        return ModelResponse(
            text=text,
            participant=self._name,
            model_id=self._model_id,
            sources=[],
            raw=response,
        )

    async def generate_structured(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> StructuredResponse:
        """Async structured generation (PASS/CONTRIBUTE JSON schema)."""
        instructions, oai_input = _build_input(messages, system_instruction, self._name)
        log.debug("[structured] Querying OpenAI model=%s with %d input items", self._model_id, len(oai_input))

        try:
            response = await self._client.responses.create(
                model=self._model_id,
                instructions=instructions,
                input=oai_input,
                reasoning={"effort": "medium", "summary": "auto"},
                text={"format": _STRUCTURED_SCHEMA, "verbosity": "medium"},
            )
        except Exception as exc:
            log.debug("OpenAI structured API error", exc_info=True)
            raise

        raw_text = response.output_text or "{}"
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            log.warning("OpenAI returned invalid JSON: %s", raw_text[:200])
            raise ValueError(f"Invalid structured response from OpenAI: {raw_text[:200]}")

        return StructuredResponse(
            internal_reasoning=data.get("internal_reasoning", ""),
            action=data.get("action", "PASS"),
            payload=data.get("payload", ""),
            participant=self._name,
            model_id=self._model_id,
        )

    # ------------------------------------------------------------------
    # Tool support
    # ------------------------------------------------------------------

    def format_tools(self, specs: list[ToolSpec]) -> list[dict]:
        """Convert ToolSpecs to OpenAI Responses API function tools."""
        tools = []
        for spec in specs:
            tools.append({
                "type": "function",
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            })
        return tools

    def extract_tool_calls(self, response) -> list[ToolCall] | None:
        """Extract function calls from an OpenAI Responses API response."""
        if not hasattr(response, "output") or not response.output:
            return None
        calls = []
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                args = {}
                if hasattr(item, "arguments") and item.arguments:
                    try:
                        args = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                calls.append(ToolCall(
                    call_id=getattr(item, "call_id", item.name),
                    name=item.name,
                    arguments=args,
                ))
        return calls if calls else None

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        system_instruction: str | None = None,
        extra_native_parts: list | None = None,
    ) -> ModelResponse:
        """Async generation with tool declarations."""
        instructions, oai_input = _build_input(messages, system_instruction, self._name)

        # Inject extra native parts (rehydrated attachments) into a user message
        if extra_native_parts:
            oai_input.append({
                "role": "user",
                "content": extra_native_parts,
            })

        oai_tools = self.format_tools(tools)
        # Add web search alongside function tools
        oai_tools.append({"type": "web_search_preview"})

        log.debug("[tools] Querying OpenAI model=%s with %d input items, %d tools",
                  self._model_id, len(oai_input), len(tools))

        try:
            response = await self._client.responses.create(
                model=self._model_id,
                instructions=instructions,
                input=oai_input,
                tools=oai_tools,
                reasoning={"effort": "none"},
            )
        except Exception as exc:
            log.debug("OpenAI API error", exc_info=True)
            raise

        tool_calls = self.extract_tool_calls(response)
        text = response.output_text or ""

        return ModelResponse(
            text=text,
            participant=self._name,
            model_id=self._model_id,
            sources=[],
            tool_calls=tool_calls,
            raw=response,
        )
