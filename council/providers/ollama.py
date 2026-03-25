"""Ollama provider — local models via HTTP API with graceful tool fallback."""

from __future__ import annotations

import json
import logging
import sys

import httpx

from council.config import DEFAULT_SYSTEM_PROMPT, format_system_prompt
from council.providers.base import ModelResponse, StructuredResponse
from council.settings import get_settings
from council.tools import ToolCall, ToolSpec

log = logging.getLogger("council")

# Timeout for Ollama requests — first call can be very slow while the model loads into memory.
# Subsequent calls are fast since the model stays loaded.
_TIMEOUT = httpx.Timeout(600.0, connect=10.0)

# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from text that may contain markdown or preamble."""
    # Try the whole string first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON between braces
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def _build_messages(
    messages: list[dict],
    system_instruction: str | None = None,
) -> list[dict]:
    """Convert council messages to Ollama chat format."""
    result: list[dict] = []

    if system_instruction:
        result.append({"role": "system", "content": system_instruction})

    for msg in messages:
        participant = msg.get("participant", "user")
        content = msg.get("content", "")
        ts = msg.get("created_at", "")

        role = "user" if participant in ("user", "operator") else "assistant"
        text = f"[{ts}] [{participant}]: {content}" if ts else f"[{participant}]: {content}"

        # Inline text file attachments
        attachments = msg.get("attachments") or []
        for att in attachments:
            if att.get("mime_type", "").startswith("text/") or att.get("name", "").endswith(
                (".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml", ".toml", ".sql")
            ):
                text += f"\n\n--- {att.get('name', 'attachment')} ---\n{att.get('content', '')}"

        result.append({"role": role, "content": text})

    return result


def _tools_as_prompt(specs: list[ToolSpec]) -> str:
    """Format tool specs as a text description for injection into the prompt."""
    lines = ["You have access to the following tools. To use a tool, respond with a JSON object:",
             '{"tool": "<tool_name>", "arguments": {<args>}}', "", "Available tools:"]
    for spec in specs:
        params = spec.parameters.get("properties", {})
        param_strs = []
        for pname, pinfo in params.items():
            param_strs.append(f"  - {pname}: {pinfo.get('description', pinfo.get('type', 'string'))}")
        lines.append(f"\n{spec.name}: {spec.description}")
        if param_strs:
            lines.extend(param_strs)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class OllamaProvider:
    """Ollama provider for local models via HTTP API."""

    def __init__(self) -> None:
        cfg = get_settings().ollama
        self._host = cfg.host.rstrip("/")
        self._model_id = cfg.model
        self._name = "ollama"

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_id(self) -> str:
        return self._model_id

    def _url(self, path: str) -> str:
        return f"{self._host}{path}"

    # -------------------------------------------------------------------
    # Sync generation
    # -------------------------------------------------------------------

    def generate_sync(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> ModelResponse:
        system = format_system_prompt(system_instruction, self._name)
        ollama_messages = _build_messages(messages, system)

        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                self._url("/api/chat"),
                json={"model": self._model_id, "messages": ollama_messages, "stream": False},
            )
            resp.raise_for_status()

        data = resp.json()
        text = data.get("message", {}).get("content", "")

        return ModelResponse(
            text=text,
            participant=self._name,
            model_id=self._model_id,
            sources=[],
            raw=data,
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
        ollama_messages = _build_messages(messages, system)

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                self._url("/api/chat"),
                json={"model": self._model_id, "messages": ollama_messages, "stream": False},
            )
            resp.raise_for_status()

        data = resp.json()
        text = data.get("message", {}).get("content", "")

        return ModelResponse(
            text=text,
            participant=self._name,
            model_id=self._model_id,
            sources=[],
            raw=data,
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
        schema_prompt = (
            f"{system}\n\n"
            "You MUST respond with ONLY a JSON object in exactly this format, no other text:\n"
            '{"internal_reasoning": "your reasoning", "action": "CONTRIBUTE" or "PASS", '
            '"payload": "your response text if CONTRIBUTE, empty if PASS"}'
        )
        ollama_messages = _build_messages(messages, schema_prompt)

        structured_schema = {
            "type": "object",
            "properties": {
                "internal_reasoning": {"type": "string"},
                "action": {"type": "string", "enum": ["CONTRIBUTE", "PASS"]},
                "payload": {"type": "string"},
            },
            "required": ["internal_reasoning", "action", "payload"],
        }

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    self._url("/api/chat"),
                    json={
                        "model": self._model_id,
                        "messages": ollama_messages,
                        "stream": False,
                        "format": structured_schema,
                    },
                )
                resp.raise_for_status()

            data = resp.json()
            text = data.get("message", {}).get("content", "")
            parsed = _extract_json(text)

            if parsed and "action" in parsed:
                return StructuredResponse(
                    internal_reasoning=parsed.get("internal_reasoning", ""),
                    action=parsed.get("action", "PASS"),
                    payload=parsed.get("payload", ""),
                    participant=self._name,
                    model_id=self._model_id,
                )
        except Exception as exc:
            log.debug("Ollama structured generation failed: %s", exc)

        # Fallback: default to PASS
        return StructuredResponse(
            internal_reasoning="Failed to parse structured response from local model",
            action="PASS",
            payload="",
            participant=self._name,
            model_id=self._model_id,
        )

    # -------------------------------------------------------------------
    # Tool support (with graceful fallback)
    # -------------------------------------------------------------------

    def format_tools(self, specs: list[ToolSpec]) -> list[dict]:
        """Convert ToolSpecs to OpenAI-compatible format (supported by Ollama for some models)."""
        tools = []
        for spec in specs:
            tools.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            })
        return tools

    def extract_tool_calls(self, response) -> list[ToolCall] | None:
        """Extract tool calls from Ollama response dict."""
        if not isinstance(response, dict):
            return None
        msg = response.get("message", {})
        calls_raw = msg.get("tool_calls")
        if not calls_raw:
            return None

        calls = []
        for i, tc in enumerate(calls_raw):
            fn = tc.get("function", {})
            calls.append(ToolCall(
                call_id=f"ollama_{i}",
                name=fn.get("name", ""),
                arguments=fn.get("arguments", {}),
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
        ollama_messages = _build_messages(messages, system)

        # Inject extra parts as text
        if extra_native_parts:
            parts_text = []
            for part in extra_native_parts:
                if isinstance(part, dict):
                    parts_text.append(part.get("content", str(part)))
            if parts_text:
                ollama_messages.append({"role": "user", "content": "\n".join(parts_text)})

        formatted_tools = self.format_tools(tools)

        # Try with native tool calling first
        use_fallback = False
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    self._url("/api/chat"),
                    json={
                        "model": self._model_id,
                        "messages": ollama_messages,
                        "tools": formatted_tools,
                        "stream": False,
                    },
                )
                resp.raise_for_status()

            data = resp.json()
            text = data.get("message", {}).get("content", "")
            tool_calls = self.extract_tool_calls(data)

            if tool_calls:
                # Model used native tool calling — return directly
                return ModelResponse(
                    text=text,
                    participant=self._name,
                    model_id=self._model_id,
                    sources=[],
                    tool_calls=tool_calls,
                    raw=data,
                )

            # Model returned 200 but no tool_calls — common with local models
            # that don't support tools. Fall through to prompt injection.
            if text:
                # Check if the text itself contains a parseable tool call
                parsed = _extract_json(text)
                if parsed and "tool" in parsed:
                    return ModelResponse(
                        text=text,
                        participant=self._name,
                        model_id=self._model_id,
                        sources=[],
                        tool_calls=[ToolCall(
                            call_id="ollama_text_0",
                            name=parsed["tool"],
                            arguments=parsed.get("arguments", {}),
                        )],
                        raw=data,
                    )
                # Plain text response with no tool call — return as-is
                return ModelResponse(
                    text=text,
                    participant=self._name,
                    model_id=self._model_id,
                    sources=[],
                    raw=data,
                )

            # Empty response — try fallback
            use_fallback = True
            log.debug("Ollama returned empty response with tools, falling back to prompt injection")

        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            use_fallback = True
            log.debug("Ollama tool calling failed, falling back to prompt injection: %s", exc)

        if not use_fallback:
            # Shouldn't reach here, but safety net
            return ModelResponse(text="", participant=self._name, model_id=self._model_id, sources=[], raw=None)

        # Fallback: inject tools as prompt text
        tools_text = _tools_as_prompt(tools)
        fallback_system = f"{system}\n\n{tools_text}"
        fallback_messages = _build_messages(messages, fallback_system)

        if extra_native_parts:
            parts_text = []
            for part in extra_native_parts:
                if isinstance(part, dict):
                    parts_text.append(part.get("content", str(part)))
            if parts_text:
                fallback_messages.append({"role": "user", "content": "\n".join(parts_text)})

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                self._url("/api/chat"),
                json={"model": self._model_id, "messages": fallback_messages, "stream": False},
            )
            resp.raise_for_status()

        data = resp.json()
        text = data.get("message", {}).get("content", "")

        # Try to parse tool calls from text
        parsed = _extract_json(text)
        tool_calls = None
        if parsed and "tool" in parsed:
            tool_calls = [ToolCall(
                call_id="ollama_fallback_0",
                name=parsed["tool"],
                arguments=parsed.get("arguments", {}),
            )]

        return ModelResponse(
            text=text,
            participant=self._name,
            model_id=self._model_id,
            sources=[],
            tool_calls=tool_calls,
            raw=data,
        )
