"""Gemini provider — sync + async, with structured output support."""

from __future__ import annotations

import json
import logging
import sys

from google import genai
from google.genai import errors as genai_errors
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    FunctionDeclaration,
    GenerateContentConfig,
    GoogleSearch,
    Part,
    Tool,
)

from council.config import DEFAULT_SYSTEM_PROMPT, format_system_prompt
from council.settings import get_settings
from council.formatting import file_parts
from council.providers.base import ModelResponse, StructuredResponse
from council.tools import ToolCall, ToolSpec

log = logging.getLogger("council")

# Suppress noisy AFC warnings from the google-genai SDK.
# The SDK uses "google_genai" (underscore) as the logger namespace, not "google.genai".
logging.getLogger("google_genai").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Structured output schema for PASS/CONTRIBUTE
# ---------------------------------------------------------------------------

_STRUCTURED_SCHEMA = {
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
}


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        cfg = get_settings().gemini
        try:
            if cfg.api_key and not cfg.vertex_ai:
                log.debug("Initializing genai client with API key")
                _client = genai.Client(api_key=cfg.api_key)
            else:
                log.debug("Initializing genai client (Vertex AI, project=%s, location=%s)", cfg.project, cfg.location)
                _client = genai.Client(
                    vertexai=True,
                    project=cfg.project,
                    location=cfg.location,
                )
        except Exception as exc:
            log.debug("Client init failed", exc_info=True)
            if cfg.api_key and not cfg.vertex_ai:
                print(
                    f"Error: Failed to initialize Gemini client.\n"
                    f"  Check your API key in ~/.council/config.toml or COUNCIL_GEMINI_API_KEY\n"
                    f"  Detail: {exc}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Error: Failed to initialize Gemini client (Vertex AI).\n"
                    f"  Project: {cfg.project}, Location: {cfg.location}\n"
                    f"  Ensure ADC is configured: gcloud auth application-default login\n"
                    f"  Detail: {exc}",
                    file=sys.stderr,
                )
            sys.exit(1)
    return _client


# ---------------------------------------------------------------------------
# Content building
# ---------------------------------------------------------------------------

def _build_contents(messages: list[dict]) -> list[Content]:
    """Convert stored messages to Gemini Content objects."""
    contents: list[Content] = []
    for msg in messages:
        participant = msg["participant"]
        timestamp = msg["created_at"]
        text = f"[{timestamp}] [{participant}]: {msg['content']}"
        role = "user" if participant == "user" else "model"
        parts: list[Part] = [Part(text=text)]
        parts.extend(file_parts(msg.get("attachments")))
        contents.append(Content(role=role, parts=parts))
    return contents


def _make_config(
    system_instruction: str | None = None,
    structured: bool = False,
    model_name: str = "Gemini",
) -> GenerateContentConfig:
    """Build GenerateContentConfig with optional structured output."""
    prompt = format_system_prompt(system_instruction, model_name)
    log.debug("System instruction: %s", prompt[:80] + "..." if len(prompt) > 80 else prompt)

    kwargs: dict = {
        "system_instruction": prompt,
        "tools": [Tool(google_search=GoogleSearch())],
    }

    if structured:
        # Enable JSON mode with schema enforcement
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = _STRUCTURED_SCHEMA
        # Google Search not compatible with structured output
        kwargs.pop("tools")

    return GenerateContentConfig(**kwargs)


def _extract_sources(response) -> list[str]:
    """Extract grounding source URLs from response metadata."""
    sources: list[str] = []
    try:
        metadata = getattr(response.candidates[0], "grounding_metadata", None)
        if metadata and getattr(metadata, "grounding_chunks", None):
            for chunk in metadata.grounding_chunks:
                web = getattr(chunk, "web", None)
                if web:
                    title = getattr(web, "title", None) or getattr(web, "domain", "") or "Source"
                    uri = getattr(web, "uri", "") or ""
                    sources.append(f"- [{title}]({uri})")
    except (IndexError, AttributeError):
        pass
    return sources


def _format_text_with_sources(text: str, sources: list[str]) -> str:
    if sources:
        text += "\n\n**Sources:**\n" + "\n".join(sources)
    return text


def _handle_api_error(exc: genai_errors.ClientError) -> None:
    """Print a user-friendly error message for Gemini API errors."""
    log.debug("Gemini API error", exc_info=True)
    status = getattr(exc, "status", None) or getattr(exc, "code", "")
    if "429" in str(status) or "RESOURCE_EXHAUSTED" in str(exc):
        print("Error: Gemini quota exceeded. Try again later.", file=sys.stderr)
    elif "404" in str(status) or "NOT_FOUND" in str(exc):
        print(
            f"Error: Model '{get_settings().gemini.model}' not found. Check model in config.\n"
            f"  Detail: {exc}",
            file=sys.stderr,
        )
    elif "403" in str(status) or "PERMISSION_DENIED" in str(exc):
        print(
            f"Error: Permission denied. Ensure your account has Vertex AI access.\n"
            f"  Project: {PROJECT}\n"
            f"  Detail: {exc}",
            file=sys.stderr,
        )
    else:
        print(f"Error: Gemini API error: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------

class GeminiProvider:
    """Gemini provider with sync and async generation."""

    def __init__(self) -> None:
        self._name = "gemini"
        self._model_id = get_settings().gemini.model

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
        """Synchronous generation — used by ask/chat subcommands."""
        client = _get_client()
        contents = _build_contents(messages)
        log.debug("Querying model=%s with %d message(s)", self._model_id, len(messages))

        try:
            response = client.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=_make_config(system_instruction, model_name=self._name),
            )
        except genai_errors.ClientError as exc:
            _handle_api_error(exc)
            sys.exit(1)
        except Exception as exc:
            log.debug("Unexpected error querying Gemini", exc_info=True)
            print(f"Error: Unexpected error: {exc}", file=sys.stderr)
            sys.exit(1)

        log.debug("Response received (%d candidates)", len(response.candidates or []))
        text = response.text or ""
        sources = _extract_sources(response)

        return ModelResponse(
            text=_format_text_with_sources(text, sources),
            participant=self._name,
            model_id=self._model_id,
            sources=sources,
            raw=response,
        )

    async def generate(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> ModelResponse:
        """Async generation — used by discuss orchestrator."""
        client = _get_client()
        contents = _build_contents(messages)
        log.debug("[async] Querying model=%s with %d message(s)", self._model_id, len(messages))

        try:
            response = await client.aio.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=_make_config(system_instruction, model_name=self._name),
            )
        except genai_errors.ClientError as exc:
            _handle_api_error(exc)
            raise
        except Exception as exc:
            log.debug("Unexpected error querying Gemini", exc_info=True)
            raise

        log.debug("[async] Response received (%d candidates)", len(response.candidates or []))
        text = response.text or ""
        sources = _extract_sources(response)

        return ModelResponse(
            text=_format_text_with_sources(text, sources),
            participant=self._name,
            model_id=self._model_id,
            sources=sources,
            raw=response,
        )

    async def generate_structured(
        self,
        messages: list[dict],
        system_instruction: str | None = None,
    ) -> StructuredResponse:
        """Async structured generation (PASS/CONTRIBUTE JSON schema)."""
        client = _get_client()
        contents = _build_contents(messages)
        log.debug("[structured] Querying model=%s with %d message(s)", self._model_id, len(messages))

        try:
            response = await client.aio.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=_make_config(system_instruction, structured=True, model_name=self._name),
            )
        except genai_errors.ClientError as exc:
            _handle_api_error(exc)
            raise
        except Exception as exc:
            log.debug("Unexpected error in structured query", exc_info=True)
            raise

        # Parse the JSON response
        raw_text = response.text or "{}"
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            log.warning("Gemini returned invalid JSON: %s", raw_text[:200])
            raise ValueError(f"Invalid structured response from Gemini: {raw_text[:200]}")

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

    def format_tools(self, specs: list[ToolSpec]) -> list[Tool]:
        """Convert ToolSpecs to Gemini FunctionDeclarations."""
        decls = []
        for spec in specs:
            decls.append(FunctionDeclaration(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters if spec.parameters.get("properties") else None,
            ))
        return [Tool(function_declarations=decls)]

    def extract_tool_calls(self, response) -> list[ToolCall] | None:
        """Extract function calls from a Gemini response."""
        if not response.candidates:
            return None
        candidate = response.candidates[0]
        if not hasattr(candidate.content, "parts"):
            return None
        calls = []
        for part in candidate.content.parts:
            fc = getattr(part, "function_call", None)
            if fc:
                calls.append(ToolCall(
                    call_id=fc.name,  # Gemini uses name as ID
                    name=fc.name,
                    arguments=dict(fc.args) if fc.args else {},
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
        client = _get_client()
        contents = _build_contents(messages)

        # Inject extra native parts (e.g. rehydrated attachments) into last user message
        if extra_native_parts:
            if contents and contents[-1].role == "user":
                contents[-1].parts.extend(extra_native_parts)
            else:
                contents.append(Content(role="user", parts=extra_native_parts))

        prompt = format_system_prompt(system_instruction, self._name)
        gemini_tools = self.format_tools(tools)
        # Include Google Search alongside function declarations
        gemini_tools.append(Tool(google_search=GoogleSearch()))

        config = GenerateContentConfig(
            system_instruction=prompt,
            tools=gemini_tools,
            # Explicitly disable AFC — we handle tool calls manually.
            # This also suppresses the "AFC is disabled" warning the SDK emits
            # when tools include FunctionDeclarations (which are incompatible
            # with AFC's requirement for Python callables).
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
        )

        log.debug("[tools] Querying model=%s with %d message(s), %d tools",
                  self._model_id, len(messages), len(tools))

        try:
            response = await client.aio.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=config,
            )
        except genai_errors.ClientError as exc:
            _handle_api_error(exc)
            raise

        # Check for tool calls first
        tool_calls = self.extract_tool_calls(response)
        text = response.text or ""
        sources = _extract_sources(response)

        return ModelResponse(
            text=_format_text_with_sources(text, sources) if text else "",
            participant=self._name,
            model_id=self._model_id,
            sources=sources,
            tool_calls=tool_calls,
            raw=response,
        )
