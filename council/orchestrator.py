"""Multi-model orchestrator — bounded 3-stage council protocol.

Stage 1 (Proposals):  All providers respond in parallel — independent, unanchored.
Stage 2 (Critique):   All providers critique peer proposals in parallel — structured tools.
Stage 3 (Synthesis):  Lead provider synthesizes the discussion into a final recommendation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from council.config import DEFAULT_SYSTEM_PROMPT
from council.db import (
    add_message, get_attachment, get_messages, get_or_create_conversation,
    list_attachments as db_list_attachments, store_attachment,
)
from council.events import EventSink, UiEvent
from council.chair import Chair
from council.formatting import file_parts, is_text_mime
from council.providers.base import ModelResponse, Provider
from council.tools import (
    CRITIQUE_TOOLS, PROPOSAL_TOOLS,
    CHAT_TOOLS, ToolCall, ToolResult, ToolSpec,
)

log = logging.getLogger("council")

_PROVIDER_TIMEOUT = 180  # seconds per provider call


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _emit(sink: EventSink | None, event: UiEvent) -> None:
    """Emit an event if a sink is present, otherwise no-op."""
    if sink is not None:
        sink.emit(event)


async def _countdown_pause(sink: EventSink | None, next_stage: str, seconds: int = 5) -> bool:
    """Show a countdown timer, auto-continue when it expires. Returns False if user cancelled."""
    import select

    _emit(sink, UiEvent(type="status", text=""))
    for remaining in range(seconds, 0, -1):
        msg = f"Continuing to {next_stage} in {remaining}s... (Enter to continue, q to end)"
        # Use \r to overwrite the line
        print(f"\r\033[K  {msg}", end="", flush=True)
        # Check for user input (non-blocking)
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 1.0)
            if ready:
                response = sys.stdin.readline().strip().lower()
                print()  # newline after the countdown line
                if response in ("q", "quit", "end", "e"):
                    return False
                return True  # Enter or any other input = continue
        except (EOFError, KeyboardInterrupt):
            print()
            return True
    print()  # newline after countdown completes
    return True


# ---------------------------------------------------------------------------
# Stage prompts
# ---------------------------------------------------------------------------

_PROPOSAL_STRUCTURE_PROMPT = (
    "\n\nProduce a decision proposal using exactly this format. "
    "The structured fields ARE the proposal — do not add a separate rationale.\n\n"
    "POSITION: one sentence — your specific recommendation.\n\n"
    "WHY:\n- 2 to 4 bullets. Strongest reasons only.\n"
    "- Focus on factors that discriminate this option from alternatives.\n\n"
    "ASSUMPTIONS:\n- Up to 3 bullets. Only uncertain premises that materially affect the recommendation.\n\n"
    "RISKS:\n- Up to 3 bullets. Concrete failure modes or blind spots.\n"
    "- Do not restate assumptions unless the risk mechanism is different.\n\n"
    "SWITCH_IF:\n- Up to 2 bullets. Specific conditions or new facts that would change your recommendation.\n\n"
    "NEXT_STEP: one concrete action.\n\n"
    "Rules:\n"
    "- No introduction, conclusion, or recap.\n"
    "- No generic caveats or filler.\n"
    "- Each bullet must add non-overlapping information.\n"
    "- Prefer concrete tradeoffs over balanced prose.\n"
)

_CRITIQUE_PROMPT = (
    "STAGE 2: CRITIQUE.\n\n"
    "You are reviewing proposals from the other council participants.\n\n"
    "Your ONLY job is to identify critical flaws, blind spots, faulty assumptions, "
    "or materially better alternatives.\n\n"
    "DO NOT praise. DO NOT say 'I agree'. DO NOT summarize. DO NOT restate your own position.\n"
    "DO NOT respond with plain text — you MUST end your turn by calling exactly one tool: "
    "either call_pass() or call_contribute().\n\n"
    "If the positions are sound, call call_pass(reason='...') with a brief reason. "
    "Silence is approval.\n\n"
    "Only call call_contribute if you can provide at least one of:\n"
    "- A concrete flaw or blind spot (kind='challenge')\n"
    "- A materially better alternative (kind='alternative')\n"
    "- A refinement that would change the recommendation (kind='refinement')\n"
    "- A question that exposes a critical ambiguity (kind='question')\n\n"
    "Keep contributions short and specific (max 3 bullet points).\n\n"
    "Other participants' positions:\n{proposals}\n"
)

_RESOLUTION_SYSTEM = (
    "You are the resolver for a multi-model advisory council.\n\n"
    "Your job is NOT to force consensus. Your job is to produce the most useful "
    "decision outcome for the user.\n\n"
    "You must choose exactly one resolution type:\n\n"
    "1. RECOMMENDATION — use when the council substantially agrees, or one position "
    "clearly dominates. Provide a single clear recommendation.\n\n"
    "2. ALTERNATIVES — use when there are 2-3 genuinely viable but materially "
    "different approaches. You MUST provide a default option and explicit decision "
    "rules for when to choose the alternatives. Do NOT merge incompatible positions "
    "into a compromise.\n\n"
    "3. QUESTION — use when a single missing piece of information would be decisive. "
    "Ask exactly one sharp question. Also provide a conditional recommendation "
    "for each likely answer.\n\n"
    "4. INVESTIGATE — use when the disagreement depends on verifiable evidence "
    "(data, research, existing state of affairs). Provide a concrete investigation "
    "plan and a provisional default.\n\n"
    "Core rules:\n"
    "- Never hedge with 'it depends' — convert that into ALTERNATIVES, QUESTION, or INVESTIGATE.\n"
    "- Prefer RECOMMENDATION when one position is clearly stronger.\n"
    "- Prefer ALTERNATIVES over mushy compromise when positions are genuinely incompatible.\n"
    "- Always be decisive. Always state the main tradeoff explicitly.\n"
    "- In ALTERNATIVES, always explain why you did not merge them.\n"
    "- QUESTION and INVESTIGATE should be rare — prefer making a recommendation.\n\n"
    "Return valid JSON only (no markdown fences). Required fields:\n"
    '{"resolution_type": "RECOMMENDATION"|"ALTERNATIVES"|"QUESTION"|"INVESTIGATE", '
    '"markdown": "<human-readable markdown rendering of the decision>"}\n\n'
    "Additional fields by type:\n"
    'RECOMMENDATION: "recommendation": {"title": str, "summary": str, "why": [str], '
    '"risks": [str], "next_steps": [str]}\n'
    'ALTERNATIVES: "default_option": str, "options": [{"id": str, "title": str, '
    '"summary": str, "best_when": [str], "risks": [str]}], '
    '"decision_rules": [str], "why_not_merged": str\n'
    'QUESTION: "question": {"text": str, "why_it_matters": str}, '
    '"conditional_recommendation": {"if_A": str, "if_B": str}\n'
    'INVESTIGATE: "investigation": {"goal": str, "items": [str]}, '
    '"provisional_default": str\n'
)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _execute_tool(
    call: ToolCall,
    db: sqlite3.Connection,
    conv_id: int,
    loaded_this_turn: set[str],
    chair: Chair | None = None,
    participant: str | None = None,
    sink: EventSink | None = None,
) -> ToolResult:
    """Execute a tool call and return the result."""
    if call.name == "list_attachments":
        atts = db_list_attachments(db, conv_id)
        content = json.dumps([
            {"id": a["id"], "name": a["name"], "mime_type": a["mime_type"],
             "size_bytes": a["size_bytes"]}
            for a in atts
        ], indent=2)
        return ToolResult(call_id=call.call_id, name=call.name, content=content)

    elif call.name == "read_attachment":
        att_id = call.arguments.get("id", "")
        att = get_attachment(db, att_id)
        if not att:
            return ToolResult(
                call_id=call.call_id, name=call.name,
                content=f"Attachment '{att_id}' not found.",
            )
        if att_id in loaded_this_turn:
            return ToolResult(
                call_id=call.call_id, name=call.name,
                content=f"Attachment '{att['name']}' is already loaded in your current context.",
            )
        if len(loaded_this_turn) >= 2:
            return ToolResult(
                call_id=call.call_id, name=call.name,
                content="Attachment limit reached (max 2 per turn). Finish with call_pass or call_contribute.",
            )

        loaded_this_turn.add(att_id)
        p = Path(att["file_path"])
        if not p.exists():
            return ToolResult(
                call_id=call.call_id, name=call.name,
                content=f"Attachment file missing from disk: {att['name']}",
            )

        if is_text_mime(att["mime_type"]):
            text = p.read_text(errors="replace")
            if len(text) > 50_000:
                text = text[:50_000] + f"\n\n[Truncated — showing first 50,000 of {len(text)} chars]"
            return ToolResult(
                call_id=call.call_id, name=call.name,
                content=f"--- File: {att['name']} ---\n{text}\n--- End: {att['name']} ---",
            )
        else:
            data = p.read_bytes()
            log.debug("Loading binary attachment %s (%d bytes) for native injection", att["name"], len(data))
            return ToolResult(
                call_id=call.call_id, name=call.name,
                content=f"[Loaded {att['name']} ({att['mime_type']}, {len(data)} bytes) into your context]",
                native_parts=[{
                    "data": data,
                    "mime_type": att["mime_type"],
                    "name": att["name"],
                }],
            )

    elif call.name == "read_file":
        path = call.arguments.get("path", "")
        if not path:
            return ToolResult(call_id=call.call_id, name=call.name, content="Error: path is required.")
        if chair:
            content = chair.read_file(path, participant or "unknown", sink)
        else:
            content = f"Error: no chair available to read files."
        return ToolResult(call_id=call.call_id, name=call.name, content=content)

    elif call.name == "write_file":
        path = call.arguments.get("path", "")
        content = call.arguments.get("content", "")
        if not path:
            return ToolResult(call_id=call.call_id, name=call.name, content="Error: path is required.")
        if chair:
            result_msg = chair.write_file(path, content, participant or "unknown", sink)
        else:
            result_msg = "Error: no chair available to write files."
        return ToolResult(call_id=call.call_id, name=call.name, content=result_msg)

    elif call.name == "edit_file":
        path = call.arguments.get("path", "")
        old_text = call.arguments.get("old_text", "")
        new_text = call.arguments.get("new_text", "")
        if not path or not old_text:
            return ToolResult(call_id=call.call_id, name=call.name, content="Error: path and old_text are required.")
        if chair:
            result_msg = chair.edit_file(path, old_text, new_text, participant or "unknown", sink)
        else:
            result_msg = "Error: no chair available to edit files."
        return ToolResult(call_id=call.call_id, name=call.name, content=result_msg)

    elif call.name == "find_files":
        pattern = call.arguments.get("pattern")
        directory = call.arguments.get("directory")
        if chair:
            result_msg = chair.find_files(pattern, directory, sink)
        else:
            result_msg = "Error: no chair available to find files."
        return ToolResult(call_id=call.call_id, name=call.name, content=result_msg)

    elif call.name == "save_file":
        # Legacy compatibility — route through write_file
        filename = call.arguments.get("filename", "") or call.arguments.get("path", "")
        content = call.arguments.get("content", "")
        if not filename:
            return ToolResult(call_id=call.call_id, name=call.name, content="Error: filename is required.")
        if chair:
            result_msg = chair.write_file(filename, content, participant or "unknown", sink)
        else:
            result_msg = "Error: no chair available to write files."
        return ToolResult(call_id=call.call_id, name=call.name, content=result_msg)

    elif call.name == "request_info":
        question = call.arguments.get("question", "")
        if not question:
            return ToolResult(
                call_id=call.call_id, name=call.name,
                content="Error: question is required.",
            )
        if chair:
            answer = chair.fulfill_request(question, participant or "unknown", sink)
        else:
            answer = "No chair available. Proceed with best judgment."
        return ToolResult(call_id=call.call_id, name=call.name, content=answer)

    elif call.name == "call_pass":
        reason = call.arguments.get("reason", "") or call.arguments.get("note", "")
        return ToolResult(
            call_id=call.call_id, name=call.name,
            is_terminal=True, terminal_action="pass",
            terminal_payload=reason,
        )

    elif call.name == "call_contribute":
        message = call.arguments.get("message", "")
        kind = call.arguments.get("kind")
        target = call.arguments.get("target")
        return ToolResult(
            call_id=call.call_id, name=call.name,
            is_terminal=True, terminal_action="contribute",
            terminal_payload=message,
            critique_kind=kind,
            critique_target=target,
        )

    elif call.name == "call_operator":
        task = call.arguments.get("task_description", "")
        return ToolResult(
            call_id=call.call_id, name=call.name,
            is_terminal=True, terminal_action="operator_request",
            terminal_payload=task,
        )

    elif call.name == "call_council":
        reason = call.arguments.get("reason", "")
        return ToolResult(
            call_id=call.call_id, name=call.name,
            is_terminal=True, terminal_action="reconvene",
            terminal_payload=reason,
        )

    else:
        return ToolResult(
            call_id=call.call_id, name=call.name,
            content=f"Unknown tool: {call.name}",
        )


async def _run_model_turn(
    provider: Provider,
    messages: list[dict],
    tools: list[ToolSpec],
    system: str | None,
    db: sqlite3.Connection,
    conv_id: int,
    chair: Chair | None = None,
    sink: EventSink | None = None,
    max_steps: int = 3,
) -> ToolResult | ModelResponse:
    """Run a model turn with tool execution loop.

    Returns either:
    - ToolResult with is_terminal=True (call_pass, call_contribute, or call_operator)
    - ModelResponse if the model responds with text (no terminal tool called)
    """
    loaded_this_turn: set[str] = set()
    extra_native_parts: list = []
    working_messages = list(messages)

    for step in range(max_steps):
        log.debug("Tool loop step %d/%d for %s", step + 1, max_steps, provider.name)

        resp = await asyncio.wait_for(
            provider.generate_with_tools(
                working_messages, tools, system_instruction=system,
                extra_native_parts=extra_native_parts if extra_native_parts else None,
            ),
            timeout=_PROVIDER_TIMEOUT,
        )

        if not resp.tool_calls:
            return resp

        for call in resp.tool_calls:
            result = _execute_tool(
                call, db, conv_id, loaded_this_turn,
                chair=chair, participant=provider.name, sink=sink,
            )

            if result.is_terminal:
                return result

            if result.content:
                working_messages.append({
                    "participant": "tool",
                    "content": f"[Tool: {result.name}] {result.content}",
                    "created_at": _now(),
                    "attachments": None,
                })

            if result.native_parts:
                extra_native_parts.extend(result.native_parts)

        extra_native_parts = []

    log.warning("%s hit max tool steps (%d), defaulting to PASS", provider.name, max_steps)
    return ToolResult(
        call_id="max_steps", name="call_pass",
        is_terminal=True, terminal_action="pass",
        terminal_payload="(max tool steps reached)",
    )


# ---------------------------------------------------------------------------
# File attachment handling
# ---------------------------------------------------------------------------

def _store_files_as_attachments(
    db: sqlite3.Connection,
    conv_id: int,
    file_paths: list[str] | None,
) -> list[dict]:
    """Store files as attachments and return metadata for the message."""
    if not file_paths:
        return []

    from council.formatting import resolve_file
    stored = []
    for fp in file_paths:
        resolved = resolve_file(fp)
        att = store_attachment(
            db, conv_id,
            source_path=resolved["path"],
            name=resolved["name"],
            mime_type=resolved["mime_type"],
        )
        stored.append(att)
    return stored


# ---------------------------------------------------------------------------
# 3-Stage Protocol
# ---------------------------------------------------------------------------

async def _stage_proposals(
    providers: list[Provider],
    messages: list[dict],
    system: str | None,
    db: sqlite3.Connection,
    conv_id: int,
    chair: Chair | None,
    sink: EventSink | None,
) -> list[ModelResponse]:
    """Stage 1: All providers respond in parallel — independent proposals with tool access."""
    _emit(sink, UiEvent(type="stage_start", stage="proposals"))

    # Append structure guidance to system prompt for proposals
    proposal_system = (system or DEFAULT_SYSTEM_PROMPT) + _PROPOSAL_STRUCTURE_PROMPT

    results: list[ModelResponse] = []

    async def _generate_one(provider: Provider) -> ModelResponse | None:
        _emit(sink, UiEvent(type="generation_start", participant=provider.name))
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                _run_model_turn(
                    provider, messages, PROPOSAL_TOOLS, proposal_system,
                    db, conv_id, chair=chair, sink=sink,
                ),
                timeout=_PROVIDER_TIMEOUT,
            )
            elapsed = time.monotonic() - t0
            _emit(sink, UiEvent(type="generation_end", participant=provider.name))
            # Extract text from either ModelResponse or ToolResult
            text = ""
            if isinstance(result, ModelResponse):
                text = result.text
                resp = result
            elif isinstance(result, ToolResult) and result.terminal_payload:
                text = result.terminal_payload
                resp = ModelResponse(
                    text=text, participant=provider.name,
                    model_id=provider.model_id,
                )
            else:
                return None
            _emit(sink, UiEvent(
                type="response", participant=provider.name,
                text=text, stage="proposals", elapsed=elapsed,
            ))
            results.append(resp)
            return resp
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            _emit(sink, UiEvent(type="generation_end", participant=provider.name))
            _emit(sink, UiEvent(
                type="error", participant=provider.name,
                text=f"{provider.name} timed out ({elapsed:.1f}s)",
            ))
            log.error("Provider %s timed out after %.1fs", provider.name, elapsed)
            return None
        except Exception as exc:
            elapsed = time.monotonic() - t0
            _emit(sink, UiEvent(type="generation_end", participant=provider.name))
            _emit(sink, UiEvent(
                type="error", participant=provider.name,
                text=f"{provider.name} failed ({elapsed:.1f}s): {exc}",
            ))
            log.error("Provider %s failed after %.1fs: %s", provider.name, elapsed, exc)
            return None

    await asyncio.gather(*[_generate_one(p) for p in providers])

    _emit(sink, UiEvent(type="stage_end", stage="proposals"))
    return results


async def _stage_critique(
    providers: list[Provider],
    messages: list[dict],
    proposals: list[ModelResponse],
    system: str | None,
    db: sqlite3.Connection,
    conv_id: int,
    chair: Chair | None,
    sink: EventSink | None,
) -> list[tuple[Provider, ToolResult | ModelResponse]]:
    """Stage 2: All providers critique peer proposals in parallel."""
    _emit(sink, UiEvent(type="stage_start", stage="critique"))

    results: list[tuple[Provider, ToolResult | ModelResponse]] = []

    async def _critique_one(provider: Provider) -> None:
        # Build per-model proposal text excluding the current provider's own proposal
        peer_proposals = [r for r in proposals if r.participant != provider.name]
        proposal_text = "\n\n".join(
            f"[{r.participant}]: {r.text}" for r in peer_proposals
        )
        # Build critique prompt: base system + critique instructions with proposals
        critique_injection = _CRITIQUE_PROMPT.format(proposals=proposal_text)
        effective_system = (system or DEFAULT_SYSTEM_PROMPT) + "\n\n" + critique_injection

        _emit(sink, UiEvent(type="generation_start", participant=provider.name))
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                _run_model_turn(
                    provider, messages, CRITIQUE_TOOLS,
                    effective_system, db, conv_id,
                    chair=chair, sink=sink,
                ),
                timeout=_PROVIDER_TIMEOUT,
            )
            elapsed = time.monotonic() - t0
            _emit(sink, UiEvent(type="generation_end", participant=provider.name))

            if isinstance(result, ToolResult) and result.is_terminal:
                if result.terminal_action == "pass":
                    _emit(sink, UiEvent(
                        type="pass", participant=provider.name,
                        text=result.terminal_payload or None,
                        stage="critique", elapsed=elapsed,
                    ))
                elif result.terminal_action == "contribute":
                    _emit(sink, UiEvent(
                        type="critique", participant=provider.name,
                        text=result.terminal_payload,
                        kind=result.critique_kind,
                        target=result.critique_target,
                        stage="critique", elapsed=elapsed,
                    ))
                elif result.terminal_action == "operator_request":
                    _emit(sink, UiEvent(
                        type="operator_request", participant=provider.name,
                        text=result.terminal_payload,
                        stage="critique", elapsed=elapsed,
                    ))
            elif isinstance(result, ModelResponse):
                # Model responded with text without using tools
                text = (result.text or "").strip()
                if len(text) > 20 and not text.startswith("("):
                    # Substantive text — treat as unstructured critique
                    log.debug(
                        "%s responded with text instead of tools in critique; "
                        "treating as refinement", provider.name,
                    )
                    _emit(sink, UiEvent(
                        type="critique", participant=provider.name,
                        text=text, kind="refinement",
                        stage="critique", elapsed=elapsed,
                    ))
                else:
                    # Too short, meta-commentary, or empty — treat as pass
                    log.debug(
                        "%s responded with non-substantive text in critique; "
                        "treating as pass: %r", provider.name, text[:80],
                    )
                    _emit(sink, UiEvent(
                        type="pass", participant=provider.name,
                        stage="critique", elapsed=elapsed,
                    ))

            results.append((provider, result))

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            _emit(sink, UiEvent(type="generation_end", participant=provider.name))
            _emit(sink, UiEvent(
                type="error", participant=provider.name,
                text=f"{provider.name} timed out in critique ({elapsed:.1f}s)",
            ))
            log.warning("Provider %s timed out in critique after %.1fs", provider.name, elapsed)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            _emit(sink, UiEvent(type="generation_end", participant=provider.name))
            _emit(sink, UiEvent(
                type="error", participant=provider.name,
                text=f"{provider.name} failed in critique ({elapsed:.1f}s): {exc}",
            ))
            log.warning("Provider %s failed in critique after %.1fs: %s", provider.name, elapsed, exc)

    await asyncio.gather(*[_critique_one(p) for p in providers])

    _emit(sink, UiEvent(type="stage_end", stage="critique"))
    return results


def _build_resolution_input(
    user_prompt: str,
    proposals: list[ModelResponse],
    critique_results: list[tuple[Provider, ToolResult | ModelResponse]],
    interactive: bool = False,
) -> str:
    """Build a structured decision record for the resolver — not raw chat history."""
    parts = [f"## User Request\n{user_prompt}"]
    parts.append(f"## Execution Mode\n{'interactive' if interactive else 'non_interactive'}")

    # Proposals (full text preserved)
    proposal_lines = []
    for r in proposals:
        proposal_lines.append(f"### {r.participant}\n{r.text}")
    parts.append("## Positions\n" + "\n\n".join(proposal_lines))

    # Critiques
    critique_lines = []
    for provider, result in critique_results:
        if isinstance(result, ToolResult) and result.is_terminal:
            if result.terminal_action == "pass":
                critique_lines.append(f"- **{provider.name}**: passed (no material objections)")
            elif result.terminal_action == "contribute" and result.terminal_payload:
                kind = result.critique_kind or "comment"
                target = f" (re: {result.critique_target})" if result.critique_target else ""
                critique_lines.append(f"- **{provider.name}** [{kind}]{target}: {result.terminal_payload}")
        elif isinstance(result, ModelResponse) and result.text:
            critique_lines.append(f"- **{provider.name}**: {result.text}")

    if critique_lines:
        parts.append("## Critiques\n" + "\n".join(critique_lines))
    else:
        parts.append("## Critiques\nNo material critiques were raised.")

    return "\n\n".join(parts)


def _parse_resolution(raw_text: str) -> dict | None:
    """Parse resolver JSON output. Tolerates markdown fences and minor formatting issues."""
    text = raw_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    log.warning("Failed to parse resolution JSON: %s", text[:200])
    return None


async def _stage_resolution(
    resolver: Provider,
    user_prompt: str,
    proposals: list[ModelResponse],
    critique_results: list[tuple[Provider, ToolResult | ModelResponse]],
    interactive: bool,
    sink: EventSink | None,
) -> dict | None:
    """Stage 3: Adaptive resolution — classifies disagreement and produces typed output."""
    _emit(sink, UiEvent(type="stage_start", stage="resolution", participant=resolver.name))

    # Build structured decision record
    decision_record = _build_resolution_input(
        user_prompt, proposals, critique_results, interactive,
    )

    # Fresh context — only the decision record as a single user message
    resolution_messages = [{
        "participant": "user",
        "content": decision_record,
        "created_at": _now(),
        "attachments": None,
    }]

    _emit(sink, UiEvent(type="generation_start", participant=resolver.name))
    t0 = time.monotonic()
    try:
        resp = await asyncio.wait_for(
            resolver.generate(resolution_messages, system_instruction=_RESOLUTION_SYSTEM),
            timeout=_PROVIDER_TIMEOUT,
        )
        elapsed = time.monotonic() - t0
        _emit(sink, UiEvent(type="generation_end", participant=resolver.name))

        # Parse the JSON resolution
        resolution = _parse_resolution(resp.text)
        if resolution is None:
            # Fallback: treat as plain text recommendation
            log.warning("Resolver did not return valid JSON, falling back to plain text")
            resolution = {
                "resolution_type": "RECOMMENDATION",
                "markdown": resp.text,
                "recommendation": {
                    "title": "Council Resolution",
                    "summary": resp.text[:200],
                    "why": [],
                    "risks": [],
                    "next_steps": [],
                },
            }

        resolution_type = resolution.get("resolution_type", "RECOMMENDATION")
        markdown = resolution.get("markdown", resp.text)

        _emit(sink, UiEvent(
            type="response", participant=resolver.name,
            text=markdown, stage="resolution", elapsed=elapsed,
            metadata={"resolution_type": resolution_type},
        ))

        return resolution

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        _emit(sink, UiEvent(type="generation_end", participant=resolver.name))
        _emit(sink, UiEvent(
            type="error", participant=resolver.name,
            text=f"Resolver timed out ({elapsed:.1f}s)",
        ))
        log.error("Resolver %s timed out after %.1fs", resolver.name, elapsed)
        return None
    except Exception as exc:
        elapsed = time.monotonic() - t0
        _emit(sink, UiEvent(type="generation_end", participant=resolver.name))
        _emit(sink, UiEvent(
            type="error", participant=resolver.name,
            text=f"Resolver failed ({elapsed:.1f}s): {exc}",
        ))
        log.error("Resolver %s failed after %.1fs: %s", resolver.name, elapsed, exc)
        return None


_LEAD_FOLLOWUP_PROMPT = (
    "You are the lead model for this council. The council has already deliberated "
    "and produced a synthesis. The user is now asking a follow-up question.\n\n"
    "If this follow-up is a request for elaboration, formatting, examples, or "
    "continuation of the existing answer, respond directly and concisely.\n\n"
    "If the follow-up fundamentally changes the premise, introduces new constraints, "
    "or requires genuinely diverse perspectives, invoke call_council(reason=...) to "
    "reconvene the full council instead of answering alone.\n\n"
    "You have the full conversation history including all proposals, critiques, and synthesis."
)


async def run_lead_followup(
    lead: Provider,
    prompt: str,
    db: sqlite3.Connection,
    conv_id: int,
    system: str | None,
    sink: EventSink | None = None,
    chair: Chair | None = None,
) -> str | None:
    """Run a lead-model follow-up. Returns 'reconvene' if the lead triggers call_council, else None."""
    from council.tools import LEAD_FOLLOWUP_TOOLS

    add_message(db, conv_id, "user", prompt, metadata={"stage": "followup"})
    messages = get_messages(db, conv_id)

    effective_system = (system or DEFAULT_SYSTEM_PROMPT) + "\n\n" + _LEAD_FOLLOWUP_PROMPT

    _emit(sink, UiEvent(type="generation_start", participant=lead.name))
    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(
            _run_model_turn(lead, messages, LEAD_FOLLOWUP_TOOLS, effective_system, db, conv_id, chair=chair, sink=sink),
            timeout=_PROVIDER_TIMEOUT,
        )
        elapsed = time.monotonic() - t0
        _emit(sink, UiEvent(type="generation_end", participant=lead.name))

        if isinstance(result, ToolResult) and result.is_terminal:
            if result.terminal_action == "reconvene":
                _emit(sink, UiEvent(
                    type="status",
                    text=f"{lead.name} is reconvening the council: {result.terminal_payload}",
                ))
                return "reconvene"

        # Regular text response
        text = ""
        if isinstance(result, ToolResult) and result.terminal_payload:
            text = result.terminal_payload
        elif isinstance(result, ModelResponse):
            text = result.text

        if text:
            _emit(sink, UiEvent(
                type="response", participant=lead.name,
                text=text, stage="followup", elapsed=elapsed,
            ))
            add_message(db, conv_id, lead.name, text,
                        metadata={"stage": "followup", "action": "CONTRIBUTE",
                                  "model_id": lead.model_id})
        return None

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        _emit(sink, UiEvent(type="generation_end", participant=lead.name))
        _emit(sink, UiEvent(type="error", text=f"{lead.name} timed out ({elapsed:.1f}s)"))
        return None
    except Exception as exc:
        elapsed = time.monotonic() - t0
        _emit(sink, UiEvent(type="generation_end", participant=lead.name))
        _emit(sink, UiEvent(type="error", text=f"{lead.name} failed ({elapsed:.1f}s): {exc}"))
        return None


# ---------------------------------------------------------------------------
# Orchestrator — main entry point
# ---------------------------------------------------------------------------

async def run_discuss(
    db: sqlite3.Connection,
    prompt: str,
    providers: list[Provider],
    session_id: str | None = None,
    system: str | None = None,
    files: list[str] | None = None,
    conversation_name: str | None = None,
    sink: EventSink | None = None,
    interactive: bool = False,
    chair: Chair | None = None,
    # Legacy parameter — ignored, kept for backward compat
    max_rounds: int = 10,
) -> None:
    """Run a bounded 3-stage council discussion.

    Stage 1 (Proposals):  All providers respond in parallel.
    Stage 2 (Critique):   All providers critique peers in parallel (PASS/CONTRIBUTE tools).
    Stage 3 (Synthesis):  Lead provider (first in list) synthesizes the discussion.

    Single-model case: skips critique and synthesis — behaves like a simple ask.
    """
    name = conversation_name or f"discuss-{_now().replace(' ', '-').replace(':', '')}"
    conv_id = get_or_create_conversation(db, name, session_id, system)
    provider_names = [p.name for p in providers]

    # Check if this is a continuation
    existing_messages = get_messages(db, conv_id)
    is_continuation = len(existing_messages) > 0

    _emit(sink, UiEvent(
        type="status",
        text=f"Council: {name} | participants: {', '.join(provider_names)}",
    ))

    # Store file attachments
    stored_atts = _store_files_as_attachments(db, conv_id, files)
    if stored_atts:
        file_note = " [attached: " + ", ".join(a["name"] for a in stored_atts) + "]"
        legacy_atts = [{"path": a["file_path"], "name": a["name"], "mime_type": a["mime_type"]}
                       for a in stored_atts]
    else:
        file_note = ""
        legacy_atts = None

    # Store user prompt
    add_message(db, conv_id, "user", prompt + file_note, legacy_atts,
                metadata={"stage": "prompt"})

    # -------------------------------------------------------------------
    # Stage 1: Proposals (parallel)
    # -------------------------------------------------------------------
    messages = get_messages(db, conv_id)
    # Create chair if not provided
    if chair is None:
        chair = Chair(Path.cwd(), interactive=interactive)

    proposals = await _stage_proposals(providers, messages, system, db, conv_id, chair, sink)

    if not proposals:
        _emit(sink, UiEvent(type="error", text="No providers responded. Aborting."))
        return

    # Persist proposals
    for resp in proposals:
        add_message(db, conv_id, resp.participant, resp.text,
                    metadata={"stage": "proposals", "action": "CONTRIBUTE",
                              "model_id": resp.model_id})

    # Single-model case: no critique or synthesis needed
    if len(providers) <= 1:
        _emit(sink, UiEvent(type="status", text=f"Discussion saved as '{name}'."))
        return

    # -------------------------------------------------------------------
    # Pause point after proposals (interactive only)
    # -------------------------------------------------------------------
    if interactive and sys.stdout.isatty():
        if not await _countdown_pause(sink, "critique", 5):
            _emit(sink, UiEvent(type="status", text=f"Ended after proposals. Saved as '{name}'."))
            return

    # -------------------------------------------------------------------
    # Stage 2: Critique (parallel)
    # -------------------------------------------------------------------
    messages = get_messages(db, conv_id)
    critique_results = await _stage_critique(
        providers, messages, proposals, system, db, conv_id, chair, sink,
    )

    # Persist critiques
    for provider, result in critique_results:
        if isinstance(result, ToolResult) and result.is_terminal:
            if result.terminal_action == "contribute" and result.terminal_payload:
                add_message(
                    db, conv_id, provider.name, result.terminal_payload,
                    metadata={
                        "stage": "critique", "action": "CONTRIBUTE",
                        "kind": result.critique_kind,
                        "target": result.critique_target,
                        "model_id": provider.model_id,
                    },
                )
            elif result.terminal_action == "pass":
                add_message(
                    db, conv_id, provider.name, "(passed)",
                    metadata={
                        "stage": "critique", "action": "PASS",
                        "note": result.terminal_payload or "",
                    },
                )
            elif result.terminal_action == "operator_request":
                add_message(
                    db, conv_id, provider.name,
                    f"[OPERATOR REQUEST]: {result.terminal_payload}",
                    metadata={
                        "stage": "critique", "action": "OPERATOR_REQUEST",
                    },
                )
        elif isinstance(result, ModelResponse) and result.text:
            add_message(
                db, conv_id, provider.name, result.text,
                metadata={
                    "stage": "critique", "action": "CONTRIBUTE",
                    "kind": "refinement",
                    "model_id": provider.model_id,
                },
            )

    # -------------------------------------------------------------------
    # Pause point after critique (interactive only)
    # -------------------------------------------------------------------
    if interactive and sys.stdout.isatty():
        if not await _countdown_pause(sink, "resolution", 5):
            _emit(sink, UiEvent(type="status", text=f"Ended after critique. Saved as '{name}'."))
            return

    # -------------------------------------------------------------------
    # Stage 3: Resolution (adaptive — replaces forced synthesis)
    # -------------------------------------------------------------------
    resolver = providers[0]  # first active model is the resolver
    resolution = await _stage_resolution(
        resolver, prompt, proposals, critique_results, interactive, sink,
    )

    if resolution:
        resolution_type = resolution.get("resolution_type", "RECOMMENDATION")
        markdown = resolution.get("markdown", "")

        # Persist resolution
        add_message(db, conv_id, resolver.name, markdown,
                    metadata={"stage": "resolution", "action": "CONTRIBUTE",
                              "resolution_type": resolution_type,
                              "model_id": resolver.model_id})

        # Route based on resolution type
        if resolution_type == "QUESTION" and interactive and sys.stdout.isatty():
            # Ask the question and feed answer back
            question = resolution.get("question", {})
            q_text = question.get("text", "") if isinstance(question, dict) else str(question)
            if q_text:
                try:
                    answer = input(f"\n  Council asks: {q_text}\n  Your answer: ").strip()
                    if answer:
                        _emit(sink, UiEvent(type="status", text=f"Received: {answer}"))
                        add_message(db, conv_id, "user", answer,
                                    metadata={"stage": "resolution_answer"})
                        # Re-run resolution with the answer
                        _emit(sink, UiEvent(type="status", text="Re-resolving with your input..."))
                        messages = get_messages(db, conv_id)
                        re_resolution = await _stage_resolution(
                            resolver, f"{prompt}\n\nUser clarification: {answer}",
                            proposals, critique_results, interactive, sink,
                        )
                        if re_resolution:
                            re_markdown = re_resolution.get("markdown", "")
                            re_type = re_resolution.get("resolution_type", "RECOMMENDATION")
                            add_message(db, conv_id, resolver.name, re_markdown,
                                        metadata={"stage": "resolution", "action": "CONTRIBUTE",
                                                  "resolution_type": re_type,
                                                  "model_id": resolver.model_id})
                except (EOFError, KeyboardInterrupt):
                    pass

        elif resolution_type == "INVESTIGATE" and chair:
            # Route investigation to chair
            investigation = resolution.get("investigation", {})
            items = investigation.get("items", []) if isinstance(investigation, dict) else []
            if items:
                _emit(sink, UiEvent(type="status", text="Investigating..."))
                evidence = []
                for item in items[:3]:  # max 3 investigation items
                    result = chair.fulfill_request(item, resolver.name, sink)
                    evidence.append(f"- {item}: {result}")
                evidence_text = "\n".join(evidence)
                add_message(db, conv_id, "chair", evidence_text,
                            metadata={"stage": "investigation"})
                # Re-run resolution with evidence
                _emit(sink, UiEvent(type="status", text="Re-resolving with evidence..."))
                re_resolution = await _stage_resolution(
                    resolver, f"{prompt}\n\nInvestigation results:\n{evidence_text}",
                    proposals, critique_results, interactive, sink,
                )
                if re_resolution:
                    re_markdown = re_resolution.get("markdown", "")
                    re_type = re_resolution.get("resolution_type", "RECOMMENDATION")
                    add_message(db, conv_id, resolver.name, re_markdown,
                                metadata={"stage": "resolution", "action": "CONTRIBUTE",
                                          "resolution_type": re_type,
                                          "model_id": resolver.model_id})

    _emit(sink, UiEvent(type="status", text=f"Discussion saved as '{name}'."))
