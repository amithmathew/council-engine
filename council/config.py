"""Configuration — re-exports from settings for backward compatibility."""

from __future__ import annotations

from pathlib import Path

from council.settings import get_settings

# Database paths (unchanged)
DB_DIR = Path.home() / ".council"
DB_PATH = DB_DIR / "conversations.db"
ATTACHMENTS_DIR = DB_DIR / "attachments"


def _s():
    return get_settings()


# Lazy properties that read from settings on first access.
# These are module-level for backward compat with `from council.config import PROJECT`.
class _ConfigProxy:
    @property
    def PROJECT(self) -> str:
        return _s().gemini.project

    @property
    def LOCATION(self) -> str:
        return _s().gemini.location

    @property
    def GEMINI_MODEL(self) -> str:
        return _s().gemini.model

    @property
    def OPENAI_MODEL(self) -> str:
        return _s().chatgpt.model

    @property
    def CLAUDE_MODEL(self) -> str:
        return _s().claude.model

    @property
    def OLLAMA_MODEL(self) -> str:
        return _s().ollama.model


_proxy = _ConfigProxy()

# Re-export as module-level names. These are read at import time by existing code
# via `from council.config import PROJECT, GEMINI_MODEL, ...`
# Since settings are loaded lazily, we need a different approach: providers should
# read from get_settings() directly. These are kept for the system prompt only.

DEFAULT_SYSTEM_PROMPT = (
    "You are {model_name}, a participant in a multi-model brainstorming council. "
    "Messages are labeled with timestamps and participant names "
    "(e.g. [user], [gemini], [chatgpt]). Multiple AI models and humans may "
    "be part of the conversation. Respond as {model_name} — use first person.\n\n"
    "Be direct, thorough, and opinionated. Don't hedge "
    "or give wishy-washy answers. When you disagree with another participant, "
    "say so and explain why.\n\n"
    "You are an active participant, not just a responder. Ask follow-up "
    "questions when the problem is underspecified. Challenge vague assumptions. "
    "Suggest angles others haven't considered. Request more context when you "
    "need it. Push the conversation forward rather than just answering and stopping.\n\n"
    "Cite sources when your response draws on search results."
)


def format_system_prompt(prompt: str | None, model_name: str) -> str:
    """Format a system prompt, injecting the model's identity."""
    # Priority: explicit prompt arg > config file > default
    if not prompt:
        cfg_prompt = get_settings().system_prompt
        prompt = cfg_prompt if cfg_prompt else None
    p = prompt or DEFAULT_SYSTEM_PROMPT
    formatted = p.replace("{model_name}", model_name)
    if "{model_name}" not in (prompt or "") and model_name not in formatted[:100]:
        formatted = f"You are {model_name}. Respond as {model_name} — use first person.\n\n" + formatted
    return formatted
