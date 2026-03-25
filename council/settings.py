"""Settings — config file + env var based configuration."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".council"
CONFIG_PATH = CONFIG_DIR / "config.toml"

# Template for council init
CONFIG_TEMPLATE = """\
# Council Engine configuration
# https://councilengine.dev
#
# Uncomment and fill in API keys for the providers you want to use.
# You need at least one provider configured. For multi-model deliberation,
# configure two or more.
#
# You can also set these as environment variables (they override this file):
#   COUNCIL_GEMINI_API_KEY, COUNCIL_OPENAI_API_KEY,
#   COUNCIL_CLAUDE_API_KEY, COUNCIL_OLLAMA_HOST

# Optional: override the default system prompt for all models.
# Use {model_name} as a placeholder — it will be replaced with the model's name.
# Stage-specific prompts (proposal structure, critique rules, resolution types)
# are part of the protocol and not configurable here.
# system_prompt = "You are {model_name}, a direct and opinionated advisor..."

[providers.gemini]
# Get an API key from Google AI Studio: https://aistudio.google.com/apikey
# Or from Vertex AI: https://console.cloud.google.com/apis/credentials
# api_key = "your-google-api-key"
# model = "gemini-3.1-pro-preview"
#
# To use Vertex AI instead of AI Studio, uncomment these:
# vertex_ai = true
# project = "your-gcp-project"
# location = "global"

[providers.chatgpt]
# Get an API key from: https://platform.openai.com/api-keys
# api_key = "your-openai-api-key"
# model = "gpt-5.4"

[providers.claude]
# Get an API key from: https://console.anthropic.com/settings/keys
# api_key = "your-anthropic-api-key"
# model = "claude-sonnet-4-6"

[providers.ollama]
# Runs models locally — no API key needed.
# Install: brew install ollama (macOS) or https://ollama.com/download
# Browse models: https://ollama.com/library
# Then: ollama serve && ollama pull <model_name>
# host = "http://localhost:11434"
# model = "llama3.1:8b"
"""


@dataclass
class GeminiConfig:
    api_key: str = ""
    model: str = "gemini-3.1-pro-preview"
    vertex_ai: bool = False
    project: str = ""
    location: str = "global"


@dataclass
class ChatGPTConfig:
    api_key: str = ""
    model: str = "gpt-5.4"


@dataclass
class ClaudeConfig:
    api_key: str = ""
    model: str = "claude-sonnet-4-6"


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"


@dataclass
class Settings:
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    chatgpt: ChatGPTConfig = field(default_factory=ChatGPTConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    system_prompt: str = ""


_settings: Settings | None = None


def _load_config_file() -> dict:
    """Load config.toml if it exists."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        return tomllib.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def _apply_config(settings: Settings, raw: dict) -> None:
    """Apply config file values to settings."""
    providers = raw.get("providers", {})

    g = providers.get("gemini", {})
    if g.get("api_key"):
        settings.gemini.api_key = g["api_key"]
    if g.get("model"):
        settings.gemini.model = g["model"]
    if g.get("vertex_ai"):
        settings.gemini.vertex_ai = True
    if g.get("project"):
        settings.gemini.project = g["project"]
    if g.get("location"):
        settings.gemini.location = g["location"]

    c = providers.get("chatgpt", {})
    if c.get("api_key"):
        settings.chatgpt.api_key = c["api_key"]
    if c.get("model"):
        settings.chatgpt.model = c["model"]

    cl = providers.get("claude", {})
    if cl.get("api_key"):
        settings.claude.api_key = cl["api_key"]
    if cl.get("model"):
        settings.claude.model = cl["model"]

    o = providers.get("ollama", {})
    if o.get("host"):
        settings.ollama.host = o["host"]
    if o.get("model"):
        settings.ollama.model = o["model"]

    if raw.get("system_prompt"):
        settings.system_prompt = raw["system_prompt"]


def _apply_env_overrides(settings: Settings) -> None:
    """Env vars override config file. Supports both new and legacy env var names."""
    # Gemini
    if v := os.environ.get("COUNCIL_GEMINI_API_KEY"):
        settings.gemini.api_key = v
    if v := os.environ.get("GEMINI_MODEL"):
        settings.gemini.model = v
    if v := os.environ.get("GOOGLE_CLOUD_PROJECT"):
        settings.gemini.project = v
        if not settings.gemini.api_key:
            settings.gemini.vertex_ai = True
    if v := os.environ.get("GOOGLE_CLOUD_LOCATION"):
        settings.gemini.location = v

    # ChatGPT
    if v := os.environ.get("COUNCIL_OPENAI_API_KEY"):
        settings.chatgpt.api_key = v
    if v := os.environ.get("OPENAI_MODEL"):
        settings.chatgpt.model = v

    # Claude
    if v := os.environ.get("COUNCIL_CLAUDE_API_KEY"):
        settings.claude.api_key = v
    if v := os.environ.get("CLAUDE_MODEL"):
        settings.claude.model = v

    # Ollama
    if v := os.environ.get("COUNCIL_OLLAMA_HOST"):
        settings.ollama.host = v
    if v := os.environ.get("COUNCIL_OLLAMA_MODEL"):
        settings.ollama.model = v

    # System prompt
    if v := os.environ.get("COUNCIL_SYSTEM_PROMPT"):
        settings.system_prompt = v


def get_settings() -> Settings:
    """Return the singleton Settings, loading on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
        raw = _load_config_file()
        _apply_config(_settings, raw)
        _apply_env_overrides(_settings)
    return _settings


def write_config(settings: Settings) -> Path:
    """Write a config file from current settings. Returns the path written."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["# Council Engine configuration\n"]

    if settings.gemini.api_key or settings.gemini.vertex_ai:
        lines.append("[providers.gemini]")
        if settings.gemini.api_key:
            lines.append(f'api_key = "{settings.gemini.api_key}"')
        lines.append(f'model = "{settings.gemini.model}"')
        if settings.gemini.vertex_ai:
            lines.append("vertex_ai = true")
            if settings.gemini.project:
                lines.append(f'project = "{settings.gemini.project}"')
            lines.append(f'location = "{settings.gemini.location}"')
        lines.append("")

    if settings.chatgpt.api_key:
        lines.append("[providers.chatgpt]")
        lines.append(f'api_key = "{settings.chatgpt.api_key}"')
        lines.append(f'model = "{settings.chatgpt.model}"')
        lines.append("")

    if settings.claude.api_key:
        lines.append("[providers.claude]")
        lines.append(f'api_key = "{settings.claude.api_key}"')
        lines.append(f'model = "{settings.claude.model}"')
        lines.append("")

    if settings.ollama.host != "http://localhost:11434" or settings.ollama.model != "llama3.1:8b":
        lines.append("[providers.ollama]")
        lines.append(f'host = "{settings.ollama.host}"')
        lines.append(f'model = "{settings.ollama.model}"')
        lines.append("")

    CONFIG_PATH.write_text("\n".join(lines))
    return CONFIG_PATH
