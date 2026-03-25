"""Provider registry — discovers and instantiates available providers."""

from __future__ import annotations

import logging
import sys

from council.providers.base import ModelResponse, Provider, StructuredResponse
from council.settings import get_settings

log = logging.getLogger("council")

__all__ = ["ModelResponse", "Provider", "StructuredResponse", "ProviderRegistry"]


class ProviderRegistry:
    """Discovers and instantiates available providers based on config."""

    _providers: dict[str, Provider] = {}

    @classmethod
    def get(cls, name: str) -> Provider:
        """Return an initialized provider by name, or exit with error."""
        if name not in cls._providers:
            cls._init_provider(name)
        return cls._providers[name]

    @classmethod
    def available(cls) -> list[str]:
        """Return names of providers that have credentials configured."""
        cfg = get_settings()
        names: list[str] = []

        # Gemini: available if API key set or Vertex AI configured
        if cfg.gemini.api_key or cfg.gemini.vertex_ai or cfg.gemini.project:
            names.append("gemini")

        # ChatGPT: available if API key set
        if cfg.chatgpt.api_key:
            names.append("chatgpt")

        # Claude: available if API key set
        if cfg.claude.api_key:
            names.append("claude")

        # Ollama: available if explicitly configured (host differs from default or model set in config)
        # We check if ollama section exists in config by checking if the settings were touched
        if _ollama_configured(cfg):
            names.append("ollama")

        return names

    @classmethod
    def _init_provider(cls, name: str) -> None:
        if name == "gemini":
            from council.providers.gemini import GeminiProvider
            cls._providers[name] = GeminiProvider()
        elif name == "chatgpt":
            from council.providers.openai import OpenAIProvider
            cls._providers[name] = OpenAIProvider()
        elif name == "claude":
            from council.providers.claude import ClaudeProvider
            cls._providers[name] = ClaudeProvider()
        elif name == "ollama":
            from council.providers.ollama import OllamaProvider
            cls._providers[name] = OllamaProvider()
        else:
            avail = cls.available()
            print(
                f"Error: Unknown provider '{name}'.\n"
                f"  Available: {', '.join(avail) if avail else 'none (run council init to configure)'}",
                file=sys.stderr,
            )
            sys.exit(1)

    @classmethod
    def reset(cls) -> None:
        """Clear cached providers (useful for testing)."""
        cls._providers.clear()


def _ollama_configured(cfg) -> bool:
    """Check if Ollama was explicitly configured (not just defaults)."""
    import tomllib
    from council.settings import CONFIG_PATH
    if not CONFIG_PATH.exists():
        # Check env vars
        import os
        return bool(os.environ.get("COUNCIL_OLLAMA_HOST") or os.environ.get("COUNCIL_OLLAMA_MODEL"))
    try:
        raw = tomllib.loads(CONFIG_PATH.read_text())
        return "ollama" in raw.get("providers", {})
    except Exception:
        return False
