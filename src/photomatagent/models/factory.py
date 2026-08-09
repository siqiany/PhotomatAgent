"""Simple provider configuration for the CLI."""

from __future__ import annotations

import os

from photomatagent.models.anthropic import AnthropicProvider
from photomatagent.models.base import ModelProvider
from photomatagent.models.fake import FakeModelProvider
from photomatagent.models.openai import OpenAIProvider


def create_provider(provider: str, model: str | None = None) -> ModelProvider:
    normalized = provider.lower()
    if normalized == "fake":
        return FakeModelProvider(auto=True)
    if normalized == "openai":
        selected = model or os.getenv("OPENAI_MODEL")
        if not selected:
            raise ValueError("OpenAI provider requires --model or OPENAI_MODEL")
        return OpenAIProvider(selected, base_url=os.getenv("OPENAI_BASE_URL") or None)
    if normalized == "anthropic":
        selected = model or os.getenv("ANTHROPIC_MODEL")
        if not selected:
            raise ValueError("Anthropic provider requires --model or ANTHROPIC_MODEL")
        return AnthropicProvider(selected)
    raise ValueError(f"unknown provider: {provider}; expected fake, openai, or anthropic")


def api_key_status(provider: str) -> str:
    if provider == "openai":
        return "configured" if os.getenv("OPENAI_API_KEY") else "missing"
    if provider == "anthropic":
        return "configured" if os.getenv("ANTHROPIC_API_KEY") else "missing"
    return "not required"
