"""Simple provider configuration for the CLI."""

from __future__ import annotations

import os
from pathlib import Path

from photomatagent.model_settings import ModelRole

from photomatagent.models.anthropic import AnthropicProvider
from photomatagent.models.base import ModelProvider
from photomatagent.models.fake import FakeModelProvider
from photomatagent.models.openai import OpenAIProvider


def create_provider(
    provider: str | None = None, model: str | None = None, *,
    role: ModelRole = "main", workspace: Path | str | None = None,
) -> ModelProvider:
    from photomatagent.model_settings import selection, validate_isolation, values

    data = values(workspace)
    normalized, selected, key, base_url = selection(data, role, provider, model)
    if normalized == "fake":
        return FakeModelProvider(auto=True)
    if role != "main":
        validate_isolation(data, role, normalized, selected, key)
    if not selected:
        raise ValueError(f"{role} provider requires a model")
    if not key:
        raise ValueError(f"{role} provider requires an API key")
    if normalized == "openai":
        return OpenAIProvider(selected, api_key=key or None, base_url=base_url or "https://api.openai.com/v1")
    if normalized == "anthropic":
        return AnthropicProvider(selected, api_key=key or None, base_url=base_url or "https://api.anthropic.com")
    raise ValueError(f"unknown provider: {normalized}")


def api_key_status(provider: str) -> str:
    if provider == "openai":
        return "configured" if os.getenv("OPENAI_API_KEY") else "missing"
    if provider == "anthropic":
        return "configured" if os.getenv("ANTHROPIC_API_KEY") else "missing"
    return "not required"
