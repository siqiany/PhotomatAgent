"""Workspace-local LLM configuration backed by a protected ``.env`` file."""

from __future__ import annotations

import os
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, set_key

PREFERRED_PROVIDER_ENV = "PHOTOMATAGENT_PROVIDER"
SUPPORTED_PROVIDERS = ("openai", "anthropic")
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

_PROVIDER_ALIASES = {
    "openai": "openai",
    "open_ai": "openai",
    "anthropic": "anthropic",
    "an": "anthropic",
    "claude": "anthropic",
}

_PROVIDER_FIELDS = {
    "openai": ("OPENAI_MODEL", "OPENAI_API_KEY", OPENAI_BASE_URL_ENV),
    "anthropic": ("ANTHROPIC_MODEL", "ANTHROPIC_API_KEY"),
}

_ENV_TEMPLATE = """# PhotomatAgent LLM configuration
# PHOTOMATAGENT_PROVIDER: openai or anthropic
PHOTOMATAGENT_PROVIDER=

# OpenAI official SDK
OPENAI_MODEL=
OPENAI_BASE_URL=
OPENAI_API_KEY=

# Anthropic official SDK
ANTHROPIC_MODEL=
ANTHROPIC_API_KEY=
"""


@dataclass(frozen=True)
class LLMConfig:
    """A complete provider selection ready for model construction."""

    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None


PromptValue = Callable[[str, bool, str | None], str]


def normalize_provider(value: str) -> str:
    """Return the canonical provider name or raise a user-facing error."""
    normalized = value.strip().lower()
    if normalized == "fake":
        return normalized
    try:
        return _PROVIDER_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError("provider must be openai or anthropic") from exc


class DotEnvConfig:
    """Create, read, and update a workspace ``.env`` without exposing secrets."""

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.path = self.workspace / ".env"

    def ensure_exists(self) -> bool:
        """Create a restrictive template if absent; return whether it was created."""
        if self.path.exists():
            if not self.path.is_file():
                raise ValueError(f"configuration path is not a file: {self.path}")
            return False
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, _ENV_TEMPLATE.encode("utf-8"))
        finally:
            os.close(descriptor)
        return True

    def values(
        self, environ: MutableMapping[str, str] | None = None
    ) -> dict[str, str]:
        """Read ``.env`` and overlay non-empty process environment values."""
        raw = dotenv_values(self.path)
        values = {key: value or "" for key, value in raw.items()}
        source = os.environ if environ is None else environ
        for key in (PREFERRED_PROVIDER_ENV, *self.all_provider_fields()):
            if source.get(key):
                values[key] = source[key]
        return values

    def set(self, key: str, value: str) -> None:
        """Persist one value and retain owner-only permissions where supported."""
        set_key(self.path, key, value, quote_mode="always")
        try:
            self.path.chmod(0o600)
        except OSError:
            # Some Windows-mounted filesystems do not implement POSIX modes.
            pass

    @staticmethod
    def all_provider_fields() -> tuple[str, ...]:
        return tuple(field for fields in _PROVIDER_FIELDS.values() for field in fields)


def resolve_llm_config(
    store: DotEnvConfig,
    *,
    prompt: PromptValue,
    provider: str | None = None,
    model: str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> tuple[LLMConfig, bool]:
    """Resolve configuration, interactively filling and persisting missing fields.

    Returns the complete configuration and whether the ``.env`` file was created.
    Explicit CLI values have highest priority, followed by process environment and
    then values loaded from ``.env``.
    """
    target_environ = os.environ if environ is None else environ
    created = store.ensure_exists()
    values = store.values(target_environ)

    selected_value = provider or values.get(PREFERRED_PROVIDER_ENV, "")
    while True:
        if not selected_value:
            selected_value = prompt(
                "请选择 LLM SDK/提供商（OpenAI 或 Anthropic）", False, None
            )
        try:
            selected = normalize_provider(selected_value)
            break
        except ValueError:
            if provider:
                raise
            selected_value = ""

    if selected == "fake":
        return LLMConfig(provider="fake", model=model or "fake"), created

    if values.get(PREFERRED_PROVIDER_ENV) != selected:
        store.set(PREFERRED_PROVIDER_ENV, selected)
    target_environ[PREFERRED_PROVIDER_ENV] = selected

    model_env, api_key_env = _PROVIDER_FIELDS[selected][:2]
    selected_model = model or values.get(model_env, "").strip()
    while not selected_model:
        selected_model = prompt(f"请输入 {selected} 模型名称", False, None).strip()
    if values.get(model_env) != selected_model:
        store.set(model_env, selected_model)
    target_environ[model_env] = selected_model

    base_url: str | None = None
    if selected == "openai":
        base_url = values.get(OPENAI_BASE_URL_ENV, "").strip()
        if not base_url:
            base_url = (
                prompt(
                    "请输入 OpenAI-compatible Base URL",
                    False,
                    DEFAULT_OPENAI_BASE_URL,
                ).strip()
                or DEFAULT_OPENAI_BASE_URL
            )
        if values.get(OPENAI_BASE_URL_ENV) != base_url:
            store.set(OPENAI_BASE_URL_ENV, base_url)
        target_environ[OPENAI_BASE_URL_ENV] = base_url

    api_key = values.get(api_key_env, "").strip()
    while not api_key:
        api_key = prompt(
            f"请输入 {selected} API Key（输入内容将隐藏）", True, None
        ).strip()
    if values.get(api_key_env) != api_key:
        store.set(api_key_env, api_key)
    target_environ[api_key_env] = api_key

    return (
        LLMConfig(
            provider=selected,
            model=selected_model,
            api_key_env=api_key_env,
            base_url=base_url,
        ),
        created,
    )


def read_preferred_config(
    store: DotEnvConfig,
    *,
    provider: str | None = None,
    model: str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> LLMConfig | None:
    """Read configuration without prompting, for diagnostics."""
    store.ensure_exists()
    target_environ = os.environ if environ is None else environ
    values = store.values(target_environ)
    selected_value = provider or values.get(PREFERRED_PROVIDER_ENV, "")
    if not selected_value:
        return None
    selected = normalize_provider(selected_value)
    if selected == "fake":
        return LLMConfig(provider="fake", model=model or "fake")
    model_env, api_key_env = _PROVIDER_FIELDS[selected][:2]
    selected_model = model or values.get(model_env, "").strip()
    api_key = values.get(api_key_env, "").strip()
    base_url = (
        values.get(OPENAI_BASE_URL_ENV, "").strip()
        if selected == "openai"
        else None
    )
    if selected_model:
        target_environ[model_env] = selected_model
    if api_key:
        target_environ[api_key_env] = api_key
    if base_url:
        target_environ[OPENAI_BASE_URL_ENV] = base_url
    target_environ[PREFERRED_PROVIDER_ENV] = selected
    return LLMConfig(
        provider=selected,
        model=selected_model,
        api_key_env=api_key_env if api_key else None,
        base_url=base_url,
    )
