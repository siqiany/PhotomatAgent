"""Role-specific model settings; secrets never enter public settings responses."""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from photomatagent.config import DotEnvConfig, normalize_provider

ModelRole = Literal["main", "judge", "feedback"]
ROLES: tuple[ModelRole, ...] = ("main", "judge", "feedback")
_LOCK = threading.RLock()


class ModelSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["openai", "anthropic", "fake"]
    model: str = Field(min_length=1, max_length=200)
    api_key: SecretStr | None = None
    base_url: str | None = Field(default=None, max_length=2000)


def fields(role: ModelRole, provider: str) -> tuple[str, str, str, str]:
    if role == "main":
        return ("PHOTOMATAGENT_PROVIDER", f"{provider.upper()}_MODEL",
                f"{provider.upper()}_API_KEY", f"{provider.upper()}_BASE_URL")
    prefix = f"PHOTOMATAGENT_{role.upper()}"
    return f"{prefix}_PROVIDER", f"{prefix}_MODEL", f"{prefix}_API_KEY", f"{prefix}_BASE_URL"


def values(workspace: Path | str | None = None) -> dict[str, str]:
    keys = {key for role in ROLES for provider in ("openai", "anthropic") for key in fields(role, provider)}
    result = {k: os.environ[k] for k in keys if os.environ.get(k)}
    if workspace is not None:
        path = DotEnvConfig(workspace).path
        if path.is_symlink():
            raise ValueError(".env must not be a symbolic link")
        for name, value in dotenv_values(path, interpolate=False).items():
            # Empty template credentials still permit explicit environment setup;
            # a saved empty URL intentionally selects the official service.
            if value or name.endswith("_BASE_URL") or name not in result:
                result[name] = value or ""
    return result


def selection(data: dict[str, str], role: ModelRole, provider: str | None = None,
              model: str | None = None) -> tuple[str, str, str, str | None]:
    provider_key = fields(role, "openai")[0]
    selected_value = provider or data.get(provider_key) or ("fake" if role == "main" else "")
    if not selected_value:
        raise ValueError(f"{role} is not configured; run configure --role {role}")
    selected = normalize_provider(selected_value)
    _, model_key, key_key, url_key = fields(role, selected)
    return selected, (model or data.get(model_key, "")).strip(), data.get(key_key, "").strip(), data.get(url_key, "").strip() or None


def validate_isolation(data: dict[str, str], role: ModelRole, provider: str,
                       model: str, key: str) -> None:
    if provider == "fake":
        return
    if not model.strip() or not key.strip():
        raise ValueError(f"{role} requires its own model and API key; configure --role {role}")
    for other in ROLES:
        if other == role:
            continue
        for other_provider in ("openai", "anthropic"):
            if key == data.get(fields(other, other_provider)[2], "").strip():
                raise ValueError(f"{role} and {other} must use different API keys")


def public_settings(workspace: Path | str) -> dict[str, dict[str, object]]:
    data = values(workspace)
    result: dict[str, dict[str, object]] = {}
    for role in ROLES:
        provider_key = fields(role, "openai")[0]
        provider = data.get(provider_key) or ""
        _, model_key, key_key, url_key = fields(role, provider)
        result[role] = {"provider": provider, "model": data.get(model_key, ""),
                        "base_url": data.get(url_key, ""), "key_configured": bool(data.get(key_key)),
                        "configured": bool(provider and (provider == "fake" or (data.get(model_key) and data.get(key_key))))}
    return result


def save_settings(workspace: Path | str, role: ModelRole, update: ModelSettingsUpdate) -> None:
    """Atomically preserve unrelated dotenv entries and protect the resulting file."""
    if role not in ROLES:
        raise ValueError("unknown model role")
    model = update.model.strip()
    key = update.api_key.get_secret_value().strip() if update.api_key is not None else None
    url = update.base_url.strip() if update.base_url is not None else None
    if not model or any("\n" in v or "\r" in v or "\x00" in v for v in (model, key or "", url or "")):
        raise ValueError("model settings must be nonempty single-line values")
    if url:
        from urllib.parse import urlsplit
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
    store = DotEnvConfig(workspace)
    with _LOCK:
        data = values(workspace)
        provider_key, model_key, key_key, url_key = fields(role, update.provider)
        provider_changed = bool(data.get(provider_key) and data[provider_key] != update.provider)
        if role != "main" and provider_changed and key is None and update.provider != "fake":
            raise ValueError("changing an auxiliary provider requires a new API key")
        changes = {provider_key: update.provider, model_key: model}
        if key is not None:
            changes[key_key] = key
        if url is not None:
            changes[url_key] = url
        elif provider_changed:
            changes[url_key] = ""
        candidate = data | changes
        validate_isolation(candidate, role, update.provider, model, candidate.get(key_key, ""))
        text = store.path.read_text(encoding="utf-8") if store.path.exists() else "# PhotomatAgent model settings\n"
        descriptor, temporary = tempfile.mkstemp(prefix=".model-settings-", dir=store.workspace)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(text)
            from dotenv import set_key
            for name, value in changes.items():
                set_key(temporary, name, value, quote_mode="always")
            os.chmod(temporary, 0o600)
            os.replace(temporary, store.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
