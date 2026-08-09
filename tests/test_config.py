from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from typer.testing import CliRunner

import photomatagent.cli.app as cli_app
import photomatagent.config as config_module
from photomatagent.config import (
    DEFAULT_OPENAI_BASE_URL,
    DotEnvConfig,
    OPENAI_BASE_URL_ENV,
    PREFERRED_PROVIDER_ENV,
    read_preferred_config,
    resolve_llm_config,
)


def test_first_start_creates_protected_env_and_collects_openai_config(
    tmp_path: Path,
) -> None:
    answers = iter(
        ["OpenAI", "gpt-test", "https://compatible.example/v1", "sk-test-secret"]
    )
    prompts: list[tuple[str, bool]] = []
    environ: dict[str, str] = {}

    def prompt(message: str, secret: bool, default: str | None) -> str:
        prompts.append((message, secret))
        return next(answers)

    store = DotEnvConfig(tmp_path)
    config, created = resolve_llm_config(store, prompt=prompt, environ=environ)

    assert created is True
    assert config.provider == "openai"
    assert config.model == "gpt-test"
    assert config.base_url == "https://compatible.example/v1"
    assert prompts[-1][1] is True
    assert environ["OPENAI_API_KEY"] == "sk-test-secret"

    values = dotenv_values(store.path)
    assert values[PREFERRED_PROVIDER_ENV] == "openai"
    assert values["OPENAI_MODEL"] == "gpt-test"
    assert values[OPENAI_BASE_URL_ENV] == "https://compatible.example/v1"
    assert values["OPENAI_API_KEY"] == "sk-test-secret"


def test_env_creation_requests_owner_only_permissions(
    tmp_path: Path, monkeypatch
) -> None:
    real_open = os.open
    requested_modes: list[int] = []

    def recording_open(path, flags, mode=0o777):
        requested_modes.append(mode)
        return real_open(path, flags, mode)

    monkeypatch.setattr(config_module.os, "open", recording_open)

    DotEnvConfig(tmp_path).ensure_exists()

    assert requested_modes == [0o600]


def test_existing_preference_prompts_only_for_missing_model_and_key(
    tmp_path: Path,
) -> None:
    store = DotEnvConfig(tmp_path)
    store.ensure_exists()
    store.set(PREFERRED_PROVIDER_ENV, "anthropic")
    answers = iter(["claude-test", "ant-test-secret"])
    prompts: list[tuple[str, bool]] = []

    def prompt(message: str, secret: bool, default: str | None) -> str:
        prompts.append((message, secret))
        return next(answers)

    config, created = resolve_llm_config(
        store, prompt=prompt, environ={}
    )

    assert created is False
    assert config.provider == "anthropic"
    assert config.model == "claude-test"
    assert len(prompts) == 2
    assert "模型名称" in prompts[0][0]
    assert prompts[1][1] is True


def test_complete_env_enters_without_prompting(tmp_path: Path) -> None:
    store = DotEnvConfig(tmp_path)
    store.ensure_exists()
    store.set(PREFERRED_PROVIDER_ENV, "openai")
    store.set("OPENAI_MODEL", "gpt-complete")
    store.set(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL)
    store.set("OPENAI_API_KEY", "sk-complete")

    def unexpected_prompt(
        message: str, secret: bool, default: str | None
    ) -> str:
        raise AssertionError(f"unexpected prompt: {message}, secret={secret}")

    environ: dict[str, str] = {}
    config, created = resolve_llm_config(
        store, prompt=unexpected_prompt, environ=environ
    )

    assert created is False
    assert config.provider == "openai"
    assert config.model == "gpt-complete"
    assert config.base_url == DEFAULT_OPENAI_BASE_URL
    assert environ["OPENAI_API_KEY"] == "sk-complete"


def test_process_environment_overrides_dotenv_without_rewriting_secret(
    tmp_path: Path,
) -> None:
    store = DotEnvConfig(tmp_path)
    store.ensure_exists()
    store.set(PREFERRED_PROVIDER_ENV, "openai")
    store.set("OPENAI_MODEL", "file-model")
    store.set(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL)
    store.set("OPENAI_API_KEY", "file-secret")
    environ = {
        "OPENAI_MODEL": "process-model",
        "OPENAI_API_KEY": "process-secret",
    }

    config, _ = resolve_llm_config(
        store,
        prompt=lambda message, secret, default: (_ for _ in ()).throw(
            AssertionError(message)
        ),
        environ=environ,
    )

    assert config.model == "process-model"
    values = dotenv_values(store.path)
    assert values["OPENAI_MODEL"] == "file-model"
    assert values["OPENAI_API_KEY"] == "file-secret"


def test_blank_openai_base_url_uses_and_persists_official_default(
    tmp_path: Path,
) -> None:
    answers = iter(["openai", "gpt-model", "", "sk-secret"])

    config, _ = resolve_llm_config(
        DotEnvConfig(tmp_path),
        prompt=lambda message, secret, default: next(answers),
        environ={},
    )

    assert config.base_url == DEFAULT_OPENAI_BASE_URL
    assert dotenv_values(tmp_path / ".env")[OPENAI_BASE_URL_ENV] == (
        DEFAULT_OPENAI_BASE_URL
    )


def test_invalid_interactive_provider_is_requested_again(tmp_path: Path) -> None:
    answers = iter(["unsupported", "An", "claude-model", "ant-secret"])
    config, _ = resolve_llm_config(
        DotEnvConfig(tmp_path),
        prompt=lambda message, secret, default: next(answers),
        environ={},
    )

    assert config.provider == "anthropic"


def test_read_preferred_config_does_not_prompt(tmp_path: Path) -> None:
    store = DotEnvConfig(tmp_path)
    config = read_preferred_config(store, environ={})

    assert config is None
    assert store.path.is_file()


def test_no_argument_command_launches_default_chat(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(cli_app, "_launch_chat", lambda **kwargs: calls.append(kwargs))

    result = CliRunner().invoke(cli_app.app, [])

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["provider"] is None
    assert calls[0]["goal"] is None
