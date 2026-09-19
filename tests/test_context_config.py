from __future__ import annotations

import pytest

from photomatagent.runtime.context_config import resolve_context_config


def test_context_config_priority_explicit_beats_env() -> None:
    config = resolve_context_config(
        context_limit_tokens=512_000,
        compact_trigger_tokens=123_456,
        environ={
            "PHOTOMATAGENT_CONTEXT_LIMIT_TOKENS": "64000",
            "PHOTOMATAGENT_COMPACT_TRIGGER_TOKENS": "32000",
            "PHOTOMATAGENT_COMPACT_TARGET_TOKENS": "16000",
            "PHOTOMATAGENT_RESPONSE_RESERVE_TOKENS": "2048",
        },
    )
    assert config.context_limit_tokens == 512_000
    assert config.compact_trigger_tokens == 123_456
    assert config.compact_target_tokens == 16_000
    assert config.response_reserve_tokens == 2_048


def test_context_config_empty_environment_uses_schema_defaults() -> None:
    config = resolve_context_config(environ={})
    assert config.context_limit_tokens == 128_000
    assert config.compact_trigger_tokens == 256_000
    assert config.compact_target_tokens == 128_000
    assert config.response_reserve_tokens == 8_192


def test_context_config_reads_non_empty_environment_values() -> None:
    config = resolve_context_config(
        environ={
            "PHOTOMATAGENT_CONTEXT_LIMIT_TOKENS": "256000",
            "PHOTOMATAGENT_COMPACT_TRIGGER_TOKENS": "",
            "PHOTOMATAGENT_COMPACT_TARGET_TOKENS": "64000",
            "PHOTOMATAGENT_RESPONSE_RESERVE_TOKENS": "4096",
        }
    )
    assert config.context_limit_tokens == 256_000
    assert config.compact_trigger_tokens == 256_000
    assert config.compact_target_tokens == 64_000
    assert config.response_reserve_tokens == 4_096


@pytest.mark.parametrize(
    "environ",
    [
        {"PHOTOMATAGENT_CONTEXT_LIMIT_TOKENS": "abc"},
        {"PHOTOMATAGENT_COMPACT_TARGET_TOKENS": "-1"},
        {
            "PHOTOMATAGENT_CONTEXT_LIMIT_TOKENS": "32000",
            "PHOTOMATAGENT_RESPONSE_RESERVE_TOKENS": "31000",
        },
    ],
)
def test_context_config_rejects_bad_values_with_field_names(environ) -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_context_config(environ=environ)
    message = str(exc_info.value)
    assert "PHOTOMATAGENT_" in message or "context_limit_tokens" in message
    assert "sk-" not in message


def test_build_runtime_uses_resolved_context_config(tmp_path, monkeypatch):
    from photomatagent.cli.chat import build_runtime

    monkeypatch.setenv("PHOTOMATAGENT_CONTEXT_LIMIT_TOKENS", "64000")
    monkeypatch.setenv("PHOTOMATAGENT_COMPACT_TRIGGER_TOKENS", "32000")
    monkeypatch.setenv("PHOTOMATAGENT_COMPACT_TARGET_TOKENS", "16000")
    runtime, logger = build_runtime(
        provider="fake",
        workspace_root=tmp_path,
        approval="deny",
        log_events=False,
    )
    assert logger is None
    assert runtime.context_engine.config.context_limit_tokens == 64_000
    assert runtime.context_engine.config.compact_trigger_tokens == 32_000
    assert runtime.context_engine.config.compact_target_tokens == 16_000


def test_fresh_evaluation_ignores_invalid_context_environment(tmp_path, monkeypatch):
    from photomatagent.cli.chat import build_runtime

    monkeypatch.setenv("PHOTOMATAGENT_CONTEXT_LIMIT_TOKENS", "not-an-int")
    runtime, _ = build_runtime(
        provider="fake",
        workspace_root=tmp_path,
        approval="deny",
        log_events=False,
        evaluation_isolation=True,
    )
    assert runtime.context_engine.config.context_limit_tokens == 128_000
    assert runtime.context_engine.config.compact_trigger_tokens == 256_000
