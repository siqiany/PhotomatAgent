"""User-facing RAG CLI and slash-router contracts."""

from __future__ import annotations

import pytest
from rich.console import Console
from typer.testing import CliRunner

from photomatagent.cli.app import app
from photomatagent.cli import rag as rag_cli
from photomatagent.cli.commands import ChatCommandRouter
from photomatagent.scientific.capabilities.base import CapabilityStatus, ProbeResult
from photomatagent.workspace import Workspace


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_rag_status_hides_api_key(cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "secret-value")
    result = cli_runner.invoke(app, ["rag", "status"])
    assert result.exit_code == 0
    assert "secret-value" not in result.stdout


def test_rag_index_external_requires_confirmation(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("PHOTOMATAGENT_RAG_ALLOW_EXTERNAL", "1")
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_MODEL", "embedding-test")
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_BASE_URL", "https://embed.test/v1")
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_API_KEY_ENV", "EMBED_TEST_KEY")
    monkeypatch.setenv("EMBED_TEST_KEY", "test-key")
    result = cli_runner.invoke(
        app,
        ["rag", "index", "--workspace", str(tmp_path)],
        input="n\n",
    )
    assert result.exit_code != 0
    assert "发送全文片段" in result.stdout


@pytest.mark.asyncio
async def test_rag_slash_status_routes_to_typer_group(tmp_path) -> None:
    calls: list[list[str]] = []

    class Runtime:
        permission_policy = object()

    router = ChatCommandRouter(Console(), Runtime(), Workspace(tmp_path))

    async def run_cli(args: list[str]) -> None:
        calls.append(args)

    router._run_cli = run_cli  # type: ignore[method-assign]
    await router.execute("/rag status")
    assert calls == [["rag", "status"]]


def test_rag_group_defaults_to_status(cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "secret-value")
    result = cli_runner.invoke(app, ["rag"])
    assert result.exit_code == 0
    assert "secret-value" not in result.stdout


def test_rag_status_shows_server_alias_and_generation_state(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProbe:
        def __init__(self, config, workspace) -> None:
            del config, workspace

        def probe(self) -> ProbeResult:
            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                detail="aliases ready; generation ready",
                version="qdrant-server=1.18.2",
            )

        def status_snapshot(self) -> dict[str, str]:
            return {
                "server_version": "1.18.2",
                "alias_state": "ready",
                "generation_state": "ready:123456789abc",
            }

    monkeypatch.setattr(rag_cli, "LiteratureProbe", FakeProbe)
    result = cli_runner.invoke(app, ["rag", "status"])

    assert result.exit_code == 0
    assert "Qdrant server" in result.stdout
    assert "Alias state" in result.stdout
    assert "Generation" in result.stdout
    assert "1.18.2" in result.stdout
