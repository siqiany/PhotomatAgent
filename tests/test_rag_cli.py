"""User-facing RAG CLI and slash-router contracts."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_rag_status_is_fail_soft_for_source_root_outside_workspace(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("PHOTOMATAGENT_LITERATURE_DIR", "/etc")

    class FakeProbe:
        def __init__(self, config, workspace) -> None:
            del config, workspace

        def probe(self) -> ProbeResult:
            return ProbeResult(
                status=CapabilityStatus.UNCONFIGURED,
                detail="source_root_missing",
                version="qdrant-server=1.18.2",
            )

        def status_snapshot(self) -> dict[str, str]:
            return {
                "server_version": "1.18.2",
                "alias_state": "ready",
                "generation_state": "ready:123456789abc",
                "source_root": "outside workspace",
            }

    monkeypatch.setattr(rag_cli, "LiteratureProbe", FakeProbe)
    result = cli_runner.invoke(app, ["rag", "status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "outside workspace" in result.stdout
    assert "1.18.2" in result.stdout


def test_rag_status_renders_credential_free_qdrant_url_and_connection_state(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "PHOTOMATAGENT_QDRANT_URL",
        "https://user:secret@qdrant.example:6333?token=another-secret",
    )
    monkeypatch.setenv("QDRANT_API_KEY", "another-secret")

    class FakeProbe:
        def __init__(self, config, workspace) -> None:
            del config, workspace

        def probe(self) -> ProbeResult:
            return ProbeResult(status=CapabilityStatus.UNCONFIGURED, detail="source missing")

        def status_snapshot(self) -> dict[str, str]:
            return {
                "server_version": "1.18.2",
                "alias_state": "ready",
                "generation_state": "ready:123456789abc",
                "tls": "enabled",
                "auth": "configured (value hidden)",
            }

    monkeypatch.setattr(rag_cli, "LiteratureProbe", FakeProbe)
    result = cli_runner.invoke(app, ["rag", "status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "https://qdrant.example:6333" in result.stdout
    assert "user:secret" not in result.stdout
    assert "another-secret" not in result.stdout
    assert "TLS" in result.stdout
    assert "Auth" in result.stdout


def test_rag_evaluate_reports_live_unavailable_instead_of_fake_quality(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        rag_cli,
        "build_literature_services",
        lambda config, workspace: (_ for _ in ()).throw(
            RuntimeError("qdrant unavailable")
        ),
    )
    result = cli_runner.invoke(app, ["rag", "evaluate", "--workspace", str(tmp_path)])

    assert result.exit_code != 0
    assert '"live_evaluation": false' in result.stdout
    assert '"passed": false' in result.stdout
    assert "qdrant unavailable" in result.stdout


@pytest.mark.asyncio
async def test_index_until_complete_stops_after_unresolved_retryable_batch(
    tmp_path, monkeypatch
) -> None:
    calls = 0

    class Ingestion:
        embedder = object()

        async def plan(self, root, boundary):
            del root, boundary
            return object()

        async def index_batch(self, plan, **kwargs):
            del plan, kwargs
            nonlocal calls
            calls += 1
            return type(
                "Stats",
                (),
                {
                    "run_id": "retry-run",
                    "discovered": 1,
                    "unchanged": 0,
                    "indexed": 0,
                    "failed": 1,
                    "deleted": 0,
                    "chunks": 0,
                    "staged_cleanup": 0,
                    "next_cursor": None,
                    "complete": False,
                    "retryable": True,
                    "errors": ("paper.pdf: failed",),
                },
            )()

    class Services:
        ingestion = Ingestion()
        store = object()

    result = await rag_cli._index_until_complete(
        Services(),
        Workspace(tmp_path),
        tmp_path,
        config=rag_cli.ScientificConfig(rag_tool_max_documents=1),
        run_id="retry-run",
    )

    assert calls == 1
    assert result["complete"] is False
    assert result["retryable"] is True


@pytest.mark.asyncio
async def test_index_until_complete_uses_source_aware_chunk_schema_generation(tmp_path) -> None:
    versions: list[int] = []

    class Embedder:
        identity = object()

    class Store:
        def ensure_generation(self, *, identity, chunk_schema_version):
            del identity
            versions.append(chunk_schema_version)
            return SimpleNamespace()

    class Ingestion:
        embedder = Embedder()

        async def plan(self, root, boundary):
            del root, boundary
            return object()

        async def index_batch(self, plan, **kwargs):
            del plan, kwargs
            return SimpleNamespace(
                run_id="run",
                discovered=0,
                unchanged=0,
                indexed=0,
                failed=0,
                deleted=0,
                chunks=0,
                staged_cleanup=0,
                next_cursor=None,
                complete=True,
                retryable=False,
                errors=(),
            )

    class Services:
        ingestion = Ingestion()
        store = Store()

    await rag_cli._index_until_complete(
        Services(),
        Workspace(tmp_path),
        tmp_path,
        config=rag_cli.ScientificConfig(rag_tool_max_documents=1),
        run_id="run",
    )

    assert versions == [2]


@pytest.mark.asyncio
async def test_index_pauses_after_requested_budget(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    calls: list[int] = []

    class Embedder:
        identity = object()

    class Store:
        def ensure_generation(self, *, identity, chunk_schema_version):
            del identity, chunk_schema_version
            return SimpleNamespace()

    class Ingestion:
        embedder = Embedder()

        async def plan(self, root, boundary):
            del root, boundary
            return SimpleNamespace(items=[object()] * 100)

        async def index_batch(self, plan, **kwargs):
            del plan
            requested = int(kwargs["max_documents"])
            calls.append(requested)
            processed = sum(calls)
            return SimpleNamespace(
                run_id="pdf-run",
                discovered=100,
                processed=processed,
                unchanged=0,
                indexed=processed,
                failed=0,
                deleted=0,
                chunks=processed,
                staged_cleanup=0,
                next_cursor=f"paper-{processed}",
                complete=False,
                retryable=False,
                errors=(),
            )

    services = SimpleNamespace(ingestion=Ingestion(), store=Store())
    result = await rag_cli._index_until_complete(
        services,
        Workspace(tmp_path),
        tmp_path,
        config=rag_cli.ScientificConfig(rag_tool_max_documents=20),
        run_id="pdf-run",
        stop_after=25,
        progress=rows.append,
    )

    assert result["paused"] is True
    assert result["processed_this_invocation"] <= 25
    assert rows
    assert calls == [20, 5]


@pytest.mark.asyncio
async def test_index_abstracts_binds_staging_generation_before_first_batch(tmp_path) -> None:
    generation = SimpleNamespace(fingerprint="a" * 64)
    selected: list[object] = []
    ensured: list[int] = []

    class Embedder:
        identity = object()

    class Store:
        def ensure_generation(self, *, identity, chunk_schema_version):
            del identity
            ensured.append(chunk_schema_version)
            return generation

        def select_staging_generation(self, value):
            selected.append(value)

    class Ingestion:
        embedder = Embedder()
        generation = None

        async def index_batch(self, **kwargs):
            del kwargs
            assert self.generation is generation
            return SimpleNamespace(
                run_id="abstract-run",
                total=1,
                processed=1,
                indexed=1,
                unchanged=0,
                failed=0,
                skipped_empty=0,
                skipped_invalid_key=0,
                passages=1,
                cursor=None,
                complete=True,
                status="complete",
                errors=(),
            )

    ingestion = Ingestion()
    result = await rag_cli._index_abstracts_until_complete(
        ingestion,
        run_id="abstract-run",
        config=rag_cli.ScientificConfig(rag_tool_max_documents=1),
        store=Store(),
    )

    assert result["complete"] is True
    assert ensured == [2]
    assert selected == [generation]


def test_index_abstracts_resume_rejects_missing_run(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    source_database = tmp_path / "abstracts.sqlite3"
    source_database.touch()
    calls: list[str] = []
    generation = SimpleNamespace(fingerprint="b" * 64)

    class Store:
        def ensure_generation(self, *, identity, chunk_schema_version):
            del identity, chunk_schema_version
            return generation

        def select_staging_generation(self, value):
            assert value is generation

        async def get_ingestion_run(self, run_id, workspace_id, **kwargs):
            del run_id, workspace_id, kwargs
            return None

    class Ingestion:
        embedder = SimpleNamespace(identity=object())
        workspace_id = "workspace"

        async def index_batch(self, **kwargs):
            calls.append(str(kwargs.get("run_id")))
            return SimpleNamespace(
                total=0,
                processed=0,
                indexed=0,
                unchanged=0,
                failed=0,
                skipped_empty=0,
                skipped_invalid_key=0,
                passages=0,
                cursor=None,
                complete=True,
                status="complete",
                errors=(),
            )

    services = SimpleNamespace(
        store=Store(),
        ingestion=SimpleNamespace(embedder=Ingestion.embedder),
        abstract_ingestion=Ingestion(),
        workspace_id="workspace",
    )
    monkeypatch.setattr(rag_cli, "_config", lambda workspace: SimpleNamespace(rag_tool_max_documents=1))
    monkeypatch.setattr(rag_cli, "_abstract_database", lambda boundary, database, config: source_database)
    monkeypatch.setattr(rag_cli, "_confirm_external", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag_cli, "build_literature_services", lambda config, workspace: services)

    result = cli_runner.invoke(
        app,
        [
            "rag",
            "index-abstracts",
            "--resume",
            "--run-id",
            "missing-run",
            "--yes",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "resume_run_missing" in result.stdout
    assert calls == []


def test_activate_validates_required_run_in_selected_generation(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    generation = SimpleNamespace(
        fingerprint="c" * 64,
        documents_physical="documents",
        passages_physical="passages",
    )

    class Store:
        selected = None

        def ensure_generation(self, *, identity, chunk_schema_version):
            del identity
            assert chunk_schema_version == 2
            return generation

        def select_staging_generation(self, value):
            self.selected = value

        async def get_ingestion_run(self, run_id, workspace_id, **kwargs):
            del run_id, workspace_id, kwargs
            assert self.selected is generation
            return SimpleNamespace(
                source_kind=rag_cli.LiteratureSourceKind.FULLTEXT,
                status="complete",
                generation_fingerprint=generation.fingerprint,
                stats=SimpleNamespace(complete=True, retryable=False),
            )

        async def activate_generation(self, value, *, allow_empty_bootstrap):
            assert value is generation
            assert self.selected is generation
            del allow_empty_bootstrap

    store = Store()
    services = SimpleNamespace(
        store=store,
        ingestion=SimpleNamespace(embedder=SimpleNamespace(identity=object())),
        workspace_id="workspace",
    )
    monkeypatch.setattr(rag_cli, "_config", lambda workspace: object())
    monkeypatch.setattr(rag_cli, "build_literature_services", lambda config, workspace: services)
    result = cli_runner.invoke(
        app,
        [
            "rag",
            "activate",
            "--require-stage",
            "pdf",
            "--pdf-run-id",
            "pdf-run",
            "--yes",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout


def test_rag_status_reports_explicit_stage_runs(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class FakeProbe:
        def __init__(self, config, workspace) -> None:
            del config, workspace

        def probe(self) -> ProbeResult:
            return ProbeResult(status=CapabilityStatus.AVAILABLE, detail="ready")

        def status_snapshot(self) -> dict[str, str]:
            return {}

    class Store:
        async def get_ingestion_run(self, run_id, workspace_id, **kwargs):
            del run_id, workspace_id
            return SimpleNamespace(
                source_kind=kwargs.get("source_kind"),
                status="complete",
                stats=SimpleNamespace(complete=True, retryable=False),
            )

    config = SimpleNamespace(
        qdrant_url="http://qdrant",
        qdrant_api_key_env="QDRANT_API_KEY",
        qdrant_collection_prefix="literature",
        embedding_provider="local",
        embedding_model="local",
        reranker_provider="disabled",
        reranker_model="",
        literature_root=tmp_path,
    )
    services = SimpleNamespace(store=Store(), workspace_id="workspace")
    monkeypatch.setattr(rag_cli, "_config", lambda workspace: config)
    monkeypatch.setattr(rag_cli, "LiteratureProbe", FakeProbe)
    monkeypatch.setattr(rag_cli, "build_literature_services", lambda config, workspace: services)
    result = cli_runner.invoke(
        app,
        [
            "rag",
            "status",
            "--pdf-run-id",
            "pdf-run",
            "--abstract-run-id",
            "abstract-run",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "PDF stage" in result.stdout
    assert "Abstract stage" in result.stdout
    assert result.stdout.count("complete") >= 2


def test_rag_status_marks_unidentified_stage_runs_as_not_supplied(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class FakeProbe:
        def __init__(self, config, workspace) -> None:
            del config, workspace

        def probe(self) -> ProbeResult:
            return ProbeResult(status=CapabilityStatus.AVAILABLE, detail="ready")

        def status_snapshot(self) -> dict[str, str]:
            return {"pdf_stage": "unknown", "abstract_stage": "unknown"}

    monkeypatch.setattr(rag_cli, "LiteratureProbe", FakeProbe)
    result = cli_runner.invoke(
        app,
        ["rag", "status", "--workspace", str(tmp_path)],
    )

    assert result.exit_code == 0, result.stdout
    assert "not supplied" in result.stdout
    stage_lines = [
        line
        for line in result.stdout.splitlines()
        if "PDF stage" in line or "Abstract stage" in line
    ]
    assert stage_lines
    assert all("unknown" not in line for line in stage_lines)


def test_rag_index_abstracts_external_confirmation_is_stage_specific(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("PHOTOMATAGENT_RAG_ALLOW_EXTERNAL", "1")
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_MODEL", "embedding-test")
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_BASE_URL", "https://embed.test/v1")
    monkeypatch.setenv("PHOTOMATAGENT_RAG_EMBEDDING_API_KEY_ENV", "EMBED_TEST_KEY")
    monkeypatch.setenv("EMBED_TEST_KEY", "test-key")
    monkeypatch.setenv("PHOTOMATAGENT_LITERATURE_DIR", str(tmp_path))
    abstract_dir = tmp_path / "abstract"
    abstract_dir.mkdir()
    (abstract_dir / "abstracts.sqlite3").touch()

    result = cli_runner.invoke(
        app,
        ["rag", "index-abstracts", "--workspace", str(tmp_path)],
        input="n\n",
    )

    assert result.exit_code != 0
    assert "发送摘要片段" in result.stdout


def test_abstract_resume_requires_run_id(cli_runner: CliRunner, tmp_path) -> None:
    result = cli_runner.invoke(
        app,
        ["rag", "index-abstracts", "--resume", "--workspace", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "resume_requires_run_id" in result.stdout


def test_activate_rejects_incomplete_required_stage(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class Store:
        async def get_ingestion_run(self, run_id, workspace_id, **kwargs):
            del run_id, workspace_id, kwargs
            return None

    services = SimpleNamespace(
        store=Store(),
        ingestion=SimpleNamespace(embedder=SimpleNamespace(identity=object())),
    )
    monkeypatch.setattr(rag_cli, "_config", lambda workspace: object())
    monkeypatch.setattr(
        rag_cli, "build_literature_services", lambda config, workspace: services
    )
    result = cli_runner.invoke(
        app,
        [
            "rag",
            "activate",
            "--require-stage",
            "pdf",
            "--require-stage",
            "abstracts",
            "--yes",
            "--workspace",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "stage_incomplete" in result.stdout


def test_rag_activate_uses_source_aware_chunk_schema_generation(
    tmp_path, monkeypatch
) -> None:
    versions: list[int] = []
    generation = SimpleNamespace(
        fingerprint="f" * 64,
        documents_physical="documents",
        passages_physical="passages",
    )

    class Store:
        def ensure_generation(self, *, identity, chunk_schema_version):
            del identity
            versions.append(chunk_schema_version)
            return generation

        async def activate_generation(self, value, *, allow_empty_bootstrap):
            assert value is generation
            assert allow_empty_bootstrap is True

    services = SimpleNamespace(
        store=Store(), ingestion=SimpleNamespace(embedder=SimpleNamespace(identity=object()))
    )
    monkeypatch.setattr(rag_cli, "_config", lambda workspace: object())
    monkeypatch.setattr(rag_cli, "build_literature_services", lambda config, workspace: services)

    rag_cli.rag_activate(yes=True, bootstrap=True, workspace=tmp_path)

    assert versions == [2]
