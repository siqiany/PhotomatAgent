"""Integration coverage for registered indexing, multi-file runs and uploads."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from io import BytesIO
import json
import tarfile

from fastapi.testclient import TestClient
import httpx
from PIL import Image
import pytest

from photomatagent.application.models import TaskCreate
from photomatagent.application.reproduction import NumericReference, ReproductionService, ReproductionSpec
from photomatagent.application.runner import RunManager
from photomatagent.application.service import WorkbenchService
from photomatagent.application.store import WorkbenchStore
from photomatagent.capability_acquisition import container_backend
from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend, _tree_sha256
from photomatagent.capability_acquisition.repository_sources import FetchRepositoryArchiveTool
from photomatagent.capability_acquisition.sandbox import SandboxService
from photomatagent.capability_acquisition.web_sources import PublicSourceClient
from photomatagent.models.openai import OpenAIProvider
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.observation import ObservationPolicy
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.literature import (
    LiteratureExtractEvidenceTool, LiteratureIndexDocumentTool, LiteratureReadObjectTool, LiteratureReadPaperTool,
    LiteratureServices,
)
from photomatagent.scientific.capabilities.literature.documents import DocumentObject, DocumentService, ParsedDocument
from photomatagent.scientific.capabilities.literature.ingestion import LiteratureIngestionService, PlanKind
from photomatagent.scientific.capabilities.literature.models import IngestState
from photomatagent.scientific.capabilities.literature.qdrant_store import CollectionGeneration, workspace_id_for
from photomatagent.scientific.capabilities.literature.providers.base import ModelIdentity
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.web.app import create_app
from photomatagent.web.artifacts import ArtifactService
from photomatagent.workspace import Workspace

from test_materials_method_integration import fixture_document


@pytest.mark.asyncio
async def test_long_document_objects_and_table_cells_have_lossless_bounded_cursors(tmp_path):
    workspace = Workspace(tmp_path)
    (tmp_path / "papers").mkdir()
    paper = tmp_path / "papers" / "long.pdf"
    paper.write_bytes(b"%PDF-original")
    paragraph = '段"\\😀\n' * 4300
    table = "|" * 25001
    cells = [{"text": "42", "start_row_offset_idx": 0},
             {"text": '长"😀\\' * 1200, "start_row_offset_idx": 1},
             {"text": "end", "start_row_offset_idx": 2}]
    objects = [
        DocumentObject(object_id="p", kind="paragraph", text=paragraph, quality="unknown"),
        DocumentObject(object_id="t", kind="table", text=table, table_cells=cells, quality="unknown"),
    ]
    documents = DocumentService(workspace, lambda _path: ParsedDocument(title="Long", objects=objects))
    config = ScientificConfig(literature_max_chars=20_000)
    reader = LiteratureReadPaperTool(config, workspace, documents)
    selected = {"file": paper.name, "object_ids": ["p", "t"], "max_chars": 2048}
    first = (await reader.execute(selected)).data
    assert first["cursor"]
    cursor = ""
    recovered = {"p": "", "t": ""}
    recovered_cells = {}
    fragments = {}
    for _ in range(100):
        page = (await reader.execute({**selected, "cursor": cursor})).data
        consumed = 0
        for row in page["objects"]:
            if "text" in row:
                assert len(recovered[row["object_id"]]) == row["text_start"]
                recovered[row["object_id"]] += row["text"]
                consumed += len(json.dumps(row["text"], ensure_ascii=False).encode())
            if "table_cells" in row:
                consumed += len(json.dumps(row["table_cells"], ensure_ascii=False, separators=(",", ":")).encode())
                recovered_cells.update(enumerate(row["table_cells"], row["table_cell_start"]))
            if "table_cell_fragment" in row:
                fragment = row["table_cell_fragment"]
                index = fragment["cell_index"]
                assert len(fragments.get(index, "")) == fragment["start"]
                fragments[index] = fragments.get(index, "") + fragment["json"]
                consumed += len(json.dumps(fragment["json"], ensure_ascii=False).encode())
        assert 0 < consumed <= 2048
        cursor = page["cursor"]
        if cursor is None:
            break
    assert cursor is None
    assert recovered == {"p": paragraph, "t": table}
    assert [recovered_cells.get(i) or json.loads(fragments[i]) for i in range(len(cells))] == cells
    object_reader = LiteratureReadObjectTool(config, workspace, documents)
    object_page = await object_reader.execute({
        "document_id": first["document_id"], "object_ids": ["t"], "max_chars": 20000,
    })
    assert object_page.data["cursor"]
    assert not ObservationPolicy().apply(object_reader.name, object_page.output).truncated
    evidence = await LiteratureExtractEvidenceTool(config, workspace, documents=documents).execute({
        "passages": [{"document_id": first["document_id"], "object_id": "p"}],
    })
    assert evidence.is_error and evidence.data["diagnostic"] == "DOCUMENT_EVIDENCE_TEXT_EXCEEDS_BUDGET"
    assert (await reader.execute({**selected, "object_ids": ["t"], "cursor": first["cursor"]})).data["error"] == "DOCUMENT_CURSOR_INVALID_OR_STALE"
    assert (await reader.execute({**selected, "cursor": first["cursor"].split(":")[0] + ":1"})).data["error"] == "DOCUMENT_CURSOR_INVALID_OR_STALE"
    paper.write_bytes(b"%PDF-new-revision")
    assert (await reader.execute({**selected, "cursor": first["cursor"]})).data["error"] == "DOCUMENT_CURSOR_INVALID_OR_STALE"


class IndexFixtureEmbedder:
    identity = ModelIdentity(provider="fixture", model="fixture", dimension=2)

    async def embed_documents(self, texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class IndexFixtureStore:
    def __init__(self):
        self.generation = CollectionGeneration(
            fingerprint="f" * 64, documents_physical="documents-f", passages_physical="passages-f",
            documents_alias="documents-current", passages_alias="passages-current",
        )
        self.documents = {}
        self.passage_upserts = []
        self.runs = {}

    def expected_generation(self, **_kwargs):
        return self.generation

    async def resolve_current_generation(self):
        return self.generation

    async def get_document_manifests(self, _workspace_id, ids, **_filters):
        return {id_: self.documents[id_] for id_ in ids if id_ in self.documents}

    async def list_document_manifests(self, _workspace_id, **_filters):
        return dict(self.documents)

    async def delete_staged_revisions(self, *_args, **_kwargs):
        return 0

    async def delete_other_revisions(self, *_args, **_kwargs):
        return None

    async def upsert_passages(self, points, **_kwargs):
        self.passage_upserts.append(list(points))

    async def count_revision(self, _document_id, _revision, state, **_kwargs):
        assert state is IngestState.STAGED
        return len(self.passage_upserts[-1])

    async def set_revision_state(self, *_args, **_kwargs):
        return None

    async def upsert_document(self, manifest, **_kwargs):
        self.documents[manifest.document_id] = manifest

    async def get_ingestion_run(self, run_id, _workspace_id):
        return self.runs.get(run_id)

    async def upsert_ingestion_run(self, run, **_kwargs):
        self.runs[run.run_id] = run


@pytest.mark.asyncio
async def test_registered_objects_index_through_existing_pipeline(tmp_path, monkeypatch):
    from photomatagent.scientific.capabilities.literature import ingestion

    ws = Workspace(tmp_path)
    store = IndexFixtureStore()
    monkeypatch.setattr(ingestion, "parse_pdf", lambda _: pytest.fail("Do not reparse indexed objects"))
    service = LiteratureIngestionService(store, IndexFixtureEmbedder())
    document_service = DocumentService(ws, fixture_document)
    artifact = document_service.register_pdf(b"%PDF-test", "https://arxiv.org/pdf/2401.01234")
    await document_service.read(document_id=artifact.document_id)
    services = LiteratureServices(ingestion=service, retriever=None, store=store, workspace_id=workspace_id_for(tmp_path))
    tool = LiteratureIndexDocumentTool(ScientificConfig(), ws, document_service, services)
    first = await tool.execute({"document_id": artifact.document_id})
    assert first.data["indexed"] == 1 and first.data["search_visible"]
    points = store.passage_upserts[0]
    assert len(points) == 2
    ref = points[0].to_payload()["structured_ref"]
    assert ref == {"document_id": artifact.document_id, "revision": first.data["revision"],
                   "object_id": "table", "text_offset": 0}
    assert points[0].page_start == 1 and points[0].page_end == 2
    reader = LiteratureReadObjectTool(ScientificConfig(), ws, document_service)
    result = await reader.execute({"document_id": ref["document_id"], "revision": ref["revision"],
                                   "object_ids": [ref["object_id"]]})
    assert result.data["objects"][0]["kind"] == "table"
    assert (await reader.execute({"document_id": ref["document_id"], "revision": "0" * 64,
                                  "object_ids": ["table"]})).is_error
    assert (await tool.execute({"document_id": artifact.document_id})).data["unchanged"] == 1
    assert len(store.passage_upserts) == 1
    article = document_service.register_html(b"<article>Public excerpt.</article>", "https://example.com/paper")
    rejected = await tool.execute({"document_id": article.document_id})
    assert rejected.data["error"] == "DOCUMENT_INDEX_COVERAGE_UNCONFIRMED"


    # A later bulk index over the workspace must preserve managed snapshots.
    plan = await service.plan(ws.root, ws)
    assert not any(item.kind is PlanKind.DELETED for item in plan.items)
    assert not any("structured-index" in item.relative_source_path for item in plan.items)


def tar_archive(*entries: tuple[str, bytes]) -> bytes:
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, data in entries:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, BytesIO(data))
    return output.getvalue()


@pytest.mark.asyncio
async def test_archive_run_binds_tree_and_recovers_terminal_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_GITHUB_DISCOVERY", "1")
    ws = Workspace(tmp_path)
    data = tar_archive(("repo-pinned/main.py", b"import helper\n"), ("repo-pinned/helper.py", b"value = 2\n"))
    archive_body = data

    async def resolver(_host):
        return ["1.1.1.1"]

    def respond(request):
        if request.headers["host"] == "api.github.com":
            return httpx.Response(302, headers={"Location": "https://codeload.github.com/example/repo/tar.gz/sha"})
        return httpx.Response(200, content=archive_body, headers={"Content-Type": "application/x-gzip"})

    transport = PublicSourceClient(frozenset({"api.github.com", "codeload.github.com"}),
                                   resolver=resolver, transport=httpx.MockTransport(respond))
    fetch = FetchRepositoryArchiveTool(ws, None, transport)
    params = {"owner": "example", "repository": "repo", "commit": "a" * 40, "entry": "main.py",
              "files": ["helper.py"], "sha256": hashlib.sha256(data).hexdigest(),
              "license": "MIT", "license_source": "https://github.com/example/repo"}
    acquired = (await fetch.execute(params)).data
    assert acquired["source_files"] == ["helper.py"]
    archive_body = tar_archive(("repo-pinned/main.py", b"import helper\n"),
                               ("repo-pinned/helper.py", b"value = 2\n"), ("repo-pinned/../../escape", b"x"))
    unsafe = await fetch.execute({**params, "sha256": hashlib.sha256(archive_body).hexdigest()})
    assert unsafe.data["diagnostic"] == "SOURCE_ARCHIVE_MEMBER_INVALID"

    env_id = "env_" + "a" * 24
    storage = tmp_path / "environments"
    env_path = storage / env_id
    py = env_path / "python"
    py.mkdir(parents=True)
    (py / "fixture").write_text("fixture")
    (env_path / "requirements.txt").write_text("")
    (env_path / "installed-manifest.json").write_text(json.dumps({
        "plan_sha256": "a" * 64, "environment_sha256": _tree_sha256(py),
        "image_ref": "python:3.12-slim@sha256:" + "b" * 64, "entrypoints": {},
        "created_at": datetime.now(UTC).isoformat(),
    }))
    commands = []
    async def process(arguments, **_limits):
        commands.append(arguments)
        if arguments[1] == "run":
            stage = next(arg.split(":/input:ro,Z")[0] for arg in arguments if arg.endswith(":/input:ro,Z"))
            assert (tmp_path / stage / "source" / "helper.py").read_bytes() == b"value = 2\n"
            assert "/input/source/main.py" in arguments
            assert "--network=none" in arguments and "--pull=never" in arguments
        return 0, b'{"result":2,"unit":"1"}', b"", False, False

    monkeypatch.setattr(container_backend, "_run_bounded_process", process)
    def service():
        backend = PodmanSandboxBackend(storage)
        backend.executable = "/usr/bin/podman"
        sandbox = SandboxService(workspace=tmp_path, trial_id="fixture", genome_sha256="c" * 64,
                                 scripts={}, backend=backend, dynamic_script_prefix="tmp/reproduction")
        sandbox.register_environment(backend.restore_environment(env_id))
        return ReproductionService(ws, sandbox)

    spec = ReproductionSpec(
        source="https://example.com/paper", paper_anchor="Example", source_commit=params["commit"],
        implementation="third_party", environment_id=env_id,
        script_path=acquired["script_path"], source_root=acquired["source_root"],
        source_files=("helper.py",), source_bundle_id=acquired["bundle_id"],
        reference=NumericReference(metric="result", expected=2, absolute_tolerance=0.0, unit="1", source_anchor="Fixture"),
    )
    reproduce = service()
    plan = reproduce.prepare(spec)
    helper = ws.resolve(acquired["source_root"] + "/helper.py")
    helper.write_text("value = 3\n")
    changed = await reproduce.execute(plan["plan_id"], plan["plan_sha256"])
    assert changed["diagnostic"] == "INPUT_BINDING_CHANGED"
    with pytest.raises(ValueError, match="SOURCE_ARCHIVE_BINDING_CHANGED"):
        reproduce.prepare(spec)
    helper.write_text("value = 2\n")
    outcome = await reproduce.execute(plan["plan_id"], plan["plan_sha256"])
    assert outcome["execution"] == "SUCCEEDED" and outcome["checker"]["result"] == "PASS"

    # The backend receipt survived a host interruption before publishing the service record.
    published = tmp_path / ".photomatagent" / "executions" / outcome["run_id"] / "record.json"
    published.unlink()
    recovered = await service().execute(plan["plan_id"], plan["plan_sha256"])
    assert recovered == outcome
    assert sum(command[1] == "run" for command in commands) == 1


def test_workbench_image_run_binds_upload_and_sends_model_input(tmp_path):
    requests = []
    class Responses:
        async def create(self, **kwargs):
            requests.append(kwargs)
            async def stream():
                yield {"type": "response.output_text.delta", "delta": "Received."}
                yield {"type": "response.completed", "response": {"output": []}}
            return stream()

    class Client:
        responses = Responses()

    ws = Workspace(tmp_path)
    store = WorkbenchStore(tmp_path / ".photomatagent" / "workbench.sqlite3")
    def runtime_builder(**kwargs):
        state = ScientificState()
        return AgentRuntime(model=OpenAIProvider("deepseek-flash", client=Client()), workspace=ws,
                           tools=create_default_registry(state, ws), scientific_state=state, budget=kwargs["budget"])
    manager = RunManager(store, runtime_builder=runtime_builder, runtime_options={"workspace_root": tmp_path})
    service = WorkbenchService(store=store, manager=manager, workspace_root=tmp_path,
                               sessions_dir=tmp_path / ".photomatagent/sessions", artifacts=ArtifactService(store, tmp_path))
    app = create_app(service=service, manager=manager)
    task = store.create_task(TaskCreate(title="Photo", goal="Read the image"))
    other = store.create_task(TaskCreate(title="Other", goal="Other"))
    png = BytesIO()
    Image.new("RGB", (8, 8)).save(png, format="PNG")
    with TestClient(app) as client:
        upload = client.post(f"/api/tasks/{task}/attachments",
                             files={"file": ("sample.png", png.getvalue(), "image/png")})
        assert upload.status_code == 201, upload.json()
        body = {"text": "Read the image", "image_attachment_ids": [upload.json()["attachment_id"]],
                "evolution": {"enabled": False}}
        foreign = client.post(f"/api/tasks/{other}/runs", json=body, headers={"Idempotency-Key": "foreign"})
        assert foreign.status_code == 404
        run = client.post(f"/api/tasks/{task}/runs", json=body, headers={"Idempotency-Key": "image"})
        assert run.status_code == 202, run.json()
        client.portal.call(manager.wait_for_idle)
        assert store.run(run.json()["run_id"]).status == "SUCCEEDED"
        assert requests[0]["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    store.close()


@pytest.mark.asyncio
async def test_pdf_object_crop_is_bound_to_model_tool_result(tmp_path):
    from pypdf import PdfWriter
    from photomatagent.models.image_assets import load_latest_images
    from photomatagent.models.openai import openai_input
    from photomatagent.models.anthropic import anthropic_input
    from photomatagent.models.types import ToolResultMessage, UserMessage
    from photomatagent.scientific.capabilities.literature.documents import DocumentObject, ParsedDocument

    pdf = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.write(pdf)
    ws = Workspace(tmp_path)
    def fixture(_path):
        return ParsedDocument(title="Figure", objects=[DocumentObject(
            object_id="figure-1", kind="figure", quality="anchored", anchors=[
                {"page": 1, "bbox": {"l": 10, "t": 20, "r": 30, "b": 40, "coord_origin": "TOPLEFT"}},
            ],
        )])

    service = DocumentService(ws, fixture)
    artifact = service.register_pdf(pdf.getvalue(), "https://arxiv.org/pdf/2401.01234")
    tool = LiteratureReadObjectTool(ScientificConfig(), ws, service)
    result = await tool.execute({"document_id": artifact.document_id, "object_ids": ["figure-1"], "include_images": True})
    assert result.images and not result.data["crop_diagnostics"]
    crop = result.data["crops"][0]
    assert crop["source_sha256"] == artifact.sha256 and crop["object_id"] == "figure-1"
    assert crop["image"]["sha256"] == result.images[0].sha256
    user = UserMessage(content="Read the figure.")
    observation = ToolResultMessage(tool_call_id="call1", tool_name=tool.name, content=result.output,
                                    images=result.images)
    history = [user, observation]
    encoded = load_latest_images(ws, history)
    assert openai_input(history, image_assets=encoded)[-1]["output"][1]["type"] == "input_image"
    assert anthropic_input(history, image_assets=encoded)[1][-1]["content"][-1]["content"][1]["type"] == "image"
    assert "base64" not in observation.model_dump_json()
    later = [*history, UserMessage(content="New task")]
    assert load_latest_images(ws, later) == {}


@pytest.mark.asyncio
async def test_federated_pages_cache_relationships_and_request_budget():
    from photomatagent.scientific.capabilities.literature.federated import FederatedLiterature

    calls = []
    async def resolver(_):
        return ["1.1.1.1"]

    def response(request):
        calls.append(request)
        return httpx.Response(200, json={"message": {"items": [{
            "DOI": "10.1234/preprint", "title": ["Preprint"],
            "relation": {"is-preprint-of": [{"id": "10.1234/final", "id-type": "doi"}]},
        }]}})

    client = PublicSourceClient(frozenset({"api.crossref.org"}), resolver=resolver,
                                transport=httpx.MockTransport(response), request_quota=1)
    service = FederatedLiterature(frozenset({"crossref"}), client=client)
    page = await service.search("optics", ("crossref",), 5, page=2)
    assert calls[0].url.params["offset"] == "5"
    assert page["hits"][0]["records"][0]["relationships"] == [
        {"relationship": "is-preprint-of", "doi": "10.1234/final"},
    ]
    assert await service.search("optics", ("crossref",), 5, page=2) == page
    assert (await service.search("optics", ("crossref",), 5, page=3))["source_status"] == {
        "crossref": "SOURCE_REQUEST_QUOTA_EXHAUSTED",
    }
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_supplement_links_are_bounded_same_policy(tmp_path):
    from photomatagent.scientific.capabilities.literature.source_fetch import LiteratureFetchSourceTool

    page = """<html><head><meta name='citation_title' content='Example'>
        <meta name='citation_doi' content='10.1234/example'></head><body><article>
        <a href='supplementary/file.pdf'>Supplementary Information</a>
        <a href='https://localhost/file.pdf'>Supplementary Information</a>
        <a href='https://[invalid'>Supplementary Information</a></article></body></html>"""
    tool = LiteratureFetchSourceTool(
        documents=DocumentService(Workspace(tmp_path), fixture_document),
        transport=httpx.MockTransport(lambda req: httpx.Response(200, text=page,
                                      headers={"content-type": "text/html"})),
    )
    result = await tool.execute({"url": "https://www.nature.com/articles/example", "content_role": "paper"})
    assert result.data["supplement_links"] == ["https://www.nature.com/articles/supplementary/file.pdf"]
