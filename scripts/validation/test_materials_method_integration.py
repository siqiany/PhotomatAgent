"""Offline regressions for the materials-method acquisition workflow."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from photomatagent.application.reproduction import (
    NumericReference,
    ReproductionService,
    ReproductionSpec,
)
from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend
from photomatagent.capability_acquisition.models import EnvironmentRecord, ExecutionRecord
from photomatagent.capability_acquisition.method_discovery import (
    MethodDiscoveryService, MethodInspectTool, MethodSearchExternalTool, MethodSearchTool,
)
from photomatagent.capability_acquisition.repository_sources import FetchRepositoryScriptTool, RepositorySearchService
from photomatagent.capability_acquisition.sandbox import SandboxService
from photomatagent.capability_acquisition.web_sources import PublicSourceClient, SourceError, WebSearchService
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.image_assets import load_latest_images, register_image
from photomatagent.models.openai import OpenAIStreamMapper, openai_input
from photomatagent.models.types import AssistantMessage, ModelCompleted, UserMessage
from photomatagent.runtime.context_engine import _working_copy
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AutoApproveHandler, PermissionDecision, default_permission_policy
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.literature import (
    LiteratureExtractEvidenceTool,
    LiteratureReadObjectTool,
    LiteratureReadPaperTool,
    LiteratureSearchLocalTool,
    LiteratureServices,
)
from photomatagent.scientific.capabilities.literature.documents import (
    DocumentObject,
    DocumentService,
    ParsedDocument,
    structured_index_path,
)
from photomatagent.scientific.capabilities.literature.models import DocumentStatus, LiteratureSourceKind
from photomatagent.scientific.capabilities.literature.qdrant_store import SearchCandidate, document_id_for
from photomatagent.scientific.capabilities.literature.retrieval import LiteratureRetriever
from photomatagent.scientific.capabilities.literature.federated import FederatedLiterature
from photomatagent.scientific.capabilities.literature.source_fetch import LiteratureFetchSourceTool
from photomatagent.scientific.evolution.targeting import TargetConstraintDraft, _eqe_bounds_conflict
from photomatagent.scientific.state import ScientificState
from photomatagent.sessions.store import load_session_snapshot, save_session_snapshot
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace


def fixture_document(path):
    assert path.read_bytes().startswith(b"%PDF-")
    anchors = [{"page": 1, "bbox": {"l": 2, "t": 3, "r": 4, "b": 5}},
               {"page": 2, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}]
    return ParsedDocument(title="Film reference", objects=[
        DocumentObject(object_id="methods", kind="heading", section_id="methods",
                       text="Methods", anchors=anchors[:1], quality="anchored"),
        DocumentObject(object_id="table", kind="table", section_id="methods",
                       text="The detector measured 0.82 A/W responsivity at 80 K.",
                       anchors=anchors, quality="anchored"),
        DocumentObject(object_id="formula", kind="equation", section_id="methods",
                       text="R=((n1-n2)/(n1+n2))^2", anchors=anchors, quality="anchored"),
    ])


@pytest.mark.asyncio
async def test_read_paper_query_scopes_semantic_index_and_paginates_objects(tmp_path):
    workspace = Workspace(tmp_path)

    def parse(path):
        parsed = fixture_document(path)
        parsed.objects[1].text *= 10
        return parsed

    documents = DocumentService(workspace, parse)
    artifact = documents.register_pdf(b"%PDF-fixture", "https://arxiv.org/pdf/example")
    revision = (await documents.read(document_id=artifact.document_id))["revision"]
    query = "optical current conversion"
    indexed_id = document_id_for("ws", structured_index_path(artifact.document_id, revision).as_posix())
    generation = object()
    reverse = False
    reranker_available = True

    def candidate(object_id, text, **changes):
        return SearchCandidate(object_id, 1.0, {
            "record_type": "passage", "workspace_id": "ws", "ingest_state": "ready", "source_kind": "fulltext",
            "document_id": indexed_id, "document_revision": artifact.sha256, "text": text,
            "structured_ref": {"document_id": artifact.document_id, "revision": revision,
                               "object_id": object_id, "text_offset": 0},
        } | changes)

    class Store:
        async def resolve_current_generation(self):
            return generation

        async def get_document_manifests(self, workspace_id, document_ids, **kwargs):
            assert (workspace_id, document_ids) == ("ws", [indexed_id])
            assert kwargs == {"generation": generation, "source_kind": LiteratureSourceKind.FULLTEXT}
            return {indexed_id: SimpleNamespace(status=DocumentStatus.READY, content_sha256=artifact.sha256)}

        async def hybrid_candidates(self, text, vector, **kwargs):
            assert text == query and vector == [1.0]
            assert kwargs == {"workspace_id": "ws", "source_kind": LiteratureSourceKind.FULLTEXT, "limit": 50,
                              "document_id": indexed_id, "document_revision": artifact.sha256}
            return [candidate("table", "responsivity"), candidate("formula", "reflection"),
                    candidate("old", "old", structured_ref={"document_id": artifact.document_id,
                              "revision": "0" * 64, "object_id": "table", "text_offset": 0}),
                    candidate("foreign", "foreign", document_id="c" * 64),
                    candidate("abstract", "abstract", source_kind="abstract")]

    class Embedder:
        async def embed_query(self, text):
            return [1.0]

    class Reranker:
        async def rerank(self, text, passages, *, top_n):
            assert set(passages) == {"responsivity", "reflection"} and top_n == 2
            if not reranker_available:
                raise RuntimeError("reranker offline")
            return [SimpleNamespace(index=i, score=float((value == "responsivity") != reverse))
                    for i, value in enumerate(passages)]

    store = Store()
    tool = LiteratureReadPaperTool(ScientificConfig(), workspace, documents,
                                  LiteratureServices(None, LiteratureRetriever(store, Embedder(), Reranker()), store, "ws"))
    arguments = {"document_id": artifact.document_id, "query": query, "max_chars": 200}
    result = await tool.execute(arguments)
    assert not result.is_error and result.data["retrieval_route"] == "document_index"
    assert result.data["retrieval_mode"] == "hybrid_rrf" and result.data["indexed"] and result.data["reranked"]
    first = result.data["objects"][0]
    assert first["object_id"] == "table" and query not in first["text"]
    assert first["matched_ref"]["revision"] == revision
    cursor = result.data["cursor"]
    assert cursor
    second = await tool.execute(arguments | {"cursor": cursor})
    assert second.data["objects"][0]["text_start"] == len(first["text"])
    reverse = True
    stale = await tool.execute(arguments | {"cursor": cursor})
    assert stale.data["error"] == "DOCUMENT_CURSOR_INVALID_OR_STALE"
    reranker_available = False
    degraded = await tool.execute(arguments)
    assert degraded.data["retrieval_route"] == "document_index" and not degraded.data["reranked"]
    assert "reranker_unavailable" in degraded.data["degraded_reasons"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("unavailable", "reason", "indexed"), [
    (False, "DOCUMENT_INDEX_NOT_AVAILABLE", False), (True, "DOCUMENT_INDEX_QUERY_UNAVAILABLE", None),
])
async def test_read_paper_query_falls_back_without_coupling_explicit_selectors(tmp_path, unavailable, reason, indexed):
    workspace = Workspace(tmp_path)
    documents = DocumentService(workspace, fixture_document)
    artifact = documents.register_pdf(b"%PDF-fixture", "https://arxiv.org/pdf/example")
    calls = []

    class Store:
        async def resolve_current_generation(self):
            calls.append("index")
            if unavailable:
                raise OSError("Qdrant offline")
            return None

    tool = LiteratureReadPaperTool(ScientificConfig(), workspace, documents, LiteratureServices(None, None, Store(), "ws"))
    arguments = {"document_id": artifact.document_id}
    for selector in ({"section_id": "methods"}, {"object_ids": ["table"]}, {}):
        result = await tool.execute(arguments | selector)
        assert not result.is_error and result.data["objects"]
    assert not calls
    result = await tool.execute(arguments | {"query": "responsivity"})
    assert result.data["retrieval_route"] == "document_text_match" and result.data["indexed"] is indexed
    assert not result.data["reranked"] and reason in result.data["degraded_reasons"]
    assert [obj["object_id"] for obj in result.data["objects"]] == ["table"]


@pytest.mark.asyncio
async def test_arxiv_location_to_bounded_object_read_without_index(tmp_path):
    workspace = Workspace(tmp_path)
    config = ScientificConfig()
    service = DocumentService(workspace, fixture_document)
    target = (await FederatedLiterature(frozenset()).resolve("arXiv:2401.01234v2"))["locations"][0]
    reader = LiteratureFetchSourceTool(documents=service, transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=b"%PDF-fixture",
                                   headers={"Content-Type": "application/pdf"})))
    fetched = await reader.execute({"url": target["url"]})
    assert fetched.data["downloaded"] and not fetched.data["parsed"]
    document_id = fetched.data["document_id"]
    catalog = await LiteratureReadPaperTool(config, workspace, service).execute({"document_id": document_id})
    assert catalog.data["indexed"] is None
    assert catalog.data["retrieval_route"] == "catalog"
    assert "text" not in next(obj for obj in catalog.data["objects"] if obj["kind"] == "table")

    read = await LiteratureReadObjectTool(config, workspace, service).execute(
        {"document_id": document_id, "object_ids": ["table", "formula"]})
    assert read.data["sha256"] == hashlib.sha256(b"%PDF-fixture").hexdigest()
    assert [anchor["page"] for anchor in read.data["objects"][0]["anchors"]] == [1, 2]
    assert "n1" in read.data["objects"][1]["text"]
    assert not read.data["reranked"]
    evidence = await LiteratureExtractEvidenceTool(config, workspace, documents=service).execute({
        "passages": [{"document_id": document_id, "object_id": "table"},
                     {"text": "Target responsivity is 1 A/W.", "page": 99}],
    })
    anchored, raw = evidence.evidence
    assert anchored.provenance["source_sha256"] == read.data["sha256"]
    assert raw.provenance["source_kind"] == "UNATTESTED_TEXT"
    assert raw.assessment_role == "proposal"
    assert raw.method == "target or proposed value"
    assert "page" not in raw.provenance


@pytest.mark.asyncio
async def test_legacy_paper_entry_path_replacement_and_pending_job(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    workspace = Workspace(local)
    outside = tmp_path / "papers"
    outside.mkdir()
    (outside / "outsider.pdf").write_bytes(b"%PDF-outside")
    monkeypatch.chdir(tmp_path)
    tool = LiteratureReadPaperTool(ScientificConfig(), workspace, DocumentService(workspace, fixture_document))
    assert (await tool.execute({"file": "outsider.pdf"})).is_error
    papers = local / "papers"
    papers.mkdir()
    (papers / "link.pdf").symlink_to(outside / "outsider.pdf")
    assert (await tool.execute({"file": "link.pdf"})).is_error
    assert (await tool.execute({"file": str(outside / "outsider.pdf")})).is_error
    filename = papers / "film.pdf"
    filename.write_bytes(b"%PDF-fixture")
    config = ScientificConfig()
    def slow_parse(path):
        time.sleep(0.04)
        return fixture_document(path)
    service = DocumentService(workspace, slow_parse, wait_seconds=0.01)
    search = LiteratureSearchLocalTool(config, workspace, service)
    assert (await search.execute({"query": "responsivity"})).data["matches"][0]["parsed"] is False
    old = LiteratureReadPaperTool(config, workspace, service)
    first = await old.execute({"file": filename.name, "max_chars": 200})
    assert first.data["parse_state"] == "RUNNING"
    await service.aclose()
    second = await old.execute({"document_id": first.data["document_id"]})
    assert second.data["parse_state"] == "READY"
    assert (await search.execute({"query": "responsivity"})).data["matches"][0]["parsed"] is True
    filename.write_bytes(b"%PDF-second")
    assert (await old.execute({"document_id": first.data["document_id"]})).data["error"] == "DOCUMENT_REVISION_CHANGED"


@pytest.mark.asyncio
async def test_public_request_pins_address_and_blocks_redirect_to_private():
    calls = []
    async def resolver(host):
        return ["1.1.1.1"] if host == "good.test" else ["127.0.0.1"]
    def response(req):
        calls.append(req)
        return httpx.Response(302, headers={"Location": "https://other.test/data"})
    client = PublicSourceClient(frozenset({"good.test", "other.test"}), resolver=resolver,
                                transport=httpx.MockTransport(response))
    with pytest.raises(SourceError, match="PRIVATE_OR_UNROUTABLE_ADDRESS"):
        await client.get("https://good.test/data")
    assert len(calls) == 1
    assert calls[0].url.host == "1.1.1.1"
    assert calls[0].headers["Host"] == calls[0].extensions["sni_hostname"] == "good.test"


@pytest.mark.asyncio
async def test_federated_search_reports_rate_limit_missing_source_and_deduplicates():
    async def resolver(host):
        return ["1.1.1.1"]
    def response(req):
        host = req.headers["Host"]
        if host == "api.semanticscholar.org":
            return httpx.Response(429)
        if host == "api.crossref.org":
            return httpx.Response(200, json={"message": {"items": [
                {"DOI": "10.1234/example", "title": ["Paper"],
                 "type": "journal-article", "published": {"date-parts": [[2026]]}}]}})
        return httpx.Response(200, json={"results": [{"id": "https://openalex.org/W1",
                              "display_name": "Paper", "doi": "https://doi.org/10.1234/example"}]})
    client = PublicSourceClient(frozenset({"api.semanticscholar.org", "api.crossref.org", "api.openalex.org"}),
                                resolver=resolver, transport=httpx.MockTransport(response))
    service = FederatedLiterature(frozenset({"openalex", "crossref", "semantic_scholar"}), client)
    result = await service.search("thin film", ("openalex", "crossref", "semantic_scholar", "zenodo"), 5)
    assert len(result["hits"]) == 1
    assert len(result["hits"][0]["records"]) == 2
    assert result["source_status"] == {"openalex": "OK", "crossref": "OK",
                                       "semantic_scholar": "RATE_LIMITED", "zenodo": "NOT_CONFIGURED"}


class OfflinePodman(PodmanSandboxBackend):
    async def run(self, request, *, output_directory):
        self.requests.append(request)
        # Independent Fresnel reference fixture, with no external code execution.
        reflected = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        now = datetime.now(UTC)
        return ExecutionRecord(execution_id=request.execution_id, backend="podman", state="COMPLETED",
                               exit_code=0, stdout=json.dumps({"R": reflected, "unit": "1"}),
                               environment_sha256="b" * 64, script_sha256=request.script_digest,
                               started_at=now, ended_at=now)


@pytest.mark.asyncio
async def test_pinned_source_freezes_once_and_does_not_claim_applicability(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_GITHUB_DISCOVERY", "1")
    workspace = Workspace(tmp_path)
    source = b'print("reference script fixture")\n'
    async def resolver(host):
        return ["1.1.1.1"]
    client = PublicSourceClient(frozenset({"raw.githubusercontent.com"}), resolver=resolver,
                                transport=httpx.MockTransport(lambda _: httpx.Response(
                                    200, content=source, headers={"Content-Type": "text/plain"})))
    downloaded = await FetchRepositoryScriptTool(workspace, "tmp/trial", client).execute({
        "owner": "example", "repository": "optics", "commit": "a" * 40, "file": "example.py",
        "sha256": hashlib.sha256(source).hexdigest(), "license": "MIT",
        "license_source": "https://github.com/example/optics/blob/main/LICENSE",
    })
    assert downloaded.data["commit"] == "a" * 40
    assert downloaded.data["license_verified"] is False
    backend = OfflinePodman(tmp_path / "environments")
    backend.requests = []
    sandbox = SandboxService(workspace=tmp_path, trial_id="trial", genome_sha256="c" * 64,
                             scripts={}, backend=backend, dynamic_script_prefix="tmp/trial")
    environment_id = "env_" + "a" * 24
    sandbox.register_environment(EnvironmentRecord(
        environment_id=environment_id, plan_sha256="a" * 64, status="READY",
        lock_ref="lock", installed_manifest_ref="manifest", entrypoints={}, backend="podman-rootless",
        created_at=datetime.now(UTC), environment_sha256="b" * 64))
    spec = ReproductionSpec(
        source="https://example.com/fresnel-reference", paper_anchor="Normal incidence, air to glass",
        source_commit="a" * 40, implementation="third_party", script_path=downloaded.data["script_path"],
        environment_id=environment_id, reference=NumericReference(
            metric="R", expected=0.04, absolute_tolerance=1e-12, unit="1", source_anchor="Test reference"))
    disabled = SandboxService(workspace=tmp_path, trial_id="trial", genome_sha256="c" * 64,
                              scripts={}, dynamic_script_prefix="tmp/trial")
    unavailable = ReproductionService(workspace, disabled).prepare(spec)
    assert unavailable["execution"] == "BLOCKED"
    service = ReproductionService(workspace, sandbox)
    plan = service.prepare(spec)
    done = await service.execute(plan["plan_id"], plan["plan_sha256"])
    assert done["execution"] == "SUCCEEDED"
    assert done["checker"]["result"] == "PASS"
    assert done["applicability"] == "UNKNOWN"
    assert (await service.execute(plan["plan_id"], plan["plan_sha256"])) == done
    assert len(backend.requests) == 1
    assert ReproductionService(workspace, disabled).inspect(run_id=plan["run_id"]) == done
    with pytest.raises(FileNotFoundError):
        ReproductionService(workspace, None).inspect(plan["plan_id"])
    changed = service.prepare(spec.model_copy(update={"source_commit": "b" * 40}))
    assert changed["plan_id"] != plan["plan_id"]
    assert changed["checker"]["result"] == "UNKNOWN"


@pytest.mark.asyncio
async def test_method_tool_runs_through_default_registry_and_frozen_surface_has_no_network(tmp_path):
    workspace = Workspace(tmp_path)
    state = ScientificState()
    model = FakeModelProvider([scripted_tool_call("method_search", {"query": "分析多层薄膜光谱"}),
                               FakeResponse(text="候选方法已检索")])
    registry = create_default_registry(state, workspace)
    runtime = AgentRuntime(model=model, tools=registry, workspace=workspace, scientific_state=state,
                           permission_policy=default_permission_policy())
    events = [event async for event in runtime.run("分析多层薄膜光谱")]
    assert any(event.kind == "tool_completed" and event.tool_name == "method_search" for event in events)
    results = [message for message in runtime.conversation_state.messages
               if message.kind == "tool_result"]
    assert "transfer-matrix" in results[0].content
    assert state.method_needs[-1].status == "searched"
    frozen = create_default_registry(ScientificState(), workspace, evaluation_isolation=True)
    assert not {"method_search", "web.search", "reproduction.execute"} & {tool.name for tool in frozen.list_tools()}
    policy = default_permission_policy()
    assert (await policy.check("web.search", {})).decision == PermissionDecision.ASK
    assert (await policy.check("reproduction.execute", {})).decision == PermissionDecision.ASK


@pytest.mark.asyncio
async def test_method_service_searches_real_adapters_and_inspects_metadata_leads(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_GITHUB_DISCOVERY", "1")
    calls = []

    async def resolve(_host):
        return ["1.1.1.1"]

    def respond(request):
        host = request.headers["Host"]
        calls.append(host)
        if host == "web.example":
            return httpx.Response(200, json={"results": [
                {"url": "https://official.example/docs", "title": "Optics documentation", "snippet": "layer model"},
            ]})
        if host == "api.github.com":
            return httpx.Response(200, json={"items": [
                {"private": False, "full_name": "optics/tool", "html_url": "https://github.com/optics/tool",
                 "license": {"spdx_id": "MIT"}},
                {"private": False, "full_name": "optics/rejected", "html_url": "https://github.com/optics/rejected",
                 "license": {"spdx_id": "GPL-3.0"}},
            ]})
        assert host == "api.crossref.org"
        return httpx.Response(200, json={"message": {"items": [
            {"DOI": "10.1234/fixture", "title": ["Thin-film model"]},
        ]}})

    def client(host):
        return PublicSourceClient(frozenset({host}), resolver=resolve, transport=httpx.MockTransport(respond))

    service = MethodDiscoveryService(
        Workspace(tmp_path), web=WebSearchService("https://web.example/search", client=client("web.example")),
        repositories=RepositorySearchService(client("api.github.com")),
        literature=FederatedLiterature(frozenset({"crossref"}), client("api.crossref.org")),
    )
    args = {"query": "thin film spectrum", "allowed_licenses": ["MIT"], "regimes": ["laterally_patterned_film"]}
    local = await MethodSearchTool(service).execute(args)
    assert not calls
    assert local.data["excluded"][0]["reason"] == "SCIENTIFIC_REGIME_INCOMPATIBLE"
    remote = MethodSearchExternalTool(service)
    args["sources"] = ["web", "github", "pypi", "crossref", "openalex"]
    result = (await remote.execute(args)).data
    assert set(calls) == {"web.example", "api.github.com", "api.crossref.org"}
    assert result["source_status"]["pypi"] == result["source_status"]["openalex"] == "NOT_CONFIGURED"
    assert [row["source"] for row in result["implementations"]] == ["web", "github"]
    assert result["literature_evidence_leads"][0]["record"]["doi"] == "10.1234/fixture"
    assert any(row["reason"] == "LICENSE_NOT_ALLOWED" for row in result["excluded"])
    lead = result["implementations"][1]
    inspected = await MethodInspectTool(service).execute({"method_id": lead["method_id"]})
    assert inspected.data["record"]["license_hint"] == "MIT"
    assert inspected.data["applicability"] == "UNKNOWN"
    assert (await remote.execute(args)).data == result and len(calls) == 3
    assert (await default_permission_policy().check(remote.name, args)).decision == PermissionDecision.ASK


@pytest.mark.asyncio
async def test_runtime_records_missing_tool_need_and_versioned_decision(tmp_path):
    workspace = Workspace(tmp_path)
    state = ScientificState()
    registry = create_default_registry(state, workspace)
    model = FakeModelProvider([
        scripted_tool_call("tool_search", {"query": "zzqmethodunknownzz"}),
        scripted_tool_call("tool_call", {"name": "decision.record", "arguments": {
            "question": "Which method applies to a core-shell dot?",
            "alternatives": [{"description": "Compare a multiband solver with a measured series"}],
            "method_need": {"query": "core-shell quantum dot", "regimes": ["core_shell_dot"]},
            "method_need_basis": "Homogeneous-sphere parameters do not describe the shell.",
        }}),
        FakeResponse(text="Method selection remains open."),
    ])
    runtime = AgentRuntime(model=model, tools=registry, workspace=workspace, scientific_state=state,
                           permission_policy=default_permission_policy(), approval_handler=AutoApproveHandler())
    events = [event async for event in runtime.run("Check quantum dot conditions")]
    assert any(event.kind == "scientific_state_updated" for event in events)
    assert len(state.decisions) == 1
    assert {row.basis for row in state.method_needs} == {
        "NO_REGISTERED_TOOL_MATCH", "Homogeneous-sphere parameters do not describe the shell.",
    }
    save_session_snapshot(tmp_path / "session", conversation=runtime.conversation_state, scientific=state)
    restored = load_session_snapshot(tmp_path / "session")
    assert restored.scientific.decisions == state.decisions
    assert restored.scientific.method_needs == state.method_needs


def test_deepseek_continuation_round_trip_and_image_asset(tmp_path):
    model = "deepseek-flash"
    mapper = OpenAIStreamMapper(model)
    events = mapper.feed({"type": "response.completed", "response": {"output": [
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "private carry"}]},
        {"type": "function_call", "id": "fc1", "call_id": "call1", "name": "echo", "arguments": "{}"},
    ]}})
    completed = next(event.response for event in events if isinstance(event, ModelCompleted))
    assert completed.continuation is not None
    assert "private carry" not in repr(completed)
    from PIL import Image
    workspace = Workspace(tmp_path)
    path = workspace.user_output_dir / "asset.png"
    Image.new("RGB", (8, 8), "blue").save(path)
    reference = register_image(workspace, "user_output/asset.png")
    conversation = ConversationState(messages=[
        UserMessage(content="Read this", images=(reference,)),
        AssistantMessage(tool_calls=completed.tool_calls, continuation=completed.continuation),
    ])
    session = tmp_path / "session"
    save_session_snapshot(session, conversation=conversation, scientific=ScientificState())
    recovered = load_session_snapshot(session).conversation.messages
    payload = openai_input(recovered, model=model, image_assets=load_latest_images(workspace, recovered))
    assert payload[0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert payload[1] == {"type": "reasoning", "content": "private carry"}
    assert not any(item.get("type") == "reasoning" for item in openai_input(
        recovered, model="another", image_assets=load_latest_images(workspace, recovered)))
    assert "base64" not in (session / "session_state.json").read_text()
    working, _ = _working_copy([*recovered, UserMessage(content="New turn")])
    assert not any(isinstance(item, AssistantMessage) and item.continuation for item in working)
    assert UserMessage.model_validate({"kind": "user", "content": "old"}).model_dump() == {
        "kind": "user", "content": "old"}


def test_eqe_lower_bounds_intersect_and_opposite_bounds_can_conflict():
    def make(op, value, unit):
        return TargetConstraintDraft(property="eqe", operator=op, value=value, unit=unit,
                                     basis="EXPLICIT_GOAL", rationale="From request", confidence=1)
    assert not _eqe_bounds_conflict((make("ge", 20, "%"), make("ge", 0.3, "fraction")))
    assert _eqe_bounds_conflict((make("gt", 30, "%"), make("le", 0.3, "fraction")))
    assert not _eqe_bounds_conflict((make("ge", 30, "%"), make("le", 0.3, "fraction")))


@pytest.mark.asyncio
async def test_process_proposal_survives_runtime_checkpoint(tmp_path):
    workspace = Workspace(tmp_path)
    state = ScientificState()
    proposal = {
        "decision_question": "Can this substrate support the film process?",
        "sample": "film on substrate",
        "steps": [{"operation": "deposition", "purpose": "Set up a thickness control",
                   "parameters": [{"name": "thickness", "unit": "nm", "bounds": [50, 60],
                                   "basis": "PROPOSED"}]}],
        "missing_inputs": ["substrate and thermal budget"],
    }
    tool_call = {"name": "process.register_plan", "arguments": proposal}
    model = FakeModelProvider([
        scripted_tool_call("tool_describe", {"name": "process.register_plan"}),
        scripted_tool_call("tool_call", tool_call),
        scripted_tool_call("tool_call", tool_call),
        FakeResponse(text="Saved proposal with unresolved inputs."),
    ])
    runtime = AgentRuntime(model=model, tools=create_default_registry(state, workspace),
                           workspace=workspace, scientific_state=state,
                           permission_policy=default_permission_policy(),
                           approval_handler=AutoApproveHandler())
    _events = [event async for event in runtime.run("Record the proposed film process.")]
    assert len(runtime.scientific_state.process_plans) == 1
    plan = runtime.scientific_state.process_plans[0]
    assert plan.missing_inputs == ("substrate and thermal budget",)
    save_session_snapshot(tmp_path / "session", conversation=runtime.conversation_state, scientific=state)
    recovered = load_session_snapshot(tmp_path / "session")
    assert recovered.scientific.process_plans[0].id == plan.id
    assert not recovered.scientific.evidence


@pytest.mark.asyncio
async def test_runtime_sends_image_and_deepseek_reasoning_past_dynamic_state(tmp_path):
    from PIL import Image
    from photomatagent.models.openai import OpenAIProvider

    workspace = Workspace(tmp_path)
    Image.new("RGB", (8, 8), "blue").save(workspace.user_output_dir / "asset.png")

    class Responses:
        def __init__(self):
            self.requests = []

        async def create(self, **kwargs):
            self.requests.append(kwargs)
            first_request = len(self.requests) == 1
            async def stream():
                if first_request:
                    yield {"type": "response.completed", "response": {"output": [
                        {"type": "reasoning", "content": "Carry one"},
                        {"type": "function_call", "id": "fc1", "call_id": "call1", "name": "method_search",
                         "arguments": '{"query":"thin film reflectance"}'},
                    ]}}
                else:
                    yield {"type": "response.output_text.delta", "delta": "Finished."}
                    yield {"type": "response.completed", "response": {"output": []}}
            return stream()

    class Client:
        base_url = "https://deepseek-fixture.invalid/"
        responses = Responses()

    state = ScientificState()
    client = Client()
    runtime = AgentRuntime(model=OpenAIProvider("deepseek-flash", client=client),
                           tools=create_default_registry(state, workspace),
                           workspace=workspace, scientific_state=state,
                           permission_policy=default_permission_policy())
    events = [event async for event in runtime.run("Read the image.", image_paths=("user_output/asset.png",))]
    assert any(event.kind == "tool_completed" and event.tool_name == "method_search" for event in events)
    first, second = [request["input"] for request in client.responses.requests]
    assert any(isinstance(item.get("content"), list) and item["content"][1]["type"] == "input_image"
               for item in first)
    assert {"type": "reasoning", "content": "Carry one"} in second
    assert any(item["type"] == "function_call_output" for item in second if "type" in item)
    assert not any("base64" in message.model_dump_json() for message in runtime.conversation_state.messages)
