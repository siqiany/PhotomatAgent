"""Acceptance contracts for managed references and sample-bound process proposals."""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from photomatagent.application.reproduction import ReproductionService, ReproductionSpec
from photomatagent.capability_acquisition.method_discovery import MethodDiscoveryService, MethodInspectTool, MethodNeed
from photomatagent.runtime.context import format_scientific_state
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.literature import LiteratureExtractEvidenceTool
from photomatagent.scientific.capabilities.literature.documents import DocumentObject, DocumentService, ParsedDocument
from photomatagent.scientific.process import ProcessPlan
from photomatagent.scientific.state import EvidenceAttestation, ScientificState
from photomatagent.tools.process_plan import RegisterDecisionTool, RegisterProcessPlanTool
from photomatagent.tools.scientific_state_inspect import ScientificStateInspectTool
from photomatagent.workspace import Workspace


REFERENCE = "The band gap was measured as 0.82 eV."
TOLERANCE = "The absolute comparison tolerance was 0.01 eV."
METHODS = "Sample A was annealed at 300–320 K for 15 min. Sample B was annealed at 350 K."
AMBIGUOUS = "Samples A and B were annealed at 300 K and 350 K, respectively."


async def document_fixture(workspace):
    def parse(_):
        return ParsedDocument(title="Synthetic process reference", objects=[
            DocumentObject(object_id=key, kind="paragraph", text=text, anchors=[{"page": 1}], quality="anchored")
            for key, text in (("value", REFERENCE), ("tolerance", TOLERANCE), ("methods", METHODS), ("mixed", AMBIGUOUS))
        ])
    service = DocumentService(workspace, parse)
    artifact = service.register_pdf(b"%PDF-reference", "https://arxiv.org/pdf/2401.01234v1")
    doc = await service.read(document_id=artifact.document_id)
    return service, artifact, doc["revision"]


async def test_reproduction_managed_value_tolerance_and_asset_bytes_are_frozen(tmp_path):
    workspace = Workspace(tmp_path)
    _, artifact, revision = await document_fixture(workspace)
    scratch = tmp_path / "tmp/reproduction"
    scratch.mkdir(parents=True)
    (scratch / "main.py").write_text("print('smoke')\n")
    data = scratch / "weights.bin"
    data.write_bytes(b"fixture")
    def anchor(object_id, sentence):
        return {"document_id": artifact.document_id, "revision": revision, "source_sha256": artifact.sha256,
                "object_id": object_id, "sentence": sentence}
    arguments = {
        "source": artifact.source_url, "paper_anchor": "Figure 1", "target": "Figure 1 band gap",
        "implementation": "third_party", "source_commit": "a" * 40, "requested_scope": "PAPER_FULL",
        "script_path": "tmp/reproduction/main.py", "environment_id": "env_" + "a" * 24,
        "argv": ["--seed", "0"], "seed": 0, "seed_flag": "--seed",
        "environment_profile": "cuda", "input_refs": {"weights": "tmp/reproduction/weights.bin"},
        "assets": [{"input_key": "weights", "role": "weights", "source_url": "https://example.org/weights.bin",
                    "sha256": hashlib.sha256(b"fixture").hexdigest(), "size_bytes": 7, "version": "v1",
                    "license": "CC0-1.0", "acquired_at": datetime.now(UTC)}],
        "reference": {"metric": "bandgap", "expected": 0.82, "absolute_tolerance": 0.01, "unit": "eV",
                      "source_anchor": "Figure 1", "managed_value": anchor("value", REFERENCE),
                      "managed_tolerance": anchor("tolerance", TOLERANCE)},
    }
    service = ReproductionService(workspace, None)
    spec = ReproductionSpec.model_validate(arguments)
    plan = service.prepare(spec)
    assert plan["scope"] == "PAPER_SUBSET" and plan["paper_reproduction"] == "UNKNOWN"
    assert plan["reference_authority"] == "MANAGED_VALUE_AND_TOLERANCE"
    assert plan["diagnostic"] == "CUDA_PROFILE_UNAVAILABLE"
    assert service.inspect(plan["plan_id"]) == plan
    supplied = service.prepare(spec.model_copy(update={"reference": spec.reference.model_copy(update={
        "managed_value": None, "managed_tolerance": None, "absolute_tolerance": 0.5,
    })}))
    assert supplied["scope"] == "EXAMPLE" and supplied["plan_id"] != plan["plan_id"]
    for changed, error in (
        ({"expected": 0.81}, "REPRODUCTION_REFERENCE_VALUE_NOT_BOUND"),
        ({"absolute_tolerance": 0.1}, "REPRODUCTION_TOLERANCE_NOT_BOUND"),
        ({"managed_value": spec.reference.managed_value.model_copy(update={"sentence": "Invented."})},
         "REPRODUCTION_REFERENCE_SENTENCE_NOT_BOUND"),
    ):
        with pytest.raises(ValueError, match=error):
            service.prepare(spec.model_copy(update={"reference": spec.reference.model_copy(update=changed)}))
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="REPRODUCTION_ASSET_BYTES_CHANGED"):
        service.prepare(spec)
    workspace.resolve(artifact.path).write_bytes(b"%PDF-changed")
    recovered = service.inspect(plan["plan_id"])
    assert recovered["scope"] == "EXAMPLE" and recovered["reference_authority"] == "MANAGED_REFERENCE_UNAVAILABLE"


async def test_process_branches_bind_sample_range_and_recheck_evidence_after_revision(tmp_path):
    workspace = Workspace(tmp_path)
    documents, artifact, _ = await document_fixture(workspace)
    state = ScientificState()
    extractor = LiteratureExtractEvidenceTool(ScientificConfig(), workspace, documents=documents)
    result = await extractor.execute({"passages": [
        {"document_id": artifact.document_id, "object_id": key} for key in ("methods", "mixed")
    ]})
    authority = object()
    state._bind_runtime_authority(authority)
    for item in result.evidence:
        state.add_evidence(item)
        state._attest_evidence(EvidenceAttestation(
            evidence_id=item.id, authority="observation", origin="trusted_builtin",
            tool_name=extractor.name, tool_call_id="fixture",
        ), capability=authority)
    row = next(item for item in result.evidence if item.property == "temperature" and item.conditions.get("sample_label") == "A")
    assert row.value == [300, 320]
    assert not any(item.conditions.get("sample_label") for item in result.evidence if item.provenance["object_id"] == "mixed")
    parameter = {"name": "temperature", "unit": "K", "bounds": [300, 320], "basis": "REPORTED",
                 "evidence_id": row.id, "sample_id": "a", "source_sample": "A"}
    plan = {"decision_question": "Can the reference anneal be used?", "sample": "film A",
            "samples": [{"sample_id": "a", "label": "A", "role": "input"},
                        {"sample_id": "annealed", "label": "A after anneal", "role": "intermediate"}],
            "steps": [
                {"step_id": "heat", "operation": "anneal", "purpose": "Reference baseline",
                 "checkpoint": "Inspect film integrity", "input_samples": ["a"], "output_samples": ["annealed"],
                 "parameters": [parameter]},
                {"step_id": "measure", "operation": "measurement", "purpose": "Measure intact film",
                 "input_samples": ["annealed"]},
                {"step_id": "quarantine", "operation": "other", "purpose": "Stop on delamination",
                 "input_samples": ["annealed"]},
            ],
            "transitions": [{"source": "heat", "target": "measure", "outcome": "pass", "condition": "Film intact"},
                            {"source": "heat", "target": "quarantine", "outcome": "fail", "condition": "Film delaminates"}]}
    tool = RegisterProcessPlanTool(state, workspace)
    accepted = await tool.execute(plan)
    assert accepted.data["scientific_validation"] == "UNKNOWN" and accepted.data["conditional"]
    assert "thermal_budget" in accepted.data["missing_inputs"]
    state.add_process_plan(accepted.state_updates[0])
    combination = plan | {"steps": [plan["steps"][0], plan["steps"][1] | {
        "operation": "anneal", "purpose": "Repeat the reported interval in a second treatment", "parameters": [parameter | {
            "sample_id": "annealed",
        }],
    }], "transitions": [plan["transitions"][0]], "samples": [
        plan["samples"][0], plan["samples"][1] | {"label": "A"},
    ], "proposal": {
        "kind": "VALIDATION", "baseline": "Single reported anneal", "change": "Combine two annealing stages",
        "rationale": "The individual interval is reported; the sequence remains a hypothesis.",
        "variables": ["Number of annealing stages"],
        "validation": {"control": "Single anneal", "procedure": "Apply each sequence to matched samples",
                       "metric": "Film integrity", "criterion": "Check whether damage increases in the repeated route",
                       "next_action": "Prepare matched samples"},
    }}
    combined = await tool.execute(combination)
    assert not combined.is_error and combined.data["proposal_delivery"] == "READY"
    assert combined.data["scientific_validation"] == "UNKNOWN"
    for changes in ({"source_sample": "B"}, {"source_conditions": {"bias_V": 2}}, {"bounds": [320, 320]}):
        bad = plan | {"steps": [plan["steps"][0] | {"parameters": [parameter | changes]}, *plan["steps"][1:]]}
        assert (await tool.execute(bad)).output == "PROCESS_PARAMETER_EVIDENCE_NOT_BOUND"
    transferred = parameter | {"basis": "TRANSFERRED", "transfer_assumption": "Compare substrate adhesion",
                               "control_experiment": "Keep a specimen on the original substrate"}
    revision = plan | {"previous_plan_id": accepted.data["plan_id"], "revision_basis": "Review adhesion constraint",
                       "steps": [plan["steps"][0] | {"parameters": [transferred]}, *plan["steps"][1:]]}
    assert (await tool.execute(revision)).data["plan_id"] != accepted.data["plan_id"]
    workspace.resolve(artifact.path).write_bytes(b"%PDF-changed")
    assert (await tool.execute(revision)).output == "PROCESS_PARAMETER_EVIDENCE_NOT_BOUND"


async def test_method_selection_uses_durable_provider_snapshot_and_controlled_terms(tmp_path):
    class Repositories:
        async def search(self, query, limit):
            return {"hits": [
                {"repository": "fixture/optics", "version": "1.2", "license_hint": "MIT", "platforms": ["linux"]},
                {"repository": "fixture/mac", "version": "1.0", "license_hint": "MIT", "platforms": ["macos"]},
                {"repository": "fixture/license", "license_hint": "GPL-3.0"},
            ]}

    workspace = Workspace(tmp_path)
    needs = MethodNeed(query="PbS +0.5 V", operation="多层薄膜光谱", known_inputs=("膜厚",), conditions=("PbS +0.5 V",),
                       required_platform="linux", allowed_licenses=("MIT",))
    discovered = await MethodDiscoveryService(workspace, repositories=Repositories()).search(needs, 5, ("github",))
    assert {row["method"]["method_id"] for row in discovered["candidates"]} >= {"transfer-matrix", "fdtd-thin-film"}
    assert all("film thickness" not in row["unconfirmed_inputs"] for row in discovered["candidates"])
    assert {row["reason"] for row in discovered["excluded"]} == {"PLATFORM_NOT_SUPPORTED", "LICENSE_NOT_ALLOWED"}
    recovered = MethodDiscoveryService(workspace)
    identifier = discovered["implementations"][0]["method_id"]
    assert (await recovered.inspect(identifier))["record"]["version"] == "1.2"
    tool = RegisterDecisionTool(ScientificState(), recovered)
    payload = {"question": "Which source should be checked?", "method_need": needs.model_dump(mode="json"),
               "method_need_basis": "Prefer Linux MIT metadata before inspecting the reference.",
               "method_disposition": "selected", "selected_method_id": identifier}
    selected = await tool.execute(payload)
    assert selected.state_updates[1].candidate_snapshot["record"]["repository"] == "fixture/optics"
    assert selected.state_updates[1].decision_id == selected.data["decision_id"]
    assert selected.state_updates[1].need.conditions == ("PbS +0.5 V",)
    assert (await tool.execute(payload | {"method_disposition": "not_needed", "selected_method_id": ""})).state_updates[1].status == "not_needed"
    changed = needs.model_copy(update={"required_platform": "macos"})
    rejected = await tool.execute(payload | {"method_need": changed.model_dump(mode="json")})
    searched = await MethodDiscoveryService(workspace, repositories=Repositories()).search(changed, 5, ("github",))
    exclusion = next(row for row in searched["excluded"] if row["method_id"] == identifier)
    assert rejected.is_error and not rejected.state_updates
    assert rejected.data["reason"] == exclusion["reason"] == "PLATFORM_NOT_SUPPORTED"


@pytest.mark.parametrize(("constraints", "reason"), [
    ({"fixed_unavailable_inputs": ["膜厚"]}, "REQUIRED_INPUT_FIXED_UNAVAILABLE"),
    ({"regimes": ["laterally_patterned_film"]}, "SCIENTIFIC_REGIME_INCOMPATIBLE"),
    ({"exclusions": ["Transfer matrix for stratified films"]}, "METHOD_EXPLICITLY_EXCLUDED"),
])
async def test_method_selection_rechecks_catalog_constraints(constraints, reason):
    service = MethodDiscoveryService()
    tool = RegisterDecisionTool(ScientificState(), service)
    payload = {"question": "Which multilayer method?", "method_disposition": "selected",
               "method_need_basis": "Compare two optical routes.", "selected_method_id": "transfer-matrix"}
    need = MethodNeed.model_validate({"query": "multilayer spectrum", **constraints})
    search = await service.search(need, 5)
    exclusion = next(row for row in search["excluded"] if row["method_id"] == "transfer-matrix")
    rejected = await tool.execute(payload | {"method_need": need.model_dump(mode="json")})
    assert rejected.is_error and not rejected.state_updates
    assert rejected.data["reason"] == exclusion["reason"] == reason

    resolved = MethodNeed(query=need.query, known_inputs=("film thickness",), required_platform="linux")
    selected = await tool.execute(payload | {"method_need": resolved.model_dump(mode="json")})
    assert not selected.is_error and selected.data["scientific_validation"] == "UNKNOWN"
    snapshot = selected.state_updates[1].candidate_snapshot
    assert snapshot["applicability"] == "UNKNOWN"
    assert snapshot["unconfirmed_inputs"] == ["complex refractive index spectra", "incidence and polarization"]
    assert snapshot["comparison"]["platform_to_verify"] == "linux"


async def test_unavailable_thickness_allows_condition_with_input_action():
    need = MethodNeed(query="multilayer spectrum", unavailable_inputs=("膜厚",))
    service = MethodDiscoveryService()
    tool = RegisterDecisionTool(ScientificState(), service)
    row = next(row for row in (await service.search(need, 5))["candidates"]
               if row["method"]["method_id"] == "transfer-matrix")
    assert row["execution_blocker"] == "REQUIRED_INPUT_UNAVAILABLE"
    args = {"question": "Which optical route?", "method_need": need.model_dump(mode="json"),
            "method_need_basis": "Compare layers.", "method_disposition": "selected",
            "selected_method_id": "transfer-matrix"}
    assert (await tool.execute(args)).data["diagnostic"] == "METHOD_SELECTION_INPUT_ACTION_REQUIRED"
    inspected = await MethodInspectTool(service).execute({"method_id": "transfer-matrix", "need": args["method_need"]})
    assert inspected.data["execution_blocker"] == row["execution_blocker"]
    selected = await tool.execute(args | {"method_input_actions": {"film thickness": "Measure or scan layer thickness."}})
    assert selected.data["conditional"] and selected.data["scientific_validation"] == "UNKNOWN"
    assert selected.state_updates[1].candidate_snapshot["unavailable_inputs"] == ["film thickness"]


@pytest.mark.parametrize("graph", [False, True])
async def test_process_anneal_gaps_are_reported_for_new_and_historical_shapes(graph):
    state = ScientificState()
    tool = RegisterProcessPlanTool(state)
    step = {"operation": "anneal", "purpose": "Improve film"}
    proposal = {"decision_question": "Prepare film", "sample": "film"}
    if graph:
        proposal["samples"] = [{"sample_id": "film", "label": "film", "role": "input"}]
        step.update({"step_id": "heat", "input_samples": ["film"]})
    proposal["steps"] = [step]
    original = ProcessPlan.model_validate(proposal)
    recovered = ProcessPlan.model_validate_json(original.model_dump_json())
    result = await tool.execute(proposal)
    assert not result.is_error and result.data["status"] == "PROPOSED"
    assert result.data["plan_id"] == recovered.id == original.id
    assert result.state_updates[0].model_dump_json() == recovered.model_dump_json()
    assert result.data["scientific_validation"] == "UNKNOWN"
    assert result.data["conditional"] and result.data["completeness"] == "INCOMPLETE"
    assert set(result.data["missing_inputs"]) == {"equipment", "temperature", "thermal_budget"}
    state.add_process_plan(recovered)
    inspection = await ScientificStateInspectTool(state).execute({"section": "process_plans"})
    assert inspection.data["items"][0]["missing_inputs"] == result.data["missing_inputs"]
    assert inspection.data["items"][0]["completeness"] == "INCOMPLETE"
    assert "missing inputs: equipment, temperature, thermal_budget" in format_scientific_state(state)

    complete = await tool.execute(proposal | {
        "requirements": {"thermal_budget": "Qualified substrate up to 350 K"},
        "steps": [step | {"equipment": "Dry tube furnace", "parameters": [
            {"name": "temperature", "unit": "K", "bounds": [300, 320], "basis": "PROPOSED"},
        ]}],
    })
    assert complete.data["missing_inputs"] == [] and not complete.data["conditional"]
    assert complete.data["completeness"] == "REQUIREMENTS_DECLARED"
    assert complete.data["scientific_validation"] == "UNKNOWN"


async def test_doi_relations_and_supplement_parent_survive_registry_restart(tmp_path):
    import httpx
    from photomatagent.scientific.capabilities.literature.federated import FederatedLiterature
    from photomatagent.scientific.capabilities.literature.source_fetch import LiteratureFetchSourceTool

    publication = "https://www.nature.com/articles/fixture"
    supplement = publication + "/supplement.pdf"
    preprint = "https://arxiv.org/pdf/2401.01234v1"
    doi = "10.1234/fixture"
    html = f"""<meta name="citation_doi" content="{doi}"><meta name="citation_title" content="Same title">
    <article>Paper Methods. <a href="{supplement}">Supplementary information</a></article>"""

    class Crossref:
        async def json(self, _url, **_kwargs):
            return {"message": {"items": [
                {"DOI": doi, "title": ["Same title"], "type": "journal-article",
                 "relation": {"has-preprint": [{"id-type": "doi", "id": "10.48550/arxiv.2401.01234"}]}},
                {"DOI": "10.1234/unrelated", "title": ["Same title"], "type": "journal-article"},
            ]}}

    def parse(_):
        return ParsedDocument(title="Same title", objects=[
            DocumentObject(object_id="methods", kind="paragraph", text=METHODS,
                           anchors=[{"page": 1}], quality="anchored"),
        ])

    def transport(request):
        if str(request.url) == publication:
            return httpx.Response(200, text=html, headers={"Content-Type": "text/html"})
        return httpx.Response(200, content=b"%PDF-fixture", headers={"Content-Type": "application/pdf"})

    workspace = Workspace(tmp_path)
    documents = DocumentService(workspace, parse)
    await FederatedLiterature(frozenset({"crossref"}), client=Crossref(), documents=documents).search("Same title", ("crossref",), 2)
    fetch = LiteratureFetchSourceTool(documents=documents, transport=httpx.MockTransport(transport))
    draft = await fetch.execute({"url": preprint})
    published = await fetch.execute({"url": publication, "content_role": "paper"})
    parent_id = published.data["document_id"]
    wrong = await fetch.execute({"url": supplement, "content_role": "supplement", "parent_document_id": draft.data["document_id"]})
    assert wrong.data["diagnostic"] == "SUPPLEMENT_NOT_LINKED_BY_PUBLICATION"
    si = await fetch.execute({"url": supplement, "content_role": "supplement", "parent_document_id": parent_id})
    restarted = DocumentService(workspace, parse)
    assert restarted.publication_context(si.data["document_id"]).parent_document_id == parent_id
    paper = await restarted.read(document_id=draft.data["document_id"])
    assert paper["publication"]["publication_id"] == "arxiv:2401.01234v1"
    assert paper["publication_relations"] == [{"source": "doi:" + doi, "relation": "has-preprint",
                                              "target": "doi:10.48550/arxiv.2401.01234", "provider": "crossref"}]
    assert restarted.publication_relations("doi:10.1234/unrelated") == []
    await restarted.read(document_id=si.data["document_id"])
    read = await LiteratureExtractEvidenceTool(ScientificConfig(), workspace, documents=restarted).execute({
        "passages": [{"document_id": si.data["document_id"], "object_id": "methods"}],
    })
    assert read.evidence[0].provenance["publication"]["parent_source_sha256"] == published.data["sha256"]


async def test_parse_job_restarts_once_for_two_document_services(tmp_path):
    import asyncio
    import json
    import time

    calls = []
    def parse(_):
        calls.append(1)
        time.sleep(0.05)
        return ParsedDocument(title="Recovered", objects=[
            DocumentObject(object_id="cell", kind="table", text="Local table", quality="anchored",
                           relations={"footnotes": ["note"]}, anchors=[{"page": 1}, {"page": 2}]),
        ])
    workspace = Workspace(tmp_path)
    left, right = (DocumentService(workspace, parse) for _ in range(2))
    artifact = left.register_pdf(b"%PDF-recovery", "https://arxiv.org/pdf/2401.01234v1")
    revision = left._revision(artifact)
    job = left._cache(f"parse-jobs/{revision}.json")
    job.parent.mkdir()
    job.write_text('{"state":"RUNNING"}')
    first, second = await asyncio.gather(*(service.read(document_id=artifact.document_id, object_ids=("cell",)) for service in (left, right)))
    assert calls == [1] and first == second
    assert first["objects"][0]["relations"]["footnotes"] == ["note"]
    assert first["objects"][0]["page_end"] == 2
    assert "DOCUMENT_PARSE_RESTARTED" in first["degraded_reasons"]
    assert json.loads(job.read_text()) == {"state": "READY", "restarted": True}


def test_optical_reference_has_independent_transfer_and_negative_checks(tmp_path):
    import cmath
    import json
    import math
    from photomatagent.application.capability_cases import cases
    from photomatagent.application.optical_reference import check_fresnel_film
    from photomatagent.application.plugin_contracts import interface_for, validate_interface
    from photomatagent.application.skills import CheckerSpec
    from jsonschema import ValidationError

    spec = CheckerSpec(checker_id="fresnel_film", output_path="result.json", field="R")
    input_path, output = tmp_path / "data.json", tmp_path / "result.json"
    for case in cases("fresnel_film", holdout=True):
        data = case["inputs"]["data"]
        contract = interface_for("fresnel_film", []).model_dump()
        if case["expected"] == "REJECT":
            with pytest.raises(ValidationError):
                validate_interface(contract, {"data": data})
            continue
        validate_interface(contract, {"data": data})
        n0, n1, n2 = (data[key] for key in ("n_incident", "n_film", "n_exit"))
        r01, r12 = (n0 - n1) / (n0 + n1), (n1 - n2) / (n1 + n2)
        phase = cmath.exp(4j * math.pi * n1 * data["thickness_nm"] / data["wavelength_nm"])
        expected = abs((r01 + r12 * phase) / (1 + r01 * r12 * phase)) ** 2
        input_path.write_text(json.dumps(data))
        result = {"R": expected, "unit": "1", "scientific_status": "UNKNOWN"}
        output.write_text(json.dumps(result))
        assert check_fresnel_film(spec, {"data": input_path}, output)["status"] == "PASS"
        output.write_text(json.dumps(result | {"R": expected + 0.1}))
        assert check_fresnel_film(spec, {"data": input_path}, output)["status"] == "FAIL"


def test_deepseek_sqlite_checkpoint_keeps_tool_pair_and_redacts_working_history(tmp_path):
    from photomatagent.application.store import WorkbenchStore
    from photomatagent.models.openai import openai_input
    from photomatagent.models.types import AssistantMessage, ProviderContinuation, ToolCall, ToolResultMessage, UserMessage
    from photomatagent.runtime.context_engine import _working_copy
    from photomatagent.runtime.state import ConversationState
    from photomatagent.sessions.sqlite_store import SessionRepository
    from photomatagent.sessions.store import SessionSnapshot

    path = tmp_path / "workbench.sqlite3"
    model = "deepseek-v4-fixture"
    endpoint = "e" * 64
    history = [UserMessage(content="Check a fixture"), AssistantMessage(
        tool_calls=[ToolCall(id="call_fixture", name="read_marker", arguments={})],
        continuation=ProviderContinuation(model=model, endpoint_key=endpoint, reasoning=("synthetic carry",)),
    ), ToolResultMessage(tool_call_id="call_fixture", tool_name="read_marker", content="marker")]
    original = WorkbenchStore(path)
    SessionRepository(original.database).checkpoint("fixture", SessionSnapshot(
        conversation=ConversationState(messages=history), scientific=ScientificState(),
    ))
    original.close()
    reopened = WorkbenchStore(path)
    try:
        restored = SessionRepository(reopened.database).load("fixture").conversation.messages
        current, _ = _working_copy(restored)
        encoded = openai_input(current, model=model, endpoint_key=endpoint)
        assert [item["type"] for item in encoded if "type" in item] == ["reasoning", "function_call", "function_call_output"]
        assert encoded[-2]["call_id"] == encoded[-1]["call_id"] == "call_fixture"
        wrong = openai_input(current, model=model, endpoint_key="f" * 64)
        assert not any(item.get("type") == "reasoning" for item in wrong)
        later, _ = _working_copy([*restored, UserMessage(content="New question")])
        assert not any(item.get("type") == "reasoning" for item in openai_input(later, model=model, endpoint_key=endpoint))
        assert SessionRepository(reopened.database).load("fixture").conversation.messages == history
    finally:
        reopened.close()


def test_reported_numeric_sign_is_preserved_for_reference_comparison():
    from photomatagent.scientific.capabilities.literature.evidence import extract_evidence_from_text

    rows = extract_evidence_from_text("Sample A dark current was measured as -0.82 A at -1 V.")
    assert rows[0].value == -0.82
    assert rows[0].conditions["bias_V"] == -1
    from photomatagent.application.reproduction import NumericReference

    with pytest.raises(ValueError, match="finite"):
        NumericReference(metric="dark_current", expected=-0.82, absolute_tolerance=float("inf"),
                         unit="A", source_anchor="Fixture")
