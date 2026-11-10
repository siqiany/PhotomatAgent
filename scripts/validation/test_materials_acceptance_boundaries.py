"""Regressions for explicit evidence substitution and solver domain boundaries."""
from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient
import pytest

from photomatagent.application.errors import ServiceError
from photomatagent.application.models import FeedbackRequest, RunRequest, TaskCreate
from photomatagent.capability_acquisition.method_discovery import MethodNeed, search_methods
from photomatagent.scientific.capabilities.kp import KdotpyRunTool
from photomatagent.scientific.capabilities.literature.documents import (
    DocumentError, DocumentObject, DocumentService, ParsedDocument, link_table_continuations,
)
from photomatagent.scientific.capabilities.optics.meep_thinfilm import MeepThinFilmTool
from photomatagent.scientific.capabilities.quantum_dot import BandgapBowingTool, BrusTransitionEnergyTool
from photomatagent.scientific.state import ScientificState
from photomatagent.runtime.state import ConversationState
from photomatagent.sessions.store import SessionSnapshot
from photomatagent.web.app import create_app
from photomatagent.workspace import Workspace

from test_materials_method_promotion import workbench


def test_cross_task_replacement_requires_a_successful_run_with_verified_inputs(tmp_path):
    host = workbench(tmp_path)
    store = host.store
    producer = host.create_task(TaskCreate(title="Upstream", goal="Evaluate reference evidence"))
    consumer = host.create_task(TaskCreate(title="Downstream", goal="Use two reference inputs"))
    stranger = host.create_task(TaskCreate(title="Unrelated", goal="Independent"))

    def run(task, key):
        return store.reserve_run(task, RunRequest(text=key), key).run_id

    def deliver(task, run_id, name, data):
        path = tmp_path / "user_output" / task / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        artifact_id = "artifact_" + name
        store.register_artifact(artifact_id=artifact_id, task_id=task, run_id=run_id,
                                path=path.relative_to(tmp_path).as_posix(), filename=path.name,
                                sha256=hashlib.sha256(data).hexdigest(), media_type="application/json", size_bytes=len(data))
        return artifact_id

    def feedback(source, name):
        plan, _ = store.reserve_revision_plan(
            producer, FeedbackRequest(source_run_id=source, text=name), name,
            source_digest="a" * 64, source_artifacts=[],
        )
        store.save_revision_plan(plan, status="READY", plan_json={"instruction": name, "issues": [
            {"status": "CORRECTION", "category": "TASK_DEFINITION", "original_feedback_span": name},
        ]})
        return plan

    def checkpoint(run_id):
        session = store.task(consumer).session_id
        host.storage.sessions.ensure(session, consumer)
        host.storage.sessions.checkpoint(
            session, SessionSnapshot(conversation=ConversationState(), scientific=ScientificState()),
            run_id=run_id, boundary="run_end",
        )

    original = run(producer, "original")
    store.finish_run(original, "SUCCEEDED")
    old_a = deliver(producer, original, "old_a", b'{"temperature":90}')
    old_b = deliver(producer, original, "old_b", b'{"bias":5}')
    url = f"/api/tasks/{consumer}/inputs"
    app = create_app(service=host, manager=host.manager)
    with TestClient(app) as client:
        a = client.post(url, json={"source_artifact_id": old_a}).json()["artifact_id"]
        b = client.post(url, json={"source_artifact_id": old_b}).json()["artifact_id"]
        history = run(consumer, "history")
        store.finish_run(history, "SUCCEEDED")
        checkpoint(history)
        correction = feedback(original, "Use 80 K and 1 V")
        assert store.run(history).result_invalidation_ids == [correction]

        repaired = store.reserve_revision_run(correction, RunRequest(text="Correct upstream data"), "upstream-repaired").run_id
        store.finish_run(repaired, "SUCCEEDED")
        new_a = deliver(producer, repaired, "new_a", b'{"temperature":80}')
        new_b = deliver(producer, repaired, "new_b", b'{"bias":1}')
        unrelated_run = run(stranger, "unrelated")
        store.finish_run(unrelated_run, "SUCCEEDED")
        foreign = deliver(stranger, unrelated_run, "foreign", b'{"temperature":80}')

        def replace(previous, source):
            return client.post(url, json={"source_artifact_id": source, "replaces_artifact_id": previous})

        assert replace(a, old_a).status_code == 409
        assert replace(a, foreign).status_code == 409
        bound_a = replace(a, new_a).json()["artifact_id"]
        assert replace(a, new_a).json()["artifact_id"] == bound_a
        assert replace(a, new_b).status_code == 409
        assert set(store.task(consumer).attachment_ids) == {bound_a, b}
        assert store.artifact(bound_a)["origin_artifact_id"] == new_a
        assert host.artifacts.content(bound_a)[0].read() == b'{"temperature":80}'

        partial = run(consumer, "partial replacement")
        assert replace(b, new_b).status_code == 409
        prompt = host.manager._runtime_input(store.run(partial))
        assert "new_a.json" in prompt and "old_b.json" not in prompt
        host.manager._confirm_replacement_inputs(consumer, partial)
        store.finish_run(partial, "SUCCEEDED")
        checkpoint(partial)
        assert store.run(partial).result_invalidation_ids == [correction]

        bound_b = replace(b, new_b).json()["artifact_id"]
        path = tmp_path / store.artifact(bound_b)["path"]
        original_snapshot = path.read_bytes()
        path.write_bytes(b'{"bias":9}')
        corrupt = run(consumer, "corrupt replacement")
        with pytest.raises(ServiceError, match="完整性"):
            host.manager._runtime_input(store.run(corrupt))
        store.finish_run(corrupt, "FAILED")
        assert store.run(corrupt).result_invalidation_ids == [correction]
        path.write_bytes(original_snapshot)
        unverified = run(consumer, "did not load inputs")
        store.finish_run(unverified, "SUCCEEDED")
        checkpoint(unverified)
        assert store.run(unverified).result_invalidation_ids == [correction]
        fresh = run(consumer, "recheck both")
        prompt = host.manager._runtime_input(store.run(fresh))
        assert "new_a.json" in prompt and "new_b.json" in prompt
        host.manager._confirm_replacement_inputs(consumer, fresh)
        store.finish_run(fresh, "SUCCEEDED")
        checkpoint(fresh)
        assert store.run(fresh).result_invalidation_ids == []
        assert store.run(history).result_invalidation_ids == [correction]
        assert store.artifact_result_invalidations(a) == [correction]

        second = feedback(repaired, "Use remeasured noise")
        assert store.run(fresh).result_invalidation_ids == [second]
        assert store.artifact_result_invalidations(bound_b) == [second]
    store.close()


@pytest.mark.parametrize(("tool", "arguments", "method", "regime"), [
    (BrusTransitionEnergyTool(), {"radius_nm": 4, "bulk_band_gap_eV": 1.3,
                                "electron_effective_mass_m0": 0.1, "hole_effective_mass_m0": 0.5},
     "effective-mass-confinement", "core_shell_dot"),
    (BandgapBowingTool(), {"x": 0.2, "band_gap_a_eV": 1.1, "band_gap_b_eV": 1.6, "bowing_parameter_eV": 0.3},
     "band-bowing", "multiphase_alloy"),
    (KdotpyRunTool(), {"args": ["bulk"]}, "kp-heterostructure", "zero_dimensional_dot"),
    (MeepThinFilmTool(), {"wavelength_um": 1.5, "thickness_um": 0.2}, "fdtd-thin-film", "laterally_patterned_film"),
])
async def test_declared_domain_exclusions_reach_real_solver_entry_points(tool, arguments, method, regime):
    result = await tool.execute({**arguments, "regimes": [regime]})
    assert result.is_error and not result.state_updates
    assert result.data["applicability"]["method_id"] == method
    assert result.data["applicability"]["status"] == "UNSUPPORTED"
    assert result.data["applicability"]["incompatible_regimes"] == [regime]
    found = search_methods(MethodNeed(query="brus bowing kdotpy fdtd", regimes=(regime,)), 8)
    assert any(row["method_id"] == method for row in found["excluded"])


async def test_unknown_domain_preserves_existing_numerical_bowing_result():
    tool = BandgapBowingTool()
    inputs = {"x": 0.2, "band_gap_a_eV": 1.1, "band_gap_b_eV": 1.6, "bowing_parameter_eV": 0.3}
    baseline = await tool.execute(inputs)
    result = await tool.execute({**inputs, "regimes": []})
    assert not result.is_error and result.data == baseline.data


def test_only_explicit_adjacent_table_continuations_with_matching_headers_are_linked():
    def fragment(index, page, caption, header):
        return [
            DocumentObject(object_id=f"caption-{index}", kind="caption", text=caption, quality="anchored"),
            DocumentObject(object_id=f"table-{index}", kind="table", quality="anchored",
                           anchors=[{"page": page}], related=[f"caption-{index}"],
                           table_cells=[{"column_header": True, "start_col_offset_idx": 0, "text": header}]),
        ]

    objects = (fragment(1, 3, "Table S1. Reference measurements", "Temperature (K)")
               + fragment(2, 4, "Table S1 (continued)", "Temperature (K)")
               + fragment(3, 5, "Table S1 (continued)", "Bias (V)")
               + fragment(4, 6, "Table S2. Separate result", "Temperature (K)"))
    linked, unresolved = link_table_continuations(objects)
    assert linked[1].related == ["caption-1", "table-2"]
    assert linked[3].related == ["caption-2", "table-1"]
    assert linked[5].related == ["caption-3"]
    assert unresolved


def test_page_break_continuation_requires_position_and_repeated_headers():
    header = [{"column_header": True, "start_col_offset_idx": col, "text": title}
              for col, title in enumerate(["Sample", "Temperature (K)", "Bias (V)"])]
    caption = DocumentObject(object_id="cap", kind="caption", text="Table 3. Conditions", quality="anchored")

    def table(name, page, top, bottom, related=()):
        return DocumentObject(
            object_id=name, kind="table", related=list(related), quality="anchored", table_cells=header,
            anchors=[{"page": page, "bbox": {"l": 70, "r": 520, "t": top, "b": bottom, "coord_origin": "BOTTOMLEFT"}}],
        )

    first = table("first", 19, 600, 74, ("cap",))
    following = table("second", 20, 733, 460)
    linked, unresolved = link_table_continuations([caption, first, following], {19: 792, 20: 792})
    assert not unresolved
    assert linked[1].table_continuations == {"second": "matching_header_at_page_break"}
    assert linked[2].table_cells == following.table_cells
    separate, _ = link_table_continuations([caption, first, table("later", 20, 450, 100)], {19: 792, 20: 792})
    assert not separate[1].table_continuations


async def test_partial_document_has_typed_coverage_and_cannot_be_indexed(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_DOCLING_TIMEOUT_SECONDS", "90")
    service = DocumentService(Workspace(tmp_path), lambda _: ParsedDocument(
        title="Timeout fixture", source_page_count=3, diagnostics=["DOCLING_PARTIAL_OR_FAILED"],
        objects=[DocumentObject(object_id="p1", kind="paragraph", text="Readable fragment",
                                quality="anchored", anchors=[{"page": 1}])],
    ))
    artifact = service.register_pdf(b"%PDF-test", "https://arxiv.org/pdf/example")
    catalog = await service.read(document_id=artifact.document_id)
    assert catalog["parse_state"] == "PARTIAL" and catalog["source_page_count"] == 3
    assert catalog["coverage"] == "document_objects_partial"
    with pytest.raises(DocumentError, match="DOCUMENT_INDEX_COVERAGE_UNCONFIRMED"):
        service.index_projection(artifact.document_id)
    monkeypatch.setenv("PHOTOMATAGENT_DOCLING_TIMEOUT_SECONDS", "480")
    with pytest.raises(DocumentError, match="DOCUMENT_REVISION_CHANGED"):
        await service.read(document_id=artifact.document_id, expected_revision=catalog["revision"])
