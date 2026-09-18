from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pymatgen.core import Lattice, Structure

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.runtime.events import StructureDerived, parse_event
from photomatagent.scientific.capabilities.structure.artifacts import publish_structures
from photomatagent.scientific.capabilities.structure.construction_models import (
    ConstructionLimits,
    OrderingRequest,
)
from photomatagent.scientific.discovery.structures import StructureDerivation, StructureRegistration
from photomatagent.scientific.capabilities.generation.lineage import CandidateLineage
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure

from conftest import collect, make_runtime


def _record(*, source: str = "user_output/task/structures/op_" + "a" * 32 + "/structure_0000.cif", structure_hash: str = "b" * 64) -> StructureDerivation:
    candidate_id = "cand_" + structure_hash[:24]
    return StructureDerivation(
        id="der_" + "d" * 20,
        candidate_id=candidate_id,
        parent_candidate_id="cand_parent",
        hypothesis_id="hyp-ordering",
        input_sha256="a" * 64,
        structure_hash=structure_hash,
        output_path=source,
        operation="enumerate_orderings",
        parameters={"matcher": {"ltol": 0.2, "stol": 0.3, "angle_tol": 5}},
        normalized_composition=(("Ag", 1), ("Na", 1)),
        lineage=CandidateLineage(
            candidate_id=candidate_id,
            parent_candidate_id="cand_parent",
            generated_by="structure_construction",
            transformation="enumerate_orderings",
        ).model_dump(mode="python"),
    )


def _published_record(tmp_path: Path) -> tuple[Path, StructureDerivation]:
    workspace = Path(tmp_path)
    from photomatagent.workspace import Workspace

    boundary = Workspace(workspace)
    request = OrderingRequest(
        path="input.cif",
        eligible_indices=[0],
        from_element="Na",
        to_element="Ag",
        replacement_count=1,
        expected_formula="AgNa",
        hypothesis_id="hyp-ordering",
        task_slug="task",
    )
    structure = Structure(
        Lattice.cubic(4), ["Na", "Ag"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    records = publish_structures(
        boundary,
        request,
        "a" * 64,
        [structure],
        structure_matcher={"ltol": 0.2, "stol": 0.3, "angle_tol": 5},
    )
    return workspace, records[0]


def _published_batch(tmp_path: Path) -> tuple[Path, list[StructureDerivation]]:
    workspace, first = _published_record(tmp_path)
    directory = workspace / first.output_path
    # Reuse the same request/artifact identity with two declared outputs.
    from photomatagent.workspace import Workspace

    boundary = Workspace(workspace)
    request = OrderingRequest(
        path="input.cif",
        eligible_indices=[0],
        from_element="Na",
        to_element="Ag",
        replacement_count=1,
        expected_formula="AgNa",
        hypothesis_id="hyp-ordering",
        task_slug="task-batch",
    )
    structure = Structure(
        Lattice.cubic(4), ["Na", "Ag"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    from photomatagent.scientific.capabilities.structure.artifacts import publish_structures

    records = publish_structures(
        boundary,
        request,
        "a" * 64,
        [structure, structure],
        structure_matcher={"ltol": 0.2, "stol": 0.3, "angle_tol": 5},
    )
    return workspace, records


def _with_hash(record: StructureDerivation, digest: str) -> StructureDerivation:
    payload = record.model_dump(mode="python")
    payload["structure_hash"] = digest
    payload["candidate_id"] = "cand_" + digest[:24]
    payload["lineage"]["candidate_id"] = payload["candidate_id"]
    return StructureDerivation.model_validate(payload)


def test_structure_state_is_idempotent_and_keeps_same_hash_from_two_sources() -> None:
    state = ScientificState()
    first = _record()
    second = first.model_copy(update={"id": "der_" + "e" * 20, "origin": {"tool_name": "other"}})
    assert state.add_structure_derivation(first) == first
    assert state.add_structure_derivation(first) == first
    assert state.add_structure_derivation(second) == second
    assert len(state.structure_derivations) == 2


def test_structure_state_revalidates_model_constructed_identity() -> None:
    state = ScientificState()
    record = _record()
    payload = record.model_dump(mode="python")
    payload["lineage"] = record.lineage
    forged = StructureDerivation.model_construct(**payload)
    object.__setattr__(forged, "candidate_id", "cand_attacker")
    with pytest.raises(ValidationError):
        state.add_structure_derivation(forged)
    assert state.structure_derivations == []


def test_structure_derived_event_is_discriminated_and_serializable() -> None:
    event = StructureDerived(
        derivation_id="der_" + "d" * 20,
        candidate_id="cand_" + "c" * 20,
        parent_candidate_id="cand_parent",
        structure_hash="b" * 64,
    )
    parsed = parse_event(event.model_dump(mode="json"))
    assert isinstance(parsed, StructureDerived)
    assert parsed.kind == "structure_derived"


class RegistrationTool(Tool):
    name = "structure.enumerate_orderings"
    source = "pymatgen"
    exposure = ToolExposure.DIRECT
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, records: list[StructureRegistration]) -> None:
        self.records = records

    async def execute(self, _arguments: dict[str, object]) -> ToolResult:
        return ToolResult(output="registered", state_updates=list(self.records))


@pytest.mark.asyncio
async def test_runtime_applies_structure_registrations_atomically_and_injects_origin(tmp_path) -> None:
    workspace, record = _published_record(tmp_path)
    runtime = make_runtime(
        FakeModelProvider([
            scripted_tool_call("structure.enumerate_orderings", {}, tool_call_id="structure-call"),
            FakeResponse(text="done"),
        ]),
        workspace=workspace,
    )
    input_origin = {"forged": "must be replaced"}
    record = record.model_validate({**record.model_dump(mode="python"), "origin": input_origin})
    input_origin["late_mutation"] = "must not leak"
    runtime._tools._tools["structure.enumerate_orderings"] = RegistrationTool([
        StructureRegistration(derivation=record)
    ])

    events = await collect(runtime, "register structure")

    assert len(runtime.scientific_state.structure_derivations) == 1
    saved = runtime.scientific_state.structure_derivations[0]
    assert saved.origin["output_sha256"] == __import__("hashlib").sha256(
        (workspace / saved.output_path).read_bytes()
    ).hexdigest()
    assert saved.origin["tool_name"] == "structure.enumerate_orderings"
    assert saved.origin["tool_call_id"] == "structure-call"
    assert saved.origin["session_id"] == runtime.session_id
    assert "forged" not in saved.origin
    assert "late_mutation" not in saved.origin
    with pytest.raises(TypeError):
        saved.origin["changed"] = "nope"
    derived = [event for event in events if isinstance(event, StructureDerived)]
    assert len(derived) == 1
    assert derived[0].derivation_id == saved.id


@pytest.mark.asyncio
async def test_runtime_retry_of_same_published_derivation_is_idempotent(tmp_path) -> None:
    workspace, record = _published_record(tmp_path)
    runtime = make_runtime(
        FakeModelProvider([
            scripted_tool_call("structure.enumerate_orderings", {}, tool_call_id="first"),
            scripted_tool_call("structure.enumerate_orderings", {}, tool_call_id="retry"),
            FakeResponse(text="done"),
        ]),
        workspace=workspace,
    )
    runtime._tools._tools["structure.enumerate_orderings"] = RegistrationTool([
        StructureRegistration(derivation=record)
    ])
    events = await collect(runtime, "retry structure")
    assert len(runtime.scientific_state.structure_derivations) == 1
    assert runtime.scientific_state.structure_derivations[0].origin["tool_call_id"] == "first"
    assert len([event for event in events if isinstance(event, StructureDerived)]) == 1


@pytest.mark.asyncio
async def test_runtime_rejects_forged_structure_batch_without_partial_state(tmp_path) -> None:
    workspace, first = _published_record(tmp_path)
    forged = _with_hash(first, "e" * 64)
    runtime = make_runtime(
        FakeModelProvider([
            scripted_tool_call("structure.enumerate_orderings", {}, tool_call_id="bad-batch"),
            FakeResponse(text="done"),
        ]),
        workspace=workspace,
    )
    runtime._tools._tools["structure.enumerate_orderings"] = RegistrationTool([
        StructureRegistration(derivation=first), StructureRegistration(derivation=forged)
    ])
    events = await collect(runtime, "reject structure batch")
    assert runtime.scientific_state.structure_derivations == []
    assert any(event.kind == "tool_failed" for event in events)
    assert not any(isinstance(event, StructureDerived) for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["missing", "sibling", "extra", "duplicate"])
async def test_runtime_rejects_incomplete_or_contaminated_manifest_batch(tmp_path, tamper: str) -> None:
    workspace, records = _published_batch(tmp_path)
    artifact = workspace / records[0].output_path
    directory = artifact.parent
    if tamper == "missing":
        (directory / "structure_0001.cif").unlink()
    elif tamper == "sibling":
        (directory / "structure_0001.cif").write_text("not a CIF", encoding="utf-8")
    elif tamper == "extra":
        (directory / "unlisted.cif").write_bytes((directory / "structure_0000.cif").read_bytes())
    else:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"].append(dict(manifest["outputs"][0]))
        manifest_path.write_text(json.dumps(manifest))
    runtime = make_runtime(
        FakeModelProvider([
            scripted_tool_call("structure.enumerate_orderings", {}, tool_call_id=f"bad-{tamper}"),
            FakeResponse(text="done"),
        ]),
        workspace=workspace,
    )
    runtime._tools._tools["structure.enumerate_orderings"] = RegistrationTool([
        StructureRegistration(derivation=records[0])
    ])
    events = await collect(runtime, f"reject manifest {tamper}")
    assert runtime.scientific_state.structure_derivations == []
    assert any(event.kind == "tool_failed" for event in events)


@pytest.mark.asyncio
async def test_runtime_accepts_exact_complete_manifest_batch(tmp_path) -> None:
    workspace, records = _published_batch(tmp_path)
    runtime = make_runtime(
        FakeModelProvider([
            scripted_tool_call("structure.enumerate_orderings", {}, tool_call_id="complete-batch"),
            FakeResponse(text="done"),
        ]),
        workspace=workspace,
    )
    runtime._tools._tools["structure.enumerate_orderings"] = RegistrationTool([
        StructureRegistration(derivation=record) for record in records
    ])
    events = await collect(runtime, "accept complete batch")
    assert len(runtime.scientific_state.structure_derivations) == 2
    assert len([event for event in events if isinstance(event, StructureDerived)]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["missing", "cif", "operation", "hash", "parameters", "symlink", "fake_hash"])
async def test_runtime_rejects_forged_artifact_or_manifest_atomically(tmp_path, tamper: str) -> None:
    workspace, record = _published_record(tmp_path)
    artifact = workspace / record.output_path
    manifest_path = artifact.parent / "manifest.json"
    if tamper == "missing":
        artifact.unlink()
    elif tamper == "cif":
        artifact.write_text("not a CIF", encoding="utf-8")
    elif tamper in {"operation", "hash", "parameters"}:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest[{"operation": "operation", "hash": "input_sha256", "parameters": "parameters"}[tamper]] = (
            "make_supercell" if tamper == "operation" else ("f" * 64 if tamper == "hash" else {"forged": True})
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "symlink":
        artifact.unlink()
        artifact.symlink_to(Path("/tmp/structure-outside.cif"))
    elif tamper == "fake_hash":
            record = _with_hash(record, "e" * 64)
    runtime = make_runtime(
        FakeModelProvider([
            scripted_tool_call("structure.enumerate_orderings", {}, tool_call_id=f"forged-{tamper}"),
            FakeResponse(text="done"),
        ]),
        workspace=workspace,
    )
    runtime._tools._tools["structure.enumerate_orderings"] = RegistrationTool([
        StructureRegistration(derivation=record)
    ])
    events = await collect(runtime, f"reject {tamper}")
    assert runtime.scientific_state.structure_derivations == []
    assert not any(isinstance(event, StructureDerived) for event in events)
    assert any(event.kind == "tool_failed" for event in events)


@pytest.mark.asyncio
async def test_runtime_rejects_structure_registration_from_non_authority_tool(tmp_path) -> None:
    workspace, record = _published_record(tmp_path)

    class NonAuthorityRegistrationTool(RegistrationTool):
        name = "test.structure_registration"

    runtime = make_runtime(
        FakeModelProvider([
            scripted_tool_call("test.structure_registration", {}, tool_call_id="wrong-source"),
            FakeResponse(text="done"),
        ]),
        workspace=workspace,
    )
    runtime._tools._tools["test.structure_registration"] = NonAuthorityRegistrationTool([
        StructureRegistration(derivation=record)
    ])
    events = await collect(runtime, "reject non-authority source")
    assert runtime.scientific_state.structure_derivations == []
    assert any(event.kind == "tool_failed" for event in events)


@pytest.mark.asyncio
async def test_runtime_rebuilds_structure_lineage_and_parent_from_manifest(tmp_path) -> None:
    workspace, record = _published_record(tmp_path)
    payload = record.model_dump(mode="python")
    payload["parent_candidate_id"] = "cand_attacker"
    payload["lineage"].update(
        {
            "candidate_id": record.candidate_id,
            "parent_candidate_id": "cand_attacker",
            "generated_by": "forged",
            "transformation": "fake",
            "validation_status": "PASS",
        }
    )
    forged = StructureDerivation.model_validate(payload)
    runtime = make_runtime(
        FakeModelProvider([
            scripted_tool_call("structure.enumerate_orderings", {}, tool_call_id="lineage"),
            FakeResponse(text="done"),
        ]),
        workspace=workspace,
    )
    runtime._tools._tools["structure.enumerate_orderings"] = RegistrationTool([
        StructureRegistration(derivation=forged)
    ])
    await collect(runtime, "rebuild lineage")
    saved = runtime.scientific_state.structure_derivations[0]
    assert saved.parent_candidate_id is None
    assert saved.lineage.parent_candidate_id is None
    assert saved.lineage.generated_by == "structure_construction"
    assert saved.lineage.transformation == "enumerate_orderings"
    assert saved.lineage.validation_status == "UNVALIDATED_GENERATED_STRUCTURE"


def test_runtime_parent_requires_current_geometry_to_match_registered_hash(
    tmp_path: Path,
) -> None:
    workspace, record = _published_record(tmp_path)
    artifact = workspace / record.output_path
    actual_sha = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()
    forged_payload = record.model_dump(mode="python")
    forged_payload.update(
        {
            "input_sha256": actual_sha,
            "structure_hash": "f" * 64,
            "candidate_id": "cand_" + "f" * 24,
        }
    )
    forged_payload["lineage"]["candidate_id"] = "cand_" + "f" * 24
    forged = StructureDerivation.model_validate(forged_payload)
    runtime = make_runtime(FakeModelProvider([]), workspace=workspace)
    runtime.scientific_state.add_structure_derivation(forged)

    child_payload = record.model_dump(mode="python")
    child_payload["input_sha256"] = actual_sha
    child_payload["output_path"] = (
        "user_output/task/structures/op_" + "b" * 32 + "/structure_0000.cif"
    )
    child = StructureDerivation.model_validate(child_payload)

    assert runtime._trusted_structure_parent(child) is None
