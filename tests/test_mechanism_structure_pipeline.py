from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from photomatagent.scientific.capabilities.chgnet import CHGNetRelaxTool, CHGNetScreenTool
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.applications.vasp.unified.executors import CollectionResult
from photomatagent.scientific.applications.vasp.unified.models import (
    PeriodicScientificSpec,
    UnifiedStage,
    UnifiedVaspManifest,
    VaspWorkflowKind,
)
from photomatagent.scientific.applications.vasp.unified.periodic import PeriodicVaspExecutor
from photomatagent.scientific.applications.vasp.unified.tool_pack import VaspCollectTool
from photomatagent.scientific.applications.vasp.tools import _evidence_from_collect
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.registry import ToolRegistry
from photomatagent.errors import ToolValidationError
from photomatagent.workspace import Workspace


NACL_CIF = """data_NaCl
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_Int_Tables_number 1
loop_
 _symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 Na Na1 0 0 0
 Cl Cl1 0.5 0.5 0.5
"""


def _write_cif(root: Path, name: str = "candidate.cif") -> str:
    (root / name).write_text(NACL_CIF, encoding="utf-8")
    return name


class _Model:
    def predict_structure(self, structure):
        return {
            "e": -1.0,
            "f": [[0.0, 0.0, 0.0] for _ in range(len(structure))],
        }


class _ChangedGeometryOptimizer:
    def relax(self, structure, **kwargs):
        changed = structure.copy()
        changed.translate_sites(0, [0.05, 0.0, 0.0], frac_coords=True)
        return {"final_structure": changed, "converged": True}


@pytest.mark.asyncio
async def test_chgnet_scope_is_recomputed_from_actual_workspace_bytes(tmp_path: Path) -> None:
    source = _write_cif(tmp_path)
    result = await CHGNetScreenTool(
        ScientificConfig(), Workspace(tmp_path), model=_Model()
    ).execute({"paths": [source]})

    assert not result.is_error, result.output
    evidence = result.evidence[0]
    assert evidence.candidate_id == result.data["results"][0]["candidate_id"]
    assert evidence.structure_hash == result.data["results"][0]["structure_hash"]
    assert evidence.provenance["input_sha256"] == hashlib.sha256(
        (tmp_path / source).read_bytes()
    ).hexdigest()


@pytest.mark.asyncio
async def test_chgnet_relax_changed_geometry_gets_fresh_structure_identity(
    tmp_path: Path,
) -> None:
    source = _write_cif(tmp_path)
    result = await CHGNetRelaxTool(
        ScientificConfig(),
        Workspace(tmp_path),
        model=_Model(),
        optimizer=_ChangedGeometryOptimizer(),
    ).execute({"path": source, "relax_cell": False})

    assert not result.is_error, result.output
    before, after = result.evidence
    assert before.structure_hash != after.structure_hash
    assert before.candidate_id != after.candidate_id
    assert after.provenance["output_sha256"] == hashlib.sha256(
        (tmp_path / result.data["output_relative_path"]).read_bytes()
    ).hexdigest()
    assert "hull" in after.limitations.casefold()
    assert "stability" in after.limitations.casefold()


@pytest.mark.asyncio
async def test_chgnet_rejects_explicit_mismatched_scope_claim(tmp_path: Path) -> None:
    source = _write_cif(tmp_path)
    result = await CHGNetScreenTool(
        ScientificConfig(), Workspace(tmp_path), model=_Model()
    ).execute(
        {
            "paths": [source],
            "candidate_ids": ["cand_" + "0" * 24],
        }
    )

    assert result.is_error
    assert result.data["error_type"] == "scope_mismatch"


@pytest.mark.asyncio
async def test_chgnet_relax_rejects_explicit_mismatched_scope_claim(tmp_path: Path) -> None:
    source = _write_cif(tmp_path)
    result = await CHGNetRelaxTool(
        ScientificConfig(),
        Workspace(tmp_path),
        model=_Model(),
        optimizer=_ChangedGeometryOptimizer(),
    ).execute(
        {
            "path": source,
            "candidate_id": "cand_" + "0" * 24,
            "relax_cell": False,
        }
    )

    assert result.is_error
    assert result.data["error_type"] == "scope_mismatch"


def test_vasp_evidence_scope_comes_from_existing_input_structure(tmp_path: Path) -> None:
    source = _write_cif(tmp_path, "vasp_input.cif")
    report = {
        "job_id": "123",
        "profile": "standard_semiconductor",
        "scheduler_state": "COMPLETED",
        "structure_path": str(tmp_path / source),
        "structure_hash": "forged",
        "candidate_id": "cand_forged",
        "scientifically_valid": True,
        "parsed": {"final_energy_eV": -10.0},
        "validation_problems": [],
    }

    evidence = _evidence_from_collect(
        report, tool="vasp.collect", workspace=Workspace(tmp_path)
    )[0]

    assert evidence.structure_hash
    assert evidence.structure_hash != "forged"
    assert evidence.candidate_id.startswith("cand_")
    assert evidence.subject == "NaCl"
    assert evidence.provenance["input_sha256"] == hashlib.sha256(
        (tmp_path / source).read_bytes()
    ).hexdigest()


def test_vasp_completed_but_invalid_result_produces_no_scientific_evidence() -> None:
    report = {
        "job_id": "123",
        "profile": "standard_semiconductor",
        "scheduler_state": "COMPLETED",
        "scientifically_valid": False,
        "parsed": {"final_energy_eV": -10.0},
    }

    assert _evidence_from_collect(report, tool="vasp.collect") == []


def test_vasp_missing_validity_or_outside_workspace_produces_no_evidence(
    tmp_path: Path,
) -> None:
    source = _write_cif(tmp_path, "vasp_input.cif")
    base = {
        "parsed": {"final_energy_eV": -10.0},
        "structure_path": str(tmp_path / source),
    }
    assert _evidence_from_collect(base, tool="vasp.collect", workspace=Workspace(tmp_path)) == []
    outside = dict(base, scientifically_valid=True, structure_path="/etc/passwd")
    assert _evidence_from_collect(
        outside, tool="vasp.collect", workspace=Workspace(tmp_path)
    ) == []


def test_chgnet_scope_identity_claims_are_registry_schema_properties(tmp_path: Path) -> None:
    screen_schema = CHGNetScreenTool.input_schema
    assert screen_schema["additionalProperties"] is False
    assert "candidate_ids" in screen_schema["properties"]
    assert "structure_hashes" in screen_schema["properties"]

    relax_schema = CHGNetRelaxTool.input_schema
    assert relax_schema["additionalProperties"] is False
    assert "candidate_id" in relax_schema["properties"]
    assert "structure_hash" in relax_schema["properties"]
    registry = ToolRegistry()
    registry.register(CHGNetScreenTool(ScientificConfig(), Workspace(tmp_path)))
    registry.register(CHGNetRelaxTool(ScientificConfig(), Workspace(tmp_path)))
    assert registry.validate_arguments(
        "chgnet.screen",
        {
            "paths": ["candidate.cif"],
            "candidate_ids": ["cand_abc"],
            "structure_hashes": ["a" * 64],
        },
    )["candidate_ids"] == ["cand_abc"]
    assert registry.validate_arguments(
        "chgnet.relax",
        {"path": "candidate.cif", "candidate_id": "cand_abc", "structure_hash": "a" * 64},
    )["structure_hash"] == "a" * 64
    with pytest.raises(ToolValidationError):
        registry.validate_arguments("chgnet.relax", {"path": "candidate.cif", "forged": True})


class _CollectRegistry:
    def get(self, request_id):
        return SimpleNamespace(job_id="123", remote_directory="remote/job")


class _CollectSession:
    def __init__(self) -> None:
        self.registry = _CollectRegistry()

    def mark_result_state(self, *args, **kwargs) -> None:
        return None


def _periodic_manifest(path: str) -> UnifiedVaspManifest:
    spec = PeriodicScientificSpec(
        structure_path=path,
        profile="standard_semiconductor",
    )
    return UnifiedVaspManifest(
        workflow_id="vasp_0123456789abcdef",
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
        scientific_fingerprint="f" * 64,
        stages=[UnifiedStage(name="relax")],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("validity", [None, False])
async def test_unified_periodic_missing_or_false_validity_has_no_evidence(
    tmp_path: Path, validity: bool | None
) -> None:
    source = _write_cif(tmp_path, "periodic.cif")
    executor = PeriodicVaspExecutor.__new__(PeriodicVaspExecutor)
    executor.application = SimpleNamespace(
        workspace=tmp_path,
        collect=lambda **kwargs: None,
    )
    executor.session = _CollectSession()

    async def collect(**kwargs):
        report = {
            "scientifically_valid": validity,
            "validation_problems": ["missing convergence"] if validity is False else [],
            "parsed": {"final_energy_eV": -10.0},
        }
        return report

    executor.application.collect = collect
    result = await executor.collect(_periodic_manifest(source))

    assert result.evidence == []
    assert result.evidence_gaps


@pytest.mark.asyncio
async def test_unified_periodic_valid_collect_scopes_evidence_to_final_geometry_and_tool(
    tmp_path: Path,
) -> None:
    source = _write_cif(tmp_path, "periodic.cif")
    executor = PeriodicVaspExecutor.__new__(PeriodicVaspExecutor)
    executor.application = SimpleNamespace(workspace=tmp_path)
    executor.session = _CollectSession()

    async def collect(**kwargs):
        from pymatgen.core import Structure

        result_dir = Path(kwargs["local_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        structure = Structure.from_file(str(tmp_path / source))
        structure.translate_sites(0, [0.05, 0.0, 0.0], frac_coords=True)
        structure.to(filename=str(result_dir / "CONTCAR"), fmt="poscar")
        return {
            "scientifically_valid": True,
            "validation_problems": [],
            "scheduler_state": "COMPLETED",
            "parsed": {"final_energy_eV": -10.0},
            "structure_path": "/tmp/forged-other-structure.cif",
        }

    executor.application.collect = collect
    result = await executor.collect(_periodic_manifest(source))

    assert result.ok
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.candidate_id.startswith("cand_")
    assert evidence.structure_hash
    from pymatgen.core import Structure
    from photomatagent.scientific.capabilities.structure.artifacts import structure_hash
    assert evidence.structure_hash != structure_hash(
        Structure.from_file(str(tmp_path / source))
    )
    assert evidence.provenance["input_structure_path"].endswith("/CONTCAR")
    assert evidence.provenance["input_sha256"] == hashlib.sha256(
        Path(evidence.provenance["input_structure_path"]).read_bytes()
    ).hexdigest()

    class Service:
        async def collect(self, workflow_id):
            return result

    tool_result = await VaspCollectTool(Service()).execute(
        {"workflow_id": "vasp_0123456789abcdef"}
    )
    assert len(tool_result.evidence) == 1


@pytest.mark.asyncio
async def test_fake_provider_uses_runtime_registry_for_hypothesis_structure_and_chgnet(
    tmp_path: Path,
) -> None:
    from pymatgen.core import Lattice, Structure

    source_path = tmp_path / "parent.cif"
    Structure(
        Lattice.cubic(6),
        ["Na", "Bi", "S", "S"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    ).to(filename=str(source_path), fmt="cif")
    source = "parent.cif"
    workspace = Workspace(tmp_path)
    scientific = ScientificState()
    registry = create_default_registry(scientific, workspace)
    model = FakeModelProvider(
        [
            scripted_tool_call(
                "tool_call",
                {
                    "name": "generation.register_hypothesis",
                    "arguments": {
                        "request_id": "pipeline-hypothesis",
                        "formula": "Na3AgBi4S8",
                        "statement": "test exact substitution ratio",
                        "design_operation": "isovalent_substitution",
                        "validation_questions": ["is this only a fixture?"],
                    },
                },
                tool_call_id="hypothesis-call",
            ),
            scripted_tool_call(
                "tool_call",
                {
                    "name": "structure.make_supercell",
                    "arguments": {
                        "path": source,
                        "scaling": [2, 2, 1],
                        "task_slug": "pipeline-supercell",
                    },
                },
                tool_call_id="supercell-call",
            ),
            scripted_tool_call(
                "tool_call",
                {
                    "name": "structure.substitute_sites",
                    "arguments": {
                        "path": "user_output/pipeline-supercell/structures/"
                        + "op_00000000000000000000000000000000/structure_0000.cif",
                        "replacements": [
                            {"index": 0, "from_element": "Na", "to_element": "Ag"}
                        ],
                        "expected_formula": "Na3AgBi4S8",
                        "hypothesis_id": "placeholder",
                        "task_slug": "pipeline-substitution",
                    },
                },
                tool_call_id="substitution-call",
            ),
            FakeResponse(text="done"),
        ]
    )
    runtime = AgentRuntime(
        model=model,
        tools=registry,
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=AllowAllPolicy(),
        budget=BudgetState(max_iterations=8),
    )
    runtime._tools._tools["chgnet.screen"] = CHGNetScreenTool(
        ScientificConfig(), workspace, model=_Model()
    )

    # Fill the runtime-produced hypothesis ID into the scripted substitution
    # request only after registration is observed by the real runtime.
    original_stream = model.stream
    chgnet_inserted = False

    async def stream(request):
        nonlocal chgnet_inserted
        last_result = next(
            (
                message
                for message in reversed(request.messages)
                if getattr(message, "tool_name", "")
            ),
            None,
        )
        last_payload = (
            json.loads(last_result.content)
            if last_result is not None and last_result.tool_name == "tool_call"
            else {}
        )
        if last_payload.get("operation") == "make_supercell":
            substitution_call = next(
                call
                for response in model._responses
                for call in (response.tool_calls or [])
                if call.arguments.get("name") == "structure.substitute_sites"
            )
            substitution_call.arguments["arguments"]["hypothesis_id"] = (
                runtime.scientific_state.material_hypotheses[0].id
            )
            substitution_call.arguments["arguments"]["path"] = (
                last_payload["derivations"][0]["output_path"]
            )
        if last_payload.get("operation") == "substitute_sites":
            if not chgnet_inserted:
                model._responses.insert(
                    0,
                    scripted_tool_call(
                        "tool_call",
                        {
                            "name": "chgnet.screen",
                            "arguments": {
                                "paths": [
                                    last_payload["derivations"][0]["output_path"]
                                ]
                            },
                        },
                        tool_call_id="chgnet-call",
                    ),
                )
                chgnet_inserted = True
        async for event in original_stream(request):
            yield event

    model.stream = stream
    events = [event async for event in runtime.run("construct and screen")]

    assert len(runtime.scientific_state.material_hypotheses) == 1
    assert len(runtime.scientific_state.structure_derivations) == 2
    assert any(event.kind == "tool_completed" for event in events)
    assert len(runtime.scientific_state.evidence) == 1
    assert runtime.scientific_state.evidence[0].structure_hash
