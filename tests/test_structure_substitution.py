from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from photomatagent.runtime.permissions import (
    DenyHandler,
    DenyAllPolicy,
)
from photomatagent.scientific.capabilities.structure.construction import (
    make_supercell,
    substitute_sites,
)
from photomatagent.scientific.capabilities.structure.construction_models import (
    ConstructionLimits,
    SiteReplacement,
)
from photomatagent.scientific.capabilities.structure.construction_tools import (
    MakeSupercellTool,
    SubstituteSitesTool,
)
from photomatagent.scientific.capabilities.structure.artifacts import structure_hash
from photomatagent.scientific.discovery.composition import composition_key
from photomatagent.scientific.discovery.models import (
    HypothesisOrigin,
    HypothesisProposal,
)
from photomatagent.scientific.discovery.registration import build_hypothesis
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace

from conftest import collect, make_runtime
from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call


def _parent() -> Structure:
    return Structure(
        Lattice.cubic(6),
        ["Na", "Bi", "S", "S"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )


def test_substitution_preserves_exact_quarter_composition_and_input() -> None:
    parent = _parent()
    before = structure_hash(parent)
    expanded = make_supercell(parent, (2, 2, 1), ConstructionLimits())
    na_index = next(i for i, site in enumerate(expanded) if site.specie.symbol == "Na")

    changed = substitute_sites(
        expanded,
        [SiteReplacement(index=na_index, from_element="Na", to_element="Ag")],
        "Na0.75Ag0.25BiS2",
    )

    assert len(parent) == 4
    assert len(changed) == 16
    assert composition_key(changed.composition.formula) == composition_key("Na3AgBi4S8")
    assert structure_hash(parent) == before
    assert structure_hash(expanded) != structure_hash(changed)


def test_substitution_validates_everything_before_mutating() -> None:
    structure = Structure(Lattice.cubic(4), ["Na", "Bi"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    before = structure_hash(structure)
    with pytest.raises(ValueError, match="input site"):
        substitute_sites(
            structure,
            [SiteReplacement(index=0, from_element="Bi", to_element="Ag")],
            "AgBi",
        )
    assert structure_hash(structure) == before


@pytest.mark.parametrize(
    "replacements, expected",
    [
        ([{"index": 99, "from_element": "Na", "to_element": "Ag"}], "AgBi"),
        ([{"index": 0, "from_element": "Na", "to_element": "Ag"}, {"index": 0, "from_element": "Na", "to_element": "Ag"}], "Ag2Bi"),
    ],
)
def test_substitution_rejects_invalid_indices(replacements, expected) -> None:
    structure = Structure(Lattice.cubic(4), ["Na", "Bi"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    with pytest.raises(ValueError):
        substitute_sites(structure, replacements, expected)


def test_substitution_rejects_expected_formula_mismatch() -> None:
    with pytest.raises(ValueError, match="expected_formula"):
        substitute_sites(
            _parent(),
            [SiteReplacement(index=0, from_element="Na", to_element="Ag")],
            "AgBi",
        )


def test_supercell_checks_atom_bound_before_operation() -> None:
    parent = _parent()
    with pytest.raises(ValueError, match="atom"):
        make_supercell(parent, (2, 2, 2), ConstructionLimits(max_atoms=16))


def test_supercell_rejects_empty_input_before_huge_scaling() -> None:
    from photomatagent.scientific.capabilities.structure.construction import (
        StructureConstructionError,
    )

    empty = Structure(Lattice.cubic(4), [], [])
    before = structure_hash(empty)
    with pytest.raises(StructureConstructionError) as exc:
        make_supercell(empty, (10**100, 1, 1), ConstructionLimits())
    assert exc.value.code == "EMPTY_STRUCTURE"
    assert structure_hash(empty) == before


def test_substitution_distinguishes_expected_and_constructed_formula_errors(monkeypatch) -> None:
    from photomatagent.scientific.capabilities.structure import construction
    from photomatagent.scientific.capabilities.structure.construction import (
        StructureConstructionError,
    )

    parent = Structure(Lattice.cubic(4), ["Na", "Bi"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    with pytest.raises(StructureConstructionError) as invalid_expected:
        substitute_sites(
            parent,
            [SiteReplacement(index=0, from_element="Na", to_element="Ag")],
            "not-a-formula",
        )
    assert invalid_expected.value.code == "INVALID_EXPECTED_FORMULA"

    original = construction.composition_key
    calls = 0

    def fail_constructed(formula: str):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic constructed composition failure")
        return original(formula)

    monkeypatch.setattr(construction, "composition_key", fail_constructed)
    with pytest.raises(StructureConstructionError) as invalid_constructed:
        substitute_sites(
            parent,
            [SiteReplacement(index=0, from_element="Na", to_element="Ag")],
            "AgBi",
        )
    assert invalid_constructed.value.code == "INVALID_CONSTRUCTED_COMPOSITION"


def test_tools_publish_deferred_artifacts_with_input_index_schema(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    input_path = tmp_path / "parent.cif"
    _parent().to(filename=str(input_path), fmt="cif")
    tool = MakeSupercellTool(workspace, limits=ConstructionLimits())
    assert tool.exposure is ToolExposure.DEFERRED
    assert "input structure" in tool.input_schema["properties"]["scaling"]["description"]

    result = __import__("asyncio").run(
        tool.execute({"path": "parent.cif", "scaling": [2, 1, 1], "task_slug": "task"})
    )
    assert not result.is_error
    assert result.artifacts
    assert all(path.startswith("user_output/task/structures/") for path in result.artifacts)
    assert list((tmp_path / "user_output").rglob("manifest.json"))


@pytest.mark.asyncio
async def test_substitution_requires_live_state_and_does_not_publish(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    _parent().to(filename=str(tmp_path / "parent.cif"), fmt="cif")
    tool = SubstituteSitesTool(workspace, scientific_state=None)
    result = await tool.execute(
        {
            "path": "parent.cif",
            "replacements": [{"index": 0, "from_element": "Na", "to_element": "Ag"}],
            "expected_formula": "AgBiS2Na",
            "hypothesis_id": "h1",
            "task_slug": "task",
        }
    )
    assert result.is_error
    assert result.data["error_type"] == "STATE_UNAVAILABLE"
    assert not list((tmp_path / "user_output").rglob("manifest.json"))


@pytest.mark.asyncio
async def test_supercell_empty_input_tool_returns_typed_error_without_artifact(
    tmp_path, monkeypatch
) -> None:
    import photomatagent.scientific.capabilities.structure.construction_tools as tools_module

    workspace = Workspace(tmp_path)
    empty = Structure(Lattice.cubic(4), [], [])
    before = structure_hash(empty)
    monkeypatch.setattr(
        tools_module,
        "load_structure_input",
        lambda *_args, **_kwargs: (empty, tmp_path / "empty.cif"),
    )
    tool = MakeSupercellTool(workspace)
    result = await tool.execute(
        {"path": "empty.cif", "scaling": [10**100, 1, 1], "task_slug": "empty"}
    )
    assert result.is_error
    assert result.data["error_type"] == "EMPTY_STRUCTURE"
    assert structure_hash(empty) == before
    assert not list((tmp_path / "user_output").rglob("manifest.json"))


@pytest.mark.asyncio
async def test_supercell_hypothesis_binding_is_optional_but_verified(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    _parent().to(filename=str(tmp_path / "parent.cif"), fmt="cif")

    no_state = MakeSupercellTool(workspace)
    missing_state = await no_state.execute(
        {
            "path": "parent.cif",
            "scaling": [1, 1, 1],
            "hypothesis_id": "h1",
            "task_slug": "no-state",
        }
    )
    assert missing_state.data["error_type"] == "STATE_UNAVAILABLE"

    wrong_state = _state_for_formula("AgBiS2")
    wrong = await MakeSupercellTool(
        workspace, scientific_state=wrong_state
    ).execute(
        {
            "path": "parent.cif",
            "scaling": [1, 1, 1],
            "hypothesis_id": wrong_state.material_hypotheses[0].id,
            "task_slug": "wrong-state",
        }
    )
    assert wrong.data["error_type"] == "HYPOTHESIS_COMPOSITION_MISMATCH"

    unknown = await MakeSupercellTool(
        workspace, scientific_state=ScientificState()
    ).execute(
        {
            "path": "parent.cif",
            "scaling": [1, 1, 1],
            "hypothesis_id": "unknown",
            "task_slug": "unknown-state",
        }
    )
    assert unknown.data["error_type"] == "HYPOTHESIS_NOT_FOUND"
    assert not list((tmp_path / "user_output").rglob("manifest.json"))

    valid_state = _state_for_formula("NaBiS2")
    valid = await MakeSupercellTool(
        workspace, scientific_state=valid_state
    ).execute(
        {
            "path": "parent.cif",
            "scaling": [1, 1, 1],
            "hypothesis_id": valid_state.material_hypotheses[0].id,
            "task_slug": "valid-state",
        }
    )
    assert not valid.is_error
    assert list((tmp_path / "user_output").rglob("manifest.json"))


@pytest.mark.asyncio
async def test_registry_supercell_uses_shared_state_for_hypothesis_binding(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    _parent().to(filename=str(tmp_path / "parent.cif"), fmt="cif")
    state = _state_for_formula("NaBiS2")
    tool = create_default_registry(state, workspace).get("structure.make_supercell")
    arguments = {
        "path": "parent.cif",
        "scaling": [1, 1, 1],
        "hypothesis_id": state.material_hypotheses[0].id,
        "task_slug": "registry-valid",
    }

    valid = await tool.execute(arguments)
    assert not valid.is_error

    unknown = await tool.execute({**arguments, "hypothesis_id": "unknown", "task_slug": "registry-unknown"})
    assert unknown.data["error_type"] == "HYPOTHESIS_NOT_FOUND"

    mismatch_state = _state_for_formula("AgBiS2")
    mismatch_registry = create_default_registry(mismatch_state, workspace)
    mismatch = await mismatch_registry.get("structure.make_supercell").execute(
        {**arguments, "hypothesis_id": mismatch_state.material_hypotheses[0].id, "task_slug": "registry-mismatch"}
    )
    assert mismatch.data["error_type"] == "HYPOTHESIS_COMPOSITION_MISMATCH"
    assert not (tmp_path / "user_output" / "registry-unknown").exists()
    assert not (tmp_path / "user_output" / "registry-mismatch").exists()


def _state_for_formula(formula: str) -> ScientificState:
    proposal = HypothesisProposal(
        request_id="h-request",
        formula=formula,
        statement="test hypothesis",
        design_operation="isovalent_substitution",
        validation_questions=["does the composition hold?"],
    )
    state = ScientificState()
    state.material_hypotheses.append(
        build_hypothesis(
            proposal,
            HypothesisOrigin(
                tool_name="test",
                tool_call_id="call",
                session_id="session",
                run_id="run",
                provider="test",
                model="test",
            ),
        )
    )
    return state


@pytest.mark.asyncio
async def test_substitution_rejects_unknown_or_mismatched_hypothesis(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    Structure(Lattice.cubic(4), ["Na", "Bi"], [[0, 0, 0], [0.5, 0.5, 0.5]]).to(
        filename=str(tmp_path / "parent.cif"), fmt="cif"
    )
    state = _state_for_formula("AgBi")
    tool = SubstituteSitesTool(workspace, scientific_state=state)
    base = {
        "path": "parent.cif",
        "replacements": [{"index": 0, "from_element": "Na", "to_element": "Ag"}],
        "expected_formula": "AgBi",
        "task_slug": "hypothesis",
    }
    missing = await tool.execute({**base, "hypothesis_id": "missing"})
    assert missing.data["error_type"] == "HYPOTHESIS_NOT_FOUND"
    hypothesis_id = state.material_hypotheses[0].id
    mismatch = await tool.execute(
        {**base, "hypothesis_id": hypothesis_id, "expected_formula": "NaBi"}
    )
    assert mismatch.data["error_type"] == "HYPOTHESIS_COMPOSITION_MISMATCH"
    assert not list((tmp_path / "user_output").rglob("manifest.json"))


@pytest.mark.asyncio
async def test_substitution_validates_hypothesis_before_publishing(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    Structure(Lattice.cubic(4), ["Na", "Bi"], [[0, 0, 0], [0.5, 0.5, 0.5]]).to(
        filename=str(tmp_path / "parent.cif"), fmt="cif"
    )
    state = _state_for_formula("AgBi")
    tool = SubstituteSitesTool(workspace, scientific_state=state)
    result = await tool.execute(
        {
            "path": "parent.cif",
            "replacements": [{"index": 0, "from_element": "Na", "to_element": "Ag"}],
            "expected_formula": "AgBi",
            "hypothesis_id": state.material_hypotheses[0].id,
            "task_slug": "hypothesis-valid",
        }
    )
    assert not result.is_error
    assert result.state_updates == []


@pytest.mark.asyncio
async def test_denied_runtime_call_does_not_publish(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    _parent().to(filename=str(tmp_path / "parent.cif"), fmt="cif")
    model = FakeModelProvider(
        [
            scripted_tool_call(
                "tool_call",
                {
                    "name": "structure.make_supercell",
                    "arguments": {"path": "parent.cif", "scaling": [2, 1, 1], "task_slug": "denied"},
                },
            ),
            FakeResponse(text="denied"),
        ]
    )
    runtime = make_runtime(model, workspace=workspace, permission_policy=DenyAllPolicy(), approval_handler=DenyHandler())
    events = await collect(runtime, "construct")
    assert any(event.kind == "tool_permission_denied" for event in events)
    assert not list((tmp_path / "user_output").rglob("manifest.json"))


def test_surface_registers_construction_tools_as_deferred(tmp_path) -> None:
    registry = create_default_registry(ScientificState(), Workspace(tmp_path))
    for name in ("structure.make_supercell", "structure.substitute_sites"):
        tool = registry.get(name)
        assert tool.exposure is ToolExposure.DEFERRED
        assert name not in {entry.name for entry in registry.definitions(ToolExposure.DIRECT)}
    assert "input-structure" in registry.get("structure.substitute_sites").input_schema["properties"]["replacements"]["description"]
