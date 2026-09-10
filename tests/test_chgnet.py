"""Focused offline tests for the CHGNet screening and relaxation pack."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from photomatagent.scientific.capabilities.chgnet import (
    CHGNetRelaxTool,
    CHGNetScreenTool,
    chgnet_pack,
)
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.registry import build_scientific_tools
from photomatagent.tools.exposure import ToolExposure
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

LIF_CIF = NACL_CIF.replace("data_NaCl", "data_LiF").replace(
    "Na Na1", "Li Li1"
).replace("Cl Cl1", "F F1")


def _write_cif(root: Path, name: str, contents: str = NACL_CIF) -> str:
    path = root / name
    path.write_text(contents, encoding="utf-8")
    return name


class FakeModel:
    def __init__(self, energies: dict[str, float]) -> None:
        self.energies = energies
        self.calls: list[str] = []

    def predict_structure(self, structure):
        formula = structure.composition.reduced_formula
        self.calls.append(formula)
        energy = self.energies[formula]
        return {
            "e": energy,
            "f": [[3.0, 4.0, 0.0], [0.0, 0.0, 0.5]],
            "s": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            "m": [0.1, -0.2],
        }


class FakeOptimizer:
    def __init__(self, final_structure=None) -> None:
        self.final_structure = final_structure
        self.calls: list[dict] = []

    def relax(self, structure, **kwargs):
        self.calls.append(kwargs)
        return {
            "final_structure": self.final_structure or structure.copy(),
            "trajectory": SimpleNamespace(energies=[-1.0, -1.25]),
        }


def test_pack_probes_missing_dependency_without_raising(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "chgnet", None)

    pack = chgnet_pack(ScientificConfig(), Workspace(tmp_path))
    result = pack.probe()

    assert result.status.value == "MISSING_DEPENDENCY"
    assert "chgnet" in result.detail.casefold()


def test_pack_tools_are_deferred_and_registered(tmp_path):
    workspace = Workspace(tmp_path)
    config = ScientificConfig.from_environment(workspace=tmp_path)

    pack_names = {tool.name for tool in chgnet_pack(config, workspace).tools()}
    tools = build_scientific_tools(config=config, workspace=workspace)
    registered = {tool.name: tool for tool in tools if tool.namespace == "chgnet"}

    assert pack_names == {"chgnet.screen", "chgnet.relax"}
    assert set(registered) == pack_names
    assert all(tool.exposure is ToolExposure.DEFERRED for tool in registered.values())


@pytest.mark.asyncio
async def test_screen_rejects_escape_before_loading_model(tmp_path):
    workspace = Workspace(tmp_path)
    loaded = False

    def model_loader(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("model must not load for invalid paths")

    tool = CHGNetScreenTool(ScientificConfig(), workspace, model_loader=model_loader)
    result = await tool.execute({"paths": ["../outside.cif"]})

    assert result.is_error
    assert result.data["error_type"] == "invalid_input"
    assert "outside workspace" in result.output
    assert loaded is False


@pytest.mark.asyncio
async def test_screen_uses_ml_potential_evidence_and_omits_cross_composition_rank(
    tmp_path,
):
    nacl = _write_cif(tmp_path, "nacl.cif")
    lif = _write_cif(tmp_path, "lif.cif", LIF_CIF)
    model = FakeModel({"NaCl": -1.0, "LiF": -2.0})
    tool = CHGNetScreenTool(ScientificConfig(), Workspace(tmp_path), model=model)

    result = await tool.execute({"paths": [nacl, lif]})

    assert not result.is_error, result.output
    assert {row["formula"] for row in result.data["results"]} == {"NaCl", "LiF"}
    assert all("rank" not in row for row in result.data["results"])
    assert result.evidence
    assert all(
        item.source_type == "ml_interatomic_potential" for item in result.evidence
    )
    assert all(item.fidelity == "ml_potential" for item in result.evidence)
    assert result.data["results"][0]["max_force_eV_A"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_screen_same_composition_ranking_is_deterministic(tmp_path):
    first = _write_cif(tmp_path, "first.cif")
    second = _write_cif(tmp_path, "second.cif")

    class PathEnergyModel(FakeModel):
        def predict_structure(self, structure):
            result = super().predict_structure(structure)
            result["e"] = -2.0 if len(self.calls) == 1 else -1.0
            return result

    model = PathEnergyModel({"NaCl": -1.0})
    tool = CHGNetScreenTool(ScientificConfig(), Workspace(tmp_path), model=model)
    result = await tool.execute({"paths": [second, first]})

    assert not result.is_error, result.output
    rows = {row["path"]: row for row in result.data["results"]}
    assert rows[second]["rank"] == 1
    assert rows[first]["rank"] == 2
    assert rows[second]["rank_within_composition"] == 1


@pytest.mark.asyncio
async def test_relax_writes_workspace_user_output_cif_and_reports_before_after(tmp_path):
    source = _write_cif(tmp_path, "candidate.cif")
    model = FakeModel({"NaCl": -1.0})
    optimizer = FakeOptimizer()
    config = ScientificConfig(chgnet_relax_fmax=0.05, chgnet_relax_steps=7)
    tool = CHGNetRelaxTool(
        config,
        Workspace(tmp_path),
        model=model,
        optimizer=optimizer,
    )

    result = await tool.execute(
        {"path": source, "fmax": 0.05, "steps": 7, "relax_cell": False}
    )

    assert not result.is_error, result.output
    artifact = Path(result.data["output_path"])
    assert artifact == tmp_path / "user_output" / "chgnet" / "candidate_relaxed.cif"
    assert artifact.is_file()
    assert artifact.read_text(encoding="utf-8").startswith("# generated using pymatgen")
    assert optimizer.calls == [
        {"fmax": 0.05, "steps": 7, "relax_cell": False, "verbose": False}
    ]
    assert result.data["before"]["energy_eV_per_atom"] == pytest.approx(-1.0)
    assert result.data["after"]["energy_eV_per_atom"] == pytest.approx(-1.0)
    assert result.artifacts == ["user_output/chgnet/candidate_relaxed.cif"]


def test_config_exposes_bounded_chgnet_defaults_and_environment_overrides(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PHOTOMATAGENT_CHGNET_MODEL_NAME", "r2scan")
    monkeypatch.setenv("PHOTOMATAGENT_CHGNET_DEVICE", "cuda:0")
    monkeypatch.setenv("PHOTOMATAGENT_CHGNET_MAX_STRUCTURES", "8")
    monkeypatch.setenv("PHOTOMATAGENT_CHGNET_RELAX_FMAX", "0.05")
    monkeypatch.setenv("PHOTOMATAGENT_CHGNET_RELAX_STEPS", "50")

    config = ScientificConfig.from_environment(workspace=tmp_path)

    assert config.chgnet_model_name == "r2scan"
    assert config.chgnet_device == "cuda:0"
    assert config.chgnet_max_structures == 8
    assert config.chgnet_relax_fmax == pytest.approx(0.05)
    assert config.chgnet_relax_steps == 50
