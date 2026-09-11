"""Offline tests for VAE formula + MatterGen candidate generation."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from photomatagent.scientific.capabilities.generation.formulas import (
    VAEFormulaGenerator,
)
from photomatagent.scientific.capabilities.generation.mattergen import (
    LocalIsolatedMatterGenProvider,
    MatterGenGenerator,
    composition_distance,
)
from photomatagent.scientific.capabilities.generation.mattergen_runner import (
    MatterGenRunSpec,
    MatterGenRunner,
)
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.errors import MissingScientificPrerequisite
from photomatagent.workspace import Workspace

VOCABULARY = ["Na", "Cl", "Hg", "Te", "Pb", "O"]


def fake_decoder(fractions: list[float]):
    vector = np.asarray(fractions, dtype=float)

    def decode(condition, count):
        return np.tile(vector, (count, 1))

    return decode


def make_generator(**kwargs) -> VAEFormulaGenerator:
    defaults = dict(
        vocabulary=VOCABULARY,
        known_formulas={"NaCl"},
        decoder=fake_decoder([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
    )
    defaults.update(kwargs)
    return VAEFormulaGenerator(**defaults)


def test_vae_generate_integer_charge_neutral_formula():
    generator = make_generator(require_novel=False)
    proposals, metadata = generator.generate(target_band_gap_eV=0.5, limit=8)
    assert proposals
    assert proposals[0].formula == "NaCl"
    assert proposals[0].charge_neutral is True
    assert proposals[0].novel_against_training_data is False  # NaCl known
    assert proposals[0].atom_counts == (1, 1)
    assert metadata["defaults_note"]
    assert "scope" in metadata


def test_vae_require_novel_filters_known_formula():
    generator = make_generator(require_novel=True)
    proposals, metadata = generator.generate(target_band_gap_eV=0.5)
    assert proposals == []  # NaCl is in known_formulas
    assert metadata["rejection_counts"]["known_formula"] > 0


def test_vae_charge_neutrality_filter():
    # Na + O at 1:1 is the lowest-error integerization but is NOT charge
    # neutral; with the filter on, nothing is proposed; with the filter off,
    # the non-neutral formula is returned (explicitly labeled).
    generator = make_generator(
        decoder=fake_decoder([0.5, 0.0, 0.0, 0.0, 0.0, 0.5]),
        require_charge_neutral=True,
        require_novel=False,
    )
    proposals, metadata = generator.generate(target_wavelength_um=3.0)
    assert proposals == []
    assert metadata["rejection_counts"]["not_charge_neutral"] > 0
    relaxed = VAEFormulaGenerator(
        vocabulary=VOCABULARY,
        require_charge_neutral=False,
        require_novel=False,
        decoder=fake_decoder([0.5, 0.0, 0.0, 0.0, 0.0, 0.5]),
    )
    proposals_relaxed, _ = relaxed.generate(target_wavelength_um=3.0)
    assert proposals_relaxed
    assert proposals_relaxed[0].charge_neutral is False


def test_vae_forbidden_elements_are_optional_user_constraint():
    generator = make_generator(
        decoder=fake_decoder([0.0, 0.0, 0.5, 0.5, 0.0, 0.0]),
        require_novel=False,
    )
    # Default: HgTe allowed (no default forbidden elements)
    allowed, _ = generator.generate(target_band_gap_eV=0.3)
    assert any("Hg" in proposal.elements for proposal in allowed)
    # Explicit user constraint: Hg forbidden -> no proposals
    blocked, metadata = generator.generate(
        target_band_gap_eV=0.3, forbidden_elements=["Hg"]
    )
    assert blocked == []
    assert metadata["rejection_counts"]["forbidden_element"] > 0


def test_vae_deterministic_seed_and_integerization():
    rng = np.random.default_rng(7)
    fractions = rng.random(len(VOCABULARY))
    fractions /= fractions.sum()
    first = make_generator(
        decoder=lambda condition, count: np.tile(fractions, (count, 1)),
        require_novel=False,
    )
    proposals_a, _ = first.generate(target_band_gap_eV=1.0, limit=4)
    proposals_b, _ = first.generate(target_band_gap_eV=1.0, limit=4)
    assert [p.formula for p in proposals_a] == [p.formula for p in proposals_b]
    for proposal in proposals_a:
        assert sum(proposal.atom_counts) > 0
        assert all(count >= 1 for count in proposal.atom_counts)


def test_vae_missing_checkpoint_is_typed_failure():
    generator = VAEFormulaGenerator(
        checkpoint_path="/nonexistent/checkpoint.pt",
        vocabulary=VOCABULARY,
    )
    with pytest.raises(MissingScientificPrerequisite) as excinfo:
        generator.generate(target_band_gap_eV=0.5)
    assert "checkpoint" in str(excinfo.value)


def test_vae_requires_a_target_and_rejects_inconsistent_gap_wavelength():
    generator = make_generator()
    with pytest.raises(ValueError):
        generator.generate()
    with pytest.raises(ValueError):
        generator.generate(target_band_gap_eV=0.5, target_wavelength_um=3.0)


def test_vae_accepts_sparse_multi_property_conditions_without_retrieval():
    fields = [
        "gap_selected_eV",
        "cutoff_wavelength_um_from_gap",
        "density_g_cm3",
        "dielectric_mean",
    ]
    generator = make_generator(
        property_fields=fields,
        condition_center=np.asarray([0.5, 2.5, 5.0, 10.0]),
        condition_scale=np.asarray([0.25, 1.0, 2.0, 5.0]),
        require_novel=False,
    )
    condition, targets, clipped = generator.condition(
        target_properties={"density": 7.0, "dielectric_mean": 20.0}
    )
    assert targets == {"density_g_cm3": 7.0, "dielectric_mean": 20.0}
    assert np.allclose(condition, [0.0, 0.0, 1.0, 2.0])
    assert clipped == []

    proposals, metadata = generator.generate(
        target_properties={"density_g_cm3": 7.0, "dielectric_mean": 20.0}
    )
    assert proposals[0].formula == "NaCl"
    assert metadata["conditioned_property_count"] == 2
    assert metadata["target_band_gap_eV"] is None
    assert "gap_selected_eV" in metadata["unspecified_properties"]


def test_vae_gap_condition_derives_consistent_cutoff_and_clips_extremes():
    generator = make_generator(
        property_fields=[
            "gap_selected_eV",
            "cutoff_wavelength_um_from_gap",
            "density_g_cm3",
        ],
        condition_center=np.asarray([0.5, 2.5, 5.0]),
        condition_scale=np.asarray([0.25, 1.0, 1.0]),
    )
    condition, targets, clipped = generator.condition(
        target_properties={"band_gap_eV": 0.5, "density_g_cm3": 100.0}
    )
    assert targets["cutoff_wavelength_um_from_gap"] == pytest.approx(
        1.239841984 / 0.5
    )
    assert condition[2] == 8.0
    assert clipped == ["density_g_cm3"]


def test_vae_tool_rejects_device_properties():
    from photomatagent.scientific.capabilities.generation.tools import (
        VAEFormulaTool,
    )

    result = asyncio.run(
        VAEFormulaTool().execute(
            {
                "target_band_gap_eV": 0.5,
                "responsivity_a_w": 0.8,
            }
        )
    )
    assert result.is_error
    assert result.data["error_type"] == "unsupported_device_property"


def test_vae_tool_rejects_nested_device_properties():
    from photomatagent.scientific.capabilities.generation.tools import (
        VAEFormulaTool,
    )

    result = asyncio.run(
        VAEFormulaTool().execute(
            {"target_properties": {"detectivity_jones": 1e10}}
        )
    )
    assert result.is_error
    assert result.data["error_type"] == "unsupported_device_property"


def test_vae_tool_schema_exposes_all_trained_material_properties():
    from photomatagent.scientific.capabilities.generation.tools import (
        VAEFormulaTool,
    )

    properties = VAEFormulaTool.input_schema["properties"][
        "target_properties"
    ]["properties"]
    assert set(properties) == {
        "gap_selected_eV",
        "cutoff_wavelength_um_from_gap",
        "formation_energy_eV_per_atom",
        "energy_above_hull_eV_per_atom",
        "density_g_cm3",
        "dielectric_mean",
        "avg_electron_mass_m0",
        "avg_hole_mass_m0",
        "bulk_modulus_GPa",
        "shear_modulus_GPa",
        "exfoliation_energy_meV_per_atom",
        "max_IR_mode_cm-1",
        "min_IR_mode_cm-1",
        "spillage",
    }


def test_vae_tool_missing_prerequisite_when_checkpoint_is_invalid():
    from photomatagent.scientific.capabilities.generation.tools import (
        VAEFormulaTool,
    )

    result = asyncio.run(
        VAEFormulaTool().execute(
            {
                "target_band_gap_eV": 0.5,
                "checkpoint_path": "/nonexistent/checkpoint.pt",
            }
        )
    )
    assert result.is_error
    assert result.data["error_type"] == "missing_prerequisites"
    assert "checkpoint" in result.output


def test_vae_checkpoint_decoder_is_loadable_and_deterministic(tmp_path):
    torch = pytest.importorskip("torch")
    from photomatagent.scientific.capabilities.generation.conditional_vae import (
        ConditionalVAE,
        VAEConfig,
    )

    config = VAEConfig(
        composition_dim=len(VOCABULARY),
        condition_dim=2,
        hidden_dim=12,
        latent_dim=4,
    )
    model = ConditionalVAE(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": {
                "composition_dim": config.composition_dim,
                "condition_dim": config.condition_dim,
                "hidden_dim": config.hidden_dim,
                "latent_dim": config.latent_dim,
            },
            "model_state_dict": model.state_dict(),
            "condition_center": torch.tensor([0.5, 2.5]),
            "condition_scale": torch.tensor([0.25, 1.0]),
            "property_fields": [
                "gap_selected_eV",
                "cutoff_wavelength_um_from_gap",
            ],
            "vocabulary": VOCABULARY,
        },
        checkpoint,
    )
    generator = VAEFormulaGenerator(
        checkpoint_path=checkpoint,
        require_novel=False,
        sample_count=8,
        random_seed=17,
    )
    generator._load_torch_decoder()
    condition, targets, clipped = generator.condition(
        target_band_gap_eV=0.5,
        target_wavelength_um=2.5,
    )
    assert targets == {
        "gap_selected_eV": 0.5,
        "cutoff_wavelength_um_from_gap": 2.5,
    }
    assert clipped == []
    first = generator.decode_samples(condition, 8)
    second = generator.decode_samples(condition, 8)
    assert first.shape == (8, len(VOCABULARY))
    assert np.allclose(first.sum(axis=1), 1.0)
    assert np.allclose(first, second)


def test_vae_asset_root_resolves_deployed_layout(tmp_path, monkeypatch):
    from photomatagent.scientific.capabilities.generation.tools import (
        _resolve_vae_assets,
    )

    checkpoint = tmp_path / "jarvis_cvae_v1" / "checkpoint.pt"
    metadata = tmp_path / "jarvis_inverse_v1" / "candidate_metadata.json"
    checkpoint.parent.mkdir()
    metadata.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    metadata.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("PHOTOMATAGENT_VAE_ASSET_ROOT", str(tmp_path))
    resolved_checkpoint, resolved_metadata = _resolve_vae_assets()
    assert resolved_checkpoint == checkpoint.resolve()
    assert resolved_metadata == metadata.resolve()


# -- MatterGen ----------------------------------------------------------------


NACL_CIF = """data_NaCl
_cell_length_a   5.6402
_cell_length_b   5.6402
_cell_length_c   5.6402
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na  Na  0.0 0.0 0.0
Cl  Cl  0.5 0.5 0.5
"""


def make_manifest(tmp_path: Path, *, candidates: list[dict] | None = None) -> Path:
    cif = tmp_path / "0001.cif"
    cif.write_text(NACL_CIF, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "pretrained_name": "dft_band_gap",
                "properties_to_condition_on": {"dft_band_gap": 0.5},
                "run_spec": {
                    "pretrained_name": "dft_band_gap",
                    "candidate_count": 8,
                    "target_band_gap_eV": 0.5,
                    "chemical_system": None,
                    "guidance_factor": 2.0,
                    "seed": 42,
                },
                "band_gap_target_source": "explicit_request",
                "candidates": candidates
                or [{"structure_path": str(cif), "candidate_id": "mg-1"}],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_mattergen_parses_manifest_and_formula():
    tmp = Path("/tmp") / "mgtest"
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        manifest = make_manifest(tmp)
        generator = MatterGenGenerator()
        candidates, metadata = generator.generate(
            target_band_gap_eV=0.5,
            manifest_path=manifest,
        )
        assert len(candidates) == 1
        assert candidates[0]["formula"] == "NaCl"
        assert candidates[0]["mattergen_generated_formula"] == "NaCl"
        assert candidates[0]["vae_proposed_formula"] is None
        assert candidates[0]["structure_validation"]["pymatgen_valid"] is True
        assert candidates[0]["lineage"]["validation_status"] == (
            "UNVALIDATED_GENERATED_STRUCTURE"
        )
        assert any("UNVALIDATED" in warning for warning in candidates[0]["warnings"])


def test_mattergen_formula_consistency_fields():
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        manifest = make_manifest(tmp)
        candidates, _ = MatterGenGenerator().generate(
            target_band_gap_eV=0.5,
            chemical_system="Na-Cl",
            proposed_formula="HgTe",  # mismatched on purpose
            manifest_path=manifest,
        )
        candidate = candidates[0]
        assert candidate["vae_proposed_formula"] == "HgTe"
        assert candidate["mattergen_generated_formula"] == "NaCl"
        assert candidate["formula_preserved"] is False
        assert candidate["composition_distance"] > 0


def test_mattergen_explicit_manifest_requires_complete_run_spec(tmp_path):
    manifest = make_manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    del raw["run_spec"]
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="complete run_spec"):
        MatterGenGenerator().generate(
            target_band_gap_eV=0.5,
            manifest_path=manifest,
        )


def test_mattergen_explicit_manifest_mismatch_is_rejected(tmp_path):
    manifest = make_manifest(tmp_path)

    with pytest.raises(ValueError, match="guidance_factor"):
        MatterGenGenerator().generate(
            target_band_gap_eV=0.5,
            guidance_factor=1.5,
            manifest_path=manifest,
        )


def test_mattergen_lineage_uses_manifest_run_spec(tmp_path):
    manifest = make_manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["run_spec"].update(
        {
            "guidance_factor": 1.5,
            "seed": 7,
        }
    )
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    candidates, _ = MatterGenGenerator().generate(
        target_band_gap_eV=0.5,
        guidance_factor=1.5,
        seed=7,
        manifest_path=manifest,
    )
    generation_parameters = candidates[0]["lineage"]["generation_parameters"]
    assert generation_parameters["guidance_factor"] == 1.5
    assert generation_parameters["seed"] == 7


def test_composition_distance_zero_for_same_formula():
    assert composition_distance("NaCl", "NaCl") == 0.0
    assert composition_distance("HgTe", "NaCl") > 0.0


def test_mattergen_missing_manifest_is_typed_failure():
    with pytest.raises(FileNotFoundError):
        MatterGenGenerator().generate(
            target_band_gap_eV=0.5,
            manifest_path="/nonexistent/manifest.json",
        )


def test_mattergen_empty_archive_fails():
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        manifest = tmp / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "run_spec": {
                        "pretrained_name": "dft_band_gap",
                        "candidate_count": 8,
                        "target_band_gap_eV": 0.5,
                        "chemical_system": None,
                        "guidance_factor": 2.0,
                        "seed": 42,
                    },
                    "properties_to_condition_on": {"dft_band_gap": 0.5},
                    "candidates": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="no usable candidates"):
            MatterGenGenerator().generate(
                target_band_gap_eV=0.5, manifest_path=manifest
            )


def test_mattergen_no_script_no_manifest_fails_cleanly():
    generator = MatterGenGenerator()
    with pytest.raises(FileNotFoundError, match="MatterGen"):
        generator.generate(target_band_gap_eV=0.5)


def test_mattergen_run_spec_requires_mode_specific_conditioning(tmp_path):
    with pytest.raises(ValueError, match="target_band_gap_eV"):
        MatterGenRunSpec(
            output_dir=tmp_path,
            pretrained_name="dft_band_gap",
            candidate_count=2,
            target_band_gap_eV=None,
            chemical_system=None,
            guidance_factor=1.0,
            seed=7,
        )

    with pytest.raises(ValueError, match="chemical_system"):
        MatterGenRunSpec(
            output_dir=tmp_path,
            pretrained_name="chemical_system",
            candidate_count=2,
            target_band_gap_eV=None,
            chemical_system=None,
            guidance_factor=1.0,
            seed=7,
        )


def test_mattergen_runner_builds_argv_without_shell_interpolation(tmp_path):
    spec = MatterGenRunSpec(
        output_dir=tmp_path / "run",
        pretrained_name="chemical_system",
        candidate_count=3,
        target_band_gap_eV=None,
        chemical_system="Na-O; touch SHOULD_NOT_RUN",
        guidance_factor=1.25,
        seed=17,
    )
    runner = MatterGenRunner(
        executable="mattergen-generate",
        workspace=Workspace(tmp_path),
    )
    command = runner.build_command(spec)

    assert isinstance(command, list)
    assert command[0] == "mattergen-generate"
    assert str(spec.output_dir) in command
    assert any("chemical_system" in item for item in command)
    assert any("Na-O; touch SHOULD_NOT_RUN" in item for item in command)
    assert all(item != "touch" for item in command)
    assert not any(item.startswith("--seed") for item in command)


def test_mattergen_runner_command_matches_generate_main_signature(tmp_path):
    spec = MatterGenRunSpec(
        output_dir=tmp_path / "run",
        pretrained_name="dft_band_gap",
        candidate_count=1,
        target_band_gap_eV=0.5,
        chemical_system=None,
        guidance_factor=1.0,
        seed=17,
    )
    command = MatterGenRunner(workspace=Workspace(tmp_path)).build_command(spec)

    # MatterGen 1.0.3's ``mattergen.scripts.generate.main`` has these flags;
    # seed is applied by the isolated adapter, not passed as an unsupported
    # Fire argument to the official executable.
    assert command[0] == "mattergen-generate"
    assert command[1] == str(spec.output_dir)
    assert {item.split("=", 1)[0] for item in command[2:]} == {
        "--pretrained_name",
        "--batch_size",
        "--num_batches",
        "--properties_to_condition_on",
        "--diffusion_guidance_factor",
    }


def test_configured_mattergen_cli_help_contract_without_generation(monkeypatch):
    executable = os.environ.get("MATTERGEN_EXECUTABLE")
    if not executable:
        pytest.skip("MATTERGEN_EXECUTABLE is not configured for the real CLI contract")
    completed = subprocess.run(
        [os.path.expanduser(executable), "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "MPLCONFIGDIR": "/tmp/photomatagent-mattergen-test-mpl"},
    )
    help_text = f"{completed.stdout}\n{completed.stderr}"
    assert "--pretrained_name" in help_text
    assert "--properties_to_condition_on" in help_text
    assert "--seed" not in help_text


def test_mattergen_runner_extracts_sorted_cifs_and_reuses_manifest(
    tmp_path, monkeypatch
):
    workspace = Workspace(tmp_path)
    output_dir = workspace.user_output_dir / "mattergen" / "reuse"
    spec = MatterGenRunSpec(
        output_dir=output_dir,
        pretrained_name="dft_band_gap",
        candidate_count=4,
        target_band_gap_eV=0.5,
        chemical_system=None,
        guidance_factor=2.0,
        seed=42,
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        staging_output = Path(command[1])
        staging_output.mkdir(parents=True, exist_ok=True)
        archive = staging_output / "generated_crystals_cif.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("z-last.cif", NACL_CIF)
            handle.writestr("a-first.cif", NACL_CIF)
            handle.writestr("empty.cif", "")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "photomatagent.scientific.capabilities.generation.mattergen_runner.subprocess.run",
        fake_run,
    )
    runner = MatterGenRunner(
        executable="/bin/true",
        workspace=workspace,
    )

    manifest_path = runner.run(spec)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert [Path(row["structure_path"]).name for row in manifest["candidates"]] == [
        "candidate-0001.cif",
        "candidate-0002.cif",
    ]
    assert manifest["properties_to_condition_on"] == {"dft_band_gap": 0.5}
    assert all(workspace.contains(Path(row["structure_path"])) for row in manifest["candidates"])

    def should_not_run(*args, **kwargs):
        raise AssertionError("matching manifest should be reused")

    monkeypatch.setattr(
        "photomatagent.scientific.capabilities.generation.mattergen_runner.subprocess.run",
        should_not_run,
    )
    assert runner.run(spec) == manifest_path


def test_mattergen_runner_rebuilds_when_cached_candidate_is_missing(
    tmp_path, monkeypatch
):
    workspace = Workspace(tmp_path)
    output_dir = workspace.user_output_dir / "mattergen" / "missing-cache"
    spec = MatterGenRunSpec(
        output_dir=output_dir,
        pretrained_name="dft_band_gap",
        candidate_count=1,
        target_band_gap_eV=0.5,
        chemical_system=None,
        guidance_factor=2.0,
        seed=42,
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        staging_output = Path(command[1])
        staging_output.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(staging_output / "generated_crystals_cif.zip", "w") as handle:
            handle.writestr("candidate.cif", NACL_CIF)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "photomatagent.scientific.capabilities.generation.mattergen_runner.subprocess.run",
        fake_run,
    )
    runner = MatterGenRunner(executable="/bin/true", workspace=workspace)
    manifest_path = runner.run(spec)
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    Path(first_manifest["candidates"][0]["structure_path"]).unlink()

    runner.run(spec)
    assert len(calls) == 2


def test_mattergen_runner_failure_does_not_replace_old_manifest_or_read_old_zip(
    tmp_path, monkeypatch
):
    workspace = Workspace(tmp_path)
    output_dir = workspace.user_output_dir / "mattergen" / "atomic-failure"
    output_dir.mkdir(parents=True)
    old_candidate = output_dir / "old-candidate.cif"
    old_candidate.write_text(NACL_CIF, encoding="utf-8")
    old_manifest = {
        "manifest_version": 1,
        "backend": "mattergen",
        "validation_status": "UNVALIDATED_GENERATED_STRUCTURE",
        "pretrained_name": "dft_band_gap",
        "properties_to_condition_on": {"dft_band_gap": 0.5},
        "run_spec": {
            "pretrained_name": "dft_band_gap",
            "candidate_count": 1,
            "target_band_gap_eV": 0.5,
            "chemical_system": None,
            "guidance_factor": 2.0,
            "seed": 41,
        },
        "candidates": [
            {
                "candidate_id": "old",
                "structure_path": str(old_candidate),
            }
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    (output_dir / "generated_crystals_cif.zip").write_bytes(b"old archive")

    def failing_run(command, **kwargs):
        assert Path(command[1]) != output_dir
        raise subprocess.CalledProcessError(
            17, command, output=b"staged failure", stderr=b"do not publish"
        )

    monkeypatch.setattr(
        "photomatagent.scientific.capabilities.generation.mattergen_runner.subprocess.run",
        failing_run,
    )
    spec = MatterGenRunSpec(
        output_dir=output_dir,
        pretrained_name="dft_band_gap",
        candidate_count=1,
        target_band_gap_eV=0.5,
        chemical_system=None,
        guidance_factor=2.0,
        seed=42,
    )
    with pytest.raises(RuntimeError, match="exit code 17"):
        MatterGenRunner(executable="/bin/true", workspace=workspace).run(spec)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == old_manifest
    assert old_candidate.is_file()


def test_mattergen_runner_rejects_cached_candidate_escape(tmp_path):
    workspace = Workspace(tmp_path)
    output_dir = workspace.user_output_dir / "mattergen" / "escape-cache"
    output_dir.mkdir(parents=True)
    outside = tmp_path.parent / "outside-cache.cif"
    outside.write_text(NACL_CIF, encoding="utf-8")
    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_spec": {
                    "pretrained_name": "dft_band_gap",
                    "candidate_count": 1,
                    "target_band_gap_eV": 0.5,
                    "chemical_system": None,
                    "guidance_factor": 2.0,
                    "seed": 42,
                },
                "candidates": [{"structure_path": str(outside)}],
            }
        ),
        encoding="utf-8",
    )
    spec = MatterGenRunSpec(
        output_dir=output_dir,
        pretrained_name="dft_band_gap",
        candidate_count=1,
        target_band_gap_eV=0.5,
        chemical_system=None,
        guidance_factor=2.0,
        seed=42,
    )
    with pytest.raises(ValueError, match="outside workspace"):
        MatterGenRunner(executable="/bin/true", workspace=workspace).run(spec)


def test_mattergen_runner_rejects_output_escape(tmp_path):
    workspace = Workspace(tmp_path)
    spec = MatterGenRunSpec(
        output_dir=tmp_path.parent / "outside",
        pretrained_name="dft_band_gap",
        candidate_count=1,
        target_band_gap_eV=0.5,
        chemical_system=None,
        guidance_factor=1.0,
        seed=0,
    )
    with pytest.raises(ValueError, match="workspace"):
        MatterGenRunner(executable="/bin/true", workspace=workspace).run(spec)


def test_mattergen_runner_fake_executable_e2e(tmp_path):
    workspace = Workspace(tmp_path)
    executable = tmp_path / "fake-mattergen"
    executable.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "import zipfile\n"
        "output = Path(sys.argv[1])\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "with zipfile.ZipFile(output / 'generated_crystals_cif.zip', 'w') as archive:\n"
        "    archive.writestr('generated/one.cif', 'data_fake\\n')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    spec = MatterGenRunSpec(
        output_dir=workspace.user_output_dir / "mattergen" / "e2e",
        pretrained_name="chemical_system",
        candidate_count=1,
        target_band_gap_eV=None,
        chemical_system="Na-O",
        guidance_factor=1.0,
        seed=5,
    )

    manifest_path = MatterGenRunner(
        executable=executable,
        workspace=workspace,
    ).run(spec)

    assert manifest_path.is_relative_to(workspace.root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["properties_to_condition_on"] == {"chemical_system": "Na-O"}
    assert Path(manifest["candidates"][0]["structure_path"]).is_file()
    assert Path(manifest["candidates"][0]["structure_path"]).parent.name != "e2e"


def test_mattergen_runner_seed_adapter_is_deterministic_without_seed_flag(tmp_path):
    workspace = Workspace(tmp_path)
    executable = tmp_path / "seeded-mattergen"
    executable.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import random\n"
        "import sys\n"
        "import zipfile\n"
        "import numpy as np\n"
        "output = Path(sys.argv[1])\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "payload = f'data_seed\\n# {random.random():.16f} {np.random.random():.16f}\\n'\n"
        "with zipfile.ZipFile(output / 'generated_crystals_cif.zip', 'w') as archive:\n"
        "    archive.writestr('generated/one.cif', payload)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    runner = MatterGenRunner(executable=executable, workspace=workspace)

    def run_at(name: str) -> str:
        spec = MatterGenRunSpec(
            output_dir=workspace.user_output_dir / "mattergen" / name,
            pretrained_name="chemical_system",
            candidate_count=1,
            target_band_gap_eV=None,
            chemical_system="Na-O",
            guidance_factor=1.0,
            seed=123,
        )
        manifest = json.loads(runner.run(spec).read_text(encoding="utf-8"))
        return Path(manifest["candidates"][0]["structure_path"]).read_text(
            encoding="utf-8"
        )

    assert run_at("seed-a") == run_at("seed-b")


def test_generation_capabilities_reports_configured_mattergen(tmp_path):
    from photomatagent.scientific.capabilities.generation.tools import (
        GenerationCapabilitiesTool,
    )

    executable = tmp_path / "mattergen-generate"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    config = ScientificConfig(
        mattergen_executable=str(executable),
        mattergen_pretrained_name="chemical_system",
        mattergen_seed=23,
    )
    result = asyncio.run(
        GenerationCapabilitiesTool(config, Workspace(tmp_path)).execute({})
    )

    assert result.data["mattergen"]["status"] == "AVAILABLE"
    assert result.data["mattergen"]["pretrained_name"] == "chemical_system"
    assert result.data["mattergen"]["seed"] == 23


def test_generation_capabilities_rejects_non_executable_configured_file(tmp_path):
    from photomatagent.scientific.capabilities.generation.tools import (
        GenerationCapabilitiesTool,
    )

    executable = tmp_path / "mattergen-generate"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    config = ScientificConfig(mattergen_executable=str(executable))
    result = asyncio.run(
        GenerationCapabilitiesTool(config, Workspace(tmp_path)).execute({})
    )
    assert result.data["mattergen"]["status"] == "UNCONFIGURED"


def test_mattergen_executable_probe_and_runner_share_path_normalization(
    tmp_path, monkeypatch
):
    from photomatagent.scientific.capabilities.generation.mattergen_runner import (
        resolve_mattergen_executable,
    )

    executable = tmp_path / "bin" / "mattergen-generate"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    resolved = resolve_mattergen_executable("bin/mattergen-generate")
    assert resolved == executable.resolve()
    assert resolve_mattergen_executable(str(executable).replace(str(Path.home()), "~")) == resolved


def test_mattergen_tool_uses_isolated_executable_and_emits_evidence(tmp_path):
    from photomatagent.scientific.capabilities.generation.tools import MatterGenTool

    executable = tmp_path / "fake-mattergen-tool"
    executable.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "import zipfile\n"
        "output = Path(sys.argv[1])\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "with zipfile.ZipFile(output / 'generated_crystals_cif.zip', 'w') as archive:\n"
        f"    archive.writestr('generated/one.cif', {NACL_CIF!r})\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    workspace = Workspace(tmp_path)
    config = ScientificConfig(
        mattergen_executable=str(executable),
        mattergen_candidate_limit=1,
    )

    result = asyncio.run(
        MatterGenTool(config, workspace).execute(
            {
                "pretrained_name": "chemical_system",
                "chemical_system": "Na-Cl",
                "candidate_count": 1,
                "seed": 11,
            }
        )
    )

    assert result.is_error is False
    assert result.data["metadata"]["pretrained_name"] == "chemical_system"
    assert result.evidence[0].source_type == "generative_model"
    assert result.evidence[0].fidelity == "ml_generated"
    assert result.artifacts[0].startswith("user_output/mattergen/")


def test_mattergen_config_reads_bounded_runner_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATAGENT_MATTERGEN_EXECUTABLE", "/opt/mattergen-generate")
    monkeypatch.setenv("PHOTOMATAGENT_MATTERGEN_HF_HOME", "/opt/hf-cache")
    monkeypatch.setenv("PHOTOMATAGENT_MATTERGEN_PRETRAINED_NAME", "chemical_system")
    monkeypatch.setenv("PHOTOMATAGENT_MATTERGEN_CANDIDATE_LIMIT", "6")
    monkeypatch.setenv("PHOTOMATAGENT_MATTERGEN_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("PHOTOMATAGENT_MATTERGEN_GUIDANCE_FACTOR", "1.75")
    monkeypatch.setenv("PHOTOMATAGENT_MATTERGEN_SEED", "19")
    config = ScientificConfig.from_environment(workspace=tmp_path)

    assert config.mattergen_executable == "/opt/mattergen-generate"
    assert config.mattergen_hf_home == "/opt/hf-cache"
    assert config.mattergen_pretrained_name == "chemical_system"
    assert config.mattergen_candidate_limit == 6
    assert config.mattergen_timeout_seconds == pytest.approx(120.0)
    assert config.mattergen_guidance_factor == pytest.approx(1.75)
    assert config.mattergen_seed == 19


def test_generation_tools_registered_deferred():
    from photomatagent.scientific.capabilities.generation.tools import (
        GenerationCapabilityPack,
    )

    pack = GenerationCapabilityPack()
    names = [tool.name for tool in pack.tools()]
    for expected in (
        "generation.capabilities",
        "generation.vae_formula",
        "generation.vae_retrieve",
        "generation.mattergen",
    ):
        assert expected in names
    assert all(tool.exposure.value == "deferred" for tool in pack.tools())
    assert pack.probe().status.value in {"MISSING_DEPENDENCY", "AVAILABLE"}


def test_legacy_mattergen_rejects_unsupported_controls_before_script(tmp_path):
    script = tmp_path / "legacy.py"
    script.write_text("raise AssertionError('must not execute')\n", encoding="utf-8")
    workspace = Workspace(tmp_path)
    provider = LocalIsolatedMatterGenProvider(
        skill_script=script,
        workspace=workspace,
    )

    with pytest.raises(ValueError, match="legacy MatterGen override.*guidance_factor"):
        provider.run(
            output_dir=workspace.user_output_dir / "mattergen" / "legacy",
            target_band_gap_eV=0.5,
            chemical_system=None,
            pretrained_name="dft_band_gap",
            guidance_factor=1.5,
            seed=42,
        )


def test_legacy_mattergen_defaults_are_explicitly_marked_non_deterministic(
    tmp_path,
):
    script = tmp_path / "legacy.py"
    script.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--output-dir', required=True)\n"
        "args, _ = parser.parse_known_args()\n"
        "output = Path(args.output_dir)\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "cif = output / 'candidate.cif'\n"
        f"cif.write_text({NACL_CIF!r}, encoding='utf-8')\n"
        "(output / 'manifest.json').write_text(json.dumps({'candidates': [{'structure_path': str(cif)}]}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    workspace = Workspace(tmp_path)
    provider = LocalIsolatedMatterGenProvider(
        skill_script=script,
        workspace=workspace,
    )
    manifest = provider.run(
        output_dir=workspace.user_output_dir / "mattergen" / "legacy",
        target_band_gap_eV=0.5,
        chemical_system=None,
        pretrained_name="dft_band_gap",
        guidance_factor=2.0,
        seed=42,
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    assert raw["run_spec"]["seed"] == 42
    assert raw["reproducibility"]["seed_applied"] is False

    candidates, metadata = MatterGenGenerator(
        provider=provider,
        output_root=workspace.user_output_dir / "mattergen",
        workspace=workspace,
    ).generate(target_band_gap_eV=0.5)
    assert candidates[0]["lineage"]["generation_parameters"]["seed_effective"] is False
    assert metadata["reproducibility"]["seed_applied"] is False
