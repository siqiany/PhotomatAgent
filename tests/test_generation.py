"""Offline tests for VAE formula + MatterGen candidate generation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from photomatagent.scientific.capabilities.generation.formulas import (
    VAEFormulaGenerator,
)
from photomatagent.scientific.capabilities.generation.mattergen import (
    MatterGenGenerator,
    composition_distance,
)
from photomatagent.scientific.errors import MissingScientificPrerequisite

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


def test_vae_target_requires_exactly_one():
    generator = make_generator()
    with pytest.raises(ValueError):
        generator.generate()
    with pytest.raises(ValueError):
        generator.generate(target_band_gap_eV=0.5, target_wavelength_um=3.0)


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


def test_vae_tool_missing_prerequisite_when_no_checkpoint():
    from photomatagent.scientific.capabilities.generation.tools import (
        VAEFormulaTool,
    )

    result = asyncio.run(
        VAEFormulaTool().execute({"target_band_gap_eV": 0.5})
    )
    assert result.is_error
    assert "checkpoint" in result.output


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
            json.dumps({"candidates": []}), encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="no usable candidates"):
            MatterGenGenerator().generate(
                target_band_gap_eV=0.5, manifest_path=manifest
            )


def test_mattergen_no_script_no_manifest_fails_cleanly():
    generator = MatterGenGenerator()
    with pytest.raises(FileNotFoundError, match="script"):
        generator.generate(target_band_gap_eV=0.5)


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
