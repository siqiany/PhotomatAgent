"""MatterGen candidate generation wrapper (donor migration, section 48-49).

Two providers:
* ``LocalIsolatedMatterGenProvider`` -- runs the MatterGen skill script in
  an isolated environment (conda/uv) via subprocess; the archive manifest is
  parsed deterministically
* tests/demos inject a fake manifest through ``manifest_path`` so CIF
  parsing, formula consistency and failure handling are covered offline

Section 49 consistency contract: when a VAE formula constrains the run, the
output records ``vae_proposed_formula``, ``vae_chemical_system``,
``mattergen_generated_formula``, ``formula_preserved`` and
``composition_distance`` -- the two formulas are never conflated.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.generation.lineage import (
    CandidateLineage,
)


def composition_distance(formula_a: str, formula_b: str) -> float:
    """Element-fraction L1 distance between two reduced formulas."""
    from pymatgen.core import Composition

    comp_a = Composition(formula_a).fractional_composition
    comp_b = Composition(formula_b).fractional_composition
    elements = set(comp_a.elements) | set(comp_b.elements)
    distance = sum(
        abs(comp_a.get_atomic_fraction(element) - comp_b.get_atomic_fraction(element))
        for element in elements
    )
    return round(float(distance), 5)


class LocalIsolatedMatterGenProvider:
    """Run MatterGen in an isolated environment (conda/uv), not the main venv."""

    def __init__(
        self,
        *,
        skill_script: str | Path | None = None,
        conda_env: str = "mattergen",
        conda_executable: str = "conda",
        mattergen_executable: str = "mattergen-generate",
        candidate_limit: int = 8,
        timeout_seconds: float = 3600.0,
        hf_home: str | Path | None = None,
    ) -> None:
        self.skill_script = (
            Path(skill_script).resolve() if skill_script else None
        )
        self.conda_env = conda_env
        self.conda_executable = conda_executable
        self.mattergen_executable = mattergen_executable
        self.candidate_limit = candidate_limit
        self.timeout_seconds = timeout_seconds
        self.hf_home = Path(hf_home).resolve() if hf_home else None

    def run(
        self,
        *,
        output_dir: Path,
        target_band_gap_eV: float | None,
        chemical_system: str | None,
    ) -> Path:
        """Run the generation; returns the manifest path (raises on failure)."""
        if self.skill_script is None or not self.skill_script.is_file():
            raise FileNotFoundError(
                "MatterGen skill script not configured; set the script path "
                "or provide a manifest"
            )
        command = [
            sys.executable,
            str(self.skill_script),
            "--candidate-count",
            str(self.candidate_limit),
            "--conda-env",
            self.conda_env,
            "--conda-executable",
            self.conda_executable,
            "--mattergen-executable",
            self.mattergen_executable,
            "--output-dir",
            str(output_dir),
        ]
        if target_band_gap_eV is not None:
            command.extend(["--band-gap-ev", str(target_band_gap_eV)])
        if chemical_system:
            command.extend(["--chemical-system", chemical_system])
        environment = os.environ.copy()
        if self.hf_home:
            environment["HF_HOME"] = str(self.hf_home)
        environment.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=environment,
        )
        manifest = output_dir / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"MatterGen manifest not produced: {manifest}"
            )
        return manifest


class MatterGenGenerator:
    """Generate structures via MatterGen and normalize candidates."""

    def __init__(
        self,
        provider: LocalIsolatedMatterGenProvider | None = None,
        *,
        output_root: str | Path = "output/mattergen",
    ) -> None:
        self.provider = provider or LocalIsolatedMatterGenProvider()
        self.output_root = Path(output_root)

    def generate(
        self,
        *,
        target_band_gap_eV: float | None = None,
        target_wavelength_um: float | None = None,
        chemical_system: str | None = None,
        proposed_formula: str | None = None,
        manifest_path: str | Path | None = None,
        output_dir_override: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Generate (or parse an existing manifest of) MatterGen candidates."""
        if (target_band_gap_eV is None) == (target_wavelength_um is None):
            raise ValueError(
                "provide exactly one of target_band_gap_eV / "
                "target_wavelength_um"
            )
        if target_band_gap_eV is not None:
            band_gap = float(target_band_gap_eV)
        else:
            assert target_wavelength_um is not None  # mutual exclusion above
            band_gap = 1.239841984 / float(target_wavelength_um)
        band_gap_float = float(band_gap)
        output_dir = (
            Path(output_dir_override).resolve()
            if output_dir_override
            else self.output_root / f"mg-{band_gap:.3f}ev"
        )
        if manifest_path is None:
            if self.provider.skill_script is None:
                raise FileNotFoundError(
                    "MatterGen skill script not configured; provide "
                    "manifest_path or configure the script"
                )
            manifest_path = self.provider.run(
                output_dir=output_dir,
                target_band_gap_eV=band_gap_float,
                chemical_system=chemical_system,
            )
        manifest_file = Path(manifest_path)
        if not manifest_file.is_file():
            raise FileNotFoundError(f"MatterGen manifest not found: {manifest_file}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        candidates: list[dict[str, Any]] = []
        raw_candidates = manifest.get("candidates", [])
        if not raw_candidates:
            raise RuntimeError("MatterGen produced no usable candidates")
        for raw in raw_candidates[: self.provider.candidate_limit]:
            path = Path(raw["structure_path"]).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"generated CIF not found: {path}")
            from pymatgen.core import Structure

            structure = Structure.from_file(path)
            generated_formula = structure.composition.reduced_formula
            formula_preserved: bool | None = None
            composition_distance_value: float | None = None
            if proposed_formula:
                formula_preserved = generated_formula == proposed_formula
                composition_distance_value = composition_distance(
                    proposed_formula, generated_formula
                )
            lineage = CandidateLineage(
                generated_by="mattergen",
                generation_parameters={
                    "target_band_gap_eV": band_gap_float,
                    "chemical_system": chemical_system,
                    "pretrained_name": manifest.get("pretrained_name"),
                    "properties_to_condition_on": manifest.get(
                        "properties_to_condition_on"
                    ),
                },
                source_artifacts=[str(manifest_file)],
                transformation="vae_formula_plus_mattergen"
                if proposed_formula
                else "mattergen",
                validation_status="UNVALIDATED_GENERATED_STRUCTURE",
            )
            candidates.append(
                {
                    "candidate_id": lineage.candidate_id,
                    "formula": generated_formula,
                    "structure_path": str(path),
                    "vae_proposed_formula": proposed_formula,
                    "vae_chemical_system": chemical_system,
                    "mattergen_generated_formula": generated_formula,
                    "formula_preserved": formula_preserved,
                    "composition_distance": composition_distance_value,
                    "structure_validation": {
                        "pymatgen_valid": structure.is_valid(),
                        "site_count": len(structure),
                        "volume_angstrom3": float(structure.volume),
                        "density_g_cm3": float(structure.density),
                    },
                    "lineage": lineage.to_evidence_dict(),
                    "warnings": [
                        "MatterGen candidate is UNVALIDATED_GENERATED_STRUCTURE: "
                        "not stable / not synthesizable / not detector-ready "
                        "without further evidence"
                    ],
                }
            )
        metadata = {
            "backend": "mattergen",
            "manifest": str(manifest_file),
            "candidate_count": len(candidates),
            "proposed_formula": proposed_formula,
            "chemical_system": chemical_system,
            "pretrained_name": manifest.get("pretrained_name"),
            "formula_consistency_note": (
                "VAE formula and MatterGen formula are separate scientific "
                "facts; formula_preserved/composition_distance record their "
                "relationship"
            ),
        }
        return candidates, metadata
