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
from photomatagent.scientific.capabilities.generation.mattergen_runner import (
    MatterGenPretrainedName,
    MatterGenRunSpec,
    MatterGenRunner,
)
from photomatagent.workspace import Workspace


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
        runner: MatterGenRunner | None = None,
        workspace: Workspace | None = None,
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
        self.runner = runner
        self.workspace = workspace

    def run(
        self,
        *,
        output_dir: Path,
        target_band_gap_eV: float | None,
        chemical_system: str | None,
        pretrained_name: MatterGenPretrainedName = "dft_band_gap",
        guidance_factor: float = 2.0,
        seed: int = 42,
    ) -> Path:
        """Run the generation; returns the manifest path (raises on failure)."""
        # The old skill-script adapter remains available as an explicit
        # compatibility override.  Normal operation uses the packaged runner
        # directly and therefore does not depend on MATTERGEN_SKILL_SCRIPT.
        if self.skill_script is None:
            runner = self.runner or MatterGenRunner(
                executable=self.mattergen_executable,
                workspace=self.workspace,
                hf_home=self.hf_home,
                timeout_seconds=self.timeout_seconds,
            )
            condition_band_gap: float | None = None
            if pretrained_name == "dft_band_gap":
                assert target_band_gap_eV is not None
                condition_band_gap = float(target_band_gap_eV)
            condition_chemical_system = (
                chemical_system if pretrained_name == "chemical_system" else None
            )
            spec = MatterGenRunSpec(
                output_dir=output_dir,
                pretrained_name=pretrained_name,
                candidate_count=self.candidate_limit,
                target_band_gap_eV=condition_band_gap,
                chemical_system=condition_chemical_system,
                guidance_factor=guidance_factor,
                seed=seed,
            )
            return runner.run(spec)
        if not self.skill_script.is_file():
            raise FileNotFoundError(f"MatterGen skill script not found: {self.skill_script}")
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
        output_root: str | Path = "user_output/mattergen",
        workspace: Workspace | None = None,
    ) -> None:
        self.provider = provider or LocalIsolatedMatterGenProvider()
        self.output_root = Path(output_root)
        self.workspace = workspace
        if self.workspace is None:
            self.workspace = getattr(self.provider, "workspace", None)

    def generate(
        self,
        *,
        target_band_gap_eV: float | None = None,
        target_wavelength_um: float | None = None,
        chemical_system: str | None = None,
        proposed_formula: str | None = None,
        manifest_path: str | Path | None = None,
        output_dir_override: str | Path | None = None,
        pretrained_name: MatterGenPretrainedName | None = None,
        guidance_factor: float = 2.0,
        seed: int = 42,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Generate (or parse an existing manifest of) MatterGen candidates."""
        if (target_band_gap_eV is not None) and (target_wavelength_um is not None):
            raise ValueError(
                "provide at most one of target_band_gap_eV / target_wavelength_um"
            )
        if pretrained_name is None:
            pretrained_name = (
                "chemical_system"
                if chemical_system and target_band_gap_eV is None and target_wavelength_um is None
                else "dft_band_gap"
            )
        if not isinstance(pretrained_name, str) or pretrained_name not in {
            "dft_band_gap",
            "chemical_system",
        }:
            raise ValueError(
                "pretrained_name must be 'dft_band_gap' or 'chemical_system'"
            )
        if target_band_gap_eV is not None:
            band_gap_float: float | None = float(target_band_gap_eV)
        elif target_wavelength_um is not None:
            wavelength = float(target_wavelength_um)
            if wavelength <= 0:
                raise ValueError("target_wavelength_um must be positive")
            band_gap_float = 1.239841984 / wavelength
        else:
            band_gap_float = None

        if pretrained_name == "dft_band_gap" and band_gap_float is None:
            raise ValueError("dft_band_gap mode requires a band-gap or wavelength target")
        if pretrained_name == "chemical_system" and not chemical_system:
            raise ValueError("chemical_system mode requires chemical_system")
        if pretrained_name == "chemical_system" and band_gap_float is not None:
            raise ValueError(
                "chemical_system mode cannot also receive a band-gap target"
            )

        output_dir = (
            Path(output_dir_override).resolve()
            if output_dir_override
            else self.output_root
            / _default_output_name(
                pretrained_name,
                band_gap_float,
                chemical_system,
            )
        )
        if manifest_path is None:
            manifest_path = self.provider.run(
                output_dir=output_dir,
                target_band_gap_eV=band_gap_float,
                # A dft_band_gap run may retain the caller's chemical system
                # as lineage context, but it is never passed as a conditioning
                # property to the dft checkpoint.
                chemical_system=(
                    chemical_system if pretrained_name == "chemical_system" else None
                ),
                pretrained_name=pretrained_name,
                guidance_factor=guidance_factor,
                seed=seed,
            )
        manifest_file = Path(manifest_path)
        if not manifest_file.is_file():
            raise FileNotFoundError(f"MatterGen manifest not found: {manifest_file}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest_mode = manifest.get("pretrained_name")
        if manifest_mode and manifest_mode != pretrained_name:
            raise ValueError(
                "MatterGen manifest conditioning mode does not match "
                f"requested {pretrained_name}: {manifest_mode}"
            )
        candidates: list[dict[str, Any]] = []
        raw_candidates = manifest.get("candidates", [])
        if not raw_candidates:
            raise RuntimeError("MatterGen produced no usable candidates")
        for raw in raw_candidates[: self.provider.candidate_limit]:
            path = Path(raw["structure_path"]).expanduser()
            if not path.is_absolute():
                path = manifest_file.parent / path
            if self.workspace is not None:
                path = self.workspace.resolve(str(path), must_exist=True)
            else:
                path = path.resolve()
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
                    "pretrained_name": manifest.get("pretrained_name", pretrained_name),
                    "properties_to_condition_on": manifest.get(
                        "properties_to_condition_on"
                    ),
                    "guidance_factor": guidance_factor,
                    "seed": seed,
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
            "pretrained_name": manifest.get("pretrained_name", pretrained_name),
            "properties_to_condition_on": manifest.get("properties_to_condition_on"),
            "guidance_factor": guidance_factor,
            "seed": seed,
            "formula_consistency_note": (
                "VAE formula and MatterGen formula are separate scientific "
                "facts; formula_preserved/composition_distance record their "
                "relationship"
            ),
        }
        return candidates, metadata


def _default_output_name(
    pretrained_name: MatterGenPretrainedName,
    target_band_gap_eV: float | None,
    chemical_system: str | None,
) -> str:
    if pretrained_name == "dft_band_gap":
        assert target_band_gap_eV is not None
        return f"mg-dft-band-gap-{target_band_gap_eV:.3f}ev"
    system = "-".join(
        token for token in (chemical_system or "chemical-system").replace(";", "-").replace(",", "-").split()
        if token
    )
    safe = "".join(character if character.isalnum() or character in "._-" else "-" for character in system)
    return f"mg-chemical-system-{safe or 'unknown'}"
