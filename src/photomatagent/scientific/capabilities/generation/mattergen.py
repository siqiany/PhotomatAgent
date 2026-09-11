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
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.generation.lineage import (
    CandidateLineage,
)
from photomatagent.scientific.capabilities.generation.mattergen_runner import (
    MatterGenPretrainedName,
    MatterGenRunSpec,
    MatterGenRunner,
    MAX_CIF_BYTES,
    _parameter_tag,
    _atomic_bytes_replace,
    _atomic_json_replace,
)
from photomatagent.workspace import Workspace


LEGACY_MATTERGEN_GUIDANCE_FACTOR = 2.0
LEGACY_MATTERGEN_SEED = 42
_LEGACY_RUN_SPEC_FIELDS = (
    "pretrained_name",
    "candidate_count",
    "target_band_gap_eV",
    "chemical_system",
    "guidance_factor",
    "seed",
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
        runner = self.runner or MatterGenRunner(
            executable=self.mattergen_executable,
            workspace=self.workspace,
            hf_home=self.hf_home,
            timeout_seconds=self.timeout_seconds,
        )
        workspace = self.workspace or runner.workspace
        condition_band_gap: float | None = None
        if pretrained_name == "dft_band_gap":
            if target_band_gap_eV is None:
                raise ValueError("dft_band_gap mode requires target_band_gap_eV")
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
        validated_output = runner._validate_output_dir(output_dir)

        # The old skill-script adapter remains available as an explicit
        # compatibility override.  Normal operation uses the packaged runner
        # directly and therefore does not depend on MATTERGEN_SKILL_SCRIPT.
        if self.skill_script is None:
            return runner.run(spec)
        if not self.skill_script.is_file():
            raise FileNotFoundError(f"MatterGen skill script not found: {self.skill_script}")
        if not math.isclose(
            float(guidance_factor),
            LEGACY_MATTERGEN_GUIDANCE_FACTOR,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "legacy MatterGen override does not support guidance_factor; "
                f"only the default {LEGACY_MATTERGEN_GUIDANCE_FACTOR:g} is allowed"
            )
        if seed != LEGACY_MATTERGEN_SEED:
            raise ValueError(
                "legacy MatterGen override does not support seed; only the "
                f"default {LEGACY_MATTERGEN_SEED} is allowed"
            )
        manifest_path = validated_output / "manifest.json"
        runner._validate_path_boundary(manifest_path, label="legacy manifest")
        legacy_parent = workspace.tmp_dir / "mattergen-legacy" / _parameter_tag(spec)
        runner._validate_path_boundary(legacy_parent, label="legacy staging parent")
        legacy_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        run_root = legacy_parent / uuid.uuid4().hex
        runner._validate_path_boundary(run_root, label="legacy staging run")
        run_root.mkdir(mode=0o700)
        staging_output = run_root / "output"
        matplotlib_dir = staging_output / ".matplotlib"
        archive = staging_output / "generated_crystals_cif.zip"
        extraction_dir = run_root / "candidates"
        published_dir = validated_output / "candidates" / run_root.name
        for path, label in (
            (staging_output, "legacy staging output"),
            (matplotlib_dir, "Matplotlib cache"),
            (archive, "legacy MatterGen archive"),
            (extraction_dir, "legacy candidate extraction"),
            (published_dir, "legacy published candidates"),
        ):
            runner._validate_path_boundary(path, label=label)
        staging_output.mkdir(parents=True, exist_ok=True, mode=0o700)
        matplotlib_dir.mkdir(mode=0o700)
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
            str(staging_output),
        ]
        if target_band_gap_eV is not None:
            command.extend(["--band-gap-ev", str(target_band_gap_eV)])
        if chemical_system:
            command.extend(["--chemical-system", chemical_system])
        environment = os.environ.copy()
        if self.hf_home:
            environment["HF_HOME"] = str(self.hf_home)
        environment["MPLCONFIGDIR"] = str(matplotlib_dir)
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=environment,
                cwd=str(staging_output),
            )
            runner._validate_path_boundary(matplotlib_dir, label="Matplotlib cache")
            manifest = staging_output / "manifest.json"
            manifest = runner._validate_file_path(
                manifest, label="legacy MatterGen manifest"
            )
            if archive.exists() or archive.is_symlink():
                runner._validate_file_path(archive, label="legacy MatterGen archive")
            normalized_manifest = self._normalize_legacy_manifest(
                manifest, spec, workspace
            )
            legacy_provenance = normalized_manifest.get("legacy_provenance")
            unknown_fields = (
                legacy_provenance.get("unknown_fields", [])
                if isinstance(legacy_provenance, dict)
                else []
            )
            if unknown_fields:
                raise ValueError(
                    "legacy MatterGen manifest cannot prove the requested "
                    "parameters; unknown/unverified: "
                    + ", ".join(str(field) for field in unknown_fields)
                )
            normalized_candidates = normalized_manifest["candidates"]
            if not normalized_candidates:
                raise RuntimeError("legacy MatterGen manifest contained no candidates")
            extraction_dir.mkdir(mode=0o700)
            for index, candidate in enumerate(normalized_candidates, start=1):
                source = Path(candidate["structure_path"])
                runner._validate_file_path(source, label="legacy candidate")
                if source.stat().st_size > MAX_CIF_BYTES:
                    raise ValueError("legacy MatterGen CIF is too large")
                target = extraction_dir / f"candidate-{index:04d}.cif"
                _atomic_bytes_replace(target, source.read_bytes())
                candidate["structure_path"] = str(target)
                candidate["relative_path"] = workspace.relative(target)
            runner._publish_candidates(
                normalized_candidates, extraction_dir, published_dir
            )
            normalized_manifest["candidates"] = normalized_candidates
            normalized_manifest["candidate_count"] = len(normalized_candidates)
            normalized_manifest["output_relative_dir"] = workspace.relative(
                validated_output
            )
            _atomic_json_replace(manifest_path, normalized_manifest)
            return manifest_path
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

    def _normalize_legacy_manifest(
        self,
        manifest_path: Path,
        spec: MatterGenRunSpec,
        workspace: Workspace,
    ) -> dict[str, Any]:
        """Add an explicit legacy contract without claiming seed determinism."""

        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"legacy MatterGen manifest is invalid: {manifest_path}") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("candidates"), list):
            raise ValueError("legacy MatterGen manifest has no candidate list")
        normalized_candidates: list[dict[str, Any]] = []
        for candidate in raw["candidates"]:
            if not isinstance(candidate, dict) or not isinstance(
                candidate.get("structure_path"), str
            ):
                raise ValueError("legacy MatterGen candidate has no structure_path")
            candidate_path = Path(candidate["structure_path"])
            if not candidate_path.is_absolute():
                candidate_path = manifest_path.parent / candidate_path
            try:
                resolved = workspace.resolve(str(candidate_path), must_exist=True)
            except Exception as exc:
                raise ValueError(
                    "legacy MatterGen candidate is outside or missing from workspace"
                ) from exc
            if resolved.suffix.lower() != ".cif":
                raise ValueError("legacy MatterGen candidates must be CIF files")
            normalized = dict(candidate)
            normalized["structure_path"] = str(resolved)
            normalized["relative_path"] = workspace.relative(resolved)
            normalized_candidates.append(normalized)
        requested = spec.manifest_parameters()
        (
            actual,
            known_fields,
            actual_properties,
            properties_known,
            raw_run_spec,
        ) = (
            _legacy_manifest_parameters(raw)
        )
        for field in _LEGACY_RUN_SPEC_FIELDS:
            if field not in known_fields:
                continue
            actual_value = actual[field]
            if not _legacy_values_match(actual_value, requested[field]):
                raise ValueError(
                    "legacy MatterGen manifest "
                    f"{field} conflicts with request: requested "
                    f"{requested[field]!r}, manifest {actual_value!r}"
                )
        expected_properties = spec.conditioning_properties()
        if properties_known and not _legacy_values_match(
            actual_properties, expected_properties
        ):
            raise ValueError(
                "legacy MatterGen manifest properties_to_condition_on conflicts "
                f"with request: requested {expected_properties!r}, "
                f"manifest {actual_properties!r}"
            )
        unknown_fields = [
            field for field in _LEGACY_RUN_SPEC_FIELDS if field not in known_fields
        ]
        if not properties_known:
            unknown_fields.append("properties_to_condition_on")
        normalized_run_spec = {
            field: actual.get(field) for field in _LEGACY_RUN_SPEC_FIELDS
        }
        raw["manifest_version"] = 1
        raw["backend"] = "mattergen-legacy"
        raw["validation_status"] = "UNVALIDATED_GENERATED_STRUCTURE"
        raw["pretrained_name"] = actual.get("pretrained_name")
        raw["properties_to_condition_on"] = actual_properties
        raw["run_spec"] = normalized_run_spec
        raw["candidate_count"] = len(normalized_candidates)
        raw["candidates"] = normalized_candidates
        raw["reproducibility"] = {
            "seed_requested": actual.get("seed"),
            "seed_applied": False,
            "reason": (
                "legacy MATTERGEN_SKILL_SCRIPT has no seed contract; "
                "the default is accepted for compatibility only"
            ),
        }
        raw["legacy_provenance"] = {
            "status": "UNVERIFIED" if unknown_fields else "VERIFIED",
            "requested_run_spec": requested,
            "actual_run_spec": normalized_run_spec,
            "unknown_fields": unknown_fields,
            "raw_run_spec": raw_run_spec,
        }
        return raw


def _legacy_manifest_parameters(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], set[str], Any, bool, dict[str, Any]]:
    """Read actual parameters from a legacy manifest without request fallback."""

    raw_run_spec_value = raw.get("run_spec")
    raw_run_spec = raw_run_spec_value if isinstance(raw_run_spec_value, dict) else {}
    aliases: dict[str, tuple[str, ...]] = {
        "pretrained_name": ("pretrained_name", "checkpoint", "checkpoint_name"),
        "candidate_count": (
            "candidate_count",
            "requested_candidate_count",
            "batch_size",
            "num_candidates",
        ),
        "target_band_gap_eV": (
            "target_band_gap_eV",
            "band_gap_target_eV",
            "target_band_gap",
        ),
        "chemical_system": ("chemical_system",),
        "guidance_factor": ("guidance_factor", "diffusion_guidance_factor"),
        "seed": ("seed",),
    }
    actual: dict[str, Any] = {}
    known_fields: set[str] = set()
    for field, keys in aliases.items():
        sources: list[tuple[str, Any]] = []
        for key in keys:
            if key in raw_run_spec:
                sources.append((f"run_spec.{key}", raw_run_spec[key]))
            if key in raw:
                sources.append((key, raw[key]))
        if not sources:
            continue
        source_name, value = sources[0]
        for other_name, other_value in sources[1:]:
            if not _legacy_values_match(value, other_value):
                raise ValueError(
                    "legacy MatterGen manifest has conflicting values for "
                    f"{field}: {source_name}={value!r}, "
                    f"{other_name}={other_value!r}"
                )
        actual[field] = value
        known_fields.add(field)

    property_sources: list[tuple[str, Any]] = []
    if "properties_to_condition_on" in raw_run_spec:
        property_sources.append(
            ("run_spec.properties_to_condition_on", raw_run_spec["properties_to_condition_on"])
        )
    if "properties_to_condition_on" in raw:
        property_sources.append(
            ("properties_to_condition_on", raw["properties_to_condition_on"])
        )
    properties_known = bool(property_sources)
    actual_properties: Any = None
    if property_sources:
        property_source, actual_properties = property_sources[0]
        for other_source, other_value in property_sources[1:]:
            if not _legacy_values_match(actual_properties, other_value):
                raise ValueError(
                    "legacy MatterGen manifest has conflicting values for "
                    "properties_to_condition_on: "
                    f"{property_source}={actual_properties!r}, "
                    f"{other_source}={other_value!r}"
                )
    elif actual.get("pretrained_name") == "dft_band_gap" and (
        "target_band_gap_eV" in known_fields
        and actual.get("target_band_gap_eV") is not None
    ):
        # This is reconstructed solely from explicit values returned by the
        # legacy manifest, never from the current request.
        actual_properties = {"dft_band_gap": actual["target_band_gap_eV"]}
        properties_known = True
    elif actual.get("pretrained_name") == "chemical_system" and (
        "chemical_system" in known_fields and actual.get("chemical_system")
    ):
        actual_properties = {"chemical_system": actual["chemical_system"]}
        properties_known = True

    return (
        actual,
        known_fields,
        actual_properties,
        properties_known,
        dict(raw_run_spec),
    )


def _legacy_values_match(actual: Any, expected: Any) -> bool:
    """Compare legacy scalar/mapping values with numeric tolerance."""

    if isinstance(expected, float):
        try:
            actual_float = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(actual_float) and math.isclose(
            actual_float, expected, rel_tol=0, abs_tol=1e-12
        )
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return False
        return all(_legacy_values_match(actual[key], expected[key]) for key in expected)
    return actual == expected


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
        manifest_file = Path(manifest_path).expanduser()
        if self.workspace is not None:
            manifest_file = self.workspace.resolve(str(manifest_file), must_exist=True)
        else:
            manifest_file = manifest_file.resolve()
        if not manifest_file.is_file():
            raise FileNotFoundError(f"MatterGen manifest not found: {manifest_file}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest_spec = _validate_manifest_run_spec(
            manifest,
            expected={
                "pretrained_name": pretrained_name,
                "candidate_count": int(self.provider.candidate_limit),
                "target_band_gap_eV": band_gap_float,
                "chemical_system": (
                    chemical_system if pretrained_name == "chemical_system" else None
                ),
                "guidance_factor": float(guidance_factor),
                "seed": seed,
            },
        )
        expected_properties = (
            {"dft_band_gap": band_gap_float}
            if pretrained_name == "dft_band_gap"
            else {"chemical_system": chemical_system}
        )
        if manifest.get("properties_to_condition_on") != expected_properties:
            raise ValueError(
                "MatterGen manifest properties_to_condition_on does not match "
                "the requested conditioning mode"
            )
        reproducibility = manifest.get("reproducibility")
        if not isinstance(reproducibility, dict):
            reproducibility = {
                "seed_requested": manifest_spec["seed"],
                "seed_applied": True,
            }
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
                    "target_band_gap_eV": manifest_spec["target_band_gap_eV"],
                    "chemical_system": manifest_spec["chemical_system"],
                    "pretrained_name": manifest_spec["pretrained_name"],
                    "properties_to_condition_on": manifest.get(
                        "properties_to_condition_on"
                    ),
                    "guidance_factor": manifest_spec["guidance_factor"],
                    "seed": manifest_spec["seed"],
                    "seed_effective": reproducibility.get("seed_applied", True),
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
            "pretrained_name": manifest_spec["pretrained_name"],
            "properties_to_condition_on": manifest.get("properties_to_condition_on"),
            "guidance_factor": manifest_spec["guidance_factor"],
            "seed": manifest_spec["seed"],
            "reproducibility": reproducibility,
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


def _validate_manifest_run_spec(
    manifest: Any,
    *,
    expected: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("MatterGen manifest must be a JSON object")
    actual = manifest.get("run_spec")
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError(
            "MatterGen explicit manifest must contain a complete run_spec "
            "with checkpoint, candidate_count, target_band_gap_eV, "
            "chemical_system, guidance_factor, and seed"
        )
    validated: dict[str, Any] = {}
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            if actual_value is None:
                raise ValueError(
                    f"MatterGen manifest run_spec {key} does not match request"
                )
            try:
                actual_float = float(actual_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"MatterGen manifest run_spec {key} does not match request"
                ) from exc
            if not math.isfinite(actual_float) or not math.isclose(
                actual_float, expected_value, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"MatterGen manifest run_spec {key} does not match request"
                )
            validated[key] = actual_float
        elif actual_value != expected_value:
            raise ValueError(
                f"MatterGen manifest run_spec {key} does not match request"
            )
        else:
            validated[key] = actual_value
    return validated
