"""Deferred ``generation.*`` tools (Sprint 3 section 44-47)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.scientific.capabilities.generation.formulas import (
    VAEFormulaGenerator,
)
from photomatagent.scientific.capabilities.generation.mattergen import (
    LocalIsolatedMatterGenProvider,
    MatterGenGenerator,
)
from photomatagent.scientific.errors import MissingScientificPrerequisite
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure

UNSUPPORTED_DEVICE_PROPERTIES = {
    "responsivity",
    "responsivity_a_w",
    "eqe",
    "detectivity",
    "detectivity_jones",
    "dark_current",
    "dark_current_a",
    "dark_current_density_a_cm2",
    "response_time_s",
    "noise_equivalent_power",
}

PACKAGED_VAE_ASSET_ROOT = (
    Path(__file__).resolve().parent / "assets" / "photoelectric_vae"
)


class GenerationCapabilitiesTool(Tool):
    name = "generation.capabilities"
    description = (
        "List candidate-generation capabilities: multi-property VAE formula "
        "generation (composition prior / formula proposal) and "
        "MatterGen structure generation (dft_band_gap / chemical_system "
        "modes, isolated environment). Reports dependency state; generated "
        "candidates are UNVALIDATED_GENERATED_STRUCTURE."
    )
    short_description = "Candidate generation capabilities and dependency state."
    exposure = ToolExposure.DEFERRED
    namespace = "generation"
    source = "capability metadata"
    tags = (
        "generation",
        "vae",
        "mattergen",
        "candidates",
        "成分生成",
        "组分生成",
        "化学式生成",
    )
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, config: Any = None) -> None:
        self.config = config

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        torch_available = importlib.util.find_spec("torch") is not None
        vae_checkpoint, vae_metadata = _resolve_vae_assets()
        if not torch_available:
            vae_status = "MISSING_DEPENDENCY"
            vae_detail = "install the generation extra to provide PyTorch"
        elif vae_checkpoint is None:
            vae_status = "UNCONFIGURED"
            vae_detail = (
                "packaged VAE assets are missing; reinstall the generation "
                "extra or set an explicit VAE_CHECKPOINT_PATH override"
            )
        elif vae_metadata is None:
            vae_status = "PARTIAL"
            vae_detail = (
                f"checkpoint: {vae_checkpoint}; set VAE_METADATA_PATH for "
                "training-set novelty filtering"
            )
        else:
            vae_status = "AVAILABLE"
            vae_detail = (
                f"checkpoint: {vae_checkpoint}; novelty metadata: {vae_metadata}"
            )
        mattergen_script = _env("MATTERGEN_SKILL_SCRIPT")
        payload = {
            "vae_formula": {
                "status": vae_status,
                "detail": vae_detail,
                "scope": (
                    "composition prior / formula proposal / candidate "
                    "retrieval only; VAE does NOT predict responsivity, EQE, "
                    "detectivity, or dark current"
                ),
                "defaults": (
                    "no default forbidden elements; no atomic-number "
                    "preference; heavy infrared elements are legitimate "
                    "candidates"
                ),
            },
            "mattergen": {
                "status": (
                    "AVAILABLE"
                    if mattergen_script
                    else "UNCONFIGURED"
                ),
                "detail": (
                    f"skill script: {mattergen_script}"
                    if mattergen_script
                    else "set MATTERGEN_SKILL_SCRIPT to enable generation"
                ),
                "modes": ["dft_band_gap", "chemical_system"],
                "execution": "isolated environment (conda/uv), never the main venv",
            },
            "cost_class": {
                "vae_formula": "CHEAP",
                "mattergen": "MODERATE",
            },
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
        )


class VAEFormulaTool(Tool):
    name = "generation.vae_formula"
    description = (
        "Propose integer, charge-balanced (optional novel) compositions "
        "conditioned on any supported combination of material properties "
        "using a 14-condition VAE. target_properties may include gap, "
        "cutoff wavelength, formation/hull energy, density, dielectric "
        "constant, carrier masses, elastic moduli, exfoliation energy, IR "
        "modes, and spillage. forbidden_elements is an OPTIONAL user "
        "constraint (default: none). Scope: composition prior / formula "
        "proposal only -- never predicts responsivity/EQE/detectivity/dark "
        "current. Loads the JARVIS conditional-VAE checkpoint packaged with "
        "PhotomatAgent; VAE_CHECKPOINT_PATH is an optional override. Reports "
        "typed missing prerequisites when model assets are unavailable."
    )
    short_description = "Generate formulas from one or more material properties."
    exposure = ToolExposure.DEFERRED
    namespace = "generation"
    source = "photomatagent VAE formula generator (donor migration)"
    tags = (
        "generation",
        "vae",
        "formula",
        "composition",
        "成分生成",
        "组分生成",
        "化学式生成",
    )
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {
            "target_properties": {
                "type": "object",
                "description": (
                    "One or more material-level conditions used directly by "
                    "the trained VAE. Unspecified fields remain unconditioned."
                ),
                "properties": {
                    "gap_selected_eV": {"type": "number", "exclusiveMinimum": 0},
                    "cutoff_wavelength_um_from_gap": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                    },
                    "formation_energy_eV_per_atom": {"type": "number"},
                    "energy_above_hull_eV_per_atom": {"type": "number"},
                    "density_g_cm3": {"type": "number", "minimum": 0},
                    "dielectric_mean": {"type": "number", "minimum": 0},
                    "avg_electron_mass_m0": {"type": "number"},
                    "avg_hole_mass_m0": {"type": "number"},
                    "bulk_modulus_GPa": {"type": "number"},
                    "shear_modulus_GPa": {"type": "number"},
                    "exfoliation_energy_meV_per_atom": {"type": "number"},
                    "max_IR_mode_cm-1": {"type": "number"},
                    "min_IR_mode_cm-1": {"type": "number"},
                    "spillage": {"type": "number"},
                },
                "additionalProperties": False,
                "minProperties": 1,
            },
            "target_band_gap_eV": {"type": "number", "minimum": 0},
            "target_wavelength_um": {"type": "number", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 32},
            "forbidden_elements": {
                "type": "array",
                "items": {"type": "string"},
            },
            "require_charge_neutral": {"type": "boolean"},
            "require_novel": {"type": "boolean"},
            "checkpoint_path": {"type": "string"},
            "metadata_path": {"type": "string"},
            "sample_count": {
                "type": "integer",
                "minimum": 8,
                "maximum": 4096,
            },
            "random_seed": {"type": "integer", "minimum": 0},
        },
    }

    def __init__(self, config: Any = None) -> None:
        self.config = config

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        target_properties = arguments.get("target_properties") or {}
        if not isinstance(target_properties, dict):
            return ScientificToolResult(
                output="target_properties must be an object",
                is_error=True,
                data={"error_type": "invalid_input"},
            )
        unsupported = sorted(
            key
            for key in set(arguments) | set(target_properties)
            if key in UNSUPPORTED_DEVICE_PROPERTIES
        )
        if unsupported:
            return ScientificToolResult(
                output=(
                    "unsupported device property: the VAE proposes "
                    "compositions only and cannot predict "
                    + ", ".join(unsupported)
                    + "; use photodetector/transport/device capabilities "
                    "for device properties"
                ),
                is_error=True,
                data={
                    "error_type": "unsupported_device_property",
                    "unsupported": unsupported,
                },
            )
        checkpoint, metadata = _resolve_vae_assets(
            checkpoint_path=arguments.get("checkpoint_path"),
            metadata_path=arguments.get("metadata_path"),
        )
        generator = VAEFormulaGenerator(
            checkpoint_path=checkpoint,
            metadata_path=metadata,
            sample_count=int(arguments.get("sample_count", 512)),
            random_seed=int(arguments.get("random_seed", 42)),
            require_charge_neutral=bool(
                arguments.get("require_charge_neutral", True)
            ),
            require_novel=bool(arguments.get("require_novel", True)),
        )
        try:
            proposals, metadata = generator.generate(
                target_properties=target_properties,
                target_band_gap_eV=(
                    float(arguments["target_band_gap_eV"])
                    if arguments.get("target_band_gap_eV") is not None
                    else None
                ),
                target_wavelength_um=(
                    float(arguments["target_wavelength_um"])
                    if arguments.get("target_wavelength_um") is not None
                    else None
                ),
                limit=int(arguments.get("limit", 8)),
                forbidden_elements=arguments.get("forbidden_elements", []),
            )
        except Exception as exc:
            is_prerequisite = isinstance(exc, MissingScientificPrerequisite)
            return ScientificToolResult(
                output=f"generation.vae_formula failed: {exc}",
                is_error=True,
                data={
                    "error_type": (
                        "missing_prerequisites"
                        if is_prerequisite
                        else type(exc).__name__
                    ),
                    "message": str(exc),
                    "missing": getattr(exc, "missing", []),
                },
            )
        payload = {
            "proposals": [proposal.as_dict() for proposal in proposals],
            "metadata": {key: value for key, value in metadata.items()},
        }
        evidence = [
            ScientificEvidence(
                subject="generated_candidates",
                property="proposed_formula",
                value=[proposal.formula for proposal in proposals],
                unit="",
                source="photomatagent VAE formula generator",
                source_type="generative_model",
                method="property-conditioned CVAE decoding + deterministic filters",
                fidelity="ml_generated",
                summary=(
                    f"{len(proposals)} VAE formula proposal(s) conditioned "
                    f"on {metadata['conditioned_property_count']} material "
                    "property value(s)"
                ),
                limitations=(
                    "proposals are UNVALIDATED_GENERATED_STRUCTURE; VAE does "
                    "not predict device performance"
                ),
                provenance={"tool": self.name},
            )
        ]
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
            evidence=evidence,
        )


class VAERetrieveTool(Tool):
    name = "generation.vae_retrieve"
    description = (
        "Retrieve known structures matching a formula or element system "
        "from the packaged JARVIS candidate metadata or an optional local "
        "formula-index CSV override. "
        "Composition prior / candidate retrieval only; no device-property "
        "prediction. Optional override path via VAE_INDEX_PATH."
    )
    short_description = "Retrieve known structures for a formula/system."
    exposure = ToolExposure.DEFERRED
    namespace = "generation"
    source = "local formula index"
    tags = ("generation", "vae", "retrieval", "index")
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {
            "formula": {"type": "string"},
            "chemical_system": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
        },
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        import csv

        index_path = _env("VAE_INDEX_PATH")
        _, packaged_metadata = _resolve_vae_assets()
        path = Path(index_path).expanduser() if index_path else packaged_metadata
        if path is None:
            return ScientificToolResult(
                output="packaged VAE candidate metadata is missing",
                is_error=True,
                data={
                    "error_type": "missing_prerequisites",
                    "missing": ["packaged VAE candidate metadata"],
                },
            )
        if not path.is_file():
            return ScientificToolResult(
                output=f"formula index not found: {path}",
                is_error=True,
                data={"error_type": "not_found"},
            )
        formula = str(arguments.get("formula", "")).strip()
        system = str(arguments.get("chemical_system", "")).strip()
        if not formula and not system:
            return ScientificToolResult(
                output="provide formula or chemical_system",
                is_error=True,
                data={"error_type": "invalid_input"},
            )
        max_results = int(arguments.get("max_results", 10))
        matches: list[dict[str, Any]] = []
        try:
            if path.suffix.lower() == ".json":
                raw_rows = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw_rows, list):
                    raise ValueError("candidate metadata must be a JSON list")
                rows = (
                    row for row in raw_rows if isinstance(row, dict)
                )
            else:
                handle = path.open("r", encoding="utf-8", newline="")
                rows = csv.DictReader(handle)
            try:
                for row_index, row in enumerate(rows):
                    row_formula = str(row.get("formula", "")).strip()
                    row_system = _row_chemical_system(row)
                    if formula and row_formula == formula:
                        matches.append({**row, "model_row_index": row_index})
                    elif system and row_system == _canonical_system(system):
                        matches.append({**row, "model_row_index": row_index})
                    if len(matches) >= max_results:
                        break
            finally:
                if path.suffix.lower() != ".json":
                    handle.close()
        except Exception as exc:
            return ScientificToolResult(
                output=f"index read failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ScientificToolResult(
            output=json.dumps(
                {
                    "matches": matches,
                    "count": len(matches),
                "source": str(path),
                "note": (
                    "packaged JARVIS database retrieval only; no "
                    "device-property claims"
                ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            data={"matches": matches, "count": len(matches)},
        )


class MatterGenTool(Tool):
    name = "generation.mattergen"
    description = (
        "Generate crystal structures with MatterGen (isolated environment): "
        "target dft_band_gap or chemical_system mode. When a VAE "
        "proposed_formula is supplied, the output records formula "
        "consistency (formula_preserved, composition_distance) -- the VAE "
        "formula and MatterGen formula are separate scientific facts. All "
        "candidates are UNVALIDATED_GENERATED_STRUCTURE. Requires "
        "MATTERGEN_SKILL_SCRIPT or an existing manifest_path."
    )
    short_description = "MatterGen structure generation (isolated env)."
    exposure = ToolExposure.DEFERRED
    namespace = "generation"
    source = "MatterGen (isolated environment)"
    tags = ("generation", "mattergen", "structure", "candidates")
    cost_class = "MODERATE"
    input_schema = {
        "type": "object",
        "properties": {
            "target_band_gap_eV": {"type": "number", "minimum": 0},
            "target_wavelength_um": {"type": "number", "minimum": 0},
            "chemical_system": {"type": "string"},
            "proposed_formula": {"type": "string"},
            "candidate_count": {"type": "integer", "minimum": 1, "maximum": 32},
            "manifest_path": {"type": "string"},
            "output_dir": {"type": "string"},
        },
    }

    def __init__(self, config: Any = None) -> None:
        self.config = config

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        provider = LocalIsolatedMatterGenProvider(
            skill_script=_env("MATTERGEN_SKILL_SCRIPT") or None,
            candidate_limit=int(arguments.get("candidate_count", 8)),
        )
        generator = MatterGenGenerator(provider=provider)
        try:
            candidates, metadata = generator.generate(
                target_band_gap_eV=(
                    float(arguments["target_band_gap_eV"])
                    if arguments.get("target_band_gap_eV") is not None
                    else None
                ),
                target_wavelength_um=(
                    float(arguments["target_wavelength_um"])
                    if arguments.get("target_wavelength_um") is not None
                    else None
                ),
                chemical_system=arguments.get("chemical_system"),
                proposed_formula=arguments.get("proposed_formula"),
                manifest_path=arguments.get("manifest_path"),
                output_dir_override=arguments.get("output_dir"),
            )
        except Exception as exc:
            return ScientificToolResult(
                output=f"generation.mattergen failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error_type": type(exc).__name__, "message": str(exc)},
            )
        payload = {
            "candidates": candidates,
            "metadata": metadata,
        }
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
        )


class GenerationCapabilityPack(CapabilityPack):
    name = "generation"
    description = "Candidate generation: VAE formula proposals + MatterGen."
    execution_mode = "subprocess/local"
    backend_name = "isolated environments (torch/conda)"

    def probe(self) -> ProbeResult:
        torch_available = importlib.util.find_spec("torch") is not None
        checkpoint, metadata = _resolve_vae_assets()
        script = _env("MATTERGEN_SKILL_SCRIPT")
        vae_available = torch_available and checkpoint is not None
        if vae_available or script:
            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                detail=(
                    f"torch={'yes' if torch_available else 'no'}; "
                    f"vae checkpoint={'set' if checkpoint else 'unset'}; "
                    f"vae metadata={'set' if metadata else 'unset'}; "
                    f"mattergen script={'set' if script else 'unset'}"
                ),
            )
        return ProbeResult(
            status=CapabilityStatus.MISSING_DEPENDENCY,
            detail=(
                "VAE runtime/assets and MATTERGEN_SKILL_SCRIPT are unavailable; "
                "generation tools return typed missing_prerequisites"
            ),
        )

    def tools(self) -> list[Tool]:
        return [
            GenerationCapabilitiesTool(),
            VAEFormulaTool(),
            VAERetrieveTool(),
            MatterGenTool(),
        ]


def _env(name: str) -> str:
    import os

    return os.environ.get(name, "").strip()


def _resolve_vae_assets(
    *,
    checkpoint_path: Any = None,
    metadata_path: Any = None,
) -> tuple[Path | None, Path | None]:
    """Resolve packaged VAE assets with optional explicit overrides."""

    explicit_checkpoint = str(
        checkpoint_path or _env("VAE_CHECKPOINT_PATH")
    ).strip()
    explicit_metadata = str(
        metadata_path or _env("VAE_METADATA_PATH")
    ).strip()

    checkpoint = _existing_file(explicit_checkpoint)
    metadata = _existing_file(explicit_metadata)
    if explicit_checkpoint and checkpoint is None:
        return None, metadata
    if checkpoint is not None and not explicit_metadata:
        metadata = _existing_file(
            checkpoint.parent.parent
            / "jarvis_inverse_v1"
            / "candidate_metadata.json"
        )
    if checkpoint is not None:
        return checkpoint, metadata

    configured_roots: list[Path] = []
    for name in (
        "PHOTOMATAGENT_VAE_ASSET_ROOT",
        "PHOTOELECTRIC_VAE_ASSET_ROOT",
        "VAE_ASSET_ROOT",
    ):
        value = _env(name)
        if value:
            configured_roots.append(Path(value).expanduser())

    roots = configured_roots or [PACKAGED_VAE_ASSET_ROOT]
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidate_checkpoint = _existing_file(
            resolved / "jarvis_cvae_v1" / "checkpoint.pt"
        )
        if candidate_checkpoint is None:
            continue
        candidate_metadata = metadata
        if not explicit_metadata:
            candidate_metadata = _existing_file(
                resolved / "jarvis_inverse_v1" / "candidate_metadata.json"
            )
        return candidate_checkpoint, candidate_metadata
    return None, metadata


def _existing_file(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_file() else None


def _canonical_system(value: str) -> str:
    normalized = value.replace(";", "-").replace(",", "-")
    return "-".join(
        sorted(item.strip() for item in normalized.split("-") if item.strip())
    )


def _row_chemical_system(row: dict[str, Any]) -> str:
    value = str(
        row.get("chemical_system") or row.get("elements") or ""
    ).strip()
    return _canonical_system(value)


def generation_pack(config: Any = None) -> GenerationCapabilityPack:
    return GenerationCapabilityPack()
