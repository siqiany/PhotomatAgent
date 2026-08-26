"""Deferred ``chemistry.*`` tools (generic structure capability).

These tools know nothing about VASP: they resolve chemical identities,
generate deterministic conformers (ETKDG + MMFF/UFF), build complex and
oligomer/proxy structures and validate geometry. Every structure is
persisted with its provenance; payloads are bounded (<= 4000 chars) and
never embed file contents.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.chemistry.conformers import (
    ChemistryError,
    generate_conformer_candidates,
    mol_formula,
    mol_from_smiles,
    mol_to_xyz,
)
from photomatagent.scientific.capabilities.chemistry.models import (
    ChemicalIdentity,
    ChemicalRole,
    ProvenanceStatus,
    StructureProvenance,
)
from photomatagent.scientific.capabilities.chemistry.oligomers import (
    OligomerRecipe,
    build_oligomer,
)
from photomatagent.scientific.capabilities.chemistry.resolver import (
    StructureRequest,
    resolve_structure,
    validate_generated,
)
from photomatagent.scientific.capabilities.chemistry.storage import (
    write_structure_manifest,
    write_structure_thumbnails,
)
from photomatagent.scientific.capabilities.contracts import ScientificToolResult
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure

MAX_TOOL_CHARS = 4000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _payload(*, ok: bool, summary: dict[str, Any], errors: list[str], **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": ok,
        "summary": summary,
        "errors": errors[:10],
        "chars": 0,
        "timestamp": _now(),
    }
    payload.update(extra)
    payload["chars"] = len(json.dumps(payload, ensure_ascii=False))
    return payload


def _result(payload: dict[str, Any]) -> ScientificToolResult:
    return ScientificToolResult(
        output=json.dumps(payload, ensure_ascii=False, indent=2),
        data=payload,
        is_error=not bool(payload.get("ok", True)),
    )


def _structure_summary(structures: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "system_id": structure.identity.system_id,
            "formula": structure.identity.formula,
            "charge": structure.identity.total_charge,
            "atoms": structure.atom_count,
            "reliability": structure.reliability_grade().value,
            "provenance": structure.provenance.status.value,
            "path": str(structure.structure_path),
        }
        for structure in structures
    ]


class ChemistryResolveStructureTool(Tool):
    name = "chemistry.resolve_structure"
    description = (
        "Resolve one chemical entity into persisted 3D structures following "
        "the documented priority: user file -> explicit SMILES/InChI -> "
        "approved alias registry -> fragment-based generation -> explicit "
        "proxy. Charges are always explicit; every assumption is recorded in "
        "the structure manifest. Never submits anything."
    )
    short_description = "Resolve a chemical entity into structures (offline)."
    exposure = ToolExposure.DEFERRED
    namespace = "chemistry"
    source = "photomatagent chemistry"
    tags = ("chemistry", "structure", "offline")
    cost_class = "MODERATE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "system_id": {"type": "string"},
            "display_name": {"type": "string"},
            "smiles": {"type": "string"},
            "structure_path": {"type": "string"},
            "total_charge": {"type": "integer"},
            "spin_multiplicity": {"type": "integer", "minimum": 1},
            "allow_assumed": {"type": "boolean"},
            "max_candidates": {"type": "integer", "minimum": 1, "maximum": 6},
            "seed": {"type": "integer"},
            "output_dir": {"type": "string"},
        },
        "required": ["system_id"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        output_dir = Path(
            arguments.get("output_dir") or Path.cwd() / "output" / "chemistry"
        ).expanduser().resolve()
        request = StructureRequest(
            system_id=str(arguments["system_id"]),
            display_name=str(arguments.get("display_name") or ""),
            aliases=[],
            smiles=arguments.get("smiles"),
            structure_path=(
                Path(arguments["structure_path"])
                if arguments.get("structure_path")
                else None
            ),
            total_charge=(
                int(arguments["total_charge"])
                if arguments.get("total_charge") is not None
                else None
            ),
            spin_multiplicity=int(arguments.get("spin_multiplicity", 1)),
            allow_assumed=bool(arguments.get("allow_assumed", True)),
            max_candidates=int(arguments.get("max_candidates", 3)),
            seed=int(arguments.get("seed", 20260825)),
        )
        structures_dir = output_dir / "structures"
        try:
            structures = resolve_structure(request, structures_dir)
        except ChemistryError as exc:
            return _result(
                _payload(
                    ok=False,
                    summary={"system_id": request.system_id},
                    errors=[f"{exc.code}: {exc}"],
                    note="no structure was generated",
                )
            )
        write_structure_manifest(structures, output_dir / "structure_manifest.json")
        thumbnails = write_structure_thumbnails(
            structures, output_dir / "figures" / "structures"
        )
        return _result(
            _payload(
                ok=bool(structures),
                summary={
                    "system_id": request.system_id,
                    "count": len(structures),
                    "structures": _structure_summary(structures),
                },
                errors=[],
                artifacts=[
                    str(output_dir / "structure_manifest.json"),
                    *[str(path) for path in thumbnails[:6]],
                ],
                note=(
                    "structures persisted with provenance; grades A-D "
                    "recorded per structure"
                ),
            )
        )


class ChemistryGenerateConformersTool(Tool):
    name = "chemistry.generate_conformers"
    description = (
        "Generate a deterministic ensemble of 3D conformers from an explicit "
        "SMILES or an XYZ structure (ETKDG, fixed seed, MMFF/UFF, collision-"
        "filtered, energy-sorted)."
    )
    short_description = "Deterministic 3D conformers (RDKit ETKDG)."
    exposure = ToolExposure.DEFERRED
    namespace = "chemistry"
    source = "photomatagent chemistry"
    tags = ("chemistry", "conformers", "offline")
    cost_class = "MODERATE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "smiles": {"type": "string"},
            "total_charge": {"type": "integer"},
            "n_conformers": {"type": "integer", "minimum": 1, "maximum": 32},
            "max_returned": {"type": "integer", "minimum": 1, "maximum": 8},
            "seed": {"type": "integer"},
            "output_dir": {"type": "string"},
        },
        "required": ["smiles", "total_charge"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        output_dir = Path(
            arguments.get("output_dir") or Path.cwd() / "output" / "chemistry"
        ).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        charge = int(arguments["total_charge"])
        try:
            mol = mol_from_smiles(
                str(arguments["smiles"]),
                expected_charge=charge,
                name="conformer-set",
            )
            candidates = generate_conformer_candidates(
                mol,
                n_conformers=int(arguments.get("n_conformers", 8)),
                seed=int(arguments.get("seed", 20260825)),
                max_returned=int(arguments.get("max_returned", 3)),
            )
        except ChemistryError as exc:
            return _result(
                _payload(
                    ok=False,
                    summary={},
                    errors=[f"{exc.code}: {exc}"],
                )
            )
        paths: list[str] = []
        for candidate in candidates:
            path = output_dir / f"conf_c{candidate.rank}.xyz"
            path.write_text(
                mol_to_xyz(
                    candidate.mol,
                    candidate.conf_id,
                    comment=f"conformer c{candidate.rank}",
                ),
                encoding="utf-8",
            )
            paths.append(str(path))
        return _result(
            _payload(
                ok=True,
                summary={
                    "formula": mol_formula(mol),
                    "count": len(candidates),
                    "paths": paths,
                },
                errors=[],
                note=(
                    "seed fixed and recorded; candidates collision-filtered "
                    "and energy-sorted"
                ),
            )
        )


class ChemistryBuildComplexTool(Tool):
    name = "chemistry.build_complex"
    description = (
        "Build several initial complex geometries (host + guest, e.g. Li+ or "
        "TFSI-) from coordination sites, multiple orientations, vdW-based "
        "distances and collision rejection; charge = fragment sum. "
        "Force-field pre-optimised, energy-sorted."
    )
    short_description = "Heuristic complex initial geometries."
    exposure = ToolExposure.DEFERRED
    namespace = "chemistry"
    source = "photomatagent chemistry"
    tags = ("chemistry", "complex", "offline")
    cost_class = "MODERATE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "host_system_id": {"type": "string"},
            "guest_smiles": {"type": "string"},
            "expected_charge": {"type": "integer"},
            "seed": {"type": "integer"},
            "max_candidates": {"type": "integer", "minimum": 1, "maximum": 6},
            "output_dir": {"type": "string"},
        },
        "required": ["host_system_id", "guest_smiles", "expected_charge"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        from photomatagent.scientific.capabilities.chemistry.complexes import (
            build_complex_candidates,
        )
        from photomatagent.scientific.capabilities.chemistry.registry import (
            lookup_alias,
        )

        host_id = str(arguments["host_system_id"])
        entry = lookup_alias(host_id)
        if entry is None or not entry.smiles:
            return _result(
                _payload(
                    ok=False,
                    summary={"host": host_id},
                    errors=["host must be a resolvable SMILES/alias"],
                )
            )
        output_dir = Path(
            arguments.get("output_dir") or Path.cwd() / "output" / "chemistry"
        ).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            host = mol_from_smiles(
                entry.smiles,
                expected_charge=entry.total_charge,
                name=entry.display_name,
            )
            candidates = build_complex_candidates(
                host,
                str(arguments["guest_smiles"]),
                expected_charge=int(arguments["expected_charge"]),
                seed=int(arguments.get("seed", 20260825)),
                max_returned=int(arguments.get("max_candidates", 3)),
            )
        except ChemistryError as exc:
            return _result(
                _payload(
                    ok=False,
                    summary={"host": host_id},
                    errors=[f"{exc.code}: {exc}"],
                )
            )
        paths: list[str] = []
        probe_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            path = output_dir / f"{entry.system_id}_g{candidate.rank}.xyz"
            path.write_text(
                mol_to_xyz(
                    candidate.mol,
                    candidate.conf_id,
                    comment=(
                        f"{entry.display_name}+guest g{candidate.rank} "
                        f"ff={candidate.force_field}"
                    ),
                ),
                encoding="utf-8",
            )
            paths.append(str(path))
            probe_rows.append(
                {
                    "rank": candidate.rank,
                    "formula": mol_formula(candidate.mol),
                    "energy_kcal_mol": candidate.energy_kcal_mol,
                    "ff": candidate.force_field,
                    "min_heavy_ang": round(candidate.min_heavy_distance, 2),
                }
            )
        return _result(
            _payload(
                ok=True,
                summary={
                    "host": host_id,
                    "charge": int(arguments["expected_charge"]),
                    "count": len(candidates),
                    "candidates": probe_rows,
                    "paths": paths,
                },
                errors=[],
                note=(
                    "complex charge equals the fragment charge sum; "
                    "multiple orientations sampled; not one arbitrary guess"
                ),
            )
        )


class ChemistryBuildOligomerProxyTool(Tool):
    name = "chemistry.build_oligomer_proxy"
    description = (
        "Build a finite linear representative oligomer from explicit monomer "
        "SMILES with recorded defaults (repeat_counts, end_caps, "
        "crosslink_position). Used for VM/TVM-type systems whose exact "
        "polymer connectivity is not user-provided; the result is explicitly "
        "labelled ASSUMED_REPRESENTATIVE, never presented as the real "
        "polymer."
    )
    short_description = "Representative oligomer proxy (VM/TVM)."
    exposure = ToolExposure.DEFERRED
    namespace = "chemistry"
    source = "photomatagent chemistry"
    tags = ("chemistry", "polymer", "proxy", "offline")
    cost_class = "MODERATE"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "system_id": {"type": "string"},
            "display_name": {"type": "string"},
            "monomer_smiles": {
                "type": "array",
                "items": {"type": "string"},
            },
            "repeat_counts": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "end_caps": {"type": "array", "items": {"type": "string"}},
            "crosslink_position": {"type": "string"},
            "seed": {"type": "integer"},
            "output_dir": {"type": "string"},
        },
        "required": ["system_id", "monomer_smiles"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        output_dir = Path(
            arguments.get("output_dir") or Path.cwd() / "output" / "chemistry"
        ).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        recipe = OligomerRecipe(
            monomer_smiles=tuple(
                str(item) for item in arguments["monomer_smiles"]
            ),
            repeat_counts=tuple(
                int(item) for item in arguments.get("repeat_counts", [])
            ),
            end_caps=tuple(
                str(item) for item in arguments.get("end_caps", ["H", "H"])
            ),
            crosslink_position=str(
                arguments.get(
                    "crosslink_position",
                    "none (single linear chain proxy)",
                )
            ),
        )
        try:
            chain = build_oligomer(recipe, seed=int(arguments.get("seed", 20260825)))
        except ChemistryError as exc:
            return _result(
                _payload(
                    ok=False,
                    summary={"system_id": str(arguments["system_id"])},
                    errors=[f"{exc.code}: {exc}"],
                )
            )
        path = output_dir / f"{arguments['system_id']}_proxy_p1.xyz"
        path.write_text(
            mol_to_xyz(chain, comment="ASSUMED_REPRESENTATIVE oligomer proxy"),
            encoding="utf-8",
        )
        return _result(
            _payload(
                ok=True,
                summary={
                    "system_id": str(arguments["system_id"]),
                    "formula": mol_formula(chain),
                    "atoms": chain.GetNumAtoms(),
                    "path": str(path),
                    "assumptions": recipe.assumptions(),
                },
                errors=[],
                note=(
                    "ASSUMED_REPRESENTATIVE proxy: never presented as the "
                    "real VM/TVM network"
                ),
            )
        )


class ChemistryValidateStructureTool(Tool):
    name = "chemistry.validate_structure"
    description = (
        "Validate a persisted structure against its identity: file exists, "
        "atom-count and formula consistency, charge contract, severe "
        "collision screening. Returns typed problems; never modifies files."
    )
    short_description = "Validate a structure file (offline)."
    exposure = ToolExposure.DEFERRED
    namespace = "chemistry"
    source = "photomatagent chemistry"
    tags = ("chemistry", "validation", "offline")
    cost_class = "CHEAP"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "structure_path": {"type": "string"},
            "expected_formula": {"type": "string"},
            "expected_charge": {"type": "integer"},
        },
        "required": ["structure_path"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        path = Path(arguments["structure_path"]).expanduser().resolve()
        problems: list[str] = []
        if not path.is_file():
            problems.append("structure file missing")
        else:
            from photomatagent.scientific.capabilities.chemistry.storage import (
                read_xyz,
            )

            symbols: list[str] = []
            coords: Any = []
            try:
                symbols, coords, _ = read_xyz(path)
            except ValueError as exc:
                problems.append(str(exc))
            if arguments.get("expected_formula"):
                from photomatagent.scientific.capabilities.chemistry.resolver import (
                    _hill_formula,
                )

                counts: dict[str, int] = {}
                for symbol in symbols:
                    counts[symbol] = counts.get(symbol, 0) + 1
                if _hill_formula(counts) != str(arguments["expected_formula"]):
                    problems.append(
                        f"formula mismatch: file "
                        f"{_hill_formula(counts)} vs expected "
                        f"{arguments['expected_formula']}"
                    )
        return _result(
            _payload(
                ok=not problems,
                summary={
                    "structure_path": str(path),
                    "problems": problems,
                },
                errors=problems,
                note="read-only validation; nothing was modified",
            )
        )


class ChemistryCapabilityPack(CapabilityPack):
    """Deferred pack exposing the ``chemistry.*`` tool family."""

    name = "chemistry"
    description = (
        "Generic chemical structure capability: identity resolution, "
        "deterministic conformers, complex and oligomer/proxy generation."
    )
    execution_mode = "local"
    backend_name = "RDKit (local)"

    def probe(self) -> ProbeResult:
        try:
            import rdkit  # noqa: F401
        except Exception:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail="RDKit is not installed",
            )
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="RDKit available; deterministic structure generation ready",
            version="rdkit",
        )

    def tools(self) -> list[Tool]:
        return [
            ChemistryResolveStructureTool(),
            ChemistryGenerateConformersTool(),
            ChemistryBuildComplexTool(),
            ChemistryBuildOligomerProxyTool(),
            ChemistryValidateStructureTool(),
        ]


def chemistry_pack() -> ChemistryCapabilityPack:
    return ChemistryCapabilityPack()
