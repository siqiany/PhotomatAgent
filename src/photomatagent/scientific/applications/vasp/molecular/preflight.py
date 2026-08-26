"""Deterministic offline preflight for isolated-molecule VASP workflows.

``run_molecular_preflight`` validates the complete WorkflowSpec (plus the
actual generated stage files when ``stage_dirs`` is provided) and returns a
structured ``PreflightReport``. ``passed`` is true ONLY when every check
passes; the full report is persisted as ``preflight.json``.

The preflight is fully offline: it never connects to SSH, never submits a
job, never runs VASP and never writes or logs POTCAR content. Only the user's
POTCAR TITEL/ZVAL/ENMAX metadata enters the report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from photomatagent.scientific.applications.vasp.molecular.models import (
    MonopoleMethod,
    PreflightConfig,
    PreflightIssue,
    PreflightReport,
    PreflightSummary,
    ResourceProfile,
    StageName,
    WorkflowSpec,
)
from photomatagent.scientific.applications.vasp.molecular.psp_metadata import (
    PspError,
    PotcarResolution,
    resolve_potcar_metadata,
)
from photomatagent.scientific.remote.lifecycle import SubmissionGate
from photomatagent.scientific.applications.vasp.molecular.render import (
    parse_bool,
    parse_float,
    parse_incar,
    parse_int,
    parse_kpoints,
    render_incar,
    render_kpoints_gamma,
)
from photomatagent.scientific.applications.vasp.molecular.structures import (
    StructureData,
    StructureError,
    center_in_cubic_box,
    formula_text,
    grouped_symbols,
    minimum_image_pairs,
    per_side_vacuum,
    read_structure,
)

# The deterministic order in which checks are executed; every executed code is
# recorded in ``report.checks``.
CHECK_ORDER = [
    "MOLECULE_SPEC_VALIDATION",
    "BLOCKED_MISSING_STRUCTURE",
    "STRUCTURE_READABILITY",
    "POSCAR_ELEMENT_BLOCKS_VALID",
    "BOX_GEOMETRY",
    "POTCAR_METADATA",
    "POTCAR_ORDER_AND_BLOCKS",
    "ELECTRON_BOOKKEEPING",
    "CORRECTION_POLICY",
    "STAGE_DAG_AND_DEPENDENCIES",
    "STAGE_INPUT_FILES",
    "KPOINTS_GAMMA_ONLY",
    "ENCUT_POLICY",
    "DIPOL_FORMAT",
    "PARITY_AND_ISPIN",
    "RESOURCE_COMPATIBILITY",
    "RESOURCE_CALIBRATION",
    "POTCAR_CONTENT_POLICY",
]


class _Collector:
    def __init__(self) -> None:
        self.errors: list[PreflightIssue] = []
        self.warnings: list[PreflightIssue] = []
        self.checks: list[str] = []

    def note(self, code: str) -> None:
        if code not in self.checks:
            self.checks.append(code)

    def error(self, code: str, message: str, path: str = "") -> None:
        self.errors.append(PreflightIssue(code=code, message=message, path=path))

    def warn(self, code: str, message: str, path: str = "") -> None:
        self.warnings.append(PreflightIssue(code=code, message=message, path=path))


def _stage_incar_text(
    workflow: WorkflowSpec,
    stage_index: int,
    stage_dirs: dict[StageName, Path] | None,
    collector: _Collector,
) -> str | None:
    stage = workflow.stages[stage_index]
    stage_dir = stage_dirs.get(stage.name) if stage_dirs else None
    if stage_dirs is not None and stage_dir is None:
        collector.error(
            "STAGE_DIR_MISSING",
            f"prepared stage directory for {stage.name.value} was not found",
            f"stages[{stage_index}].name",
        )
        return None
    if stage_dir is not None:
        incar = stage_dir / "INCAR"
        if not incar.is_file():
            collector.error(
                "STAGE_INPUT_MISSING",
                f"INCAR missing in prepared stage directory {stage_dir}",
                f"stages[{stage_index}].directory",
            )
            return None
        return incar.read_text(encoding="utf-8", errors="replace")
    return render_incar(stage.incar)


def _stage_kpoints_text(
    workflow: WorkflowSpec,
    stage_index: int,
    stage_dirs: dict[StageName, Path] | None,
) -> str | None:
    stage = workflow.stages[stage_index]
    stage_dir = stage_dirs.get(stage.name) if stage_dirs else None
    if stage_dir is not None:
        kpoints = stage_dir / "KPOINTS"
        if not kpoints.is_file():
            return None
        return kpoints.read_text(encoding="utf-8", errors="replace")
    return render_kpoints_gamma()


def _available_upstream_outputs(
    workflow: WorkflowSpec, stage_index: int
) -> set[str]:
    """Union of produced outputs reachable through the depends_on chain."""
    available: set[str] = set()
    stage = workflow.stages[stage_index]
    if stage.depends_on is None:
        return available
    for upstream in workflow.stages[:stage_index]:
        if upstream.name == stage.depends_on:
            available.update(upstream.produced_outputs)
            available.update(
                _available_upstream_outputs(
                    workflow, workflow.stages.index(upstream)
                )
            )
    return available


def _transitive_ancestors(
    workflow: WorkflowSpec, stage_index: int
) -> set[StageName]:
    """All stage names reachable through the depends_on chain."""
    ancestors: set[StageName] = set()
    previous: dict[StageName, StageName | None] = {}
    for stage in workflow.stages:
        previous[stage.name] = stage.depends_on
    cursor = workflow.stages[stage_index].depends_on
    while cursor is not None:
        if cursor in ancestors:
            break
        ancestors.add(cursor)
        cursor = previous.get(cursor)
    return ancestors


def run_molecular_preflight(
    workflow: WorkflowSpec,
    *,
    psp_dir: str | Path | None = None,
    potcar_path: str | Path | None = None,
    stage_dirs: dict[StageName, Path] | None = None,
    output_dir: str | Path | None = None,
) -> PreflightReport:
    """Run every preflight check; returns the structured report."""
    collector = _Collector()
    config = workflow.preflight_config
    molecule = workflow.molecule

    # -- 0. model-level sanity ------------------------------------------------
    collector.note("MOLECULE_SPEC_VALIDATION")
    try:
        _ = WorkflowSpec.model_validate(workflow.model_dump(mode="python"))
    except ValidationError as exc:
        for issue in exc.errors():
            collector.error(
                "MOLECULE_SPEC_INVALID",
                f"workflow model invalid: {issue['msg']}",
                ".".join(str(item) for item in issue["loc"]),
            )

    # -- 1. VM/TVM gate -------------------------------------------------------
    collector.note("BLOCKED_MISSING_STRUCTURE")
    blocked = molecule.polymer_kind.value in {"vm", "tvm"}
    if blocked and (
        molecule.structure_path is None
        or molecule.polymerization is None
        or not molecule.polymerization.connectivity.strip()
        or not molecule.polymerization.polymerization_sites
        or not molecule.polymerization.repeat_units
        or not molecule.polymerization.end_caps
    ):
        reason = molecule.blocked_reason or (
            "VM/TVM require an explicit structure file plus declared "
            "connectivity, polymerization sites, repeat units and end caps"
        )
        collector.error(
            "BLOCKED_MISSING_STRUCTURE",
            reason,
            "molecule.polymerization",
        )
        return _finish(collector, workflow, None, output_dir)
    if molecule.structure_path is None:
        collector.error(
            "BLOCKED_MISSING_STRUCTURE",
            "no structure file provided; supply an XYZ/SDF/MOL/POSCAR file",
            "molecule.structure_path",
        )
        return _finish(collector, workflow, None, output_dir)

    # -- 2. structure readability ----------------------------------------------
    collector.note("STRUCTURE_READABILITY")
    structure: StructureData | None = None
    try:
        structure = read_structure(
            molecule.structure_path,
            kind=molecule.structure_kind,
            conformer_index=_conformer_index(molecule.conformer_id),
        )
    except (StructureError, OSError) as exc:
        code = getattr(exc, "code", "STRUCTURE_UNREADABLE")
        collector.error(code, str(exc), "molecule.structure_path")
        return _finish(collector, workflow, None, output_dir)
    elements, counts = grouped_symbols(structure.symbols)

    # -- 3. POSCAR element blocks / counts (POSCAR inputs only) -----------------
    collector.note("POSCAR_ELEMENT_BLOCKS_VALID")
    if structure.source_kind == "poscar":
        if structure.elements is None or structure.counts is None:
            collector.error(
                "POSCAR_ELEMENT_BLOCKS_INVALID",
                "POSCAR without a grouped VASP5 element/counts block",
                "molecule.structure_path",
            )
        elif _poscar_contract_violation(structure, elements, counts):
            collector.error(
                "POSCAR_ELEMENT_BLOCKS_INVALID",
                "POSCAR element/counts block is invalid (repeated symbols or "
                "sum/count mismatch); write the grouped VASP5 form",
                "molecule.structure_path",
            )

    # -- 4. box geometry -------------------------------------------------------
    collector.note("BOX_GEOMETRY")
    box_ang = molecule.box_ang
    if structure.box_ang is not None and abs(structure.box_ang - molecule.box_ang) > 1e-6:
        collector.error(
            "BOX_LATTICE_MISMATCH",
            f"POSCAR lattice length {structure.box_ang:g} A differs from "
            f"declared box_ang {molecule.box_ang:g} A",
            "molecule.box_ang",
        )
        box_ang = structure.box_ang
    raw_positions = np.asarray(structure.positions, dtype=float)
    # The generator always re-centers the molecule in the fixed box; geometry
    # checks must mirror the coordinates that VASP will actually read.
    positions = center_in_cubic_box(raw_positions, box_ang)
    if structure.source_kind == "poscar":
        frac_raw = raw_positions / box_ang
        if np.any(frac_raw <= 0.0) or np.any(frac_raw >= 1.0):
            collector.error(
                "ATOM_OUTSIDE_CELL",
                "an atom of the supplied POSCAR lies on or outside the "
                "periodic cell boundary; the molecule must be fully inside "
                "the fixed vacuum box",
                "molecule.structure_path",
            )
    sides = per_side_vacuum(positions, box_ang)
    min_side = float(sides.min())
    if min_side < config.min_vacuum_per_side_ang:
        collector.error(
            "BOX_VACUUM_INSUFFICIENT",
            f"vacuum thickness {min_side:.2f} A on one side is below the "
            f"required {config.min_vacuum_per_side_ang:g} A per side",
            "molecule.box_ang",
        )
    centroid_frac = positions.mean(axis=0) / box_ang
    if np.any(centroid_frac < 0.25) or np.any(centroid_frac > 0.75):
        collector.warn(
            "MOLECULE_OFF_CENTER",
            f"molecular centroid at fractional {centroid_frac.round(3)} is "
            f"far from the box center; DIPOL must be set accordingly",
            "molecule.structure_path",
        )
    collisions = minimum_image_pairs(
        positions, box_ang, config.min_interatomic_distance_ang
    )
    if collisions:
        pairs = ", ".join(
            f"({i},{j}:{distance:.2f}A)" for i, j, distance in collisions[:5]
        )
        collector.error(
            "INTERATOMIC_COLLISION",
            f"abnormally short atom distances under PBC (first pairs): {pairs}",
            "molecule.structure_path",
        )

    # -- 5. POTCAR metadata ----------------------------------------------------
    collector.note("POTCAR_METADATA")
    resolution: PotcarResolution | None = None
    try:
        resolution = resolve_potcar_metadata(
            molecule,
            elements,
            psp_dir=psp_dir,
            potcar_path=potcar_path,
        )
    except PspError as exc:
        collector.error(getattr(exc, "code", "PSP_METADATA_UNREADABLE"), str(exc))

    # -- 6. POTCAR order / duplicate blocks -------------------------------------
    collector.note("POTCAR_ORDER_AND_BLOCKS")
    if resolution is not None and potcar_path is not None:
        actual = [block.element for block in resolution.blocks]
        if len(actual) != len(elements):
            collector.error(
                "POTCAR_BLOCK_COUNT_MISMATCH",
                f"POTCAR stream has {len(actual)} dataset blocks but the "
                f"molecule needs {len(elements)} ({', '.join(elements)})",
                "potcar_path",
            )
        duplicates = sorted(
            {element for element in actual if actual.count(element) > 1}
        )
        if duplicates:
            collector.error(
                "POTCAR_DUPLICATE_ELEMENT",
                f"the same element appears in multiple POTCAR blocks: "
                f"{', '.join(duplicates)}",
                "potcar_path",
            )
        if actual != elements:
            collector.error(
                "POTCAR_ORDER_MISMATCH",
                f"POTCAR block order {actual} does not match the POSCAR "
                f"element order {elements}"
                + (
                    f" (missing elements: "
                    f"{', '.join(element for element in elements if element not in actual)})"
                    if any(element not in actual for element in elements)
                    else ""
                ),
                "potcar_path",
            )
    if resolution is not None:
        for block in resolution.blocks:
            if block.enmax is None:
                collector.error(
                    "PSP_METADATA_UNREADABLE",
                    f"no ENMAX metadata for {block.element} in {block.source}",
                    f"potcar.datasets[{block.element}].enmax",
                )

    # -- 7. electron bookkeeping ------------------------------------------------
    collector.note("ELECTRON_BOOKKEEPING")
    neutral_electrons: float | None = None
    nelect: float | None = None
    if resolution is not None:
        zvals = {block.element: block.zval for block in resolution.blocks}
        if all(element in zvals for element in elements):
            neutral_electrons = sum(
                zvals[element] * count
                for element, count in zip(elements, counts, strict=True)
            )
            nelect = neutral_electrons - molecule.total_charge
            if nelect < 0:
                collector.error(
                    "NELECT_NEGATIVE",
                    f"NELECT = {nelect:g} is negative; charge "
                    f"{molecule.total_charge} exceeds the neutral valence "
                    f"electron count {neutral_electrons:g}",
                    "molecule.total_charge",
                )
        # else: the POTCAR element-order audit above already reported the
        # missing datasets; electron numbers are unknown and no further
        # electron bookkeeping is attempted.

    summary = _build_summary(
        workflow, elements, counts, box_ang, nelect, neutral_electrons
    )

    # -- 8. correction policy ---------------------------------------------------
    collector.note("CORRECTION_POLICY")
    policy = workflow.correction_policy
    charged = molecule.total_charge != 0
    if charged and policy.monopole_method is MonopoleMethod.NONE:
        collector.error(
            "CHARGED_CORRECTION_UNDECLARED",
            f"charged molecule (q={molecule.total_charge}) must declare a "
            f"monopole correction method (LMONO); NELECT alone leaves "
            f"periodic-image artifacts",
            "correction_policy.monopole_method",
        )
    if (
        charged
        and policy.monopole_method is MonopoleMethod.LMONO
        and policy.dipole
    ):
        collector.error(
            "CONFLICTING_CORRECTIONS",
            "LMONO must not be combined with the dipole correction "
            "(LDIPOL/IDIPOL); the methods conflict",
            "correction_policy",
        )
    if (
        not charged
        and policy.monopole_method is MonopoleMethod.LMONO
    ):
        collector.warn(
            "LMONO_ON_NEUTRAL",
            "LMONO is declared for a neutral molecule; the monopole "
            "correction is only meaningful for charged cells",
            "correction_policy.monopole_method",
        )
    if policy.monopole_method is MonopoleMethod.LMONO:
        collector.warn(
            "LMONO_UNVERIFIED",
            "LMONO support on the installed VASP binary must be smoke-tested "
            "before production submission",
            "correction_policy.monopole_method",
        )

    # -- 9. DAG and dependency contracts ----------------------------------------
    collector.note("STAGE_DAG_AND_DEPENDENCIES")
    _check_dependency_contracts(workflow, collector)

    # -- 10-16. per-stage checks ------------------------------------------------
    for stage_index, stage in enumerate(workflow.stages):
        prefix = f"stages[{stage_index}]"
        incar_text = _stage_incar_text(workflow, stage_index, stage_dirs, collector)
        if incar_text is None:
            continue
        parsed = parse_incar(incar_text)

        collector.note("STAGE_INPUT_FILES")
        collector.note("KPOINTS_GAMMA_ONLY")
        kpoints_text = _stage_kpoints_text(workflow, stage_index, stage_dirs)
        if kpoints_text is None:
            collector.error(
                "STAGE_INPUT_MISSING",
                "KPOINTS missing in prepared stage directory",
                f"{prefix}.directory",
            )
        else:
            kpoints = parse_kpoints(kpoints_text)
            gamma_ok = False
            if kpoints["mode"] == "gamma_auto":
                grid = kpoints.get("grid")
                gamma_ok = grid is not None and grid == [1, 1, 1]
            elif kpoints["mode"] == "explicit":
                points = kpoints.get("points", [])
                gamma_ok = len(points) == 1 and all(
                    abs(value) < 1e-9 for value in points[0]
                )
            if not gamma_ok:
                collector.error(
                    "KPOINTS_NOT_GAMMA_ONLY",
                    "isolated-molecule stages require Gamma-only 1x1x1 "
                    "k-points",
                    f"{prefix}.kpoints",
                )

        collector.note("ENCUT_POLICY")
        encut_raw = parsed.get("ENCUT") or stage.incar.get("ENCUT")
        encut = parse_float(str(encut_raw)) if encut_raw is not None else None
        if encut is None:
            collector.error(
                "ENCUT_MISSING",
                "stage INCAR does not define ENCUT",
                f"{prefix}.incar.ENCUT",
            )
        else:
            if encut < config.encut_floor_ev - 1e-9:
                collector.error(
                    "ENCUT_BELOW_FLOOR",
                    f"ENCUT = {encut:g} eV is below the workflow floor of "
                    f"{config.encut_floor_ev:g} eV",
                    f"{prefix}.incar.ENCUT",
                )
            elif (
                resolution is not None
                and resolution.max_enmax is not None
                and encut
                < config.encut_max_enmax_ratio * resolution.max_enmax - 1e-9
            ):
                collector.error(
                    "ENCUT_BELOW_ENMAX_RATIO",
                    f"ENCUT = {encut:g} eV is below "
                    f"{config.encut_max_enmax_ratio:g} x max ENMAX "
                    f"({resolution.max_enmax:g} eV = "
                    f"{config.encut_max_enmax_ratio * resolution.max_enmax:g} eV)",
                    f"{prefix}.incar.ENCUT",
                )

        collector.note("DIPOL_FORMAT")
        dipol_value = parsed.get("DIPOL")
        if dipol_value is not None:
            if "[" in dipol_value or "]" in dipol_value:
                collector.error(
                    "DIPOL_LIST_FORMAT",
                    f"DIPOL must render as 'DIPOL = 0.5 0.5 0.5'; found Python "
                    f"list syntax {dipol_value!r}",
                    f"{prefix}.incar.DIPOL",
                )
            else:
                tokens = dipol_value.split()
                if len(tokens) != 3 or any(
                    parse_float(token) is None for token in tokens
                ):
                    collector.error(
                        "DIPOL_LIST_FORMAT",
                        f"DIPOL must be three space-separated numbers, found "
                        f"{dipol_value!r}",
                        f"{prefix}.incar.DIPOL",
                    )
        ldipol = parse_bool(parsed.get("LDIPOL", "")) if "LDIPOL" in parsed else None
        if ldipol is True and dipol_value is None:
            collector.warn(
                "LDIPOL_WITHOUT_DIPOL",
                "LDIPOL = .TRUE. but DIPOL is not set; VASP would fall back to "
                "an implicit dipole center",
                f"{prefix}.incar.DIPOL",
            )

        collector.note("PARITY_AND_ISPIN")
        ispin_raw = parsed.get("ISPIN")
        ispin = (
            parse_int(ispin_raw)
            if ispin_raw is not None
            else config.default_ispin
        )
        # ISPIN is a VASP tag, never the spin multiplicity: only {1, 2}
        # exist, and this is checked even when electron metadata is missing.
        if ispin is not None and ispin not in {1, 2}:
            collector.error(
                "ISPIN_INVALID",
                f"ISPIN = {ispin} is not a valid VASP value; ISPIN must "
                "be 1 or 2 (spin_multiplicity is NOT ISPIN)",
                f"{prefix}.incar.ISPIN",
            )
        elif (
            ispin is not None
            and ispin == 1
            and workflow.molecule.spin_multiplicity > 1
        ):
            collector.error(
                "SPIN_POLARIZATION_REQUIRED",
                f"spin_multiplicity = "
                f"{workflow.molecule.spin_multiplicity} (> 1) with "
                f"ISPIN = 1 in this stage; declare ISPIN = 2 (or "
                "spin_polarized=True)",
                f"{prefix}.incar.ISPIN",
            )
        if nelect is not None and resolution is not None:
            if ispin is not None and nelect % 2 == 1 and ispin == 1:
                collector.error(
                    "ELECTRON_PARITY_MISMATCH",
                    f"NELECT = {nelect:g} is odd but ISPIN = 1 in this stage; "
                    f"odd-electron systems require ISPIN = 2",
                    f"{prefix}.incar.ISPIN",
                )
            explicit_nelect = parsed.get("NELECT")
            if explicit_nelect is not None:
                explicit = parse_float(explicit_nelect)
                if explicit is not None and abs(explicit - nelect) > 1e-6:
                    collector.error(
                        "NELECT_MISMATCH",
                        f"explicit NELECT = {explicit:g} contradicts the "
                        f"charge metadata ({nelect:g} = neutral "
                        f"{neutral_electrons:g} - q {molecule.total_charge})",
                        f"{prefix}.incar.NELECT",
                    )
            nupdown_raw = parsed.get("NUPDOWN")
            if nupdown_raw is not None:
                nupdown = parse_int(nupdown_raw)
                if nupdown is None or nupdown < 1:
                    collector.error(
                        "NUPDOWN_INVALID",
                        f"NUPDOWN = {nupdown_raw} must be a positive integer",
                        f"{prefix}.incar.NUPDOWN",
                    )
                elif nupdown is not None and nelect is not None:
                    if nupdown > nelect:
                        collector.error(
                            "NUPDOWN_PARITY_INCONSISTENT",
                            f"NUPDOWN = {nupdown} exceeds NELECT = {nelect:g}",
                            f"{prefix}.incar.NUPDOWN",
                        )
                    elif (int(nelect) - nupdown) % 2:
                        collector.error(
                            "NUPDOWN_PARITY_INCONSISTENT",
                            f"NELECT - NUPDOWN = {int(nelect)} - {nupdown} is "
                            "odd; no integer magnetization fits",
                            f"{prefix}.incar.NUPDOWN",
                        )
                    expected = workflow.molecule.spin_multiplicity - 1
                    if (
                        workflow.molecule.spin_multiplicity > 1
                        and nupdown != expected
                    ):
                        collector.warn(
                            "NUPDOWN_MAPPING_MISMATCH",
                            f"NUPDOWN = {nupdown} does not match the recorded "
                            f"nupdown = multiplicity - 1 = {expected} mapping "
                            "assumption for "
                            f"multiplicity {workflow.molecule.spin_multiplicity}",
                            f"{prefix}.incar.NUPDOWN",
                        )

        collector.note("RESOURCE_COMPATIBILITY")
        resource_violations = workflow.resource_plan.violations(len(workflow.stages))
        for violation in resource_violations:
            collector.error(
                "RESOURCE_PLAN_VIOLATION",
                violation,
                "workflow.resource_plan",
            )
        total_tasks = (
            workflow.resource_ceiling.nodes
            * workflow.resource_ceiling.tasks_per_node
        )
        for key, code in (("NCORE", "NCORE_INCOMPATIBLE"), ("NPAR", "NPAR_INCOMPATIBLE")):
            raw = parsed.get(key) or stage.incar.get(key)
            if raw is None:
                continue
            value = parse_int(str(raw))
            if value is None or value < 1 or total_tasks % value:
                collector.error(
                    code,
                    f"{key} = {raw} must divide total Slurm tasks "
                    f"({workflow.resource_ceiling.nodes} x "
                    f"{workflow.resource_ceiling.tasks_per_node} = "
                    f"{total_tasks})",
                    f"{prefix}.incar.{key}",
                )
        if "NCORE" not in parsed and "NPAR" not in parsed:
            collector.warn(
                "RESOURCE_PARALLELISM_UNDECLARED",
                "neither NCORE nor NPAR is declared in this stage; VASP "
                "parallelization defaults may not match the Slurm task count",
                f"{prefix}.incar",
            )

        collector.note("CORRECTION_POLICY")
        if policy.monopole_method is MonopoleMethod.LMONO:
            lmono = parse_bool(parsed.get("LMONO", "")) if "LMONO" in parsed else None
            if lmono is not True:
                collector.error(
                    "INCAR_CORRECTION_MISMATCH",
                    "correction policy declares LMONO but this stage INCAR "
                    "does not set LMONO = .TRUE.",
                    f"{prefix}.incar.LMONO",
                )
        if policy.dipole:
            idipol = parse_int(parsed.get("IDIPOL", "")) if "IDIPOL" in parsed else None
            ldipol = parse_bool(parsed.get("LDIPOL", "")) if "LDIPOL" in parsed else None
            preconverge = stage.name is StageName.STATIC_PRECONVERGE
            if preconverge and ldipol is not True:
                # The preconvergence stage intentionally skips the dipole
                # correction by default (cheap restart source); the corrected
                # static stage carries the declared dipole correction.
                collector.warn(
                    "PRECONVERGE_WITHOUT_DIPOLE",
                    "corrected_static stage declares the dipole correction "
                    "but static_preconverge runs without LDIPOL; energies "
                    "from the preconvergence stage must never be used as "
                    "production values",
                    f"{prefix}.incar",
                )
            elif idipol != 4 or ldipol is not True:
                collector.error(
                    "INCAR_DIPOLE_MISMATCH",
                    "correction policy declares a dipole correction but this "
                    "stage INCAR does not set IDIPOL = 4 and LDIPOL = .TRUE.",
                    f"{prefix}.incar",
                )
            if parsed.get("DIPOL") is None and not preconverge:
                collector.error(
                    "INCAR_DIPOLE_MISMATCH",
                    "dipole correction declared but DIPOL is not set",
                    f"{prefix}.incar.DIPOL",
                )

    # -- electron parity at the system level ------------------------------------
    if nelect is not None:
        multiplicity_parity = (molecule.spin_multiplicity - 1) % 2
        electron_parity = int(nelect) % 2
        if multiplicity_parity != electron_parity:
            collector.error(
                "ELECTRON_PARITY_MISMATCH",
                f"spin_multiplicity {molecule.spin_multiplicity} is "
                f"inconsistent with NELECT = {nelect:g} (parity "
                f"{multiplicity_parity} vs {electron_parity})",
                "molecule.spin_multiplicity",
            )

    # -- resource calibration scope --------------------------------------------
    collector.note("RESOURCE_CALIBRATION")
    calibration_record = workflow.resource_plan.calibration
    if (
        workflow.resource_plan.profile is ResourceProfile.PRODUCTION
        and calibration_record is not None
    ):
        from photomatagent.scientific.applications.vasp.molecular.calibration import (
            calibration_applicable,
        )

        plan_box = molecule.box_ang
        plan_encut: float | None = None
        if workflow.stages:
            first_incar = workflow.stages[0].incar.get("ENCUT")
            parsed_encut = parse_float(str(first_incar)) if first_incar is not None else None
            plan_encut = parsed_encut
        ok, reasons = calibration_applicable(
            calibration_record,
            formula=summary.formula if summary is not None else None,
            atom_count=len(structure.symbols) if structure is not None else None,
            box_ang=plan_box,
            encut_ev=plan_encut,
        )
        if not ok:
            collector.error(
                "RESOURCE_CALIBRATION_MISMATCH",
                "CalibrationRecord does not apply to the planned run: "
                + "; ".join(reasons),
                "workflow.resource_plan.calibration",
            )

    # -- POTCAR content policy ---------------------------------------------------
    collector.note("POTCAR_CONTENT_POLICY")
    if stage_dirs is not None:
        for stage in workflow.stages:
            stage_dir = stage_dirs.get(stage.name)
            if stage_dir is None:
                continue
            potcar = stage_dir / "POTCAR"
            if potcar.is_file() and not workflow.potcar_materialized:
                collector.error(
                    "POTCAR_CONTENT_PRESENT",
                    f"POTCAR was found in the prepared inputs ({potcar}); "
                    f"POTCAR content must not be written into prepared "
                    f"directories, logs or Git",
                    f"stages[{workflow.stages.index(stage)}].directory",
                )

    return _finish(collector, workflow, summary, output_dir)


def _poscar_contract_violation(
    structure: StructureData,
    elements: list[str],
    counts: list[int],
) -> bool:
    if structure.elements != elements or structure.counts != counts:
        return True
    return sum(counts) != len(structure.symbols)


def _conformer_index(conformer_id: str | None) -> int:
    if conformer_id is None:
        return 0
    if conformer_id.isdigit():
        return int(conformer_id)
    return 0


def _build_summary(
    workflow: WorkflowSpec,
    elements: list[str],
    counts: list[int],
    box_ang: float,
    nelect: float | None,
    neutral_electrons: float | None,
) -> PreflightSummary:
    molecule = workflow.molecule
    return PreflightSummary(
        formula=formula_text(elements, counts),
        charge=molecule.total_charge,
        nelect=_plain_number(nelect) if nelect is not None else 0.0,
        elements=elements,
        neutral_valence_electrons=(
            _plain_number(neutral_electrons)
            if neutral_electrons is not None
            else 0.0
        ),
        box_ang=box_ang,
        potcar_set=molecule.potcar_set,
    )


def _plain_number(value: float) -> float:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return float(round(value))
    return value


def _check_dependency_contracts(workflow: WorkflowSpec, collector: _Collector) -> None:
    for stage_index, stage in enumerate(workflow.stages):
        prefix = f"stages[{stage_index}]"
        required = set(stage.required_upstream_outputs)
        if stage.depends_on is StageName.RELAX:
            if "CONTCAR" not in required:
                collector.error(
                    "RELAX_DOWNSTREAM_MISSING_CONTCAR",
                    f"stage {stage.name.value} depends on relax but does not "
                    f"require CONTCAR; downstream stages must use the "
                    f"relaxed structure",
                    f"{prefix}.required_upstream_outputs",
                )
        if stage.name in {
            StageName.RESTART,
            StageName.CORRECTED_STATIC,
        }:
            missing = [item for item in ("WAVECAR", "CHGCAR") if item not in required]
            if missing:
                collector.error(
                    (
                        "RESTART_MISSING_RESTART_FILES"
                        if stage.name is StageName.RESTART
                        else "CORRECTED_STATIC_MISSING_RESTART_FILES"
                    ),
                    f"restart stage requires {', '.join(missing)} as an "
                    f"upstream output",
                    f"{prefix}.required_upstream_outputs",
                )
        if stage.name in {
            StageName.ORBITAL,
            StageName.ORBITAL_HOMO,
            StageName.ORBITAL_LUMO,
        }:
            iband = stage.incar.get("IBAND")
            if iband is None:
                collector.error(
                    "ORBITAL_IBAND_MISSING",
                    "orbital stage requires an explicit IBAND (HOMO/LUMO band "
                    "indices read from the converged static run)",
                    f"{prefix}.incar.IBAND",
                )
            wavecar = "WAVECAR" in required
            if not wavecar:
                upstream = _available_upstream_outputs(workflow, stage_index)
                wavecar = "WAVECAR" in upstream
            if not wavecar:
                collector.error(
                    "ORBITAL_WAVECAR_MISSING",
                    "orbital stage (LPARD) requires a WAVECAR from an "
                    "upstream stage",
                    f"{prefix}.required_upstream_outputs",
                )
        if stage.name is StageName.ESP:
            lvhar = stage.incar.get("LVHAR")
            locpot = "LOCPOT" in set(stage.produced_outputs)
            if lvhar is not True and not locpot:
                collector.error(
                    "ESP_SPEC_MISSING",
                    "ESP stage must declare LVHAR = .TRUE. (LOCPOT output)",
                    f"{prefix}.incar.LVHAR",
                )
            via_contcar = (
                "CONTCAR" in required
                and StageName.RELAX
                in _transitive_ancestors(workflow, stage_index)
            )
            source_note = (
                workflow.molecule.structure_source or ""
            ).lower()
            declared_relaxed = "contcar" in source_note or "relaxed" in source_note
            if not via_contcar and not declared_relaxed:
                collector.error(
                    "ESP_STRUCTURE_SOURCE_MISSING",
                    "ESP stage must use the optimized geometry from the "
                    "relax stage (CONTCAR) or declare its structure source",
                    f"{prefix}.required_upstream_outputs",
                )
        if stage.name is StageName.STATIC_HSE:
            decision = workflow.screen_decision
            conformer_id = workflow.molecule.conformer_id
            if decision is None:
                collector.error(
                    "HSE06_SCREENING_REQUIRED",
                    "HSE06 stage requires a recorded PBE screen_decision "
                    "(lowest-energy conformer); HSE06 is never run on "
                    "unscreened structures",
                    "workflow.screen_decision",
                )
            elif conformer_id is None or decision.conformer_id != conformer_id:
                collector.error(
                    "HSE06_SCREENING_REQUIRED",
                    f"HSE06 screen_decision names conformer "
                    f"{decision.conformer_id!r} but the submitted molecule "
                    f"has conformer_id {conformer_id!r}",
                    "molecule.conformer_id",
                )


def _finish(
    collector: _Collector,
    workflow: WorkflowSpec,
    summary: PreflightSummary | None,
    output_dir: str | Path | None,
) -> PreflightReport:
    for code in CHECK_ORDER:
        collector.note(code)
    report = PreflightReport(
        passed=not collector.errors,
        errors=collector.errors,
        warnings=collector.warnings,
        summary=summary,
        checks=collector.checks,
    )
    if output_dir is not None:
        save_preflight_report(report, output_dir)
    return report


def save_preflight_report(
    report: PreflightReport, output_dir: str | Path
) -> Path:
    """Persist the full structured report as preflight.json."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "preflight.json"
    target.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return target


def render_agent_text(report: PreflightReport) -> str:
    """Short agent-facing text; never dumps source code or file contents."""
    summary = report.summary
    head = (
        f"formula={summary.formula} q={summary.charge} "
        f"NELECT={_plain_number(summary.nelect):g} "
        f"elements={','.join(summary.elements)}"
        if summary is not None
        else "summary unavailable"
    )
    if report.passed:
        return (
            f"PREFLIGHT PASSED ({len(report.checks)} checks, "
            f"{len(report.warnings)} warnings); {head}. "
            f"Full report in preflight.json."
        )
    lines = [
        f"PREFLIGHT FAILED ({len(report.errors)} errors, "
        f"{len(report.warnings)} warnings); {head}."
    ]
    for issue in report.errors[:20]:
        suffix = f" ({issue.path})" if issue.path else ""
        lines.append(f"  [{issue.code}] {issue.message}{suffix}")
    if len(report.errors) > 20:
        lines.append(f"  ... and {len(report.errors) - 20} more errors")
    for issue in report.warnings[:5]:
        lines.append(f"  warning [{issue.code}] {issue.message}")
    lines.append("Full report in preflight.json.")
    return "\n".join(lines)


def preflight_gate(
    report: PreflightReport, report_path: str | None = None
) -> SubmissionGate:
    """Convert a molecular preflight report into the submission gate.

    ``passed`` must be true before any remote upload or sbatch call; the
    gateway enforces this deterministically in ``submit_once``.
    """
    return SubmissionGate(
        passed=report.passed,
        report_path=report_path,
        errors=[issue.message for issue in report.errors],
        summary=(
            report.summary.model_dump(mode="json")
            if report.summary is not None
            else {}
        ),
    )
