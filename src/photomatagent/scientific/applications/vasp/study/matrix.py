"""Calculation-matrix construction: expand, deduplicate, order, estimate.

Rules:
* one unique chemical system (identity + charge + spin) is calculated once,
  no matter how many properties or binding groups need it;
* complexes expand into their fragments, and shared references (Li+,
  TFSI-, VM/TVM themselves) are deduplicated automatically;
* charges are always explicit identity data; a binding group is only valid
  when complex charge == fragment charge sum;
* a dependency DAG orders fragment workflows before complex workflows;
* core-hours, job counts and a coarse disk footprint are estimated.
"""

from __future__ import annotations

from typing import Any

from photomatagent.scientific.applications.vasp.study.models import (
    BindingGroup,
    CalculationMatrix,
    CalculationTask,
    PropertyRequest,
    StudySystem,
    StudyTaskState,
    VaspStudyRequest,
)
from photomatagent.scientific.capabilities.chemistry.models import (
    GeneratedStructure,
)

BASE_STAGES = 3  # relax, static_preconverge, corrected_static
TASKS_PER_NODE = 8
WALLTIME_MINUTES = 20
DISK_GB_PER_WORKFLOW = 8.0


def _profile_budget(request: VaspStudyRequest) -> tuple[int, int, float, float]:
    """(tasks, walltime_minutes, disk_gb) from profile/calibration, never
    from hard-coded production guesses: smoke keeps the verified 8-core /
    20-min / 8 GB baseline; production requires a CalibrationRecord and
    derives tasks/walltime/disk from its measured values (capped by the
    hard ceiling enforced at execution time)."""
    method = request.method
    profile = method.profile()
    calibration = method.calibration_record()
    if profile.value == "production" and calibration is None:
        # Plan honestly: mark the imprecise default so the executor refuses
        # to submit without a record; the numbers stay capped by budget.
        return TASKS_PER_NODE, WALLTIME_MINUTES, DISK_GB_PER_WORKFLOW, 0.0
    if calibration is not None:
        tasks = calibration.tasks or TASKS_PER_NODE
        disk_gb = max(DISK_GB_PER_WORKFLOW, calibration.max_rss_bytes / 1e9 * 3.0)
        if calibration.elapsed_seconds > 0:
            walltime = max(
                WALLTIME_MINUTES,
                int(calibration.elapsed_seconds / 60.0 * 1.5) + 5,
            )
        else:
            walltime = WALLTIME_MINUTES
        return tasks, walltime, round(disk_gb, 1), float(calibration.elapsed_seconds)
    return TASKS_PER_NODE, WALLTIME_MINUTES, DISK_GB_PER_WORKFLOW, 0.0


def _system_properties(
    system: StudySystem, request: VaspStudyRequest
) -> list[PropertyRequest]:
    if system.properties:
        return list(system.properties)
    return list(request.property_requests)


def _estimate_core_hours(assists: list[Any]) -> float:
    return _estimate_core_hours_for(assists, TASKS_PER_NODE, WALLTIME_MINUTES)


def _estimate_core_hours_for(
    assists: list[Any], tasks: int, walltime_minutes: int
) -> float:
    stages = BASE_STAGES
    values = {str(getattr(item, "value", item)) for item in assists}
    if PropertyRequest.HOMO_LUMO.value in values:
        stages += 2  # orbital_homo, orbital_lumo
    if PropertyRequest.ESP.value in values:
        stages += 1  # esp
    return stages * tasks * walltime_minutes / 60.0


def build_calculation_matrix(
    request: VaspStudyRequest,
    resolved: dict[str, list[GeneratedStructure]],
) -> CalculationMatrix:
    """Build the deduplicated matrix from resolved structures."""
    tasks_per_node, walltime_minutes, disk_gb, _ = _profile_budget(request)
    tasks: dict[str, CalculationTask] = {}
    binding_groups: list[BindingGroup] = []
    notes: list[str] = []

    for system in request.systems:
        properties = _system_properties(system, request)
        structures = resolved.get(system.system_id) or []
        usable = [
            structure
            for structure in structures
            if structure.format in {"xyz", "sdf", "mol"}
            and structure.atom_count > 0
        ]
        if not usable:
            skipped_id = system.system_id.strip().lower()
            tasks[skipped_id] = CalculationTask(
                task_id=skipped_id,
                system_id=skipped_id,
                display_name=system.display_name or system.system_id,
                role=(
                    structures[0].identity.role.value
                    if structures
                    else "proxy"
                ),
                total_charge=system.total_charge or 0,
                reliability=(
                    structures[0].reliability_grade().value
                    if structures
                    else "D"
                ),
                structure_status=(
                    structures[0].provenance.status.value
                    if structures
                    else "GENERATION_FAILED"
                ),
                state=StudyTaskState.SKIPPED_PROXY.value,
                error="no structure resolved; task skipped (study continues)",
            )
            notes.append(
                f"{skipped_id}: missing structure -> SKIPPED_PROXY"
            )
            continue
        structure = usable[0]
        identity = structure.identity
        key = identity.canonical_key()
        task = tasks.get(key)
        if task is None:
            task = CalculationTask(
                task_id=key,
                system_id=identity.system_id,
                display_name=identity.display_name,
                role=identity.role.value,
                formula=identity.formula,
                total_charge=identity.total_charge,
                spin_multiplicity=identity.spin_multiplicity,
                structure_path=structure.structure_path,
                structure_candidates=[
                    str(candidate.structure_path)
                    for candidate in usable
                    if candidate.structure_path
                    and candidate.structure_path.is_file()
                ],
                reliability=structure.reliability_grade().value,
                structure_status=structure.provenance.status.value,
                assists=[],
                estimated_core_hours=0.0,
            )
            tasks[key] = task
        task.assists.extend(
            property_request
            for property_request in properties
            if property_request not in task.assists
            and property_request is not PropertyRequest.BINDING_ENERGY
        )
        task.estimated_core_hours = _estimate_core_hours_for(
            task.assists, tasks_per_node, walltime_minutes
        )

    # -- binding groups: expand complexes into fragment references --------
    for system in request.systems:
        if PropertyRequest.BINDING_ENERGY not in _system_properties(
            system, request
        ):
            continue
        structures = resolved.get(system.system_id) or []
        if not structures:
            # Preserve the declared binding relationship even when the
            # complex itself could not be resolved.  Downstream reporting
            # must fail closed with the missing identifier, never fabricate a
            # reference or silently omit the group.
            if PropertyRequest.BINDING_ENERGY in _system_properties(system, request):
                missing = [system.system_id.strip().lower()]
                binding_groups.append(
                    BindingGroup(
                        complex_task_id=missing[0],
                        fragment_task_ids=[],
                        missing_fragment_ids=missing,
                        label=f"E({system.display_name or system.system_id}) - sum E(fragments)",
                        total_charge=system.total_charge or 0,
                        state=StudyTaskState.FAILED.value,
                        error="missing complex structure; binding not computable",
                    )
                )
            continue
        complex_structure = structures[0]
        complex_identity = complex_structure.identity
        complex_key = complex_identity.canonical_key()
        complex_task = tasks.get(complex_key)
        if complex_task is None:
            complex_task = CalculationTask(
                task_id=complex_key,
                system_id=complex_identity.system_id,
                display_name=complex_identity.display_name,
                role=complex_identity.role.value,
                formula=complex_identity.formula,
                total_charge=complex_identity.total_charge,
                spin_multiplicity=complex_identity.spin_multiplicity,
                structure_path=complex_structure.structure_path,
                reliability=complex_structure.reliability_grade().value,
                structure_status=complex_structure.provenance.status.value,
                assists=[],
            )
            tasks[complex_key] = complex_task
        fragment_ids: list[str] = []
        fragment_charges: list[int] = []
        missing_fragment_ids: list[str] = []
        for parent_id in complex_structure.provenance.parent_structures:
            fragment_structures = resolved.get(parent_id) or []
            if not fragment_structures:
                missing_fragment_ids.append(parent_id.strip().lower())
                notes.append(
                    f"binding fragment {parent_id} of "
                    f"{complex_identity.system_id} has no structure"
                )
                continue
            fragment = fragment_structures[0]
            fragment_identity = fragment.identity
            fragment_key = fragment_identity.canonical_key()
            if fragment_key not in tasks:
                tasks[fragment_key] = CalculationTask(
                    task_id=fragment_key,
                    system_id=fragment_identity.system_id,
                    display_name=fragment_identity.display_name,
                    role=fragment_identity.role.value,
                    formula=fragment_identity.formula,
                    total_charge=fragment_identity.total_charge,
                    spin_multiplicity=fragment_identity.spin_multiplicity,
                    structure_path=fragment.structure_path,
                    reliability=fragment.reliability_grade().value,
                    structure_status=fragment.provenance.status.value,
                    assists=[],
                )
            fragment_ids.append(fragment_key)
            fragment_charges.append(fragment_identity.total_charge)
        if missing_fragment_ids:
            notes.append(
                f"{complex_identity.system_id}: binding group has missing "
                f"fragment IDs {', '.join(missing_fragment_ids)}"
            )
        if (
            not missing_fragment_ids
            and fragment_charges
            and sum(fragment_charges) != complex_identity.total_charge
        ):
            notes.append(
                f"{complex_identity.system_id}: charge contract violated "
                f"(complex {complex_identity.total_charge:+d} vs fragments "
                f"{sum(fragment_charges):+d}); binding group marked invalid"
            )
        if (
            fragment_ids
            or missing_fragment_ids
            or not complex_structure.provenance.parent_structures
        ):
            complex_task.depends_on = list(
                dict.fromkeys([*complex_task.depends_on, *fragment_ids])
            )
            binding_state = (
                StudyTaskState.PLANNED.value
                if (
                    not missing_fragment_ids
                    and bool(complex_structure.provenance.parent_structures)
                    and sum(fragment_charges) == complex_identity.total_charge
                )
                else StudyTaskState.FAILED.value
            )
            if missing_fragment_ids:
                binding_error = "missing fragment structures; binding not computed"
            elif not complex_structure.provenance.parent_structures:
                binding_error = "no declared fragment references; binding not computed"
            elif sum(fragment_charges) != complex_identity.total_charge:
                binding_error = "charge contract violated; binding not computed"
            else:
                binding_error = ""
            binding_groups.append(
                BindingGroup(
                    complex_task_id=complex_key,
                    fragment_task_ids=list(dict.fromkeys(fragment_ids)),
                    missing_fragment_ids=list(dict.fromkeys(missing_fragment_ids)),
                    label=(
                        f"E({complex_identity.display_name}) - sum E(fragments)"
                    ),
                    total_charge=complex_identity.total_charge,
                    state=binding_state,
                    error=binding_error,
                )
            )

    for task in tasks.values():
        if not task.estimated_core_hours:
            task.estimated_core_hours = _estimate_core_hours_for(
                task.assists, tasks_per_node, walltime_minutes
            )
    ordered = list(tasks.values())
    # Depth-first topological order: fragments before complexes.
    ordered.sort(
        key=lambda task: (
            len(task.depends_on),
            task.task_id,
        )
    )
    total_core_hours = sum(task.estimated_core_hours for task in ordered)
    return CalculationMatrix(
        tasks=ordered,
        binding_groups=binding_groups,
        total_core_hours=round(total_core_hours, 2),
        total_jobs=sum(
            (
                BASE_STAGES
                + (2 if PropertyRequest.HOMO_LUMO in task.assists else 0)
                + (1 if PropertyRequest.ESP in task.assists else 0)
            )
            for task in ordered
            if not task.structure_path or task.state not in {
                StudyTaskState.SKIPPED_PROXY.value,
                StudyTaskState.FAILED.value,
            }
        ),
        estimated_disk_gb=round(
            sum(
                disk_gb
                for task in ordered
                if task.structure_path
            ),
            1,
        ),
        notes=notes,
    )
