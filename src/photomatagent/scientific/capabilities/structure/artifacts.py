"""Safe structure loading, identity, and no-clobber artifact publication."""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast

from photomatagent.scientific.capabilities.generation.lineage import CandidateLineage
from photomatagent.scientific.capabilities.structure.construction_models import (
    OrderingRequest,
    SubstitutionRequest,
    SupercellRequest,
)
from photomatagent.scientific.discovery.composition import normalize_composition
from photomatagent.scientific.discovery.structures import Operation, StructureDerivation
from photomatagent.workspace import Workspace

if TYPE_CHECKING:
    from pymatgen.core import Structure

MAX_STRUCTURE_INPUT_BYTES = 10 * 1024 * 1024
MAX_OUTPUTS, MAX_ATOMS = 32, 512
HASH_VERSION, HASH_PRECISION = "structure-hash-v1", 8
ARTIFACT_INCOMPLETE, ARTIFACT_CONFLICT = "ARTIFACT_INCOMPLETE", "ARTIFACT_CONFLICT"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_OPERATION_NAMES = {"SupercellRequest": "make_supercell", "SubstitutionRequest": "substitute_sites", "OrderingRequest": "enumerate_orderings"}

class StructureArtifactError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

file_sha256 = _sha256_file

def resolve_structure_input(workspace: Workspace, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("structure path must be workspace-relative without parent traversal")
    try:
        resolved = workspace.resolve(path, must_exist=True)
    except Exception as exc:
        raise ValueError("structure input is outside workspace or missing") from exc
    if not resolved.is_file():
        raise ValueError("structure input must be a file")
    if resolved.stat().st_size > MAX_STRUCTURE_INPUT_BYTES:
        raise StructureArtifactError("INPUT_TOO_LARGE", "structure input exceeds 10 MiB")
    return resolved

def load_structure_input(workspace: Workspace, path: str, *, max_atoms: int | None = None) -> tuple[Any, Path]:
    resolved = resolve_structure_input(workspace, path)
    return _parse_structure_file(resolved, max_atoms=max_atoms)

def _parse_structure_file(resolved: Path, *, max_atoms: int | None = None) -> tuple[Any, Path]:
    try:
        from pymatgen.core import Structure
        structure = Structure.from_file(str(resolved))
        ordered = structure.is_ordered
        values = list(structure.lattice.matrix.flat) + list(structure.frac_coords.flat)
    except Exception as exc:
        raise StructureArtifactError("INVALID_STRUCTURE", f"cannot parse structure: {exc}") from exc
    if not ordered:
        raise StructureArtifactError("PARTIAL_OCCUPANCY_UNSUPPORTED", "disordered sites are unsupported")
    if not all(math.isfinite(float(value)) for value in values):
        raise StructureArtifactError("INVALID_STRUCTURE", "lattice and coordinates must be finite")
    limit = MAX_ATOMS if max_atoms is None else min(max_atoms, MAX_ATOMS)
    if len(structure) > limit:
        message = f"structure has {len(structure)} atoms; limit is {limit}"
        raise StructureArtifactError("ATOM_LIMIT_EXCEEDED", message)
    return structure, resolved

def structure_hash(structure: Any) -> str:
    def rounded(value: float) -> float:
        value %= 1.0
        if abs(value) < 0.5 * 10 ** -HASH_PRECISION or abs(value - 1.0) < 0.5 * 10 ** -HASH_PRECISION:
            return 0.0
        return round(value, HASH_PRECISION)
    sites = [(site.specie.symbol, tuple(rounded(float(v)) for v in site.frac_coords)) for site in structure]
    lattice = [[0.0 if abs(float(v)) < 0.5 * 10 ** -HASH_PRECISION else round(float(v), HASH_PRECISION) for v in row] for row in structure.lattice.matrix]
    payload = {"version": HASH_VERSION, "precision": HASH_PRECISION, "lattice": lattice, "sites": sorted(sites)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _request_parameters(request: Any) -> dict[str, Any]:
    values = request.model_dump(mode="json")
    values.pop("path", None)
    if "replacements" in values:
        values["replacements"] = sorted(
            values["replacements"], key=lambda item: item["index"]
        )
    if "eligible_indices" in values:
        values["eligible_indices"] = sorted(values["eligible_indices"])
    return values

def operation_id(request: Any, input_sha256: str) -> str:
    payload = {
        "input_sha256": input_sha256,
        "operation": type(request).__name__,
        "parameters": _request_parameters(request),
        "algorithm_version": HASH_VERSION,
    }
    return "op_" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


def _operation_id_from_manifest_fields(
    operation: str, input_sha256: str, parameters: dict[str, Any]
) -> str:
    """Recompute the Task 1 operation identity from trusted manifest fields."""

    request_type = next(
        (name for name, value in _OPERATION_NAMES.items() if value == operation),
        None,
    )
    if request_type is None:
        raise StructureArtifactError(ARTIFACT_CONFLICT, "unknown structure operation")
    payload = {
        "input_sha256": input_sha256,
        "operation": request_type,
        "parameters": parameters,
        "algorithm_version": HASH_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return "op_" + digest[:32]

def _validate_output_cap(max_outputs: int) -> int:
    if isinstance(max_outputs, bool) or not isinstance(max_outputs, int):
        raise StructureArtifactError(
            "OUTPUT_LIMIT_EXCEEDED", "max_outputs must be an integer between 1 and 32"
        )
    if not 1 <= max_outputs <= MAX_OUTPUTS:
        raise StructureArtifactError(
            "OUTPUT_LIMIT_EXCEEDED", "max_outputs must be between 1 and 32"
        )
    return max_outputs


def verify_operation_manifest(
    workspace: Workspace,
    directory: Path,
    *,
    operation_id: str,
    operation: str,
    input_sha256: str,
    parameters: dict[str, Any],
    structure_matcher: dict[str, float] | None,
    max_outputs: int = MAX_OUTPUTS,
) -> list[dict[str, Any]]:
    """Verify one complete, immutable operation directory and all outputs."""

    _validate_output_cap(max_outputs)
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or not workspace.contains(directory)
        or directory.resolve(strict=False) != directory
    ):
        raise StructureArtifactError(ARTIFACT_CONFLICT, "operation directory is not a regular workspace directory")
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink():
        raise StructureArtifactError(ARTIFACT_CONFLICT, "manifest must not be a symlink")
    if not manifest_path.exists():
        raise StructureArtifactError(ARTIFACT_INCOMPLETE, "operation manifest is missing")
    if not manifest_path.is_file():
        raise StructureArtifactError(ARTIFACT_CONFLICT, "operation manifest must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StructureArtifactError(ARTIFACT_CONFLICT, "operation manifest is invalid") from exc
    required = {
        "status", "operation_id", "operation", "input_sha256", "parameters",
        "algorithm_version", "hash_version", "hash_precision", "structure_matcher", "outputs",
    }
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise StructureArtifactError(ARTIFACT_CONFLICT, "operation manifest is incomplete")
    expected_operation_id = _operation_id_from_manifest_fields(
        operation, input_sha256, parameters
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("operation_id") != expected_operation_id
        or operation_id != expected_operation_id
        or directory.name != expected_operation_id
        or manifest.get("operation") != operation
        or manifest.get("input_sha256") != input_sha256
        or manifest.get("parameters") != parameters
        or manifest.get("algorithm_version") != HASH_VERSION
        or manifest.get("hash_version") != HASH_VERSION
        or manifest.get("hash_precision") != HASH_PRECISION
        or manifest.get("structure_matcher") != structure_matcher
    ):
        raise StructureArtifactError(ARTIFACT_CONFLICT, "operation manifest does not match the requested operation")
    entries = manifest.get("outputs")
    if not isinstance(entries, list) or not 1 <= len(entries) <= max_outputs:
        code = (
            "OUTPUT_LIMIT_EXCEEDED"
            if isinstance(entries, list) and len(entries) > max_outputs
            else ARTIFACT_CONFLICT
        )
        raise StructureArtifactError(code, "operation output count exceeds the current cap")

    filenames: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise StructureArtifactError(ARTIFACT_CONFLICT, "operation output entry is invalid")
        filename = entry.get("filename")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".cif")
            or filename in filenames
        ):
            raise StructureArtifactError(ARTIFACT_CONFLICT, "operation output filenames must be unique safe CIF names")
        filenames.append(filename)

    for filename in filenames:
        path = directory / filename
        if not path.exists() and not path.is_symlink():
            raise StructureArtifactError(ARTIFACT_INCOMPLETE, "declared structure output is missing")
    allowed = {"manifest.json", *filenames}
    try:
        directory_items = list(directory.iterdir())
    except OSError as exc:
        raise StructureArtifactError(ARTIFACT_CONFLICT, "operation directory cannot be enumerated") from exc
    if {item.name for item in directory_items} != allowed:
        raise StructureArtifactError(ARTIFACT_CONFLICT, "operation directory contains unlisted files")
    for filename in filenames:
        path = directory / filename
        if path.is_symlink():
            raise StructureArtifactError(ARTIFACT_CONFLICT, "declared structure output must not be a symlink")
        if not path.is_file():
            raise StructureArtifactError(ARTIFACT_CONFLICT, "declared structure output must be a regular file")
        try:
            resolved_path = path.resolve(strict=True)
        except OSError as exc:
            raise StructureArtifactError(
                ARTIFACT_CONFLICT, "declared structure output cannot be resolved"
            ) from exc
        if resolved_path != path or not workspace.contains(path):
            raise StructureArtifactError(ARTIFACT_CONFLICT, "declared structure output escapes workspace")
        try:
            reparsed, _ = _parse_structure_file(path)
            actual = {
                "sha256": _sha256_file(path),
                "structure_hash": structure_hash(reparsed),
                "formula": reparsed.composition.reduced_formula,
                "normalized_composition": [
                    list(item) for item in normalize_composition(reparsed.composition.formula)
                ],
                "n_atoms": len(reparsed),
            }
        except (OSError, StructureArtifactError) as exc:
            raise StructureArtifactError(ARTIFACT_CONFLICT, "declared structure output is invalid") from exc
        entry = next(item for item in entries if item["filename"] == filename)
        if any(entry.get(key) != value for key, value in actual.items()):
            raise StructureArtifactError(ARTIFACT_CONFLICT, "declared structure output does not match its manifest entry")
    return entries


def _manifest_state(
    workspace: Workspace,
    directory: Path,
    request: Any,
    input_sha256: str,
    op_id: str,
    structure_matcher: dict[str, float] | None = None,
    *,
    max_outputs: int = MAX_OUTPUTS,
) -> tuple[str, list[dict[str, Any]]]:
    try:
        return "valid", verify_operation_manifest(
            workspace,
            directory,
            operation_id=op_id,
            operation=_OPERATION_NAMES[type(request).__name__],
            input_sha256=input_sha256,
            parameters=_request_parameters(request),
            structure_matcher=structure_matcher,
            max_outputs=max_outputs,
        )
    except StructureArtifactError as exc:
        return exc.code, []


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish one complete directory without replacing an existing target."""
    if sys.platform != "linux":
        raise StructureArtifactError(ARTIFACT_CONFLICT, "atomic no-replace directory publish is unsupported on this platform")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = getattr(libc, "syscall", None)
    if syscall is None:
        raise StructureArtifactError(ARTIFACT_CONFLICT, "atomic no-replace directory publish is unavailable")
    syscall_number = {"x86_64": 316, "aarch64": 276}.get(os.uname().machine)
    if syscall_number is None:
        raise StructureArtifactError(ARTIFACT_CONFLICT, "atomic no-replace publish is unsupported on this architecture")
    result = syscall(syscall_number, -100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise StructureArtifactError(ARTIFACT_CONFLICT, "operation directory appeared before publish")
        raise OSError(error, os.strerror(error))

def publish_structures(
    workspace: Workspace,
    request: SupercellRequest | SubstitutionRequest | OrderingRequest,
    input_sha256: str,
    structures: Sequence["Structure"],
    *,
    structure_matcher: dict[str, float] | None = None,
    max_outputs: int = MAX_OUTPUTS,
) -> list[StructureDerivation]:
    _validate_output_cap(max_outputs)
    if not _SHA.fullmatch(input_sha256):
        raise ValueError("input_sha256 must be a lowercase SHA-256 digest")
    if not 1 <= len(structures) <= max_outputs:
        raise StructureArtifactError(
            "OUTPUT_LIMIT_EXCEEDED",
            f"at most {max_outputs} structures may be published",
        )
    for structure in structures:
        if len(structure) > MAX_ATOMS:
            raise StructureArtifactError("ATOM_LIMIT_EXCEEDED", "structure exceeds publication boundary")
        if not structure.is_ordered:
            raise StructureArtifactError("PARTIAL_OCCUPANCY_UNSUPPORTED", "disordered sites are unsupported")
        values = list(structure.lattice.matrix.flat) + list(structure.frac_coords.flat)
        if not all(math.isfinite(float(value)) for value in values):
            raise StructureArtifactError("INVALID_STRUCTURE", "lattice and coordinates must be finite")
    op_id = operation_id(request, input_sha256)
    relative_destination = Path("user_output") / request.task_slug / "structures" / op_id
    # Resolve existing parent components before creating anything.  This rejects
    # a task or structures directory that is a symlink to outside the workspace.
    try:
        destination = workspace.resolve(relative_destination.as_posix(), must_exist=False)
    except Exception as exc:
        raise ValueError("structure output path escapes workspace") from exc
    if not workspace.contains(destination):
        raise ValueError("structure output path escapes workspace")
    if destination.exists():
        state, entries = _manifest_state(
            workspace,
            destination,
            request,
            input_sha256,
            op_id,
            structure_matcher,
            max_outputs=max_outputs,
        )
        if state == "valid":
            return _derivations_from_manifest(entries, destination, request, input_sha256, op_id)
        raise StructureArtifactError(state, "existing operation artifacts are incomplete or conflict with the requested operation")
    tmp_root = workspace.resolve("tmp", must_exist=True)
    stage = tmp_root / f"structures-{op_id}-{uuid.uuid4().hex}"
    lock = destination.parent / f".{op_id}.lock"
    stage.mkdir(parents=True)
    lock_acquired = False
    published = False
    try:
        try:
            lock.mkdir(parents=True)
            lock_acquired = True
        except FileExistsError as exc:
            raise StructureArtifactError(ARTIFACT_CONFLICT, "operation is being published concurrently") from exc
        if destination.exists():
            raise StructureArtifactError(ARTIFACT_CONFLICT, "operation directory appeared during publication")
        outputs: list[dict[str, Any]] = []
        for index, structure in enumerate(structures):
            filename = f"structure_{index:04d}.cif"
            path = stage / filename
            structure.to(filename=str(path), fmt="cif")
            reparsed, _ = _parse_structure_file(path)
            expected_formula = getattr(request, "expected_formula", None)
            if expected_formula is not None and normalize_composition(reparsed.composition.formula) != normalize_composition(expected_formula):
                raise StructureArtifactError("INVALID_STRUCTURE", "output composition does not match expected_formula")
            if structure_hash(reparsed) != structure_hash(structure):
                raise StructureArtifactError("INVALID_STRUCTURE", "structure hash changed during serialization")
            outputs.append(
                {
                    "filename": filename,
                    "sha256": _sha256_file(path),
                    "structure_hash": structure_hash(reparsed),
                    "formula": reparsed.composition.reduced_formula,
                    "normalized_composition": [
                        list(x)
                        for x in normalize_composition(reparsed.composition.formula)
                    ],
                    "n_atoms": len(reparsed),
                }
            )
        operation = _OPERATION_NAMES[type(request).__name__]
        manifest = {
            "status": "complete",
            "operation_id": op_id,
            "operation": operation,
            "input_sha256": input_sha256,
            "parameters": _request_parameters(request),
            "algorithm_version": HASH_VERSION,
            "hash_version": HASH_VERSION,
            "hash_precision": HASH_PRECISION,
            "structure_matcher": structure_matcher,
            "outputs": outputs,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        # The staged directory is complete.  Publish it with one directory
        # rename; destination is deliberately never created before this point.
        if destination.exists():
            raise StructureArtifactError(ARTIFACT_CONFLICT, "operation directory appeared before publish")
        _rename_noreplace(stage, destination)
        published = True
        return _derivations_from_manifest(outputs, destination, request, input_sha256, op_id)
    except Exception:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if lock_acquired: shutil.rmtree(lock, ignore_errors=True)

def _derivations_from_manifest(entries: list[dict[str, Any]], destination: Path, request: Any, input_sha256: str, op_id: str) -> list[StructureDerivation]:
    result = []
    for ordinal, entry in enumerate(entries):
        normalized = tuple(tuple(x) for x in entry["normalized_composition"])
        candidate = f"cand_{entry['structure_hash'][:24]}"
        output_path = (destination / entry["filename"]).relative_to(destination.parents[3]).as_posix()
        operation = _OPERATION_NAMES[type(request).__name__]
        lineage = CandidateLineage(candidate_id=candidate, generated_by="structure_construction", transformation=operation)
        result.append(StructureDerivation(id=f"der_{op_id[3:]}_{entry['structure_hash'][:16]}_{ordinal:04d}", candidate_id=candidate, parent_candidate_id=None, hypothesis_id=getattr(request, "hypothesis_id", None), input_sha256=input_sha256, structure_hash=entry["structure_hash"], output_path=output_path, operation=cast(Operation, operation), parameters=_request_parameters(request), normalized_composition=normalized, lineage=cast(Any, lineage.model_dump(mode="python"))))
    return result


def _resolve_derivation_output(
    workspace: Workspace, record: StructureDerivation
) -> Path:
    raw_path = Path(record.output_path)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise StructureArtifactError(ARTIFACT_CONFLICT, "structure output path is not workspace-relative")
    try:
        output = workspace.resolve(record.output_path, must_exist=True)
    except Exception as exc:
        raise StructureArtifactError(
            ARTIFACT_INCOMPLETE, "structure output is missing or outside workspace"
        ) from exc
    if output.is_symlink() or not output.is_file() or not workspace.contains(output):
        raise StructureArtifactError(
            ARTIFACT_CONFLICT, "structure output must be a regular workspace file"
        )
    if workspace.relative(output) != record.output_path:
        raise StructureArtifactError(
            ARTIFACT_CONFLICT, "structure output path resolves differently"
        )
    return output


def _rebuild_verified_derivation(
    workspace: Workspace,
    record: StructureDerivation,
    output: Path,
    entries: list[dict[str, Any]],
) -> StructureDerivation:
    matches = [
        (ordinal, item)
        for ordinal, item in enumerate(entries)
        if item.get("filename") == output.name
    ]
    if len(matches) != 1:
        raise StructureArtifactError(
            ARTIFACT_CONFLICT,
            "derivation output is not uniquely listed in manifest",
        )
    ordinal, entry = matches[0]
    actual_hash = entry["structure_hash"]
    normalized = tuple(tuple(item) for item in entry["normalized_composition"])
    candidate_id = f"cand_{actual_hash[:24]}"
    operation = str(record.operation)
    manifest_hypothesis_id = record.parameters.get("hypothesis_id")
    if record.hypothesis_id != manifest_hypothesis_id:
        raise StructureArtifactError(
            ARTIFACT_CONFLICT,
            "derivation hypothesis_id does not match verified manifest parameters",
        )
    if (
        record.structure_hash != actual_hash
        or tuple(record.normalized_composition) != normalized
        or record.id
        != f"der_{output.parent.name[3:]}_{actual_hash[:16]}_{ordinal:04d}"
        or record.candidate_id != candidate_id
    ):
        raise StructureArtifactError(
            ARTIFACT_CONFLICT,
            "derivation identity does not match published structure",
        )
    lineage = CandidateLineage(
        candidate_id=candidate_id,
        parent_candidate_id=None,
        generated_by="structure_construction",
        generation_parameters={},
        source_artifacts=[],
        transformation=operation,
        validation_status="UNVALIDATED_GENERATED_STRUCTURE",
    )
    payload = record.model_dump(mode="python")
    payload.update(
        {
            "parent_candidate_id": None,
            "hypothesis_id": manifest_hypothesis_id,
            "operation": operation,
            "input_sha256": record.input_sha256,
            "parameters": record.parameters,
            "structure_hash": actual_hash,
            "normalized_composition": normalized,
            "output_path": workspace.relative(output),
            "candidate_id": candidate_id,
            "lineage": lineage.model_dump(mode="python"),
        }
    )
    return StructureDerivation.model_validate(payload)


def verify_structure_derivations(
    workspace: Workspace, records: Sequence[StructureDerivation]
) -> list[StructureDerivation]:
    """Verify complete manifests and rebuild an exact registration batch."""

    if not records:
        return []
    resolved = [_resolve_derivation_output(workspace, record) for record in records]
    groups: dict[Path, list[tuple[StructureDerivation, Path]]] = {}
    for record, output in zip(records, resolved, strict=True):
        groups.setdefault(output.parent, []).append((record, output))
    rebuilt: dict[str, StructureDerivation] = {}
    for directory, group in groups.items():
        first = group[0][0]
        operation = str(first.operation)
        matcher = (
            {"ltol": 0.2, "stol": 0.3, "angle_tol": 5}
            if operation == "enumerate_orderings"
            else None
        )
        if any(
            str(record.operation) != operation
            or record.input_sha256 != first.input_sha256
            or record.parameters != first.parameters
            for record, _ in group
        ):
            raise StructureArtifactError(
                ARTIFACT_CONFLICT,
                "registration batch fields must match one operation manifest",
            )
        op_id = directory.name
        entries = verify_operation_manifest(
            workspace,
            directory,
            operation_id=op_id,
            operation=operation,
            input_sha256=first.input_sha256,
            parameters=first.parameters,
            structure_matcher=matcher,
        )
        declared = {entry["filename"] for entry in entries}
        supplied = [output.name for _, output in group]
        if len(supplied) != len(set(supplied)) or set(supplied) != declared:
            raise StructureArtifactError(
                ARTIFACT_CONFLICT,
                "registration batch must correspond exactly to manifest outputs",
            )
        for record, output in group:
            rebuilt[record.output_path] = _rebuild_verified_derivation(
                workspace, record, output, entries
            )
    return [rebuilt[record.output_path] for record in records]


def verify_structure_derivation(
    workspace: Workspace, record: StructureDerivation
) -> StructureDerivation:
    """Verify one registration whose batch must contain the complete manifest."""

    return verify_structure_derivations(workspace, [record])[0]
