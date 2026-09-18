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
    payload = {"input_sha256": input_sha256, "operation": type(request).__name__, "parameters": _request_parameters(request), "algorithm_version": HASH_VERSION}
    return "op_" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]

def _manifest_state(
    workspace: Workspace,
    directory: Path,
    request: Any,
    input_sha256: str,
    op_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink():
        return ARTIFACT_CONFLICT, []
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
    except OSError:
        return ARTIFACT_CONFLICT, []
    if resolved_manifest != manifest_path or not workspace.contains(resolved_manifest):
        return ARTIFACT_CONFLICT, []
    if not manifest_path.is_file():
        return ARTIFACT_INCOMPLETE, []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError): return ARTIFACT_CONFLICT, []
    required = {"status", "operation_id", "operation", "input_sha256", "parameters", "algorithm_version", "hash_version", "hash_precision", "structure_matcher", "outputs"}
    if not required <= manifest.keys() or manifest.get("status") != "complete":
        return ARTIFACT_CONFLICT, []
    operation = _OPERATION_NAMES[type(request).__name__]
    actual_header = (manifest["operation_id"], manifest["operation"], manifest["input_sha256"], manifest["parameters"], manifest["algorithm_version"])
    expected_header = (op_id, operation, input_sha256, _request_parameters(request), HASH_VERSION)
    if actual_header != expected_header:
        return ARTIFACT_CONFLICT, []
    entries = manifest["outputs"]
    valid_manifest = (
        manifest["hash_version"] == HASH_VERSION
        and manifest["hash_precision"] == HASH_PRECISION
        and manifest["structure_matcher"] is None
        and isinstance(entries, list)
        and 1 <= len(entries) <= MAX_OUTPUTS
    )
    if not valid_manifest:
        return ARTIFACT_CONFLICT, []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            return ARTIFACT_CONFLICT, []
        filename = entry.get("filename")
        valid_filename = (
            isinstance(filename, str)
            and filename not in seen
            and Path(filename).name == filename
            and filename.endswith(".cif")
        )
        if not valid_filename:
            return ARTIFACT_CONFLICT, []
        assert isinstance(filename, str)
        seen.add(filename)
        path = directory / filename
        if not path.exists() and not path.is_symlink():
            return ARTIFACT_INCOMPLETE, []
        if path.is_symlink():
            return ARTIFACT_CONFLICT, []
        try:
            resolved_output = path.resolve(strict=True)
        except OSError:
            return ARTIFACT_CONFLICT, []
        if resolved_output != path or not workspace.contains(resolved_output):
            return ARTIFACT_CONFLICT, []
        if not path.is_file():
            return ARTIFACT_INCOMPLETE, []
        try:
            reparsed, _ = _parse_structure_file(path)
            actual = {"sha256": _sha256_file(path), "structure_hash": structure_hash(reparsed), "formula": reparsed.composition.reduced_formula, "normalized_composition": [list(x) for x in normalize_composition(reparsed.composition.formula)], "n_atoms": len(reparsed)}
        except StructureArtifactError:
            return ARTIFACT_CONFLICT, []
        if any(entry.get(key) != value for key, value in actual.items()):
            return ARTIFACT_CONFLICT, []
    return "valid", entries


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
) -> list[StructureDerivation]:
    if not _SHA.fullmatch(input_sha256):
        raise ValueError("input_sha256 must be a lowercase SHA-256 digest")
    if not 1 <= len(structures) <= MAX_OUTPUTS:
        raise StructureArtifactError("OUTPUT_LIMIT_EXCEEDED", "at most 32 structures may be published")
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
            workspace, destination, request, input_sha256, op_id
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
            "structure_matcher": None,
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
