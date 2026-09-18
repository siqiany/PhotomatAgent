"""Deferred model-facing tools for bounded structure construction."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from photomatagent.scientific.capabilities.contracts import ScientificToolResult
from photomatagent.scientific.capabilities.structure.artifacts import (
    StructureArtifactError,
    file_sha256,
    load_structure_input,
    publish_structures,
)
from photomatagent.scientific.capabilities.structure.construction import (
    StructureConstructionError,
    enumerate_orderings,
    make_supercell,
    substitute_sites,
)
from photomatagent.scientific.capabilities.structure.construction_models import (
    ConstructionLimits,
    OrderingRequest,
    SubstitutionRequest,
    SupercellRequest,
)
from photomatagent.scientific.discovery.composition import normalize_composition
from photomatagent.scientific.discovery.structures import StructureRegistration
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace


def _error(code: str, message: str) -> ScientificToolResult:
    payload = {"error": code, "error_type": code, "message": message}
    return ScientificToolResult(
        output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        is_error=True,
        data=payload,
    )


def _success(
    operation: str,
    derivations: list[Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> ScientificToolResult:
    records = [record.model_dump(mode="json") for record in derivations]
    artifacts = [record["output_path"] for record in records]
    payload = {"operation": operation, "derivations": records, "artifacts": artifacts}
    if metadata:
        payload.update(metadata)
    return ScientificToolResult(
        output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        data=payload,
        artifacts=artifacts,
        state_updates=[
            StructureRegistration(derivation=record) for record in derivations
        ],
    )


def _bind_hypothesis(
    state: ScientificState | None,
    hypothesis_id: str,
    expected_formula: str,
) -> None:
    if state is None:
        raise StructureConstructionError(
            "STATE_UNAVAILABLE",
            "substitution requires a live ScientificState for hypothesis binding",
        )
    hypothesis = next(
        (item for item in state.material_hypotheses if item.id == hypothesis_id),
        None,
    )
    if hypothesis is None:
        raise StructureConstructionError(
            "HYPOTHESIS_NOT_FOUND", f"unknown hypothesis_id {hypothesis_id!r}"
        )
    try:
        expected = normalize_composition(expected_formula)
    except (TypeError, ValueError) as exc:
        raise StructureConstructionError("INVALID_EXPECTED_FORMULA", str(exc)) from exc
    if expected != hypothesis.normalized_composition:
        raise StructureConstructionError(
            "HYPOTHESIS_COMPOSITION_MISMATCH",
            "expected_formula does not match the bound hypothesis composition",
        )


def _bind_supercell_hypothesis(
    state: ScientificState | None,
    hypothesis_id: str | None,
    structure: Any,
) -> None:
    """Validate an optional supercell lineage reference without mutating state."""

    if hypothesis_id is None:
        return
    if state is None:
        raise StructureConstructionError(
            "STATE_UNAVAILABLE",
            "hypothesis-bound supercell construction requires a live ScientificState",
        )
    hypothesis = next(
        (item for item in state.material_hypotheses if item.id == hypothesis_id),
        None,
    )
    if hypothesis is None:
        raise StructureConstructionError(
            "HYPOTHESIS_NOT_FOUND", f"unknown hypothesis_id {hypothesis_id!r}"
        )
    try:
        actual = normalize_composition(structure.composition.formula)
    except (TypeError, ValueError) as exc:
        raise StructureConstructionError(
            "INVALID_INPUT_COMPOSITION", str(exc)
        ) from exc
    if actual != hypothesis.normalized_composition:
        raise StructureConstructionError(
            "HYPOTHESIS_COMPOSITION_MISMATCH",
            "input structure composition does not match the bound hypothesis",
        )


class MakeSupercellTool(Tool):
    name = "structure.make_supercell"
    description = (
        "Create a bounded supercell from a workspace structure file. "
        "The input structure is never modified; scaling is a positive integer "
        "factor for each lattice axis."
    )
    short_description = "Create a bounded supercell and publish its CIF artifact."
    exposure = ToolExposure.DEFERRED
    namespace = "structure"
    source = "pymatgen"
    tags = ("structure", "supercell", "construction")
    input_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative input structure file.",
            },
            "scaling": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "integer", "minimum": 1},
                "description": "Positive integer factors for the input structure's a, b, c axes.",
            },
            "task_slug": {
                "type": "string",
                "description": "Safe task directory name for the published artifact.",
            },
            "hypothesis_id": {
                "type": ["string", "null"],
                "description": "Optional hypothesis reference; if provided, it must exist in ScientificState and match the input structure composition.",
            },
        },
        "required": ["path", "scaling", "task_slug"],
    }

    def __init__(
        self,
        workspace: Workspace,
        *,
        scientific_state: ScientificState | None = None,
        limits: ConstructionLimits | None = None,
    ) -> None:
        self._workspace = workspace
        self._limits = limits or ConstructionLimits()
        self._scientific_state = scientific_state

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            request = SupercellRequest.model_validate(arguments)
            structure, path = load_structure_input(
                self._workspace, request.path, max_atoms=self._limits.max_atoms
            )
            _bind_supercell_hypothesis(
                self._scientific_state, request.hypothesis_id, structure
            )
            result = make_supercell(structure, request.scaling, self._limits)
            derivations = publish_structures(
                self._workspace,
                request,
                file_sha256(path),
                [result],
            )
            return _success("make_supercell", derivations)
        except ValidationError as exc:
            return _error("INVALID_INPUT", str(exc))
        except StructureArtifactError as exc:
            code = "MISSING_DEPENDENCY" if isinstance(exc.__cause__, ImportError) else exc.code
            return _error(code, str(exc))
        except (StructureConstructionError, ValueError, OSError) as exc:
            return _error(getattr(exc, "code", type(exc).__name__), str(exc))
        except OverflowError as exc:
            return _error("SCALING_LIMIT_EXCEEDED", str(exc))
        except ImportError as exc:
            return _error("MISSING_DEPENDENCY", str(exc))


class SubstituteSitesTool(Tool):
    name = "structure.substitute_sites"
    description = (
        "Replace explicitly indexed sites in a workspace structure and publish "
        "a CIF only when the canonical output composition matches expected_formula. "
        "Every index is 0-based and refers to the input structure file passed to this call."
    )
    short_description = "Substitute explicitly indexed host sites at exact composition."
    exposure = ToolExposure.DEFERRED
    namespace = "structure"
    source = "pymatgen"
    tags = ("structure", "substitution", "composition", "construction")
    input_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative input structure file."},
            "replacements": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "0-based index in the input structure file, before this call's substitutions.",
                        },
                        "from_element": {"type": "string", "description": "Required host element at that input index."},
                        "to_element": {"type": "string", "description": "Replacement element."},
                    },
                    "required": ["index", "from_element", "to_element"],
                },
                "description": "Explicit 0-based input-structure site replacements; indices must be unique.",
            },
            "expected_formula": {
                "type": "string",
                "description": "Canonical target composition; equivalent reduced formulas are accepted.",
            },
            "hypothesis_id": {
                "type": "string",
                "description": "Existing ScientificState hypothesis whose composition must match expected_formula.",
            },
            "task_slug": {"type": "string", "description": "Safe task directory name for the published artifact."},
        },
        "required": ["path", "replacements", "expected_formula", "hypothesis_id", "task_slug"],
    }

    def __init__(
        self,
        workspace: Workspace,
        *,
        scientific_state: ScientificState | None = None,
        limits: ConstructionLimits | None = None,
    ) -> None:
        self._workspace = workspace
        self._scientific_state = scientific_state
        self._limits = limits or ConstructionLimits()

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            request = SubstitutionRequest.model_validate(arguments)
            _bind_hypothesis(
                self._scientific_state, request.hypothesis_id, request.expected_formula
            )
            structure, path = load_structure_input(
                self._workspace, request.path, max_atoms=self._limits.max_atoms
            )
            result = substitute_sites(
                structure, request.replacements, request.expected_formula
            )
            derivations = publish_structures(
                self._workspace,
                request,
                file_sha256(path),
                [result],
            )
            return _success("substitute_sites", derivations)
        except ValidationError as exc:
            return _error("INVALID_INPUT", str(exc))
        except StructureArtifactError as exc:
            code = "MISSING_DEPENDENCY" if isinstance(exc.__cause__, ImportError) else exc.code
            return _error(code, str(exc))
        except (StructureConstructionError, ValueError, OSError) as exc:
            return _error(getattr(exc, "code", type(exc).__name__), str(exc))
        except OverflowError as exc:
            return _error("SCALING_LIMIT_EXCEEDED", str(exc))
        except ImportError as exc:
            return _error("MISSING_DEPENDENCY", str(exc))


class EnumerateOrderingsTool(Tool):
    name = "structure.enumerate_orderings"
    description = (
        "Enumerate a bounded set of deterministic fixed-count substitutions over "
        "explicit 0-based eligible input sites, deduplicate equivalent structures, "
        "and publish the resulting CIF artifacts."
    )
    short_description = "Enumerate bounded ordered substitutions with deduplication."
    exposure = ToolExposure.DEFERRED
    namespace = "structure"
    source = "pymatgen"
    tags = ("structure", "ordering", "construction", "deduplication")
    input_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative input structure file."},
            "eligible_indices": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "integer", "minimum": 0},
                "description": "Unique 0-based input-structure site indices whose host species is from_element.",
            },
            "from_element": {"type": "string", "description": "Host element at every eligible index."},
            "to_element": {"type": "string", "description": "Replacement element for each selected combination."},
            "replacement_count": {"type": "integer", "minimum": 1, "description": "Number of eligible sites replaced in each raw combination."},
            "expected_formula": {"type": "string", "description": "Exact canonical composition required for every output."},
            "hypothesis_id": {"type": "string", "description": "Existing ScientificState hypothesis bound to expected_formula."},
            "task_slug": {"type": "string", "description": "Safe task directory name for published artifacts."},
        },
        "required": [
            "path", "eligible_indices", "from_element", "to_element",
            "replacement_count", "expected_formula", "hypothesis_id", "task_slug",
        ],
    }

    def __init__(
        self,
        workspace: Workspace,
        *,
        scientific_state: ScientificState | None = None,
        limits: ConstructionLimits | None = None,
    ) -> None:
        self._workspace = workspace
        self._scientific_state = scientific_state
        self._limits = limits or ConstructionLimits()

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            request = OrderingRequest.model_validate(arguments)
            _bind_hypothesis(
                self._scientific_state, request.hypothesis_id, request.expected_formula
            )
            structure, path = load_structure_input(
                self._workspace, request.path, max_atoms=self._limits.max_atoms
            )
            result = enumerate_orderings(structure, request, self._limits)
            derivations = publish_structures(
                self._workspace,
                request,
                file_sha256(path),
                result,
                structure_matcher=result.matcher,
            )
            return _success(
                "enumerate_orderings",
                derivations,
                metadata={
                    "total": result.total,
                    "scanned": result.scanned,
                    "discarded": result.discarded,
                    "truncated": result.truncated,
                    "exhaustive": not result.truncated,
                    "structure_matcher": result.matcher,
                },
            )
        except ValidationError as exc:
            return _error("INVALID_INPUT", str(exc))
        except StructureArtifactError as exc:
            code = "MISSING_DEPENDENCY" if isinstance(exc.__cause__, ImportError) else exc.code
            return _error(code, str(exc))
        except (StructureConstructionError, ValueError, OSError) as exc:
            return _error(getattr(exc, "code", type(exc).__name__), str(exc))
        except ImportError as exc:
            return _error("MISSING_DEPENDENCY", str(exc))
