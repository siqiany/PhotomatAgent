"""CHGNet-backed structure screening and pre-relaxation.

CHGNet is deliberately an optional, lazy-loaded capability.  The model is a
cheap machine-learning interatomic potential: its values are useful for
within-composition screening and pre-relaxation, but are not DFT or
thermodynamic validation.  Both model-visible tools stay on the normal
deferred scientific-tool path.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import math
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.config import ScientificConfig
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace


DEFAULT_MODEL_NAME = "0.3.0"
DEFAULT_DEVICE = "cpu"
MAX_SCREEN_OUTPUT_CHARS = 12_000
MAX_SUMMARY_VALUES = 32
MAX_STRUCTURE_COUNT = 32
MAX_RELAX_STEPS = 200
MIN_RELAX_FMAX = 1e-4
MAX_RELAX_FMAX = 1.0

ModelLoader = Callable[..., Any]
OptimizerFactory = Callable[..., Any]
Row = TypeVar("Row", bound=Mapping[str, Any])


class CHGNetCapabilityPack(CapabilityPack):
    """Capability-pack owner for the two CHGNet deferred tools."""

    name = "chgnet"
    description = (
        "CHGNet machine-learning interatomic-potential screening and "
        "pre-relaxation (not DFT validation)."
    )
    execution_mode = "local"
    backend_name = "CHGNet"

    def __init__(self, config: ScientificConfig, workspace: Workspace) -> None:
        self._config = config
        self._workspace = workspace

    def probe(self) -> ProbeResult:
        try:
            importlib.import_module("chgnet")
            importlib.import_module("chgnet.model")
        except ImportError as exc:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail=(
                    "chgnet is not importable "
                    f"({type(exc).__name__}: {exc}); install "
                    "photomatagent[chgnet]"
                ),
            )
        except Exception as exc:
            return ProbeResult(
                status=CapabilityStatus.ERROR,
                detail=f"chgnet probe failed: {type(exc).__name__}: {exc}",
            )
        version = _chgnet_version()
        version_note = f" (version {version})" if version else ""
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail=(
                "CHGNet is importable; values are ML-potential estimates "
                "and are not DFT or thermodynamic validation"
                f"{version_note}"
            ),
            version=version,
        )

    def tools(self) -> list[Tool]:
        # Keep metadata and deferred entry points available even if the
        # optional dependency is missing.  Calls then return a typed error.
        return [
            CHGNetScreenTool(self._config, self._workspace),
            CHGNetRelaxTool(self._config, self._workspace),
        ]


CHGNetProbe = CHGNetCapabilityPack


def chgnet_pack(config: ScientificConfig, workspace: Workspace) -> CapabilityPack:
    """Build the CHGNet capability pack without importing CHGNet itself."""

    return CHGNetCapabilityPack(config, workspace)


def _chgnet_version() -> str:
    try:
        return importlib.metadata.version("chgnet")
    except Exception:
        return ""


def _load_chgnet_model(config: ScientificConfig) -> Any:
    """Load the configured pretrained model lazily."""

    try:
        from chgnet.model import CHGNet
    except ImportError:
        from chgnet.model.model import CHGNet

    try:
        model = CHGNet.load(model_name=config.chgnet_model_name)
    except TypeError:
        model = CHGNet.load(config.chgnet_model_name)
    to = getattr(model, "to", None)
    if callable(to):
        moved = to(config.chgnet_device)
        if moved is not None:
            model = moved
    return model


def _default_optimizer_factory(model: Any, config: ScientificConfig) -> Any:
    try:
        from chgnet.model import StructOptimizer
    except ImportError:
        from chgnet.model.dynamics import StructOptimizer
    try:
        return StructOptimizer(model=model, use_device=config.chgnet_device)
    except TypeError:
        return StructOptimizer(model, use_device=config.chgnet_device)


class _CHGNetTool(Tool):
    """Shared lazy model-loading and error-handling behavior."""

    def __init__(
        self,
        config: ScientificConfig,
        workspace: Workspace,
        *,
        model: Any | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self._config = config
        self._workspace = workspace
        self._model = model
        self._model_loader = model_loader or _load_chgnet_model

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = _invoke_model_loader(self._model_loader, self._config)
        if self._model is None:
            raise RuntimeError("CHGNet model loader returned no model")
        return self._model

    @staticmethod
    def _error(
        message: str,
        *,
        error_type: str = "execution_error",
        details: Mapping[str, Any] | None = None,
    ) -> ScientificToolResult:
        data: dict[str, Any] = {"error_type": error_type}
        if details:
            data.update(details)
        return ScientificToolResult(output=message, is_error=True, data=data)


class CHGNetScreenTool(_CHGNetTool):
    """Run bounded CHGNet single-point predictions on local structures."""

    name = "chgnet.screen"
    description = (
        "Screen one to 32 workspace-contained crystal structures with a "
        "pretrained CHGNet model. Returns energy, maximum force, stress, and "
        "magnetic-moment summaries; energy ranking is only within identical "
        "reduced compositions and is never a DFT stability claim."
    )
    short_description = "Screen crystal structures with CHGNet (same-composition ranking only)."
    exposure = ToolExposure.DEFERRED
    namespace = "chgnet"
    source = "CHGNet"
    tags = (
        "chgnet",
        "machine learning potential",
        "ml interatomic potential",
        "structure screening",
        "force",
        "energy",
        "pre-relaxation",
    )
    cost_class = "CHEAP"
    input_schema = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One to 32 relative workspace paths to CIF/POSCAR "
                    "structures."
                ),
            },
            "rank": {
                "type": "boolean",
                "description": (
                    "Whether to add deterministic same-composition energy "
                    "ranks (default true)."
                ),
            },
        },
        "required": ["paths"],
    }

    def __init__(
        self,
        config: ScientificConfig,
        workspace: Workspace,
        *,
        model: Any | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        super().__init__(
            config, workspace, model=model, model_loader=model_loader
        )

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            paths = _screen_input_paths(arguments)
            if not paths:
                raise ValueError("provide at least one structure path")
            if len(paths) > min(
                int(self._config.chgnet_max_structures), MAX_STRUCTURE_COUNT
            ):
                raise ValueError(
                    "structure count exceeds the configured CHGNet screen "
                    f"limit ({self._config.chgnet_max_structures})"
                )
            resolved = [
                _resolve_input_file(path_value, self._workspace)
                for path_value in paths
            ]
            rank = arguments.get("rank", True)
            if not isinstance(rank, bool):
                raise ValueError("rank must be a boolean")
            structures = [_load_structure(path) for path in resolved]
        except Exception as exc:
            return self._error(
                f"chgnet.screen invalid input: {exc}",
                error_type="invalid_input",
            )

        try:
            model = self._get_model()
            rows: list[dict[str, Any]] = []
            evidence: list[ScientificEvidence] = []
            for path, structure in zip(resolved, structures, strict=True):
                prediction = _predict_structure(model, structure)
                row = _prediction_summary(
                    prediction,
                    structure,
                    self._workspace.relative(path),
                )
                rows.append(row)
                evidence.append(
                    _screen_evidence(
                        row,
                        model_name=self._config.chgnet_model_name,
                        tool_name=self.name,
                    )
                )
            _apply_same_composition_ranks(rows, enabled=rank)
        except Exception as exc:
            if isinstance(exc, ImportError):
                return self._error(
                    "chgnet.screen missing dependency: install photomatagent[chgnet]",
                    error_type="missing_dependency",
                    details={"error": "MISSING_DEPENDENCY", "dependency": "chgnet"},
                )
            return self._error(
                f"chgnet.screen failed: {type(exc).__name__}: {exc}",
                error_type="chgnet_error",
            )

        payload: dict[str, Any] = {
            "count": len(rows),
            "model_name": self._config.chgnet_model_name,
            "device": self._config.chgnet_device,
            "results": rows,
        }
        groups = _composition_groups(rows)
        if rank and len(groups) == 1:
            payload["ranking"] = {
                "scope": "same_reduced_composition",
                "composition": next(iter(groups)),
                "order": [
                    row["path"]
                    for row in sorted(
                        rows,
                        key=lambda row: (
                            _energy_sort_key(row.get("energy_eV_per_atom")),
                            str(row["path"]),
                        ),
                    )
                ],
            }
        elif rank and len(groups) > 1:
            # A mixed-composition batch may still have per-group ranks, but
            # deliberately has no global rank or cross-composition comparison.
            payload["ranking_note"] = (
                "Cross-composition ranking omitted; ranks, when present, "
                "are only within identical reduced-composition groups."
            )
        return ScientificToolResult(
            output=_bounded_json(payload),
            data=payload,
            evidence=evidence,
        )


class CHGNetRelaxTool(_CHGNetTool):
    """Pre-relax one structure and save a bounded CIF artifact."""

    name = "chgnet.relax"
    description = (
        "Pre-relax one workspace-contained structure with CHGNet using "
        "bounded force and step limits, optionally relaxing the cell. Writes "
        "a CIF below user_output/chgnet/ and returns before/after summaries; "
        "the result is an ML-potential pre-relaxation, not DFT validation."
    )
    short_description = "Pre-relax one structure with CHGNet and write a CIF artifact."
    exposure = ToolExposure.DEFERRED
    namespace = "chgnet"
    source = "CHGNet"
    tags = (
        "chgnet",
        "machine learning potential",
        "ml interatomic potential",
        "relaxation",
        "pre-relaxation",
        "cif",
    )
    cost_class = "MODERATE"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative workspace path to a crystal structure.",
            },
            "fmax": {
                "type": "number",
                "minimum": MIN_RELAX_FMAX,
                "maximum": MAX_RELAX_FMAX,
                "description": "Maximum force threshold in eV/angstrom.",
            },
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_RELAX_STEPS,
                "description": "Maximum optimizer steps.",
            },
            "relax_cell": {
                "type": "boolean",
                "description": "Whether to relax the unit cell (default true).",
            },
            "output_name": {
                "type": "string",
                "description": "Optional safe filename below user_output/chgnet/.",
            },
        },
        "required": ["path"],
    }

    def __init__(
        self,
        config: ScientificConfig,
        workspace: Workspace,
        *,
        model: Any | None = None,
        model_loader: ModelLoader | None = None,
        optimizer: Any | None = None,
        optimizer_factory: OptimizerFactory | None = None,
    ) -> None:
        super().__init__(
            config, workspace, model=model, model_loader=model_loader
        )
        self._optimizer = optimizer
        self._optimizer_factory = optimizer_factory or _default_optimizer_factory

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        # Resolve every model-visible argument before loading either CHGNet
        # model or optimizer.  This keeps invalid paths/limits side-effect free.
        try:
            path_value = arguments.get("path")
            path = _resolve_input_file(path_value, self._workspace)
            fmax = _validated_fmax(
                arguments.get("fmax", self._config.chgnet_relax_fmax)
            )
            steps = _validated_steps(
                arguments.get("steps", self._config.chgnet_relax_steps)
            )
            relax_cell = arguments.get("relax_cell", True)
            if not isinstance(relax_cell, bool):
                raise ValueError("relax_cell must be a boolean")
            output_name = _output_name(path, arguments.get("output_name"))
            output_path = self._workspace.resolve(
                f"user_output/chgnet/{output_name}", must_exist=False
            )
            if output_path.parent != self._workspace.root / "user_output" / "chgnet":
                raise ValueError("output path must stay below user_output/chgnet")
            structure = _load_structure(path)
        except Exception as exc:
            if isinstance(exc, ImportError):
                return self._error(
                    "chgnet.relax missing dependency: install photomatagent[chgnet]",
                    error_type="missing_dependency",
                    details={"error": "MISSING_DEPENDENCY", "dependency": "chgnet"},
                )
            return self._error(
                f"chgnet.relax invalid input: {exc}",
                error_type="invalid_input",
            )

        try:
            model = self._get_model()
            before_prediction = _predict_structure(model, structure)
            before = _prediction_summary(
                before_prediction, structure, self._workspace.relative(path)
            )
            optimizer = self._optimizer
            if optimizer is None:
                optimizer = _invoke_optimizer_factory(
                    self._optimizer_factory, model, self._config
                )
            relaxed = _run_optimizer(
                optimizer,
                structure,
                fmax=fmax,
                steps=steps,
                relax_cell=relax_cell,
            )
            final_structure = _final_structure(relaxed)
            after_prediction = _predict_structure(model, final_structure)
            after = _prediction_summary(
                after_prediction, final_structure, self._workspace.relative(path)
            )
            cif = _structure_to_cif(final_structure)
            _atomic_write_text(output_path, cif)
        except Exception as exc:
            return self._error(
                f"chgnet.relax failed: {type(exc).__name__}: {exc}",
                error_type="chgnet_error",
            )

        max_force = after.get("max_force_eV_A")
        converged = (
            bool(max_force is not None and float(max_force) <= fmax)
            if max_force is not None
            else None
        )
        output_relative = self._workspace.relative(output_path)
        payload = {
            "input_path": self._workspace.relative(path),
            "output_path": str(output_path),
            "output_relative_path": output_relative,
            "model_name": self._config.chgnet_model_name,
            "device": self._config.chgnet_device,
            "fmax_eV_A": fmax,
            "steps": steps,
            "relax_cell": relax_cell,
            "converged": converged,
            "before": before,
            "after": after,
        }
        evidence = [
            _relax_evidence(
                before,
                phase="before",
                model_name=self._config.chgnet_model_name,
                tool_name=self.name,
            ),
            _relax_evidence(
                after,
                phase="after",
                model_name=self._config.chgnet_model_name,
                tool_name=self.name,
            ),
        ]
        return ScientificToolResult(
            output=_bounded_json(payload),
            data=payload,
            artifacts=[output_relative],
            evidence=evidence,
        )


def _invoke_model_loader(loader: ModelLoader, config: ScientificConfig) -> Any:
    """Call an injected loader while accepting common test-double signatures."""

    if _accepts_n_positional(loader, 2):
        return loader(config.chgnet_model_name, config.chgnet_device)
    return loader(config)


def _invoke_optimizer_factory(
    factory: OptimizerFactory, model: Any, config: ScientificConfig
) -> Any:
    if _accepts_n_positional(factory, 2):
        return factory(model, config)
    return factory(model)


def _accepts_n_positional(function: Callable[..., Any], count: int) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    return has_varargs or len(positional) >= count


def _screen_input_paths(arguments: Mapping[str, Any]) -> list[str]:
    raw: Any = arguments.get("paths")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("paths must be a list of strings")
    paths = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("every structure path must be a non-empty string")
        paths.append(item)
    return paths


def _resolve_input_file(path_value: Any, workspace: Workspace) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("path must be a non-empty string")
    raw = Path(path_value).expanduser()
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"path is outside workspace: {path_value}")
    return workspace.resolve(path_value, must_exist=True)


def _load_structure(path: Path) -> Any:
    from pymatgen.core import Structure

    return Structure.from_file(str(path))


def _predict_structure(model: Any, structure: Any) -> Mapping[str, Any]:
    predictor = getattr(model, "predict_structure", None)
    if not callable(predictor):
        raise TypeError("CHGNet model does not expose predict_structure")
    prediction = predictor(structure)
    if isinstance(prediction, Sequence) and not isinstance(
        prediction, (str, bytes, bytearray, Mapping)
    ):
        if len(prediction) != 1:
            raise TypeError("single-structure CHGNet prediction was not scalar")
        prediction = prediction[0]
    if isinstance(prediction, Mapping):
        return prediction
    attributes = {
        key: getattr(prediction, key)
        for key in (
            "e",
            "f",
            "s",
            "m",
            "energy",
            "forces",
            "stress",
            "magmom",
            "magmoms",
        )
        if hasattr(prediction, key)
    }
    if attributes:
        return attributes
    raise TypeError("CHGNet prediction must be a mapping")


def _prediction_summary(
    prediction: Mapping[str, Any], structure: Any, relative_path: str
) -> dict[str, Any]:
    energy_raw = _first_mapping_value(
        prediction, "e", "energy", "energy_per_atom", "energy_eV_per_atom"
    )
    force_raw = _first_mapping_value(
        prediction, "f", "forces", "force", "forces_eV_A"
    )
    stress_raw = _first_mapping_value(
        prediction, "s", "stress", "stress_GPa"
    )
    magmom_raw = _first_mapping_value(
        prediction, "m", "magmom", "magmoms", "magnetic_moments"
    )
    energy = _finite_scalar(energy_raw)
    max_force = _maximum_force(force_raw)
    stress = _bounded_numeric_tree(stress_raw, MAX_SUMMARY_VALUES)
    magmom_values = _flatten_numbers(magmom_raw, MAX_SUMMARY_VALUES)
    try:
        formula = str(structure.composition.reduced_formula)
    except Exception:
        formula = ""
    try:
        n_atoms = int(len(structure))
    except Exception:
        n_atoms = None
    magmom_summary: Any
    if len(magmom_values) == 1:
        magmom_summary = magmom_values[0]
    else:
        magmom_summary = magmom_values
    row: dict[str, Any] = {
        "path": relative_path,
        "formula": formula,
        "reduced_composition": formula,
        "composition_group": formula,
        "n_atoms": n_atoms,
        "energy_eV_per_atom": _round_or_none(energy),
        "max_force_eV_A": _round_or_none(max_force),
        "max_force_eV_per_A": _round_or_none(max_force),
        "max_force_eV_per_angstrom": _round_or_none(max_force),
        "stress_GPa": stress,
        "magmom_mu_B": magmom_summary,
        "max_abs_magmom_mu_B": _round_or_none(
            max((abs(value) for value in magmom_values), default=None)
        ),
    }
    return row


def _first_mapping_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _as_python(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        try:
            value = numpy()
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            value = tolist()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _as_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_python(item) for item in value]
    return value


def _finite_scalar(value: Any) -> float | None:
    value = _as_python(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            scalar = _finite_scalar(item)
            if scalar is not None:
                return scalar
        return None
    if isinstance(value, bool):
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _flatten_numbers(value: Any, limit: int = MAX_SUMMARY_VALUES) -> list[float]:
    value = _as_python(value)
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            if len(result) >= limit:
                break
            result.extend(_flatten_numbers(item, limit - len(result)))
        return result[:limit]
    scalar = _finite_scalar(value)
    return [] if scalar is None else [scalar]


def _maximum_force(value: Any) -> float | None:
    value = _as_python(value)
    if value is None:
        return None
    rows: list[list[float]] = []
    if isinstance(value, (list, tuple)):
        if value and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        ):
            scalar_values = _flatten_numbers(value, 3)
            if len(scalar_values) == 3:
                rows.append(scalar_values)
            elif scalar_values:
                return max(abs(item) for item in scalar_values)
        else:
            for item in value[:MAX_SUMMARY_VALUES]:
                if isinstance(item, (list, tuple)):
                    row = _flatten_numbers(item, 3)
                    if row:
                        rows.append(row)
    else:
        scalar = _finite_scalar(value)
        return abs(scalar) if scalar is not None else None
    norms = [
        math.sqrt(sum(component * component for component in row))
        for row in rows
        if row
    ]
    return max(norms, default=None)


def _bounded_numeric_tree(value: Any, limit: int) -> Any:
    value = _as_python(value)
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_numeric_tree(item, limit)
            for key, item in list(value.items())[:limit]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_numeric_tree(item, limit)
            for item in list(value)[:limit]
        ]
    scalar = _finite_scalar(value)
    return _round_or_none(scalar) if scalar is not None else None


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(float(value), 8)


def _composition_groups(rows: Sequence[Row]) -> dict[str, list[Row]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("reduced_composition", ""))].append(row)
    return dict(groups)


def _energy_sort_key(value: Any) -> tuple[int, float]:
    scalar = _finite_scalar(value)
    return (0, scalar) if scalar is not None else (1, math.inf)


def _apply_same_composition_ranks(
    rows: list[dict[str, Any]], *, enabled: bool
) -> None:
    if not enabled:
        return
    groups = _composition_groups(rows)
    for members in groups.values():
        ordered = sorted(
            members,
            key=lambda row: (
                _energy_sort_key(row.get("energy_eV_per_atom")),
                str(row.get("path", "")),
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            row["rank_within_composition"] = rank
            if len(groups) == 1:
                row["rank"] = rank


def _screen_evidence(
    row: Mapping[str, Any], *, model_name: str, tool_name: str
) -> ScientificEvidence:
    return ScientificEvidence(
        subject=str(row.get("formula") or row.get("path") or "structure"),
        property="chgnet_screen",
        value={
            "energy_eV_per_atom": row.get("energy_eV_per_atom"),
            "max_force_eV_A": row.get("max_force_eV_A"),
            "stress_GPa": row.get("stress_GPa"),
            "magmom_mu_B": row.get("magmom_mu_B"),
        },
        unit="",
        source="CHGNet",
        source_type="ml_interatomic_potential",
        method=f"CHGNet {model_name} single-point prediction",
        fidelity="ml_potential",
        summary=(
            f"CHGNet screened {row.get('formula', 'structure')}: "
            f"energy={row.get('energy_eV_per_atom')} eV/atom, "
            f"maximum force={row.get('max_force_eV_A')} eV/angstrom"
        ),
        limitations=(
            "ML interatomic-potential estimate only; raw energy may be "
            "ranked within identical reduced composition, not across "
            "compositions, and is not DFT formation energy, hull energy, "
            "thermodynamic stability, synthesizability, or detector validation."
        ),
        provenance={
            "tool": tool_name,
            "model": model_name,
            "path": row.get("path", ""),
        },
    )


def _relax_evidence(
    row: Mapping[str, Any],
    *,
    phase: str,
    model_name: str,
    tool_name: str,
) -> ScientificEvidence:
    return ScientificEvidence(
        subject=str(row.get("formula") or row.get("path") or "structure"),
        property=f"chgnet_relax_{phase}",
        value={
            "energy_eV_per_atom": row.get("energy_eV_per_atom"),
            "max_force_eV_A": row.get("max_force_eV_A"),
        },
        unit="",
        source="CHGNet",
        source_type="ml_interatomic_potential",
        method=f"CHGNet {model_name} {phase} relaxation summary",
        fidelity="ml_potential",
        summary=(
            f"CHGNet {phase} relaxation summary for "
            f"{row.get('formula', 'structure')}"
        ),
        limitations=(
            "ML-potential pre-relaxation only; the CIF requires downstream "
            "DFT validation and is not a stability or detector-performance claim."
        ),
        provenance={"tool": tool_name, "model": model_name, "phase": phase},
    )


def _validated_fmax(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("fmax must be a number")
    try:
        fmax = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("fmax must be a number") from exc
    if not math.isfinite(fmax) or not MIN_RELAX_FMAX <= fmax <= MAX_RELAX_FMAX:
        raise ValueError(
            f"fmax must be between {MIN_RELAX_FMAX} and {MAX_RELAX_FMAX} eV/angstrom"
        )
    return fmax


def _validated_steps(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("steps must be an integer")
    if not 1 <= value <= MAX_RELAX_STEPS:
        raise ValueError(f"steps must be between 1 and {MAX_RELAX_STEPS}")
    return value


def _output_name(path: Path, requested: Any) -> str:
    if requested is None:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
        return f"{stem or 'structure'}_relaxed.cif"
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("output_name must be a non-empty filename")
    candidate = Path(requested)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or len(candidate.parts) != 1
        or candidate.name != requested
    ):
        raise ValueError("output_name must be a filename below user_output/chgnet")
    if candidate.suffix.lower() != ".cif":
        raise ValueError("output_name must end with .cif")
    return candidate.name


def _run_optimizer(
    optimizer: Any,
    structure: Any,
    *,
    fmax: float,
    steps: int,
    relax_cell: bool,
) -> Any:
    relax = getattr(optimizer, "relax", None)
    if not callable(relax):
        raise TypeError("CHGNet optimizer does not expose relax")
    try:
        return relax(
            structure,
            fmax=fmax,
            steps=steps,
            relax_cell=relax_cell,
            verbose=False,
        )
    except TypeError as exc:
        if "verbose" not in str(exc):
            raise
        return relax(
            structure,
            fmax=fmax,
            steps=steps,
            relax_cell=relax_cell,
        )


def _final_structure(result: Any) -> Any:
    if isinstance(result, Mapping):
        for key in ("final_structure", "structure", "final_atoms", "atoms"):
            if key in result and result[key] is not None:
                result = result[key]
                break
        else:
            raise ValueError("optimizer result has no final structure")
    try:
        from pymatgen.core import Structure

        if isinstance(result, Structure):
            return result
    except Exception:
        pass
    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        return AseAtomsAdaptor.get_structure(result)
    except Exception as exc:
        raise TypeError("optimizer did not return a pymatgen Structure") from exc


def _structure_to_cif(structure: Any) -> str:
    to = getattr(structure, "to", None)
    if not callable(to):
        raise TypeError("relaxed structure cannot be serialized as CIF")
    try:
        cif = to(fmt="cif")
    except TypeError:
        cif = to("cif")
    if isinstance(cif, bytes):
        cif = cif.decode("utf-8")
    if not isinstance(cif, str) or not cif.strip():
        raise ValueError("relaxed structure produced an empty CIF")
    return cif


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _bounded_json(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= MAX_SCREEN_OUTPUT_CHARS:
        return text
    compact = {
        key: value
        for key, value in payload.items()
        if key not in {"results", "before", "after"}
    }
    compact["truncated"] = True
    compact_text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(compact_text) <= MAX_SCREEN_OUTPUT_CHARS:
        return compact_text
    return json.dumps(
        {"truncated": True, "count": payload.get("count")},
        ensure_ascii=False,
        separators=(",", ":"),
    )


__all__ = [
    "CHGNetCapabilityPack",
    "CHGNetProbe",
    "CHGNetRelaxTool",
    "CHGNetScreenTool",
    "chgnet_pack",
]
