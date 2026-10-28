"""Deterministic, host-owned checks of sealed QD trial outputs.

``artifact_path`` is an attempt-relative path bound by the trusted trial to an
exact sealed output. Presence proves only sealed bytes exist, not
scientific correctness. Accepted parameter keys are closed: schema takes only
artifact_path and uses contract.output_schema; finite adds optional JSON Pointer
value_path; numeric_range adds finite minimum and/or maximum; reference_error
requires value_path, opaque reference_id and finite max_absolute_error;
required_evidence takes only artifact_path; human_review takes none.
References stay host side. quality_metric names a check ID with a finite
measured value; otherwise quality is unset.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from photomatagent.scientific.evolution.qd.models import (
    AcceptanceCheck, ArtifactDigest, CheckOutcome, EvaluationContract,
    TrialRecord, ValidationRecord, WorkflowGenome, canonical_genome_sha256,
)
from photomatagent.errors import ToolExecutionError
from photomatagent.workspace import Workspace

Verdict = Literal["PASS", "FAIL", "UNKNOWN"]
CHECKER_VERSION = "qd-contract-v1"
_KEYS = {
    "schema": {"artifact_path"},
    "finite": {"artifact_path", "value_path"},
    "numeric_range": {"artifact_path", "value_path", "minimum", "maximum"},
    "reference_error": {"artifact_path", "value_path", "reference_id", "max_absolute_error"},
    "required_evidence": {"artifact_path"},
    "human_review": set(),
    "registered_checker": {"artifact_path", "checker", "input_refs"},
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def canonical_contract_sha256(contract: EvaluationContract) -> str:
    """Hash the frozen contract, ordered checks and suite digest."""
    return hashlib.sha256(_canonical(contract.model_dump(mode="json"))).hexdigest()


def aggregate_required(outcomes: Iterable[CheckOutcome]) -> Verdict:
    required = [item.verdict for item in outcomes if item.required]
    if "FAIL" in required:
        return "FAIL"
    if not required or "UNKNOWN" in required:
        return "UNKNOWN"
    return "PASS"


def _outcome(check: str, required: bool, verdict: Verdict, reason: str,
             digest: str | None = None, measured: float | None = None,
             additional_digests: Iterable[str] = ()) -> CheckOutcome:
    evidence_digests = tuple(dict.fromkeys(([digest] if digest else []) + list(additional_digests)))
    return CheckOutcome(check_id=check, required=required, verdict=verdict,
                        reason_code=reason, measured_value=measured,
                        evidence_refs=tuple(f"sha256:{item}" for item in evidence_digests))


def _safe_path(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and not value.startswith("/")
            and not re.match(r"^[A-Za-z]:", value) and chr(92) not in value
            and all(part not in ("", ".", "..") for part in value.split("/")))


def _bound_artifact_path(trial: TrialRecord, logical_path: str) -> str | None:
    """Resolve a frozen attempt-relative output path within this trial only."""
    if logical_path.startswith(trial.attempt_dir + "/"):
        return logical_path
    if logical_path.startswith("attempts/"):
        return None
    return f"{trial.attempt_dir}/{logical_path}"


def _pointer_valid(value: object) -> bool:
    return (isinstance(value, str) and (not value or value.startswith("/"))
            and re.search(r"~(?![01])", value) is None)


def _pointer(value: Any, pointer: str) -> Any:
    for encoded in pointer.split("/")[1:] if pointer else ():
        part = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, Mapping):
            value = value[part]
        elif isinstance(value, list) and part.isdecimal() and (part == "0" or not part.startswith("0")):
            value = value[int(part)]
        else:
            raise KeyError(part)
    return value


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant: {value}")


def _parameters_valid(check: AcceptanceCheck) -> bool:
    kind, params = check.kind, check.parameters
    if kind not in _KEYS or not set(params).issubset(_KEYS[kind]):
        return False
    if kind == "human_review":
        return not params
    if not _safe_path(params.get("artifact_path")):
        return False
    if kind in ("finite", "numeric_range", "reference_error") and not _pointer_valid(params.get("value_path", "")):
        return False
    if kind == "numeric_range":
        bounds = {"minimum", "maximum"} & set(params)
        return (bool(bounds) and all(_finite(params[key]) for key in bounds)
                and not ("minimum" in params and "maximum" in params
                         and params["minimum"] > params["maximum"]))
    if kind == "reference_error":
        ref = params.get("reference_id")
        tolerance = params.get("max_absolute_error")
        return ("value_path" in params and isinstance(ref, str)
                and re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", ref) is not None
                and _finite(tolerance) and isinstance(tolerance, (int, float)) and tolerance >= 0)
    if kind == "registered_checker":
        checker = params.get("checker")
        input_refs = params.get("input_refs")
        return (set(params) == {"artifact_path", "checker", "input_refs"}
                and isinstance(checker, Mapping)
                and isinstance(input_refs, Mapping)
                and 0 < len(input_refs) <= 32
                and all(isinstance(key, str)
                        and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", key)
                        and _safe_path(value) for key, value in input_refs.items()))
    return True


class ContractValidator:
    """Use trusted loaders and sealed trial refs; never execute candidate code."""

    def __init__(self, workspace: Workspace, *,
                 genome_loader: Callable[[str], WorkflowGenome | None],
                 reference_provider: Callable[[str], float | None] | None = None) -> None:
        self.workspace = workspace
        self.genome_loader = genome_loader
        self.reference_provider = reference_provider

    def _read(self, trial: TrialRecord, ref: ArtifactDigest) -> tuple[bytes | None, str | None]:
        if not _safe_path(ref.path) or not ref.path.startswith(trial.attempt_dir + "/"):
            return None, "ARTIFACT_PATH_UNSAFE"
        current = self.workspace.root
        for part in Path(ref.path).parts:
            current /= part
            if current.is_symlink():
                return None, "ARTIFACT_PATH_UNSAFE"
        try:
            attempt = self.workspace.resolve(trial.attempt_dir, must_exist=False)
            path = self.workspace.resolve(ref.path, must_exist=False)
            if attempt not in path.parents or not path.is_file():
                return None, "EVIDENCE_MISSING"
            payload = path.read_bytes()
        except (OSError, ValueError, RuntimeError, ToolExecutionError):
            return None, "ARTIFACT_PATH_UNSAFE"
        if len(payload) != ref.size_bytes or hashlib.sha256(payload).hexdigest() != ref.sha256:
            return None, "ARTIFACT_CHANGED"
        return payload, None

    def _check(self, check: AcceptanceCheck, contract: EvaluationContract,
               trial: TrialRecord, refs: Mapping[str, ArtifactDigest],
               sealed: Mapping[str, tuple[bytes | None, str | None]]) -> CheckOutcome:
        def unavailable() -> CheckOutcome:
            return _outcome(check.check_id, check.required, "UNKNOWN", "CHECK_UNAVAILABLE")
        if not _parameters_valid(check):
            return unavailable()
        if check.kind == "human_review":
            return _outcome(check.check_id, check.required, "UNKNOWN", "HUMAN_REVIEW_REQUIRED")
        if check.kind == "schema":
            try:
                import jsonschema  # type: ignore[import-untyped]
                jsonschema.Draft202012Validator.check_schema(contract.model_dump(mode="json")["output_schema"])
            except ImportError:
                return unavailable()
            except jsonschema.exceptions.SchemaError:
                return unavailable()
        logical_path = check.parameters["artifact_path"]
        if not isinstance(logical_path, str):
            return unavailable()
        artifact_path = _bound_artifact_path(trial, logical_path)
        if artifact_path is None:
            return unavailable()
        ref = refs.get(artifact_path)
        if ref is None:
            return _outcome(check.check_id, check.required, "UNKNOWN", "EVIDENCE_MISSING")
        payload, issue = sealed[artifact_path]
        if issue:
            return _outcome(check.check_id, check.required,
                            "FAIL" if issue == "ARTIFACT_CHANGED" else "UNKNOWN", issue)
        assert payload is not None
        digest = ref.sha256
        if check.kind == "registered_checker":
            from photomatagent.application.skills import CheckerSpec, CHECKERS
            try:
                spec = CheckerSpec.model_validate(dict(check.parameters["checker"]))
                if spec.output_path != logical_path:
                    return unavailable()
                input_payloads: dict[str, bytes] = {}
                input_digests: list[str] = []
                sealed_inputs = {item.path: item for item in trial.input_digests}
                for key, relative in check.parameters["input_refs"].items():
                    path = f"{trial.attempt_dir}/{relative}"
                    input_ref = sealed_inputs[path]
                    content, input_issue = self._read(trial, input_ref)
                    if input_issue or content is None:
                        return unavailable()
                    input_payloads[key] = content
                    input_digests.append(input_ref.sha256)
                with tempfile.TemporaryDirectory(prefix="photomatagent-check-") as directory:
                    snapshot_root = Path(directory)
                    inputs: dict[str, Path] = {}
                    for key, content in input_payloads.items():
                        input_file = snapshot_root / f"input-{key}"
                        input_file.write_bytes(content)
                        inputs[key] = input_file
                    output_file = snapshot_root / "output.json"
                    output_file.write_bytes(payload)
                    result = CHECKERS[spec.checker_id](spec, inputs, output_file)
                verdict = result["status"]
                if verdict not in {"PASS", "FAIL", "UNKNOWN"}:
                    return unavailable()
                return _outcome(check.check_id, check.required, verdict,
                                str(result.get("gap") or "REGISTERED_" + verdict), digest,
                                1.0 if verdict == "PASS" else 0.0,
                                additional_digests=input_digests)
            except (ValueError, KeyError, TypeError, OSError):
                return unavailable()
        if check.kind == "required_evidence":
            return _outcome(check.check_id, check.required, "PASS", "EVIDENCE_PRESENT", digest)
        try:
            document = json.loads(payload, parse_constant=_reject_non_json_constant)
        except (UnicodeDecodeError, ValueError):
            return _outcome(check.check_id, check.required, "FAIL", "JSON_INVALID", digest)
        if check.kind == "schema":
            try:
                import jsonschema  # type: ignore[import-untyped]
                schema = contract.model_dump(mode="json")["output_schema"]
                jsonschema.Draft202012Validator(schema).validate(document)
            except ImportError:
                return unavailable()
            except jsonschema.exceptions.SchemaError:
                return unavailable()
            except jsonschema.exceptions.ValidationError:
                return _outcome(check.check_id, check.required, "FAIL", "SCHEMA_VIOLATION", digest)
            return _outcome(check.check_id, check.required, "PASS", "SCHEMA_VALID", digest)
        if check.kind not in ("finite", "numeric_range", "reference_error"):
            return unavailable()
        try:
            value = _pointer(document, str(check.parameters.get("value_path", "")))
        except (KeyError, IndexError, TypeError):
            return _outcome(check.check_id, check.required, "UNKNOWN", "EVIDENCE_MISSING", digest)
        if not _finite(value):
            return _outcome(check.check_id, check.required, "FAIL", "NONFINITE_VALUE", digest)
        measured = float(value)
        if check.kind == "finite":
            return _outcome(check.check_id, check.required, "PASS", "FINITE_VALUE", digest, measured)
        if check.kind == "numeric_range":
            params = check.parameters
            within = (("minimum" not in params or measured >= params["minimum"])
                      and ("maximum" not in params or measured <= params["maximum"]))
            return _outcome(check.check_id, check.required, "PASS" if within else "FAIL",
                            "IN_RANGE" if within else "OUT_OF_RANGE", digest, measured)
        if self.reference_provider is None:
            return unavailable()
        try:
            expected = self.reference_provider(str(check.parameters["reference_id"]))
        except (KeyError, LookupError, ValueError):
            return unavailable()
        if not _finite(expected) or expected is None:
            return unavailable()
        within = abs(measured - float(expected)) <= check.parameters["max_absolute_error"]
        return _outcome(check.check_id, check.required, "PASS" if within else "FAIL",
                        "REFERENCE_MATCH" if within else "REFERENCE_MISMATCH", digest, measured)

    def validate(self, *, trial: TrialRecord, contract: EvaluationContract) -> ValidationRecord:
        contract_digest = canonical_contract_sha256(contract)
        refs = {ref.path: ref for ref in trial.output_refs}
        sealed = {path: self._read(trial, ref) for path, ref in refs.items()}
        outcomes: list[CheckOutcome] = []
        if len(refs) != len(trial.output_refs):
            outcomes.append(_outcome("artifact_manifest_preflight", True, "UNKNOWN", "CHECK_UNAVAILABLE"))
        if trial.contract_sha256 != contract_digest:
            outcomes.append(_outcome("contract_preflight", True, "UNKNOWN", "CONTRACT_MISMATCH"))
        if trial.seed_set != contract.seed_set:
            outcomes.append(_outcome("suite_preflight", True, "UNKNOWN", "SUITE_MISMATCH"))
        try:
            genome = self.genome_loader(trial.genome_id)
            if (genome is None or genome.genome_id != trial.genome_id
                    or genome.task_family != contract.task_family
                    or canonical_genome_sha256(genome) != genome.content_sha256):
                genome = None
        except (KeyError, ValueError, OSError):
            genome = None
        if genome is None:
            outcomes.append(_outcome("genome_preflight", True, "UNKNOWN", "GENOME_CONTENT_UNAVAILABLE"))
        seen: set[str] = set()
        for check in contract.checks:
            if check.check_id in seen:
                outcomes.append(_outcome("contract_preflight", True, "UNKNOWN", "CHECK_UNAVAILABLE"))
            seen.add(check.check_id)
            outcomes.append(self._check(check, contract, trial, refs, sealed))
        artifact_required: dict[str, bool] = {}
        for check in contract.checks:
            path = check.parameters.get("artifact_path")
            if isinstance(path, str):
                bound_path = _bound_artifact_path(trial, path)
                if bound_path is not None:
                    artifact_required[bound_path] = artifact_required.get(bound_path, False) or check.required
        for path, (_, issue) in sealed.items():
            if issue:
                outcomes.append(_outcome(f"artifact_preflight_{len(outcomes)}",
                                         artifact_required.get(path, False),
                                         "FAIL" if issue == "ARTIFACT_CHANGED" else "UNKNOWN", issue))
        if not contract.checks:
            outcomes.append(_outcome("contract_preflight", True, "UNKNOWN", "CHECK_UNAVAILABLE"))
        quality = next((outcome.measured_value for outcome in outcomes
                        if outcome.check_id == contract.quality_metric and outcome.verdict == "PASS"
                        and outcome.measured_value is not None), None)
        bundle = hashlib.sha256(_canonical([ref.model_dump(mode="json") for ref in
                                            sorted(trial.output_refs, key=lambda item: item.path)])).hexdigest()
        validation_id = "v_" + hashlib.sha256(_canonical({
            "trial": trial.trial_id, "genome": genome.content_sha256 if genome else None,
            "bundle": bundle, "contract": contract_digest, "checker": CHECKER_VERSION,
            "outcomes": [item.model_dump(mode="json") for item in outcomes],
        })).hexdigest()[:32]
        return ValidationRecord(
            validation_id=validation_id, trial_id=trial.trial_id, genome_id=trial.genome_id,
            genome_sha256=genome.content_sha256 if genome else "0" * 64,
            bundle_sha256=bundle, contract_sha256=contract_digest,
            suite_sha256=contract.evaluation_suite_sha256, checker_version=CHECKER_VERSION,
            qualification=aggregate_required(outcomes), outcomes=tuple(outcomes),
            quality=quality, cost=trial.cost,
            provenance={"checker": CHECKER_VERSION, "trial_status": trial.status},
            resource_profile=trial.resource_profile,
            seed_set=trial.seed_set,
            runtime_samples_seconds=((trial.cost["runtime_seconds"],)
                                     if "runtime_seconds" in trial.cost else ()),
            direct_dependency_count=(sum(len(step.depends_on) for step in genome.steps)
                                     if (genome := self.genome_loader(trial.genome_id)) is not None else 0),
        )
