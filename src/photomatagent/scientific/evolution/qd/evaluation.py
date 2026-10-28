"""Frozen-skill evaluation contracts with per-case workspace isolation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from pathlib import PurePosixPath
from typing import Literal, Protocol

from pydantic import Field

from photomatagent.scientific.evolution.qd.budget import BudgetAmounts
from photomatagent.scientific.evolution.qd.checkers import EvaluationCheckerResult, EvaluationCheckerSpec
from photomatagent.scientific.evolution.qd.config import EvolutionBudget
from photomatagent.scientific.evolution.qd.models import ArtifactDigest, Identifier, QDModel
from photomatagent.workspace import Workspace

EvaluationCondition = Literal[
    "STATIC", "FEEDBACK_CHAIN", "RANDOM_CANDIDATE", "DIRECTED_MUTATION", "FULL_QD_SKILLS",
]
_REQUIRED_CONDITIONS: tuple[EvaluationCondition, ...] = (
    "STATIC", "FEEDBACK_CHAIN", "RANDOM_CANDIDATE", "DIRECTED_MUTATION", "FULL_QD_SKILLS",
)


class EvaluationCase(QDModel):
    case_id: Identifier
    task_group_id: Identifier
    signal_family: str
    source_id: str
    input_ref: ArtifactDigest
    reference_ref: ArtifactDigest


class FrozenSkillRef(QDModel):
    skill_id: Identifier
    bundle_sha256: str
    relative_path: str


class FrozenEvaluationManifest(QDModel):
    evaluation_id: Identifier
    model_id: str
    software_sha256: str
    dataset_sha256: str
    seed: int
    budget: EvolutionBudget
    cases: tuple[EvaluationCase, ...]
    development_groups: tuple[str, ...] = ()
    frozen_skills: tuple[FrozenSkillRef, ...] = ()
    conditions: tuple[EvaluationCondition, ...] = _REQUIRED_CONDITIONS
    ablations: tuple[str, ...] = ()

    @classmethod
    def required_conditions(cls) -> tuple[EvaluationCondition, ...]:
        return _REQUIRED_CONDITIONS

    @property
    def content_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest()

    def validate_experiment(self) -> None:
        if not self.model_id or not self.cases:
            raise ValueError("EVALUATION_MANIFEST_INVALID")
        if self.evaluation_id in {".", ".."} or "/" in self.evaluation_id or "\\" in self.evaluation_id:
            raise ValueError("EVALUATION_ID_INVALID")
        if any(case.case_id in {".", ".."} or "/" in case.case_id or "\\" in case.case_id for case in self.cases):
            raise ValueError("EVALUATION_CASE_ID_INVALID")
        if self.conditions != _REQUIRED_CONDITIONS:
            raise ValueError("EVALUATION_CONDITIONS_INCOMPLETE")
        if len({item.case_id for item in self.cases}) != len(self.cases):
            raise ValueError("EVALUATION_CASE_DUPLICATE")
        if len({item.task_group_id for item in self.cases}) < 2:
            raise ValueError("EVALUATION_GROUPS_INSUFFICIENT")
        evaluation_groups = {item.task_group_id for item in self.cases}
        if evaluation_groups.intersection(self.development_groups):
            raise ValueError("EVALUATION_GROUP_LEAKAGE")
        group_families: dict[str, tuple[str, str]] = {}
        for item in self.cases:
            group = (item.signal_family, item.source_id)
            prior = group_families.setdefault(item.task_group_id, group)
            if prior != group:
                raise ValueError("EVALUATION_GROUP_MIXED_SOURCE")
        if len({item.bundle_sha256 for item in self.frozen_skills}) != len(self.frozen_skills):
            raise ValueError("EVALUATION_SKILL_DUPLICATE")
        for skill in self.frozen_skills:
            parts = PurePosixPath(skill.relative_path).parts
            if parts != (".photomatagent", "qd", "published-skills", skill.skill_id, skill.bundle_sha256):
                raise ValueError("EVALUATION_SKILL_PATH_INVALID")
        if dataset_identity_sha256(self.cases) != self.dataset_sha256:
            raise ValueError("EVALUATION_DATASET_HASH_MISMATCH")


class FrozenEvaluationManifestV2(FrozenEvaluationManifest):
    """Explicit schema for Web evaluations with frozen condition subsets.

    V1 remains a complete five-condition contract. This distinct version adds
    registered checker configuration and may select a non-empty subset.
    """

    schema_version: Literal[2]
    conditions: tuple[EvaluationCondition, ...] = ("STATIC",)
    checker: EvaluationCheckerSpec

    def validate_experiment(self) -> None:
        if not self.conditions or len(self.conditions) != len(set(self.conditions)):
            raise ValueError("EVALUATION_CONDITIONS_INVALID")
        # Run all shared identity, dataset, held-out-group, and frozen bundle
        # checks while presenting the legacy validator its required condition
        # tuple. The V2 object and its canonical identity are never rewritten.
        shared_view = self.model_copy(update={"conditions": _REQUIRED_CONDITIONS})
        FrozenEvaluationManifest.validate_experiment(shared_view)


class EvaluationInvocation(QDModel):
    evaluation_id: str
    manifest_sha256: str
    condition: EvaluationCondition
    case_id: str
    task_group_id: str
    signal_family: str
    source_id: str
    seed: int
    attempt_index: int = 0
    feedback: tuple[str, ...] = ()
    previous_candidate: str | None = Field(default=None, max_length=4000)
    previous_verdict: Literal["PASS", "FAIL", "UNKNOWN"] | None = None
    software_sha256: str = ""
    task_workspace: str
    input_path: str
    allowed_skill_sha256: tuple[str, ...]
    remaining_budget: BudgetAmounts
    publication_enabled: Literal[False] = False
    development_history_visible: Literal[False] = False


class EvaluationExecution(QDModel):
    output: bytes
    cost: BudgetAmounts
    provenance: Literal["fake", "runtime"]
    diagnostics: tuple[str, ...] = ()
    completed: bool = True
    strategy_trace: tuple[str, ...] = ()


class EvaluationCaseResult(QDModel):
    case_id: str
    task_group_id: str
    condition: EvaluationCondition
    verdict: Literal["PASS", "FAIL", "UNKNOWN"]
    output_sha256: str | None = None
    cost: BudgetAmounts
    provenance: Literal["fake", "runtime"]
    diagnostics: tuple[str, ...] = ()
    strategy_trace: tuple[str, ...] = ()
    checker: dict[str, object] | None = None


class EvaluationReport(QDModel):
    evaluation_id: str
    manifest_sha256: str
    results: tuple[EvaluationCaseResult, ...]
    condition_success: dict[str, float | None]
    condition_metrics: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    condition_cost: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    total_cost: BudgetAmounts
    evidence_class: Literal["fake_engineering_only", "runtime_observation"]
    publication_enabled: Literal[False] = False
    development_history_visible: Literal[False] = False


class EvaluationRunner(Protocol):
    async def execute(self, invocation: EvaluationInvocation) -> EvaluationExecution: ...


ReferenceChecker = Callable[
    [EvaluationCase, bytes, bytes], bool | None | EvaluationCheckerResult
]


class FrozenSkillEvaluator:
    """Run each condition/case in a new workspace against one pinned manifest."""

    def __init__(
        self, *, workspace: Workspace,
        manifest: FrozenEvaluationManifest | FrozenEvaluationManifestV2,
        expected_manifest_sha256: str, runner: EvaluationRunner,
        checker: ReferenceChecker,
    ) -> None:
        manifest.validate_experiment()
        if manifest.content_sha256 != expected_manifest_sha256:
            raise ValueError("EVALUATION_MANIFEST_CHANGED")
        self.workspace = workspace
        self.manifest = manifest
        self.manifest_sha256 = expected_manifest_sha256
        self.runner = runner
        self.checker = checker
        self.evaluation_root = workspace.root / "tmp" / "qd-evaluations" / manifest.evaluation_id

    async def run(self) -> EvaluationReport:
        unsupported = set(getattr(self.runner, "unsupported_conditions", ()))
        requested_unsupported = set(self.manifest.conditions) & unsupported
        if requested_unsupported:
            raise ValueError("EVALUATION_CONDITION_UNSUPPORTED:" + ",".join(sorted(requested_unsupported)))
        results: list[EvaluationCaseResult] = []
        total = BudgetAmounts()
        caps = BudgetAmounts(
            tokens=self.manifest.budget.max_model_tokens,
            wall_seconds=self.manifest.budget.max_wall_seconds,
            trials=self.manifest.budget.max_trials,
            installs=self.manifest.budget.max_installs,
            download_bytes=self.manifest.budget.max_download_bytes,
            model_calls=self.manifest.budget.max_model_calls,
            proposals=self.manifest.budget.max_proposals,
        )
        condition_costs = {condition: BudgetAmounts() for condition in self.manifest.conditions}
        cancelled_run = False
        for condition_index, condition in enumerate(self.manifest.conditions):
            for case_index, case in enumerate(self.manifest.cases):
                condition_cost = condition_costs[condition]
                if _exhausted(condition_cost, caps):
                    results.append(EvaluationCaseResult(
                        case_id=case.case_id, task_group_id=case.task_group_id,
                        condition=condition, verdict="UNKNOWN", cost=BudgetAmounts(),
                        provenance="fake", diagnostics=("BUDGET_EXHAUSTED",),
                    ))
                    continue
                input_data = self._read_ref(case.input_ref)
                reference_data = self._read_ref(case.reference_ref)
                case_root = self.evaluation_root / condition.lower() / case.case_id
                self._prepare_case_workspace(case_root, input_data, include_skills=condition == "FULL_QD_SKILLS")
                skills = self.manifest.frozen_skills if condition == "FULL_QD_SKILLS" else ()
                accumulated = BudgetAmounts()
                traces: list[str] = []
                diagnostics: list[str] = []
                checker_result: dict[str, object] | None = None
                qualification: Literal["PASS", "FAIL", "UNKNOWN"] = "UNKNOWN"
                digest: str | None = None
                provenance: Literal["fake", "runtime"] = "runtime"
                feedback: list[str] = []
                previous_candidate: str | None = None
                previous_verdict: Literal["PASS", "FAIL", "UNKNOWN"] | None = None
                remaining_trials = _remaining(caps, condition_cost).trials
                attempts = min(3, remaining_trials) if condition in ("FEEDBACK_CHAIN", "RANDOM_CANDIDATE") else 1
                for attempt_index in range(attempts):
                    remaining = _remaining(caps, _add(condition_cost, accumulated))
                    if remaining.trials <= 0 or remaining.model_calls <= 0 or remaining.tokens <= 0 or remaining.wall_seconds <= 0:
                        diagnostics.append("BUDGET_EXHAUSTED")
                        break
                    seed_material = f"{self.manifest.seed}\0{condition}\0{case.case_id}\0{attempt_index}\0{self.manifest_sha256}"
                    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
                    invocation = EvaluationInvocation(
                        evaluation_id=self.manifest.evaluation_id, manifest_sha256=self.manifest_sha256,
                        condition=condition, case_id=case.case_id, task_group_id=case.task_group_id,
                        signal_family=case.signal_family, source_id=case.source_id,
                        seed=seed, attempt_index=attempt_index, feedback=tuple(feedback),
                        previous_candidate=previous_candidate, previous_verdict=previous_verdict,
                        software_sha256=self.manifest.software_sha256,
                        task_workspace=str(case_root), input_path="input/data.json",
                        allowed_skill_sha256=tuple(item.bundle_sha256 for item in skills),
                        remaining_budget=remaining,
                    )
                    started = time.monotonic()
                    try:
                        execution = await asyncio.wait_for(
                            self.runner.execute(invocation), timeout=remaining.wall_seconds,
                        )
                    except TimeoutError:
                        elapsed = max(0.0, time.monotonic() - started)
                        charge = BudgetAmounts(
                            tokens=remaining.tokens, wall_seconds=max(remaining.wall_seconds, elapsed),
                            trials=1, installs=remaining.installs, download_bytes=remaining.download_bytes,
                            model_calls=remaining.model_calls, proposals=remaining.proposals,
                        )
                        accumulated = _add(accumulated, charge)
                        diagnostics.append("EVALUATION_CASE_TIMEOUT")
                        traces.append(f"{condition}:attempt={attempt_index}:timeout:seed={seed}")
                        break
                    except asyncio.CancelledError:
                        elapsed = max(0.0, time.monotonic() - started)
                        charge = BudgetAmounts(
                            tokens=remaining.tokens, wall_seconds=max(remaining.wall_seconds, elapsed),
                            trials=1, installs=remaining.installs, download_bytes=remaining.download_bytes,
                            model_calls=remaining.model_calls, proposals=remaining.proposals,
                        )
                        accumulated = _add(accumulated, charge)
                        diagnostics.append("EVALUATION_CASE_CANCELLED")
                        traces.append(f"{condition}:attempt={attempt_index}:cancelled:seed={seed}")
                        cancelled_run = True
                        break
                    except Exception:
                        elapsed = max(0.0, time.monotonic() - started)
                        charge = BudgetAmounts(
                            tokens=remaining.tokens, wall_seconds=elapsed, trials=1,
                            installs=remaining.installs, download_bytes=remaining.download_bytes,
                            model_calls=remaining.model_calls, proposals=remaining.proposals,
                        )
                        accumulated = _add(accumulated, charge)
                        diagnostics.append("EVALUATION_EXECUTION_FAILED")
                        traces.append(f"{condition}:attempt={attempt_index}:failed:seed={seed}")
                        break
                    elapsed = max(0.0, time.monotonic() - started)
                    observed = BudgetAmounts(
                        tokens=execution.cost.tokens, wall_seconds=max(execution.cost.wall_seconds, elapsed),
                        trials=max(1, execution.cost.trials), installs=execution.cost.installs,
                        download_bytes=execution.cost.download_bytes, model_calls=execution.cost.model_calls,
                        proposals=execution.cost.proposals,
                    )
                    accumulated = _add(accumulated, observed)
                    diagnostics.extend(execution.diagnostics)
                    traces.extend(execution.strategy_trace)
                    provenance = execution.provenance
                    if execution.completed:
                        try:
                            verdict = self.checker(case, execution.output, reference_data)
                            if isinstance(verdict, EvaluationCheckerResult):
                                qualification = verdict.verdict
                                checker_result = verdict.model_dump(mode="json")
                            else:
                                qualification = "UNKNOWN" if verdict is None else "PASS" if verdict else "FAIL"
                        except Exception:
                            qualification = "UNKNOWN"
                            diagnostics.append("EVALUATION_CHECKER_FAILED")
                        digest = hashlib.sha256(execution.output).hexdigest()
                        if condition == "FEEDBACK_CHAIN":
                            previous_candidate = execution.output[:4000].decode("utf-8", errors="replace")
                            previous_verdict = qualification
                    else:
                        qualification = "UNKNOWN"
                    if _exceeds(accumulated, _remaining(caps, condition_cost)):
                        qualification = "UNKNOWN"
                        diagnostics.append("BUDGET_OVERRUN")
                        break
                    if qualification == "PASS":
                        break
                    if condition == "FEEDBACK_CHAIN" and execution.completed:
                        if qualification == "FAIL":
                            feedback.append(
                                "Checker verdict FAIL: the independent checker found the candidate did not pass. "
                                "Use the prior candidate and its actual checker verdict to revise it."
                            )
                        elif qualification == "UNKNOWN":
                            feedback.append(
                                "Checker verdict UNKNOWN: available evidence was insufficient to determine success. "
                                "This does not establish that the candidate is scientifically wrong; do not change "
                                "scientific parameters solely because of UNKNOWN. Improve checkable evidence or output."
                            )
                    if condition not in ("FEEDBACK_CHAIN", "RANDOM_CANDIDATE"):
                        break
                condition_costs[condition] = _add(condition_cost, accumulated)
                total = _add(total, accumulated)
                results.append(EvaluationCaseResult(
                    case_id=case.case_id, task_group_id=case.task_group_id,
                    condition=condition, verdict=qualification, output_sha256=digest,
                    cost=accumulated, provenance=provenance,
                    diagnostics=tuple(dict.fromkeys(diagnostics)), strategy_trace=tuple(traces),
                    checker=checker_result,
                ))
                if cancelled_run:
                    pending = [
                        (condition, item) for item in self.manifest.cases[case_index + 1:]
                    ] + [
                        (next_condition, item)
                        for next_condition in self.manifest.conditions[condition_index + 1:]
                        for item in self.manifest.cases
                    ]
                    results.extend(EvaluationCaseResult(
                        case_id=item.case_id, task_group_id=item.task_group_id,
                        condition=pending_condition, verdict="UNKNOWN", cost=BudgetAmounts(),
                        provenance="runtime", diagnostics=("EVALUATION_CANCELLED",),
                    ) for pending_condition, item in pending)
                    break
            if cancelled_run:
                break
        success: dict[str, float | None] = {}
        metrics: dict[str, dict[str, float | int]] = {}
        for condition in self.manifest.conditions:
            selected = [item for item in results if item.condition == condition]
            count = len(selected)
            unknown_count = sum(item.verdict == "UNKNOWN" for item in selected)
            exhausted_count = sum(
                any(code in {
                    "BUDGET_EXHAUSTED", "BUDGET_OVERRUN", "EVALUATION_CASE_TIMEOUT",
                } for code in item.diagnostics)
                for item in selected
            )
            success[condition] = (
                sum(item.verdict == "PASS" for item in selected) / count if count else None
            )
            metrics[condition] = {
                "cases": count,
                "unknown_rate": unknown_count / count if count else 0.0,
                "budget_exhaustion_rate": exhausted_count / count if count else 0.0,
            }
        classes = {item.provenance for item in results}
        evidence_class: Literal["fake_engineering_only", "runtime_observation"] = (
            "runtime_observation" if classes == {"runtime"} else "fake_engineering_only"
        )
        return EvaluationReport(
            evaluation_id=self.manifest.evaluation_id,
            manifest_sha256=self.manifest_sha256, results=tuple(results),
            condition_success=success, condition_metrics=metrics,
            condition_cost={key: value.to_dict() for key, value in condition_costs.items()},
            total_cost=total, evidence_class=evidence_class,
        )

    def _read_ref(self, ref: ArtifactDigest) -> bytes:
        relative = Path(ref.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("EVALUATION_DATA_PATH_INVALID")
        current = self.workspace.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("EVALUATION_DATA_SYMLINK")
        path = self.workspace.resolve(ref.path, must_exist=True)
        data = path.read_bytes()
        if len(data) != ref.size_bytes or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ValueError("EVALUATION_DATA_CHANGED")
        return data

    def _prepare_case_workspace(self, root: Path, input_data: bytes, *, include_skills: bool) -> None:
        if root.exists():
            raise ValueError("EVALUATION_WORKSPACE_ALREADY_USED")
        current = self.workspace.root
        for part in root.relative_to(self.workspace.root).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("EVALUATION_WORKSPACE_SYMLINK")
        root.mkdir(parents=True)
        (root / "input").mkdir()
        (root / "input" / "data.json").write_bytes(input_data)
        if include_skills and self.manifest.frozen_skills:
            skills_root = root / "frozen-skills"
            skills_root.mkdir()
            for skill in self.manifest.frozen_skills:
                source = self.workspace.resolve(skill.relative_path, must_exist=True)
                source_rel = source.relative_to(self.workspace.root)
                current_source = self.workspace.root
                for part in source_rel.parts:
                    current_source = current_source / part
                    if current_source.is_symlink():
                        raise ValueError("EVALUATION_SKILL_SYMLINK")
                from photomatagent.skills.published import PublishedSkillSource

                published_root = self.workspace.root / ".photomatagent" / "qd" / "published-skills"
                PublishedSkillSource(published_root)._verify_bundle(source, skill.bundle_sha256)
                target = skills_root / skill.skill_id
                shutil.copytree(source, target, symlinks=False)
                PublishedSkillSource(published_root)._verify_bundle(target, skill.bundle_sha256)


def _add(left: BudgetAmounts, right: BudgetAmounts) -> BudgetAmounts:
    return BudgetAmounts(**{
        key: getattr(left, key) + getattr(right, key)
        for key in ("tokens", "wall_seconds", "trials", "installs", "download_bytes", "model_calls", "proposals")
    })


def _exceeds(total: BudgetAmounts, cap: BudgetAmounts) -> bool:
    return any(
        getattr(total, key) > getattr(cap, key)
        for key in ("tokens", "wall_seconds", "trials", "installs", "download_bytes", "model_calls", "proposals")
    )


def _exhausted(total: BudgetAmounts, cap: BudgetAmounts) -> bool:
    return any(
        getattr(total, key) >= getattr(cap, key)
        for key in ("tokens", "wall_seconds", "trials", "installs", "download_bytes", "model_calls", "proposals")
    )


def _remaining(cap: BudgetAmounts, used: BudgetAmounts) -> BudgetAmounts:
    return BudgetAmounts(**{
        key: max(0, getattr(cap, key) - getattr(used, key))
        for key in ("tokens", "wall_seconds", "trials", "installs", "download_bytes", "model_calls", "proposals")
    })


def dataset_identity_sha256(cases: tuple[EvaluationCase, ...]) -> str:
    payload = [
        {
            "case_id": case.case_id, "task_group_id": case.task_group_id,
            "signal_family": case.signal_family, "source_id": case.source_id,
            "input_sha256": case.input_ref.sha256,
            "reference_sha256": case.reference_ref.sha256,
        }
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _directory_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("EVALUATION_SKILL_UNAVAILABLE")
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("EVALUATION_SKILL_SYMLINK")
        if path.is_file():
            payload = path.read_bytes()
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "media_type": _media_type(path),
            })
    return hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _media_type(path: Path) -> str:
    if path.suffix == ".py":
        return "text/x-python"
    if path.suffix == ".json":
        return "application/json"
    if path.suffix in {".md", ".lock"}:
        return "text/plain; charset=utf-8"
    return "application/octet-stream"
