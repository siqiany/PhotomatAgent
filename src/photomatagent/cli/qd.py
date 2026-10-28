"""Explicit command surface for persisted quality-diversity tasks."""
from __future__ import annotations

import json
import os
import re
import asyncio
import tempfile
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.console import Console

from photomatagent.scientific.evolution.qd.config import EvolutionBudget, QDConfig
from photomatagent.scientific.evolution.qd.models import (
    EvaluationContract, RevisionIssue, SkillPackage, WorkflowGenome,
)
from photomatagent.scientific.evolution.qd.validation import canonical_contract_sha256
from photomatagent.workspace import Workspace

qd_app = typer.Typer(help="Run bounded quality-diversity workflow evolution.")
qd_skills_app = typer.Typer(help="Inspect, publish, or revoke learned skills.")
qd_app.add_typer(qd_skills_app, name="skills")
console = Console()


from photomatagent.scientific.evolution.qd.runtime_adapter import QDTaskRecord

def _task_path(workspace: Workspace, task_id: str) -> Path:
    if not re.fullmatch(r"qdt_[0-9a-f]{32}", task_id):
        raise ValueError("QD_TASK_NOT_FOUND")
    return _safe_workspace_path(
        workspace, Path(".photomatagent/qd/tasks") / task_id / "task.json", must_exist=True,
    )


def _load_task(workspace: Workspace, task_id: str) -> QDTaskRecord:
    try:
        return QDTaskRecord.model_validate_json(_task_path(workspace, task_id).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("QD_TASK_NOT_FOUND") from exc


def _write_task(workspace: Workspace, task: QDTaskRecord, *, create: bool = False) -> None:
    path = _safe_workspace_path(
        workspace, Path(".photomatagent/qd/tasks") / task.task_id / "task.json", must_exist=False,
    )
    current = workspace.root
    for part in path.parent.relative_to(workspace.root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("QD_TASK_PATH_UNSAFE")
        current.mkdir(exist_ok=True)
    lock_path = path.parent / ".task.lock"
    if lock_path.is_symlink():
        raise ValueError("QD_TASK_PATH_UNSAFE")
    payload = task.model_dump_json(indent=2).encode("utf-8") + b"\n"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if create:
                with path.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                return
            current_task = QDTaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
            if current_task.revision + 1 != task.revision:
                raise ValueError("QD_TASK_REVISION_CONFLICT")
            fd, name = tempfile.mkstemp(prefix=".task-", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, path)
            except BaseException:
                Path(name).unlink(missing_ok=True)
                raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_workspace_json(workspace: Workspace, path: Path) -> Any:
    return json.loads(_safe_workspace_path(workspace, path, must_exist=True).read_text(encoding="utf-8"))


def _safe_workspace_path(workspace: Workspace, path: Path, *, must_exist: bool) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace.root / candidate
    try:
        relative = candidate.absolute().relative_to(workspace.root)
    except ValueError as exc:
        raise ValueError("QD_PATH_OUTSIDE_WORKSPACE") from exc
    if ".." in relative.parts or "." in relative.parts:
        raise ValueError("QD_PATH_INVALID")
    current = workspace.root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("QD_PATH_SYMLINK")
    return workspace.resolve(str(candidate), must_exist=must_exist)


def _read_qd_config(workspace: Workspace) -> QDConfig:
    path = _safe_workspace_path(workspace, Path(".photomatagent/qd.json"), must_exist=False)
    if not path.exists():
        return QDConfig()
    return QDConfig.model_validate_json(path.read_text(encoding="utf-8"))


@qd_app.command("start")
def qd_start(
    contract: Path = typer.Option(..., "--contract", exists=True, dir_okay=False),
    workflow: Path = typer.Option(..., "--workflow", exists=True, dir_okay=False),
    issues_file: Path | None = typer.Option(None, "--issues-file", exists=True, dir_okay=False),
    approval: str = typer.Option("ask", "--approval", help="ask | auto | deny"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Validate and persist a frozen QD task; execution remains a separate action."""
    if approval not in {"ask", "auto", "deny"}:
        raise typer.BadParameter("--approval must be ask | auto | deny")
    try:
        boundary = Workspace(workspace)
        qd_config = _read_qd_config(boundary)  # Invalid opt-in configuration must be diagnosed.
        contract_model = EvaluationContract.model_validate(_read_workspace_json(boundary, contract))
        genome = WorkflowGenome.model_validate(_read_workspace_json(boundary, workflow))
        if genome.task_family != contract_model.task_family:
            raise ValueError("CONTRACT_WORKFLOW_FAMILY_MISMATCH")
        raw_issues = _read_workspace_json(boundary, issues_file) if issues_file else []
        if not isinstance(raw_issues, list):
            raise ValueError("ISSUES_SCHEMA_INVALID")
        issues = tuple(RevisionIssue.model_validate(item) for item in raw_issues)
        known_checks = {check.check_id for check in contract_model.checks}
        if any(not set(issue.acceptance_check_ids).issubset(known_checks) for issue in issues):
            raise ValueError("ISSUE_CHECK_REFERENCE_INVALID")
        task_id = f"qdt_{uuid4().hex}"
        task = QDTaskRecord(
            task_id=task_id, contract=contract_model.model_dump(mode="json"),
            baseline_genome=genome.model_dump(mode="json"),
            issues=tuple(issue.model_dump(mode="json") for issue in issues),
            approval_mode=approval, budget=qd_config.budget,
            environment_id=qd_config.environment_id,
            mutable_parameters=tuple(
                item.model_dump(mode="json") for item in qd_config.mutable_parameters
            ),
        )
        _write_task(boundary, task, create=True)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        console.print(f"[red]QD_START_INVALID: {exc}[/]")
        raise typer.Exit(code=2) from None
    console.print(f"Task: {task.task_id}\nStatus: READY\nApproval: {approval}")
    console.print(f"Contract: {canonical_contract_sha256(contract_model)}")
    console.print("Run: photomatagent evolve qd run " + task.task_id)


@qd_app.command("status")
def qd_status(
    task_id: str = typer.Argument(...),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    try:
        task = _load_task(Workspace(workspace), task_id)
    except (OSError, ValueError, ValidationError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from None
    contract = EvaluationContract.model_validate(task.contract)
    genome = WorkflowGenome.model_validate(task.baseline_genome)
    print(json.dumps({
        "task_id": task.task_id, "status": task.status, "revision": task.revision,
        "task_family": contract.task_family, "contract_sha256": canonical_contract_sha256(contract),
        "baseline_genome": genome.genome_id, "open_issues": sum(item.get("status") == "open" for item in task.issues),
        "budget": task.budget.model_dump(mode="json"), "approval_mode": task.approval_mode,
        "model_provider": task.model_provider, "model_id": task.model_id,
        "event_outcome": task.event_outcome,
    }, indent=2))


@qd_app.command("feedback")
def qd_feedback(
    task_id: str = typer.Argument(...),
    issues_file: Path = typer.Option(..., "--issues-file", exists=True, dir_okay=False),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    try:
        boundary = Workspace(workspace)
        task = _load_task(boundary, task_id)
        raw = _read_workspace_json(boundary, issues_file)
        if not isinstance(raw, list):
            raise ValueError("ISSUES_SCHEMA_INVALID")
        issues = tuple(RevisionIssue.model_validate(item) for item in raw)
        known = {item.check_id for item in EvaluationContract.model_validate(task.contract).checks}
        if any(not set(issue.acceptance_check_ids).issubset(known) for issue in issues):
            raise ValueError("ISSUE_CHECK_REFERENCE_INVALID")
        _write_task(boundary, task.model_copy(update={
            "issues": tuple(issue.model_dump(mode="json") for issue in issues),
            "revision": task.revision + 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }))
    except (OSError, ValueError, ValidationError) as exc:
        console.print(f"[red]QD_FEEDBACK_INVALID: {exc}[/]")
        raise typer.Exit(code=2) from None
    console.print(f"Updated issues for {task_id}; revision {task.revision + 1}")


def _run_or_resume(task_id: str, workspace: Path, *, resume: bool = False) -> None:
    started = False
    try:
        boundary = Workspace(workspace)
        task = _load_task(boundary, task_id)
    except (OSError, ValueError, ValidationError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from None
    if task.status == "CANCELLED":
        console.print("[yellow]QD_TASK_CANCELLED[/]")
        raise typer.Exit(code=1)
    try:
        config = _read_qd_config(boundary)
        if not config.enabled:
            raise ValueError("QD_DISABLED")
        if task.environment_id is None and config.environment_id:
            task = _update_task(
                boundary, task_id, status=task.status, event_outcome=task.event_outcome,
                environment_id=config.environment_id,
            )
        if task.status in {"SOLVED", "BUDGET_EXHAUSTED"} and not resume:
            raise ValueError("QD_TASK_ALREADY_COMPLETE")
        task_path = _task_path(boundary, task_id)
        lock_path = task_path.parent / ".run.lock"
        if lock_path.is_symlink():
            raise ValueError("QD_TASK_PATH_UNSAFE")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("QD_TASK_ALREADY_RUNNING") from exc
            task = _update_task(boundary, task_id, status="RUNNING", event_outcome=None)
            started = True
            outcome = asyncio.run(_run_coordinator(boundary, task, config))
            event_outcome = outcome.model_dump(mode="json")
            latest = _load_task(boundary, task_id)
            status = "CANCELLED" if latest.status == "CANCEL_REQUESTED" else outcome.status
            _update_task(boundary, task_id, status=status, event_outcome=event_outcome)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    except (OSError, ValueError, ValidationError, RuntimeError) as exc:
        if started:
            try:
                _update_task(boundary, task_id, status="BLOCKED", event_outcome={"error": str(exc)[:500]})
            except (OSError, ValueError, ValidationError):
                pass
        console.print(f"[red]QD_RUN_FAILED: {exc}[/]")
        raise typer.Exit(code=1) from None
    console.print(json.dumps(event_outcome, ensure_ascii=False, indent=2))
    if resume and event_outcome.get("status") == "BLOCKED":
        console.print("[yellow]QD_RESUME_BLOCKED: check persisted trial state; ambiguous trials require reconciliation before retry[/]")


async def _run_coordinator(workspace: Workspace, task: QDTaskRecord, config: QDConfig):
    cancel_event = asyncio.Event()
    adapter = _make_runtime_adapter(workspace, task, config, cancel_event)
    provider = getattr(adapter.model, "provider", None)
    model_id = getattr(adapter.model, "model", None)
    if ((task.model_provider and task.model_provider != provider)
            or (task.model_id and task.model_id != model_id)):
        raise ValueError("QD_MODEL_CHANGED")
    _update_task(
        workspace, task.task_id, status=task.status, event_outcome=task.event_outcome,
        model_provider=str(provider), model_id=str(model_id),
    )
    operation = asyncio.create_task(adapter.run_event())
    while not operation.done():
        done, _ = await asyncio.wait({operation}, timeout=0.25)
        if done:
            break
        current = _load_task(workspace, task.task_id)
        if current.status == "CANCEL_REQUESTED":
            cancel_event.set()
    return await operation


def _make_runtime_adapter(workspace: Workspace, task: QDTaskRecord, config: QDConfig, cancel_event: asyncio.Event):
    from photomatagent.scientific.evolution.qd.runtime_adapter import QDRuntimeAdapter
    return QDRuntimeAdapter(
        workspace=workspace, task=task, config=config, cancel_event=cancel_event,
    )


def _update_task(
    workspace: Workspace, task_id: str, *, status: str,
    event_outcome: dict[str, Any] | None, environment_id: str | None = None,
    model_provider: str | None = None, model_id: str | None = None,
) -> QDTaskRecord:
    current = _load_task(workspace, task_id)
    changes: dict[str, Any] = {
        "status": status, "event_outcome": event_outcome,
        "revision": current.revision + 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if environment_id is not None:
        changes["environment_id"] = environment_id
    if model_provider is not None:
        changes["model_provider"] = model_provider
    if model_id is not None:
        changes["model_id"] = model_id
    updated = current.model_copy(update=changes)
    _write_task(workspace, updated)
    return updated


@qd_app.command("run")
def qd_run(task_id: str = typer.Argument(...), workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False)) -> None:
    _run_or_resume(task_id, workspace)


@qd_app.command("resume")
def qd_resume(task_id: str = typer.Argument(...), workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False)) -> None:
    _run_or_resume(task_id, workspace, resume=True)


@qd_app.command("cancel")
def qd_cancel(task_id: str = typer.Argument(...), workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False)) -> None:
    try:
        boundary = Workspace(workspace)
        task = _load_task(boundary, task_id)
        lock_path = _task_path(boundary, task_id).parent / ".run.lock"
        if lock_path.is_symlink():
            raise ValueError("QD_TASK_PATH_UNSAFE")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                status = "CANCELLED"
            except BlockingIOError:
                status = "CANCEL_REQUESTED"
            _update_task(boundary, task_id, status=status, event_outcome=task.event_outcome)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(descriptor)
    except (OSError, ValueError, ValidationError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from None
    console.print(
        f"Cancellation requested for {task_id}" if status == "CANCEL_REQUESTED"
        else f"Cancelled {task_id}"
    )


@qd_app.command("evaluate")
def qd_evaluate(
    manifest_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    """Run one hash-pinned frozen evaluation through AgentRuntime."""
    try:
        boundary = Workspace(workspace)
        envelope = _read_workspace_json(boundary, manifest_file)
        if not isinstance(envelope, dict) or set(envelope) != {"manifest", "expected_manifest_sha256"}:
            raise ValueError("EVALUATION_MANIFEST_ENVELOPE_INVALID")
        from photomatagent.scientific.evolution.qd.evaluation import (
            FrozenEvaluationManifest, FrozenSkillEvaluator,
        )
        from photomatagent.scientific.evolution.qd.evaluation_runtime import RuntimeEvaluationRunner
        from photomatagent.scientific.evolution.qd.runtime_adapter import _configured_model

        manifest = FrozenEvaluationManifest.model_validate(envelope["manifest"])
        expected = str(envelope["expected_manifest_sha256"])
        model = _configured_model(boundary.root)
        if model.model != manifest.model_id:
            raise ValueError("EVALUATION_MODEL_MISMATCH")
        evaluator = FrozenSkillEvaluator(
            workspace=boundary, manifest=manifest,
            expected_manifest_sha256=expected,
            runner=RuntimeEvaluationRunner(model=model),
            checker=_exact_json_reference_checker,
        )
        report = asyncio.run(evaluator.run())
        report_path = _safe_workspace_path(
            boundary,
            Path(".photomatagent/qd/evaluations") / manifest.evaluation_id / "report.json",
            must_exist=False,
        )
        if report_path.exists():
            raise ValueError("EVALUATION_REPORT_ALREADY_EXISTS")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_fd, temporary_name = tempfile.mkstemp(prefix=".report-", dir=report_path.parent)
        try:
            with os.fdopen(temporary_fd, "w", encoding="utf-8") as stream:
                stream.write(report.model_dump_json(indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, report_path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        console.print(f"[red]QD_EVALUATION_FAILED: {exc}[/]")
        raise typer.Exit(code=2) from None
    console.print(report.model_dump_json(indent=2))


def _exact_json_reference_checker(_case: Any, output: bytes, reference: bytes) -> bool:
    def parse(payload: bytes) -> Any:
        return json.loads(payload.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    try:
        actual = parse(output)
        expected = parse(reference)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return False
    return _json_equal(actual, expected)


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_json_equal(left[key], value) for key, value in right.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


@qd_app.command("archive")
def qd_archive(task_family: str = typer.Option(..., "--task-family"), workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False)) -> None:
    from photomatagent.scientific.evolution.qd.store import QDStore, QDStoreError

    boundary = Workspace(workspace)

    def no_trial_factory(_event_id: str, _genome_id: str):
        raise RuntimeError("read-only archive access")

    store = QDStore(boundary, trial_factory=no_trial_factory)
    root = store.root / "archives"
    found: list[dict[str, Any]] = []
    if root.is_dir() and not root.is_symlink():
        for archive_dir in root.iterdir():
            if archive_dir.is_symlink() or not archive_dir.is_dir():
                continue
            try:
                for entry in store.load_archive_entries(archive_dir.name):
                    genome = store.load_genome(entry.genome_id)
                    if genome.task_family == task_family:
                        found.append(entry.model_dump(mode="json"))
            except (OSError, ValueError, QDStoreError):
                continue
    print(json.dumps({"task_family": task_family, "entries": found}, indent=2))


@qd_skills_app.command("list")
def qd_skills_list(workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False)) -> None:
    from photomatagent.skills.loader import SkillLoader

    loader = SkillLoader(published_root=workspace / ".photomatagent" / "qd" / "published-skills")
    print(json.dumps([
        {"name": item.name, "description": item.description, "source": item.source,
         "bundle_sha256": item.bundle_sha256}
        for item in loader.load_index() if item.source == "qd-published"
    ], indent=2))


@qd_skills_app.command("inspect")
def qd_skills_inspect(candidate_id: str = typer.Argument(...), workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False)) -> None:
    if not re.fullmatch(r"candidate_[0-9a-f]{24}", candidate_id):
        raise typer.Exit(code=2)
    try:
        boundary = Workspace(workspace)
        path = _safe_workspace_path(
            boundary, Path(".photomatagent/qd/skill-candidates") / candidate_id / "candidate.json",
            must_exist=True,
        )
        package = SkillPackage.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as exc:
        console.print(f"[red]SKILL_CANDIDATE_NOT_FOUND: {exc}[/]")
        raise typer.Exit(code=2) from None
    print(package.model_dump_json(indent=2))


def _publisher(workspace: Workspace):
    from photomatagent.scientific.evolution.qd.experience_bridge import QDExperienceBridge
    from photomatagent.scientific.evolution.qd.publication import SkillPublisher
    from photomatagent.scientific.evolution.qd.skill_package import SkillPackageBuilder
    from photomatagent.scientific.evolution.qd.store import QDStore
    from photomatagent.scientific.evolution.store import EvolutionStore

    def no_trial_factory(_event_id: str, _genome_id: str):
        raise RuntimeError("read-only publication access")

    qd_store = QDStore(workspace, trial_factory=no_trial_factory)
    builder = SkillPackageBuilder(workspace=workspace, qd_store=qd_store, lock_loader=lambda _: b"")
    evolution_store = EvolutionStore(workspace)
    bridge = QDExperienceBridge(
        workspace=workspace, qd_store=qd_store, evolution_store=evolution_store,
        package_builder=builder,
    )
    publisher = SkillPublisher(
        workspace=workspace, package_builder=builder,
        experience_loader=evolution_store.load_experience,
        reviewed_links=bridge.links_for_bundle,
        verify_reviewed_link=bridge.verify_link,
    )
    return builder, publisher


@qd_skills_app.command("publish")
def qd_skills_publish(
    candidate_id: str = typer.Argument(...),
    approve: bool = typer.Option(False, "--approve", help="Explicitly approve this displayed bundle for publication."),
    actor: str = typer.Option("local-user", "--actor"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    try:
        boundary = Workspace(workspace)
        builder, publisher = _publisher(boundary)
        package = builder.load_candidate(candidate_id)
        console.print(f"Candidate {candidate_id}\nBundle SHA-256: {package.bundle_sha256}\nStatus: {package.status}")
        if not approve:
            console.print("Approval required: rerun with --approve after reviewing this bundle.")
            raise typer.Exit(code=1)
        approval_record = publisher.approve(candidate_id, actor=actor)
        published = publisher.publish(candidate_id=candidate_id, approval_id=approval_record.approval_id)
    except typer.Exit:
        raise
    except (OSError, ValueError, ValidationError) as exc:
        console.print(f"[red]SKILL_PUBLISH_REJECTED: {exc}[/]")
        raise typer.Exit(code=1) from None
    console.print(f"Published {published.skill_id} {published.bundle_sha256}")


@qd_skills_app.command("revoke")
def qd_skills_revoke(
    skill_id: str = typer.Argument(...),
    reason: str = typer.Option(..., "--reason"),
    actor: str = typer.Option("local-user", "--actor"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", exists=True, file_okay=False),
) -> None:
    try:
        _, publisher = _publisher(Workspace(workspace))
        publisher.revoke(skill_id=skill_id, reason=reason, actor=actor)
    except (OSError, ValueError) as exc:
        console.print(f"[red]SKILL_REVOKE_REJECTED: {exc}[/]")
        raise typer.Exit(code=2) from None
    console.print(f"Revoked {skill_id}")
