"""Task 4: user-only VASP decision approval command."""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from typer.testing import CliRunner

from photomatagent.cli.app import app
from photomatagent.cli.commands import ChatCommandRouter
from photomatagent.models.fake import FakeModelProvider
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import (
    ApprovalScope,
    ApprovalSettings,
    DenyAllPolicy,
    SwitchablePermissionPolicy,
)
from photomatagent.scientific.applications.vasp.unified.approvals import (
    ApprovalKind,
    ApprovalReceiptStore,
    pending_decision,
)
from photomatagent.scientific.applications.vasp.unified.fingerprints import scientific_fingerprint
from photomatagent.scientific.applications.vasp.unified.models import (
    PeriodicScientificSpec,
    UnifiedStage,
    UnifiedVaspManifest,
    VaspWorkflowKind,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace


def seed_pending(tmp_path, decision_id: str = "dec-1") -> None:
    store = ApprovalReceiptStore(tmp_path)
    spec = PeriodicScientificSpec(structure_path="structure.cif", profile="standard_semiconductor")
    manifest = UnifiedVaspManifest(
        workflow_id="wf-1",
        workflow_kind=VaspWorkflowKind.PERIODIC,
        scientific_spec=spec,
        scientific_fingerprint=scientific_fingerprint(spec),
        execution_fingerprint="e" * 64,
        stages=[UnifiedStage(name="relax", execution_fingerprint="e" * 64)],
    )
    store.record_pending(
        pending_decision(
            manifest=manifest,
            kind=ApprovalKind.SCIENTIFIC,
            summary="Change ENCUT",
            stage="relax",
            decision_id=decision_id,
            changes=[{
                "parameter": "encut_ev",
                "old_value": 500,
                "new_value": 520,
                "reason": "convergence",
            }],
        )
    )


def receipt_count(tmp_path) -> int:
    store = ApprovalReceiptStore(tmp_path)
    with store._connect() as connection:
        return connection.execute(
            "SELECT COUNT(*) AS n FROM approval_receipts"
        ).fetchone()["n"]


def test_unknown_decision_ids_fail_without_writing_receipt(tmp_path):
    result = CliRunner().invoke(
        app,
        ["scientific", "approve", "missing-decision", "--workspace", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "unknown pending decision" in result.output
    assert receipt_count(tmp_path) == 0


def test_declining_confirmation_writes_nothing(tmp_path):
    seed_pending(tmp_path)
    result = CliRunner().invoke(
        app,
        ["scientific", "approve", "dec-1", "--workspace", str(tmp_path)],
        input="n\n",
    )
    assert result.exit_code == 0
    assert "已取消" in result.output
    assert receipt_count(tmp_path) == 0


def test_approval_records_local_user_session_source(tmp_path):
    seed_pending(tmp_path)
    result = CliRunner().invoke(
        app,
        ["scientific", "approve", "dec-1", "--workspace", str(tmp_path)],
        input="y\n",
    )
    assert result.exit_code == 0
    assert "已批准" in result.output
    assert receipt_count(tmp_path) == 1
    store = ApprovalReceiptStore(tmp_path)
    with store._connect() as connection:
        row = connection.execute(
            "SELECT approved_by FROM approval_receipts WHERE decision_id='dec-1'"
        ).fetchone()
    assert row["approved_by"].startswith("local:")


def test_approve_always_does_not_approve_pending_scientific_decisions(tmp_path):
    seed_pending(tmp_path)
    workspace = Workspace(tmp_path)
    scientific = ScientificState()
    policy = SwitchablePermissionPolicy(
        DenyAllPolicy(), settings=ApprovalSettings(tmp_path)
    )
    runtime = AgentRuntime(
        model=FakeModelProvider([]),
        tools=create_default_registry(scientific, workspace),
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=policy,
    )
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120)
    router = ChatCommandRouter(console, runtime, workspace)

    import asyncio

    asyncio.run(router.execute("/approve -a"))

    # Runtime allow-all is not an application-level scientific approval.
    assert policy.scope is ApprovalScope.ALWAYS
    assert receipt_count(tmp_path) == 0
