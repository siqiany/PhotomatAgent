"""One production composition root for the unified VASP lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.vasp.application import VaspApplication
from photomatagent.scientific.applications.vasp.molecular.runtime import (
    default_molecular_runtime,
)
from photomatagent.scientific.applications.vasp.unified.approvals import (
    ApprovalReceiptStore,
)
from photomatagent.scientific.applications.vasp.unified.molecular import (
    MolecularVaspExecutorAdapter,
)
from photomatagent.scientific.applications.vasp.unified.models import (
    VaspWorkflowKind,
)
from photomatagent.scientific.applications.vasp.unified.periodic import (
    PeriodicVaspExecutor,
)
from photomatagent.scientific.applications.vasp.unified.repository import (
    ManifestRepository,
)
from photomatagent.scientific.applications.vasp.unified.resources import (
    ResourceAuthorizationService,
)
from photomatagent.scientific.applications.vasp.unified.router import (
    UnifiedVaspRouter,
)
from photomatagent.scientific.applications.vasp.unified.service import (
    UnifiedVaspService,
)
from photomatagent.scientific.applications.vasp.unified.study import (
    VaspStudyExecutorAdapter,
)
from photomatagent.scientific.applications.vasp.unified.tool_pack import (
    VaspUnifiedCapabilityPack,
)
from photomatagent.workspace import Workspace


@dataclass(frozen=True)
class UnifiedVaspGraph:
    """Named owner-scoped unified VASP service graph."""

    workspace: Workspace
    repository: ManifestRepository
    approvals: ApprovalReceiptStore
    resources: ResourceAuthorizationService
    router: UnifiedVaspRouter
    service: UnifiedVaspService
    periodic: PeriodicVaspExecutor | None
    molecular: MolecularVaspExecutorAdapter
    study: VaspStudyExecutorAdapter
    tool_pack: VaspUnifiedCapabilityPack


def build_unified_vasp_graph(
    *,
    application: VaspApplication | None,
    workspace: Workspace | Path | str,
    approval_root: Path | str | None = None,
) -> UnifiedVaspGraph:
    """Build the single graph shared by one pack or MCP server owner.

    This keeps VASP's public surfaces as adapters over the same application
    service and deliberately preserves an unconfigured molecular/study graph.
    """
    resolved_workspace = (
        workspace
        if isinstance(workspace, Workspace)
        else Workspace(Path(workspace))
    )
    repository = ManifestRepository(resolved_workspace)
    approvals = ApprovalReceiptStore(approval_root or resolved_workspace.root)
    resources = ResourceAuthorizationService(
        approvals, policy=application.policy if application is not None else None
    )

    periodic = None
    if application is not None and application.backend is not None:
        periodic_application = VaspApplication(
            application.backend,
            workspace=resolved_workspace.root,
            psp_dir=str(application.generator.psp_dir),
            jobs_local_dir=application.generator.jobs_local_dir,
            policy=application.policy,
            module_name=application.module_name,
            env_script=application.env_script,
            remote_root=application.remote_root,
            remote_psp_dir=application.remote_psp_dir,
        )
        periodic = PeriodicVaspExecutor(periodic_application)

    molecular_runtime = default_molecular_runtime(
        resolved_workspace.root, application=application
    )
    molecular = MolecularVaspExecutorAdapter(
        molecular_runtime, workspace=resolved_workspace
    )
    router = UnifiedVaspRouter(periodic=periodic, molecular=molecular)
    service = UnifiedVaspService(repository, approvals, router, resources=resources)
    study = VaspStudyExecutorAdapter(molecular_runtime, child_service=service)
    router.register(VaspWorkflowKind.STUDY, study)
    tool_pack = VaspUnifiedCapabilityPack(service)
    return UnifiedVaspGraph(
        workspace=resolved_workspace,
        repository=repository,
        approvals=approvals,
        resources=resources,
        router=router,
        service=service,
        periodic=periodic,
        molecular=molecular,
        study=study,
        tool_pack=tool_pack,
    )
