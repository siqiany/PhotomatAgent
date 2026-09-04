"""Runtime assembly for the isolated-molecule VASP tool surface.

The agent never builds ``JobRegistry`` / ``SubmitOnceSession`` /
``MolecularVaspTools`` by hand with bash. This factory assembles one
workspace-scoped runtime from the environment:

    * the configured SCNet backend (``default_vasp_application``),
    * a local SQLite :class:`JobRegistry` under the workspace,
    * a :class:`SubmitOnceSession` bound to that registry + backend,
    * the pseudopotential configuration (``PMG_VASP_PSP_DIR`` /
      ``SCNET_VASP_PSP_DIR``),
    * workspace-scoped workflow / log / registry paths.

When SCNet is unconfigured the runtime is ``configured=False``: offline tools
(prepare/preflight/analyze_*) still work, while any submission tool reports a
typed ``missing_prerequisites`` diagnostic instead of silently touching a
fallback backend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.vasp.application import (
    VaspApplication,
    default_vasp_application,
)
from photomatagent.scientific.applications.vasp.molecular.tools import (
    MolecularVaspTools,
)
from photomatagent.scientific.remote.fake import FakeSCNetBackend
from photomatagent.scientific.remote.lifecycle import SubmitOnceSession
from photomatagent.scientific.remote.models import ResourcePolicy
from photomatagent.scientific.remote.registry import JobRegistry


class MolecularVaspRuntime:
    """One assembled, workspace-scoped molecular VASP runtime.

    Construction is cheap: the SQLite registry and session are created
    lazily on first submission so offline discovery never touches disk state
    beyond the workspace paths below.
    """

    def __init__(
        self,
        *,
        backend: Any = None,
        application: VaspApplication | None = None,
        configured: bool = True,
        psp_dir: str | Path | None = None,
        workflow_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        registry_path: str | Path | None = None,
        module_name: str = "",
        env_script: str = "",
        remote_psp_dir: str = "",
        remote_root: str = "~/photomatagent",
    ) -> None:
        self.application = application
        self.configured = configured and backend is not None
        self.backend = backend
        self.psp_dir = Path(psp_dir).expanduser().resolve() if psp_dir else None
        self.workflow_dir = (
            Path(workflow_dir).expanduser().resolve() if workflow_dir else None
        )
        self.log_dir = (
            Path(log_dir).expanduser().resolve() if log_dir else None
        )
        self._registry_path = (
            Path(registry_path).expanduser().resolve()
            if registry_path
            else None
        )
        self.module_name = module_name
        self.env_script = env_script
        self.remote_psp_dir = remote_psp_dir
        self.remote_root = remote_root.rstrip("/")
        self._session: SubmitOnceSession | None = None
        self._facade: MolecularVaspTools | None = None

    # -- lazy state --------------------------------------------------------

    @property
    def registry_path(self) -> Path:
        if self._registry_path is None:
            base = (
                self.workflow_dir.parent
                if self.workflow_dir is not None
                else Path.cwd()
            )
            self._registry_path = (
                base / ".photomatagent" / "state" / "jobs.sqlite3"
            )
        return self._registry_path

    def registry(self) -> JobRegistry:
        return JobRegistry(self.registry_path)

    @property
    def session(self) -> SubmitOnceSession:
        if self._session is None:
            backend = self.backend
            if backend is None:
                # Submission-facing tools refuse at the tool layer when not
                # configured; this strict stub is only a safe last resort.
                backend = FakeSCNetBackend(
                    policy=ResourcePolicy(allow_hpc_submit=False), strict=True
                )
            self._session = SubmitOnceSession(
                self.registry(), backend, remote_root=self.remote_root
            )
        return self._session

    def facade(self) -> MolecularVaspTools:
        if self._facade is None:
            self._facade = MolecularVaspTools(
                session=self.session,
                backend=self.backend,
                psp_dir=self.psp_dir,
                workflow_dir=self.workflow_dir,
                log_dir=self.log_dir,
                module_name=self.module_name,
                env_script=self.env_script,
                remote_psp_dir=self.remote_psp_dir,
                configured=self.configured,
            )
        return self._facade

    def capabilities_payload(self) -> dict[str, Any]:
        from photomatagent.scientific.applications.vasp.molecular.slurm import (
            local_potcar_materializable,
        )
        from photomatagent.scientific.applications.vasp.psp import (
            resolve_local_psp_library,
        )

        resolved_local = (
            resolve_local_psp_library(self.psp_dir)
            if self.psp_dir is not None
            else None
        )
        if self.remote_psp_dir:
            selected_potcar_mode = "remote"
        elif local_potcar_materializable(self.psp_dir):
            selected_potcar_mode = "local"
        else:
            selected_potcar_mode = "none"
        return {
            "configured": self.configured,
            "psp_dir": str(self.psp_dir) if self.psp_dir else None,
            "psp_layout": (
                resolved_local[1] if resolved_local is not None else None
            ),
            "psp_resolved_library": (
                str(resolved_local[0]) if resolved_local is not None else None
            ),
            "module": self.module_name or None,
            "env_script_configured": bool(self.env_script),
            "remote_psp_dir_configured": bool(self.remote_psp_dir),
            "workflow_dir": str(self.workflow_dir) if self.workflow_dir else None,
            "log_dir": str(self.log_dir) if self.log_dir else None,
            "registry": str(self.registry_path),
            "remote_root": self.remote_root,
            "selected_potcar_mode": selected_potcar_mode,
            "potcar_policy": (
                "remote: POTCAR assembled on SCNet from SCNET_VASP_PSP_DIR in "
                "POSCAR element order; local: POTCAR assembled from the "
                "resolved local PAW-PBE library and uploaded to the unique "
                "remote job directory only; none: submission refused. "
                "POTCAR content is never logged or returned"
            ),
        }


def default_molecular_runtime(
    workspace: str | Path | None = None,
    *,
    application: VaspApplication | None = None,
) -> MolecularVaspRuntime:
    """Build the runtime from the process environment and workspace root."""
    workspace_path = (
        Path(workspace).expanduser().resolve() if workspace else Path.cwd()
    )
    app = application if application is not None else default_vasp_application()
    backend = getattr(app, "backend", None) if app is not None else None
    configured = app is not None and backend is not None
    psp_dir = (
        os.environ.get("PMG_VASP_PSP_DIR", "").strip()
        or (str(app.generator.psp_dir) if app is not None else "")
        or None
    )
    module_name = (
        os.environ.get("SCNET_VASP_MODULE", "").strip()
        or (app.module_name if app is not None else "")
    )
    env_script = (
        os.environ.get("SCNET_VASP_ENV_SCRIPT", "").strip()
        or (app.env_script if app is not None else "")
    )
    remote_psp_dir = (
        os.environ.get("SCNET_VASP_PSP_DIR", "").strip()
        or (app.remote_psp_dir if app is not None else "")
    )
    remote_root = (
        os.environ.get("SCNET_REMOTE_ROOT", "").strip()
        or (app.remote_root if app is not None else "")
        or "~/photomatagent"
    )
    return MolecularVaspRuntime(
        application=app,
        backend=backend,
        configured=configured,
        psp_dir=psp_dir,
        workflow_dir=workspace_path / "output" / "vasp_molecule",
        log_dir=workspace_path / "output" / "molecule_logs",
        registry_path=workspace_path / ".photomatagent" / "state" / "jobs.sqlite3",
        module_name=module_name,
        env_script=env_script,
        remote_psp_dir=remote_psp_dir,
        remote_root=remote_root,
    )
