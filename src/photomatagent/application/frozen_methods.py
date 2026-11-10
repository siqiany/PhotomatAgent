"""No-install frozen method adapter for isolated evaluations."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Any
from photomatagent.application.skills import LocalSkillService, SkillReplay
from photomatagent.application.errors import ServiceError
from photomatagent.capability_acquisition.models import EnvironmentRecord
from photomatagent.capability_acquisition.sandbox import SandboxBackend, SandboxService


class FrozenBackend:
    def __init__(self, backend: Any, environment: EnvironmentRecord) -> None:
        self.backend = backend
        self.environment = environment
        from photomatagent.capability_acquisition.models import InstallPlan
        self.plan: InstallPlan
    async def install(self, plan: Any) -> EnvironmentRecord:
        if plan.plan_sha256 != self.environment.plan_sha256:
            raise ValueError('FROZEN_ENVIRONMENT_MISMATCH')
        return self.environment
    async def run(self, request: Any, *, output_directory: Path) -> Any:
        return await self.backend.run(request, output_directory=output_directory)
    async def reconcile(self, execution_id: str) -> Any:
        return await self.backend.reconcile(execution_id)


class FrozenEvaluationSkills(LocalSkillService):
    frozen_evaluation = True
    def __init__(self, root: Path, source: LocalSkillService, skill_ids: list[str], *, allow_unqualified: bool = False, environment_method_ids: list[str] | None = None) -> None:
        self.allow_unqualified = allow_unqualified
        self.allowed: dict[str, str] = {}
        self.qualified_reproductions: set[str] = set()
        backends: dict[str, FrozenBackend] = {}
        for skill_id in dict.fromkeys(skill_ids + (environment_method_ids or [])):
            item = source.read(skill_id)
            if skill_id in skill_ids:
                self.allowed[skill_id] = item['contract_sha256']
                if item.get('reproduction') and source._reproduction_qualified(item):
                    self.qualified_reproductions.add(item['contract_sha256'])
                destination = root / '.photomatagent/local-skills' / skill_id
                destination.mkdir(parents=True, exist_ok=True)
                for name in ('candidate.json', 'uses.jsonl', *item['files']):
                    path = source._directory(skill_id) / name
                    if path.exists():
                        target = destination / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
            provenance = json.loads((source.workspace.root / '.photomatagent/executions' / item['source']['execution_id'] / 'request.json').read_text())
            environment = EnvironmentRecord.model_validate(provenance['environment'])
            if source._custom_backend_factory:
                backend = source.backend_factory(source._directory(skill_id) / 'environment', item['plan']['execution_backend'])
            elif item['plan']['execution_backend'] == 'local_dev':
                from photomatagent.capability_acquisition.local_backend import LocalDevBackend
                backend = LocalDevBackend(source.workspace.root / '.photomatagent/capabilities' / item['source']['task_id'])
                restored = backend.restore_environment(environment.environment_id)
                if restored.environment_sha256 != environment.environment_sha256:
                    raise ValueError('FROZEN_ENVIRONMENT_CHANGED')
            else:
                from photomatagent.scientific.evolution.qd.config import QDConfig
                from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend
                config = QDConfig.model_validate_json((source.workspace.root / '.photomatagent/qd.json').read_text())
                backend = PodmanSandboxBackend(source.workspace.resolve(config.environment_storage))
                restored = backend.restore_environment(environment.environment_id)
                if restored.environment_sha256 != environment.environment_sha256:
                    raise ValueError('FROZEN_ENVIRONMENT_CHANGED')
            frozen_backend=FrozenBackend(backend,environment)
            from photomatagent.capability_acquisition.models import InstallPlan
            from datetime import datetime, timezone, timedelta
            frozen_backend.plan=InstallPlan.model_validate(item['plan'] | {'expires_at':datetime.now(timezone.utc)+timedelta(days=1)})
            backends[environment.plan_sha256]=frozen_backend
        self.frozen_backends = backends
        super().__init__(root, backend_factory=lambda path, kind: _SelectedFrozenBackend(backends))

    def _reproduction_qualified(self, data: dict[str, Any]) -> bool:
        return data['contract_sha256'] in self.qualified_reproductions

    def _source_tree_backend(self, sandbox: SandboxService, environment: EnvironmentRecord) -> SandboxBackend | None:
        frozen = self.frozen_backends[environment.plan_sha256]
        if frozen.environment != environment:
            raise ServiceError('FROZEN_ENVIRONMENT_MISMATCH', '冻结环境与方法合同不同。')
        return frozen.backend

    def generic_sandbox(self, task_id: str, run_id: str) -> Any:
        import hashlib
        from photomatagent.capability_acquisition.sandbox import SandboxService
        from photomatagent.capability_acquisition.models import InstallPlan
        from datetime import datetime, timezone, timedelta
        sandbox=SandboxService(workspace=self.workspace.root,trial_id=run_id,genome_sha256=hashlib.sha256(run_id.encode()).hexdigest(),
            scripts={},dynamic_script_prefix=f'tmp/workbench/{task_id}',backend=_SelectedFrozenBackend(self.frozen_backends))
        for frozen in self.frozen_backends.values():
            # Already installed immutable environment. This never authorizes a
            # download or host package installation in held-out evaluation.
            sandbox.register_environment(frozen.environment,plan=frozen.plan)
        return sandbox

    async def replay(self, skill_id: str, payload: SkillReplay, *, model_requested: bool = False, run_id: str | None = None) -> dict[str, Any]:
        if skill_id not in self.allowed or self.read(skill_id)['contract_sha256'] != self.allowed[skill_id]:
            raise ServiceError('FROZEN_METHOD_NOT_ALLOWED', '评测只能调用清单中冻结的方法版本。')
        return await super().replay(skill_id, payload, model_requested=model_requested and not self.allow_unqualified, run_id=run_id)


class _SelectedFrozenBackend:
    def __init__(self, backends: dict[str, FrozenBackend]) -> None:
        self.backends = backends
        self.selected: FrozenBackend | None = None
    async def install(self, plan: Any) -> EnvironmentRecord:
        self.selected = self.backends[plan.plan_sha256]
        return await self.selected.install(plan)
    async def run(self, request: Any, *, output_directory: Path) -> Any:
        backend=next((value for value in self.backends.values() if value.environment.environment_id==request.environment_id),None)
        if backend is None:raise ValueError('FROZEN_ENVIRONMENT_UNBOUND')
        self.selected=backend
        return await backend.run(request,output_directory=output_directory)
    async def reconcile(self, execution_id: str) -> Any:
        return await self.selected.reconcile(execution_id) if self.selected else None
